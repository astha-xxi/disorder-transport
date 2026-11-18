"""DisorderTransport: three-dimensional split-step Fourier quantum dynamics.

The solver uses dimensionless units unless the caller supplies a consistent
physical unit system: hbar, mass, length, energy, and time must agree.
The Fourier kinetic step implies periodic boundaries. An optional smooth
absorbing layer can reduce wrap-around, at the cost of probability norm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import fftn, ifftn

Array = np.ndarray


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical and physical parameters for a cubic 3-D simulation."""
    n: int = 64
    length: float = 40.0
    dt: float = 0.01
    n_steps: int = 300
    snapshot_every: int = 5
    hbar: float = 1.0
    mass: float = 1.0

    def __post_init__(self):
        if self.n < 16:
            raise ValueError("n must be at least 16")
        if self.length <= 0 or self.dt <= 0 or self.mass <= 0 or self.hbar <= 0:
            raise ValueError("length, dt, mass, and hbar must be positive")
        if self.n_steps < 0 or self.snapshot_every < 1:
            raise ValueError("n_steps must be nonnegative and snapshot_every positive")

    @property
    def dx(self) -> float:
        return self.length / self.n

    @property
    def cell_volume(self) -> float:
        return self.dx ** 3


@dataclass
class SimulationResult:
    """Saved wavefunction snapshots and diagnostics."""
    times: Array
    wavefunctions: Array  # (n_frames, n, n, n), complex
    norms: Array
    msd: Array
    ipr: Array
    edge_probability: Array

    @property
    def densities(self) -> Array:
        return np.abs(self.wavefunctions) ** 2


def coordinate_grid(config: SimulationConfig):
    """Return x, y, z coordinate arrays and a 3-D mesh (ij indexing)."""
    x = np.linspace(-config.length / 2, config.length / 2, config.n, endpoint=False)
    return np.meshgrid(x, x, x, indexing="ij")


def gaussian_wavepacket(config: SimulationConfig, center=(-10., 0., 0.),
                        sigma=2., momentum=(2., 0., 0.)) -> Array:
    """Create a normalized 3-D Gaussian packet with a plane-wave phase."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    X, Y, Z = coordinate_grid(config)
    r2 = (X-center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2
    psi = np.exp(-r2 / (4 * sigma**2)).astype(np.complex128)
    psi *= np.exp(1j * (momentum[0]*X + momentum[1]*Y + momentum[2]*Z))
    norm = np.sqrt(np.sum(np.abs(psi)**2) * config.cell_volume)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("wavepacket has invalid norm")
    return psi / norm


def _standardize(field: Array) -> Array:
    field = np.asarray(field, dtype=float)
    field = field - np.mean(field)
    sd = np.std(field)
    return field / sd if sd > 0 else np.zeros_like(field)


def generate_potential(config: SimulationConfig, family="multiscale", *,
                       seed=20260903, level=4, amplitude=0.8,
                       custom: Optional[Array] = None) -> Array:
    """Generate a 3-D random, multiscale, binary, or user-defined potential.

    ``level`` is an index from 0 to 7 controlling correlation/smoothing; it is
    not itself a universal complexity measure. ``custom`` must be an n-cubed
    array. Binary fields retain sharp interfaces; no hidden smoothing is
    applied after thresholding.
    """
    if amplitude < 0:
        raise ValueError("amplitude must be nonnegative")
    if custom is not None:
        v = np.asarray(custom, dtype=float)
        if v.shape != (config.n,)*3 or not np.all(np.isfinite(v)):
            raise ValueError("custom potential must be a finite (n,n,n) array")
        return v.copy()
    if family not in {"random", "multiscale", "binary", "ordered"}:
        raise ValueError(f"unknown potential family: {family}")
    if not 0 <= level <= 7:
        raise ValueError("level must be between 0 and 7")
    rng = np.random.default_rng(seed)
    if family == "ordered":
        X, Y, Z = coordinate_grid(config)
        return amplitude * (0.5 + 0.5*np.cos(2*np.pi*X/5))
    q = level / 7
    if family == "random":
        field = rng.normal(size=(config.n,)*3)
        field = gaussian_filter(field, sigma=0.5 + 4*q, mode="wrap")
    elif family == "multiscale":
        scales = (0.5 + 2*q, 1.0 + 5*q, 2.0 + 8*q)
        weights = (1-q, q, 0.7*q)
        field = sum(w*gaussian_filter(rng.normal(size=(config.n,)*3), s, mode="wrap")
                    for w, s in zip(weights, scales))
    else:
        field = gaussian_filter(rng.normal(size=(config.n,)*3), 0.5 + 5*q, mode="wrap")
        field = (field > np.median(field)).astype(float)
    return amplitude * _standardize(field)


def _kinetic_phase(config: SimulationConfig) -> Array:
    k = 2*np.pi*np.fft.fftfreq(config.n, d=config.dx)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    k2 = kx*kx + ky*ky + kz*kz
    return np.exp(-1j * config.dt * config.hbar * k2 / (2*config.mass))


def _absorbing_mask(config: SimulationConfig, width_fraction=0.12, strength=0.15) -> Array:
    """Smooth edge absorber. Applying it makes propagation non-unitary."""
    if not 0 < width_fraction < 0.5 or strength < 0:
        raise ValueError("invalid absorber parameters")
    u = np.abs(np.linspace(-1, 1, config.n, endpoint=False))
    start = 1 - 2*width_fraction
    ramp = np.clip((u-start)/(1-start), 0, 1)
    one_d = np.exp(-strength*ramp**4)
    return one_d[:, None, None] * one_d[None, :, None] * one_d[None, None, :]


def observables(psi: Array, config: SimulationConfig, edge_cells=3):
    """Calculate norm, 3-D MSD, IPR, and probability in edge cells."""
    rho = np.abs(psi)**2
    dV = config.cell_volume
    norm = float(np.sum(rho)*dV)
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("wavefunction has invalid probability norm")
    # Marginal densities avoid allocating three additional n-cubed coordinate grids.
    x = np.linspace(-config.length/2, config.length/2, config.n, endpoint=False)
    marginals = (rho.sum(axis=(1, 2)), rho.sum(axis=(0, 2)), rho.sum(axis=(0, 1)))
    means = [float(np.dot(x, marginal)*dV/norm) for marginal in marginals]
    second = [float(np.dot(x*x, marginal)*dV/norm) for marginal in marginals]
    msd = float(sum(s-m*m for s, m in zip(second, means)))
    ipr = float(np.sum(rho*rho)*dV)
    e = max(1, min(int(edge_cells), config.n//2))
    mask = np.zeros(rho.shape, dtype=bool)
    mask[:e] = mask[-e:] = True
    mask[:, :e] = mask[:, -e:] = True
    mask[:, :, :e] = mask[:, :, -e:] = True
    edge = float(np.sum(rho[mask])*dV)
    return norm, msd, ipr, edge


def propagate(psi0: Array, potential: Array, config: SimulationConfig, *,
              absorber=False, absorber_strength=0.15,
              callback: Optional[Callable[[int, float, Array], None]]=None,
              max_snapshot_bytes: int = 512 * 1024**2) -> SimulationResult:
    """Evolve TDSE with second-order Strang split-step Fourier propagation.

    The potential is static and real. FFT propagation imposes periodic
    boundaries. ``absorber=True`` multiplies by a smooth edge mask each step
    to reduce wraparound; norm loss then represents absorption, not numerical
    nonconservation. Callback receives (step, time, read-only psi view).
    """
    shape = (config.n,)*3
    psi = np.array(psi0, dtype=np.complex128, copy=True)
    v = np.asarray(potential, dtype=float)
    if psi.shape != shape or v.shape != shape:
        raise ValueError(f"psi0 and potential must both have shape {shape}")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(psi)):
        raise ValueError("inputs must contain only finite values")
    n_frames = config.n_steps // config.snapshot_every + 1
    if config.n_steps % config.snapshot_every:
        n_frames += 1
    estimated = n_frames * np.prod(shape) * np.dtype(np.complex128).itemsize
    if estimated > max_snapshot_bytes:
        raise MemoryError(f"Snapshots need about {estimated/1024**3:.2f} GiB; "
                           f"limit is {max_snapshot_bytes/1024**3:.2f} GiB. "
                           "Increase snapshot_every, reduce n, or raise max_snapshot_bytes.")
    phase_v = np.exp(-1j*v*config.dt/(2*config.hbar))
    phase_k = _kinetic_phase(config)
    mask = _absorbing_mask(config, strength=absorber_strength) if absorber else None
    frames, times, norms, msds, iprs, edges = [], [], [], [], [], []
    for step in range(config.n_steps+1):
        if step % config.snapshot_every == 0 or step == config.n_steps:
            norm, msd, ipr, edge = observables(psi, config)
            frames.append(psi.copy()); times.append(step*config.dt)
            norms.append(norm); msds.append(msd); iprs.append(ipr); edges.append(edge)
        if step == config.n_steps:
            break
        psi *= phase_v
        psi = ifftn(fftn(psi, workers=-1)*phase_k, workers=-1)
        psi *= phase_v
        if mask is not None:
            psi *= mask
        if callback is not None:
            callback(step+1, (step+1)*config.dt, psi)
    return SimulationResult(np.asarray(times), np.asarray(frames), np.asarray(norms),
                            np.asarray(msds), np.asarray(iprs), np.asarray(edges))


def export_result(result: SimulationResult, path: str) -> None:
    """Save compact numerical diagnostics and time series to compressed NPZ."""
    np.savez_compressed(path, times=result.times, norms=result.norms,
                        msd=result.msd, ipr=result.ipr,
                        edge_probability=result.edge_probability)
