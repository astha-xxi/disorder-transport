"""Minimal reproducible 3-D simulation."""
import numpy as np
from DisorderTransport import SimulationConfig, gaussian_wavepacket, generate_potential, propagate

cfg = SimulationConfig(n=32, length=40, dt=0.01, n_steps=50, snapshot_every=5)
V = generate_potential(cfg, "binary", seed=20261118, level=3, amplitude=0.8)
psi0 = gaussian_wavepacket(cfg, center=(-8, 0, 0), sigma=2, momentum=(1.5, 0, 0))
out = propagate(psi0, V, cfg, absorber=False)
print(f"frames={len(out.times)}")
print(f"max norm drift={np.max(np.abs(out.norms-out.norms[0])):.3e}")
print(f"final MSD={out.msd[-1]:.6g}; final IPR={out.ipr[-1]:.6g}")
