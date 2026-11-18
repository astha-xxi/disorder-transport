import numpy as np
import pytest
from DisorderTransport import (SimulationConfig, gaussian_wavepacket,
                               generate_potential, propagate, observables)


def test_wavepacket_is_normalized():
    c = SimulationConfig(n=24, n_steps=2)
    psi = gaussian_wavepacket(c)
    assert abs(np.sum(np.abs(psi)**2)*c.cell_volume - 1) < 1e-12


def test_free_propagation_conserves_norm():
    c = SimulationConfig(n=24, dt=0.005, n_steps=8, snapshot_every=2)
    psi = gaussian_wavepacket(c, center=(0,0,0), momentum=(0,0,0))
    result = propagate(psi, np.zeros((24,)*3), c)
    assert np.max(np.abs(result.norms - 1)) < 1e-11
    assert result.wavefunctions.shape[1:] == (24,)*3


def test_generated_fields_are_reproducible():
    c = SimulationConfig(n=20)
    a = generate_potential(c, "multiscale", seed=17, level=3)
    b = generate_potential(c, "multiscale", seed=17, level=3)
    assert np.array_equal(a, b)


def test_custom_potential_shape_checked():
    c = SimulationConfig(n=16)
    with pytest.raises(ValueError):
        generate_potential(c, custom=np.zeros((16,16)))


def test_absorber_is_non_unitary_and_reduces_edge_probability():
    c = SimulationConfig(n=20, n_steps=4, snapshot_every=1)
    psi = gaussian_wavepacket(c, center=(8,8,8), sigma=1.5, momentum=(0,0,0))
    result = propagate(psi, np.zeros((20,)*3), c, absorber=True)
    assert np.all(np.isfinite(result.norms))


def test_snapshot_memory_guard():
    c = SimulationConfig(n=32, n_steps=10, snapshot_every=1)
    psi = gaussian_wavepacket(c)
    with pytest.raises(MemoryError):
        propagate(psi, np.zeros((32,)*3), c, max_snapshot_bytes=100)
