# Original 2-D study: provenance and scope

The archived study, *The Role of Structural Complexity in Quantum Transport Across Artificial Landscapes*, is a two-dimensional TDSE computation. Its reference configuration is a 128 × 128 periodic grid over a length-40 domain, dt = 0.01, 500 steps, and snapshots every 10 steps. The initial state is a normalized Gaussian packet centered at (-10, 0), with width 2 and momentum (2, 0). The Hamiltonian uses hbar = mass = 1.

The ensemble comprises three landscape families (random, multiscale, binary), eight smoothing/correlation levels, and eight realizations per level: 3 × 8 × 8 = 192 independent generated landscapes. The base seed is 20260903, with deterministic offsets by family, level, and realization in the notebook.

The archived CSV files retain the corrected summary and time-series records. The original reported maximum probability-normalization error is approximately 1.394 × 10^-13 for the original computation. That figure is a result of the 2-D implementation and must not be attributed to the new 3-D solver.

The original notebook uses Fourier propagation, hence periodic boundaries. The code's Gaussian filtering for generated fields also uses periodic wrapping. The original study is finite in space and time; its correlations are associations in the sampled parameter regime, not proof of an asymptotic localization transition or a universal law. The smoothing/correlation level is a generator control, not itself a direct measurement of complexity. The composite descriptor was explicitly exploratory.

When publishing derived analyses, retain the raw data and notebook, state any filtering or exclusions, and distinguish the 192 generated landscapes from the 9,792 time-series rows (51 saved time points per landscape).
