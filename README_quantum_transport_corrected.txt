QUANTUM TRANSPORT PROJECT: CORRECTED WORKING PACKAGE

Files
-----
quantum_transport_corrected_notebook.ipynb
    Corrected, runnable notebook. Run all cells from the top in a fresh kernel.

quantum_transport_raw_data_corrected.csv
    Newly regenerated data from 192 corrected simulations, with 51 time points per run (9,792 rows).

experiment_summary_corrected.csv
    One row per independent landscape realization (192 rows).

Figures
-------
landscape_overview.png
transport_by_smoothing_level.png
landscape_descriptors_by_smoothing.png
representative_simulation.png
wavepacket_evolution.png
spatial_variance_test.png
ipr_test.png

Major corrections
-----------------
1. The propagator now copies the input wavefunction. The old propagator modified psi0 in place. Because a single test run occurred before the 192-run experiment, the old ensemble used a test-evolved initial state rather than the intended original Gaussian.
2. Every realization now regenerates the intended Gaussian initial wavepacket.
3. Gaussian filtering uses periodic wrapping, consistent with the Fourier propagator's periodic boundary conditions.
4. The observable previously called MSD is now correctly called spatial variance. It is not mean-squared displacement from the initial position.
5. The eight experimental levels are called smoothing levels. Increasing the level increases spatial smoothing/correlation length; it does not increase complexity.
6. The landscape descriptors are recalculated with fixed-category entropy normalization and a multiscale residual normalized by the field standard deviation. Their equal-weight composite is explicitly exploratory, not a validated universal complexity measure.
7. The analysis emphasizes outcomes by smoothing level and reports within-level associations descriptively rather than relying on a pooled correlation that confounds the between-level trend.
8. An edge-probability diagnostic was added to monitor probability near the periodic boundaries.
9. Numerical checks were added: free-particle analytical benchmark, free-particle spatial-grid refinement, and a nonzero-potential time-step refinement test.

Corrected-run diagnostics
-------------------------
- 192 independent landscapes: 3 families x 8 smoothing levels x 8 realizations.
- 9,792 time-resolved rows: 51 snapshots per simulation.
- Maximum norm drift across the ensemble: approximately 1.38e-13.
- Maximum probability mass in the four-cell boundary band: approximately 0.0100.
- Free-particle benchmark at N=128, T=2: x-position error approximately 3.39e-6; total spatial-variance error approximately 7.54e-6.
- Free-particle spatial-grid refinement (N=64, 128, 256): spatial-variance errors approximately 1.06e-6, 7.54e-6, and 8.73e-6.
- One representative nonzero-potential time-step refinement (dt=0.01 versus dt=0.005, T=2): L2 state difference approximately 1.83e-5.

Interpretation and remaining limitations
----------------------------------------
The corrected ensemble is a valid regenerated computational dataset for the stated numerical setup, but it does not yet establish a general causal effect of "complexity" on transport.

The landscape families are not matched for correlation length, spectral structure, or other statistical properties. The exploratory composite descriptor is not independently validated. The time-step refinement is for one representative potential, and the grid refinement benchmark is a free-particle test, not a full grid-convergence study for every disordered landscape. The simulations use periodic boundaries. The final-time spatial variance and IPR do not by themselves establish Anderson localization.

The defensible current framing is a computational comparison of wavepacket transport across three families of spatially correlated potential landscapes, with smoothing level as the controlled generator parameter. Do not describe the results as proof of Anderson localization or as an isolated causal effect of structural complexity.
