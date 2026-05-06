# SCM triplet diagnostic campaign

This folder contains a complete set of Python scripts for diagnostic calculations of three dielectric spheres in a rotating electric field using the validated **matrix SCM** implementation.

The goal is not yet to produce the final BEM benchmark. The goal is to identify which triplet geometries and distances are physically interesting for the article: where dipoles are insufficient, where higher multipoles matter, and where the non-pairwise three-body contribution is strongest.

## Physical setup

Spherical dielectric particles:

- particle permittivity: `eps1_r = 3.9`
- medium permittivity: `eps2_r = 81.0`
- radius: `a = 1.0e-6 m`
- diameter: `d = 2a`
- field magnitude: `E0 = 1.0e5 V/m`

Rotating field in the `xy` plane:

```text
E_k = E0 (cos theta_k, sin theta_k, 0)
theta_k = 2*pi*k/n_orient
n_orient = 8
```

SCM multipole orders:

```text
l_max = 1, 2, 3, 4, 5, 6
```

`l_max=1` is the self-consistent dipole level. Larger `l_max` values include quadrupoles and higher multipoles.

## What is calculated

For a cluster of three particles:

```text
Phi3 = U3 - 3 U1
```

where `U3` is the full cluster energy and `U1` is the single-sphere energy in the same field orientation and at the same `l_max`.

The pairwise estimate is

```text
Phi_pairwise = sum_{i<j} phi_pair(r_ij)
```

where

```text
phi_pair(r) = U2(r) - 2 U1.
```

The non-pairwise three-body contribution is

```text
Delta3 = Phi3 - Phi_pairwise.
```

This is the key physical diagnostic. If `Delta3` is small, the triplet is nearly pair-additive. If it is large, the configuration has a genuine many-body polarization contribution.

The scripts also save local particle contributions `Phi_i = U_i - U1`. These are useful diagnostics, especially for the central particle in the linear and angular triplets, but the strict physical conclusions should be based on `Phi3` and `Delta3`.

## Geometry families

### 1. Linear triplet

```text
r1 = (-r, 0, 0)
r2 = ( 0, 0, 0)  central particle
r3 = (+r, 0, 0)
```

Here `r` is the central-neighbor distance. Pair distances are `r`, `r`, and `2r`.

Distance grid:

```text
r/d = 1.05, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00
```

Pairwise estimate:

```text
Phi_pairwise_linear = 2 phi_pair(r) + phi_pair(2r)
```

### 2. Equilateral triplet

Three particles at the vertices of an equilateral triangle with edge `r` and center of mass at the origin.

Distance grid is the same as for the linear triplet.

Pairwise estimate:

```text
Phi_pairwise_equilateral = 3 phi_pair(r)
```

After phase averaging, the three local particle contributions should be nearly equal. This is used as a symmetry check.

### 3. Angular triplet

```text
r1 = (0, 0, 0)  central particle
r2 = (r, 0, 0)
r3 = (r cos gamma, r sin gamma, 0)
```

The neighbor-neighbor distance is

```text
r23 = 2 r sin(gamma/2)
```

Distance grid:

```text
r/d = 1.05, 1.20, 2.00
```

Angle grid:

```text
gamma = 60, 90, 120, 150, 180 degrees
```

Pairwise estimate:

```text
Phi_pairwise_angle = 2 phi_pair(r) + phi_pair(r23)
```

The endpoints are internal code checks:

- `gamma=60 deg` should match the equilateral triplet for the same `r/d`.
- `gamma=180 deg` should match the linear triplet for the same `r/d`.

## Files

### Core files

- `scm_core.py`  
  Reusable matrix SCM implementation and geometry builders.

- `scm_config.py`  
  Central configuration: physical parameters, grids, output paths.

### Run scripts

Run these in order:

```bash
python run_00_pair_cache.py
python run_01_triplet_linear.py
python run_02_triplet_equilateral.py
python run_03_triplet_angle.py
python analyze_triplet_results.py
```

### Output directory

All numerical results are saved to:

```text
results_triplet_scm/
```

Figures and CSV tables are saved to:

```text
results_triplet_scm/analysis_figures/
```

## Output NPZ files

### Pair cache

```text
results_triplet_scm/matrix_scm_pair_cache_lmax1_6.npz
```

Important arrays:

```python
r_pair_list              # physical pair distances [m]
r_pair_over_d            # pair distances / d
lmax_list                # [1,2,3,4,5,6]
U_single_lk              # shape (n_lmax, n_orient)
U_single_analytic_k      # shape (n_orient,)
U_pair_lrk               # shape (n_lmax, n_r_pair, n_orient)
U_pair_parts_lrkp        # shape (n_lmax, n_r_pair, n_orient, 2)
phi_pair_lrk             # U2 - 2U1 for every orientation
phi_pair_avg_lr          # phase-averaged pair excess energy
```

The pair cache includes all pair distances required by the triplets. For the linear triplet, it includes `2r`, so the cache reaches `r/d=12` even though the triplet scan reaches only `r/d=6`.

### Linear triplet

```text
results_triplet_scm/matrix_scm_triplet_linear_lmax1_6.npz
```

Important arrays:

```python
r_over_d
lmax_list
U_single_lk
U_total_lrk
U_parts_lrkp
Phi3_lrk
Phi3_lr
Phi_particles_lrkp
Phi_particles_lrp
Phi_center_lrk
Phi_center_lr
Phi_pairwise_lr
Phi_center_pairwise_lr
Phi_edge_pairwise_lr
Delta3_lr
eta3_pair_lr
eta3_sym_lr
```

Shapes:

```python
U_total_lrk.shape   == (n_lmax, n_r, n_orient)
U_parts_lrkp.shape  == (n_lmax, n_r, n_orient, 3)
Phi3_lr.shape       == (n_lmax, n_r)
Delta3_lr.shape     == (n_lmax, n_r)
```

### Equilateral triplet

```text
results_triplet_scm/matrix_scm_triplet_equilateral_lmax1_6.npz
```

Important arrays:

```python
r_over_d
lmax_list
U_single_lk
U_total_lrk
U_parts_lrkp
Phi3_lrk
Phi3_lr
Phi_particles_lrkp
Phi_particles_lrp
Phi_particles_maxdev_lr
Phi_pairwise_lr
Delta3_lr
eta3_pair_lr
eta3_sym_lr
```

`Phi_particles_maxdev_lr` is a symmetry diagnostic. It should be small after phase averaging.

### Angular triplet

```text
results_triplet_scm/matrix_scm_triplet_angle_lmax1_6.npz
```

Important arrays:

```python
r_over_d_values
gamma_deg
lmax_list
U_single_lk
U_total_lrgk
U_parts_lrgkp
Phi3_lrgk
Phi3_lrg
Phi_particles_lrgkp
Phi_particles_lrgp
Phi_center_lrgk
Phi_center_lrg
Phi_pairwise_lrg
Phi_center_pairwise_lrg
Phi_neighbor_pairwise_lrg
Delta3_lrg
eta3_pair_lrg
eta3_sym_lrg
r23_over_d_rg
```

Shapes:

```python
U_total_lrgk.shape  == (n_lmax, n_r, n_gamma, n_orient)
U_parts_lrgkp.shape == (n_lmax, n_r, n_gamma, n_orient, 3)
Phi3_lrg.shape      == (n_lmax, n_r, n_gamma)
Delta3_lrg.shape    == (n_lmax, n_r, n_gamma)
```

## Analysis figures

The analysis script creates figures such as:

```text
fig_pair_phi_vs_r.png
fig_linear_Phi3_vs_r.png
fig_linear_Delta3_vs_r.png
fig_linear_eta3_pair_vs_r.png
fig_linear_local_parts_lmax6.png
fig_equilateral_Phi3_vs_r.png
fig_equilateral_Delta3_vs_r.png
fig_equilateral_eta3_pair_vs_r.png
fig_equilateral_local_parts_lmax6.png
fig_equilateral_symmetry_error.png
fig_angle_Phi3_vs_gamma_lmax6.png
fig_angle_center_vs_gamma_lmax6.png
fig_angle_Delta3_vs_gamma_lmax6.png
fig_angle_eta3_pair_vs_gamma_lmax6.png
fig_convergence_lmax_linear_equilateral.png
summary_cross_checks.txt
```

## What to check after running

### 1. Single-sphere validation

In `run_00_pair_cache.py`, the printed ratio

```text
U1 / U1_analytic
```

should be close to 1. The single sphere is a basic check that the response normalization and energy convention are consistent.

### 2. Equilateral local symmetry

Open:

```text
fig_equilateral_symmetry_error.png
```

The local diagnostic contributions should become nearly equal after phase averaging. A visible discrepancy can indicate insufficient quadrature/projection accuracy or a bug in geometry/orientation averaging.

### 3. Geometry endpoint checks

Open:

```text
summary_cross_checks.txt
```

Expected:

```text
angle gamma=60 deg  ≈ equilateral
angle gamma=180 deg ≈ linear
```

The differences should be small compared with the absolute energy scale of the corresponding configuration.

### 4. Multipole convergence

Open:

```text
fig_convergence_lmax_linear_equilateral.png
```

For `r/d >= 2`, the increments `|Phi(lmax=6)-Phi(lmax=5)|` and `|Phi(lmax=5)-Phi(lmax=4)|` should be much smaller than the total energy. In the near zone `r/d=1.05...1.2`, larger differences are expected and physically meaningful.

### 5. Non-pairwise contribution

Open:

```text
fig_linear_Delta3_vs_r.png
fig_equilateral_Delta3_vs_r.png
fig_angle_Delta3_vs_gamma_lmax6.png
```

Expected physics:

- `Delta3` should be largest at small distances.
- `Delta3` should decay with increasing `r/d`.
- Angular geometry should strongly affect `Delta3`.
- The strongest candidates for later BEM validation are the points with large absolute `Delta3`, large relative nonadditivity, or poor `lmax` convergence.

## Recommended next step after diagnostics

After these SCM calculations, select a small set of BEM validation points. A reasonable first set is:

```text
linear:       r/d = 1.10, 1.20, 2.00
equilateral: r/d = 1.10, 1.20, 2.00
angle:        r/d = 1.20, gamma = 60, 90, 180 deg
```

The final selection should be based on the actual SCM diagnostics:

- maximum `|Delta3|`,
- maximum `eta3_pair`,
- strongest difference between `lmax=1` and `lmax=6`,
- weakest convergence between `lmax=5` and `lmax=6`.

## Notes on computational cost

The code reuses the transfer matrix for all field orientations at fixed geometry and `lmax`, so it is much faster than rebuilding the full matrix for every orientation.

The most expensive settings are:

```text
n_quad = 8000
lmax = 6
near-contact geometries r/d = 1.05
```

For a quick smoke test, you can temporarily reduce in `scm_config.py`:

```python
PARAMS = SCMParams(..., n_quad=1000)
LMAX_LIST = np.array([1, 2], dtype=int)
```

Do not use such reduced settings for final diagnostics.
