"""
scm3d_config.py

Configuration for the 3D orientation-dependent SCM map campaign.

This module is intended to be placed inside the existing repository
`article_scm_triplets/` next to `scm_core.py`.
Run scripts should be launched from the repository root as, for example:

    python -m scm_3d_maps.scripts.run_00_pair_orientation_map

Distances are stored in SI units in output arrays, and dimensionless r/d grids
are also saved for convenient plotting.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from scm_core import SCMParams

# -----------------------------------------------------------------------------
# Physical and numerical parameters
# -----------------------------------------------------------------------------
PARAMS = SCMParams(
    eps1_r=3.9,
    eps2_r=81.0,
    a=1.0e-6,
    E0=1.0e5,
    eps0=8.854187817e-12,
    n_orient=8,
    n_quad=8000,
)

LMAX_MAIN = 6
LMAX_LIST = np.array([1, 2, 3, 4, 5, 6], dtype=int)

# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------
OUT_DIR = Path("results_scm_3d_maps")
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = OUT_DIR / "logs"

for _p in (OUT_DIR, FIG_DIR, TABLE_DIR, LOG_DIR):
    _p.mkdir(parents=True, exist_ok=True)

PAIR_ORIENTATION_FILE = OUT_DIR / "scm_pair_orientation_map_lmax6.npz"
TRIANGLE_ISOSCELES_FILE = OUT_DIR / "scm_triangle_isosceles_map_lmax6.npz"
TRIANGLE_FULL_FILE = OUT_DIR / "scm_triangle_full_map_lmax6.npz"
# Backward-compatible name used by run_02 and analysis modules.
TRIANGLE_ASYMMETRIC_FILE = TRIANGLE_FULL_FILE
LMAX_CONVERGENCE_FILE = OUT_DIR / "scm_lmax_convergence_checks.npz"

# -----------------------------------------------------------------------------
# Stage 1: pair orientation map
# -----------------------------------------------------------------------------
R_PAIR_OVER_D = np.array(
    [1.10, 1.20, 1.35, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 8.00, 10.00],
    dtype=float,
)

BETA_DEG = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0], dtype=float)

# -----------------------------------------------------------------------------
# Stage 2: main isosceles triangle map, r12 = r13 = r
# -----------------------------------------------------------------------------
R_TRI_OVER_D = np.array(
    [1.10, 1.20, 1.35, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00],
    dtype=float,
)

GAMMA_DEG = np.array([60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0], dtype=float)

# psi: tilt of the triangle plane relative to the field-rotation plane xy.
PSI_DEG = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0], dtype=float)

# alpha: in-plane azimuth of the triangle before tilting. For psi=0 the result
# should be independent of alpha after phase averaging; for psi>0 alpha matters.
ALPHA_DEG = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0], dtype=float)

# -----------------------------------------------------------------------------
# Stage 3: full general-triangle map
# -----------------------------------------------------------------------------
# Full scan over distances from particle 1 to particles 2 and 3.
# Particle labels 2 and 3 are physically equivalent, but the full square is
# intentionally kept to allow a direct numerical symmetry check:
#     Delta3(r12,r13,...) ≈ Delta3(r13,r12,...)
# after the corresponding in-plane relabelling/orientation is taken into account.
FULL_R12_OVER_D = R_TRI_OVER_D.copy()
FULL_R13_OVER_D = R_TRI_OVER_D.copy()
FULL_GAMMA_DEG = GAMMA_DEG.copy()
FULL_PSI_DEG = PSI_DEG.copy()
FULL_ALPHA_DEG = ALPHA_DEG.copy()

# Backward-compatible aliases for the existing run_02 file name.
ASYM_R12_OVER_D = FULL_R12_OVER_D
ASYM_R13_OVER_D = FULL_R13_OVER_D
ASYM_GAMMA_DEG = FULL_GAMMA_DEG
ASYM_PSI_DEG = FULL_PSI_DEG
ASYM_ALPHA_DEG = FULL_ALPHA_DEG

# -----------------------------------------------------------------------------
# Convergence checks
# -----------------------------------------------------------------------------
CONV_PAIR_CASES = [
    # (r_over_d, beta_deg)
    (1.20, 0.0),
    (1.20, 45.0),
    (1.20, 90.0),
]

CONV_TRIANGLE_CASES = [
    # (r12_over_d, r13_over_d, gamma_deg, psi_deg, alpha_deg)
    (1.20, 1.20, 60.0, 0.0, 0.0),
    (1.20, 1.20, 60.0, 45.0, 0.0),
    (1.20, 1.20, 60.0, 90.0, 0.0),
    (1.20, 1.20, 90.0, 0.0, 0.0),
    (1.20, 1.20, 90.0, 45.0, 45.0),
    (1.20, 1.20, 90.0, 90.0, 90.0),
    (1.20, 1.20, 120.0, 0.0, 0.0),
    (1.20, 1.20, 120.0, 45.0, 45.0),
    (1.20, 1.20, 120.0, 90.0, 90.0),
    (1.20, 1.20, 180.0, 0.0, 0.0),
    (1.20, 1.20, 180.0, 45.0, 45.0),
    (1.20, 1.20, 180.0, 90.0, 90.0),
]
