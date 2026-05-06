"""
scm_config.py

Central configuration file for the SCM triplet diagnostic campaign.
Edit this file if you need to change physical parameters, distance grids, or output paths.
"""

import os
import numpy as np

from scm_core import SCMParams

# -----------------------------------------------------------------------------
# Output directory
# -----------------------------------------------------------------------------
OUT_DIR = "results_triplet_scm"
os.makedirs(OUT_DIR, exist_ok=True)

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

# l_max scan requested for diagnostics.
LMAX_LIST = np.array([1, 2, 3, 4, 5, 6], dtype=int)

# Main distance grid for linear and equilateral triplets.
R_OVER_D_MAIN = np.array(
    [1.05, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00],
    dtype=float,
)

# Angle scan: central particle + two neighbors at equal distance r.
R_OVER_D_ANGLE = np.array([1.05, 1.20, 2.00], dtype=float)
GAMMA_DEG = np.array([60.0, 90.0, 120.0, 150.0, 180.0], dtype=float)

# Files
PAIR_CACHE_FILE = os.path.join(OUT_DIR, "matrix_scm_pair_cache_lmax1_6.npz")
LINEAR_FILE = os.path.join(OUT_DIR, "matrix_scm_triplet_linear_lmax1_6.npz")
EQUILATERAL_FILE = os.path.join(OUT_DIR, "matrix_scm_triplet_equilateral_lmax1_6.npz")
ANGLE_FILE = os.path.join(OUT_DIR, "matrix_scm_triplet_angle_lmax1_6.npz")
ANALYSIS_DIR = os.path.join(OUT_DIR, "analysis_figures")
os.makedirs(ANALYSIS_DIR, exist_ok=True)
