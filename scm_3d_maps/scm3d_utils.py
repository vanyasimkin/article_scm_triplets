"""
scm3d_utils.py

Shared numerical utilities for 3D SCM maps.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from scm_core import MatrixSCMSystem, analytic_single_sphere_bem_like_energy, rotating_field_k, timestamp


def compute_single_energy_k(lmax: int, normals: np.ndarray, params) -> Tuple[np.ndarray, np.ndarray]:
    """Compute U1(k) by SCM and analytic single-sphere check."""
    centers = np.array([[0.0, 0.0, 0.0]], dtype=float)
    system = MatrixSCMSystem(centers=centers, lmax=int(lmax), normals=normals, params=params)

    U_single_k = np.full(params.n_orient, np.nan, dtype=float)
    U_single_analytic_k = np.full(params.n_orient, np.nan, dtype=float)
    for k in range(params.n_orient):
        E = rotating_field_k(k, params)
        U_single_k[k], _ = system.energy_parts(E)
        U_single_analytic_k[k] = analytic_single_sphere_bem_like_energy(E, params)
    return U_single_k, U_single_analytic_k


def compute_cluster_energy_k(centers: np.ndarray, lmax: int, normals: np.ndarray, params) -> Tuple[np.ndarray, np.ndarray]:
    """Compute total energy and particle-local contributions for all field phases."""
    system = MatrixSCMSystem(centers=centers, lmax=int(lmax), normals=normals, params=params)
    n_particles = np.asarray(centers).shape[0]
    U_total_k = np.full(params.n_orient, np.nan, dtype=float)
    U_parts_kp = np.full((params.n_orient, n_particles), np.nan, dtype=float)
    for k in range(params.n_orient):
        E = rotating_field_k(k, params)
        U_total_k[k], U_parts_kp[k] = system.energy_parts(E)
    return U_total_k, U_parts_kp


def bilinear_interp_pair_phi(
    r: float,
    beta_deg: float,
    r_grid: np.ndarray,
    beta_grid: np.ndarray,
    phi_grid_rb: np.ndarray,
) -> float:
    """Bilinear interpolation of phi_pair(r,beta) on a rectangular grid.

    r_grid must be sorted in physical units, beta_grid in degrees.
    """
    r_grid = np.asarray(r_grid, dtype=float)
    beta_grid = np.asarray(beta_grid, dtype=float)
    phi = np.asarray(phi_grid_rb, dtype=float)

    rr = float(r)
    bb = abs(float(beta_deg))
    bb = min(max(bb, float(beta_grid[0])), float(beta_grid[-1]))

    if rr < r_grid[0] - 1e-14 or rr > r_grid[-1] + 1e-14:
        raise ValueError(
            f"Pair distance {rr:.6e} is outside pair map range "
            f"[{r_grid[0]:.6e}, {r_grid[-1]:.6e}]."
        )

    # Exact / boundary handling.
    ir_hi = int(np.searchsorted(r_grid, rr, side="left"))
    if ir_hi == 0:
        ir0 = ir1 = 0
        tr = 0.0
    elif ir_hi >= len(r_grid):
        ir0 = ir1 = len(r_grid) - 1
        tr = 0.0
    elif abs(r_grid[ir_hi] - rr) < 1e-14:
        ir0 = ir1 = ir_hi
        tr = 0.0
    else:
        ir0 = ir_hi - 1
        ir1 = ir_hi
        tr = (rr - r_grid[ir0]) / (r_grid[ir1] - r_grid[ir0])

    ib_hi = int(np.searchsorted(beta_grid, bb, side="left"))
    if ib_hi == 0:
        ib0 = ib1 = 0
        tb = 0.0
    elif ib_hi >= len(beta_grid):
        ib0 = ib1 = len(beta_grid) - 1
        tb = 0.0
    elif abs(beta_grid[ib_hi] - bb) < 1e-12:
        ib0 = ib1 = ib_hi
        tb = 0.0
    else:
        ib0 = ib_hi - 1
        ib1 = ib_hi
        tb = (bb - beta_grid[ib0]) / (beta_grid[ib1] - beta_grid[ib0])

    f00 = phi[ir0, ib0]
    f10 = phi[ir1, ib0]
    f01 = phi[ir0, ib1]
    f11 = phi[ir1, ib1]

    return float((1 - tr) * (1 - tb) * f00 + tr * (1 - tb) * f10 + (1 - tr) * tb * f01 + tr * tb * f11)


def load_pair_map(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "r_over_d": data["r_over_d"],
        "r": data["r"],
        "beta_deg": data["beta_deg"],
        "phi_pair_avg_rb": data["phi_pair_avg_rb"],
        "U_single_k": data["U_single_k"],
    }


def pairwise_energy_from_edges(edge_distances: np.ndarray, edge_betas_deg: np.ndarray, pair_map: Dict[str, np.ndarray]) -> float:
    total = 0.0
    for dist, beta in zip(edge_distances, edge_betas_deg):
        total += bilinear_interp_pair_phi(
            r=float(dist),
            beta_deg=float(beta),
            r_grid=pair_map["r"],
            beta_grid=pair_map["beta_deg"],
            phi_grid_rb=pair_map["phi_pair_avg_rb"],
        )
    return float(total)


def eta_pair(delta3: float, phi_pairwise: float, eps: float = 1e-300) -> float:
    return float(abs(delta3) / (abs(phi_pairwise) + eps))


def eta_sym(delta3: float, phi3: float, phi_pairwise: float, eps: float = 1e-300) -> float:
    return float(2.0 * abs(delta3) / (abs(phi3) + abs(phi_pairwise) + eps))


def save_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
