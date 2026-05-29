"""
scm3d_pair_map.py

Compute the orientation-dependent pair excess energy map

    phi_pair(r, beta) = < U2(r,beta,k) - 2 U1(k) >_k.

The output file is the required pairwise baseline for all 3D triangle maps.
"""

from __future__ import annotations

import time

import numpy as np

from scm_core import fibonacci_sphere_points, timestamp

from .scm3d_config import BETA_DEG, LMAX_MAIN, PAIR_ORIENTATION_FILE, PARAMS, R_PAIR_OVER_D
from .scm3d_geometry import pair_oriented_centers
from .scm3d_utils import compute_cluster_energy_k, compute_single_energy_k, print_header


def run_pair_orientation_map() -> None:
    t_start = time.time()
    print_header("SCM 3D Stage 1: pair orientation map")

    normals = fibonacci_sphere_points(PARAMS.n_quad)
    r_grid = R_PAIR_OVER_D * PARAMS.d
    beta_grid = BETA_DEG.copy()

    n_r = len(r_grid)
    n_b = len(beta_grid)
    n_k = PARAMS.n_orient

    U_single_k, U_single_analytic_k = compute_single_energy_k(LMAX_MAIN, normals, PARAMS)
    U_pair_rbk = np.full((n_r, n_b, n_k), np.nan, dtype=float)
    phi_pair_rbk = np.full((n_r, n_b, n_k), np.nan, dtype=float)
    phi_pair_avg_rb = np.full((n_r, n_b), np.nan, dtype=float)

    print("lmax         =", LMAX_MAIN)
    print("n_quad       =", PARAMS.n_quad)
    print("n_orient     =", PARAMS.n_orient)
    print("r/d grid     =", R_PAIR_OVER_D)
    print("beta grid    =", BETA_DEG)
    print("U1/analytic  =", U_single_k / U_single_analytic_k)

    for ir, r in enumerate(r_grid):
        for ib, beta in enumerate(beta_grid):
            print(f"\nPair {ir+1}/{n_r}, {ib+1}/{n_b}: r/d={r/PARAMS.d:.3f}, beta={beta:.1f} deg")
            centers = pair_oriented_centers(float(r), float(beta))
            t0 = time.time()
            U_pair_k, _ = compute_cluster_energy_k(centers, LMAX_MAIN, normals, PARAMS)
            U_pair_rbk[ir, ib] = U_pair_k
            phi_pair_rbk[ir, ib] = U_pair_k - 2.0 * U_single_k
            phi_pair_avg_rb[ir, ib] = float(np.mean(phi_pair_rbk[ir, ib]))
            print(f"  phi_pair_avg = {phi_pair_avg_rb[ir, ib]:+.8e} J, time={time.time()-t0:.2f} s")

        # Save after every distance row so long runs are restart-friendly.
        save_pair_orientation_map(
            r_grid,
            beta_grid,
            U_single_k,
            U_single_analytic_k,
            U_pair_rbk,
            phi_pair_rbk,
            phi_pair_avg_rb,
        )
        print("Saved intermediate:", PAIR_ORIENTATION_FILE)

    save_pair_orientation_map(
        r_grid,
        beta_grid,
        U_single_k,
        U_single_analytic_k,
        U_pair_rbk,
        phi_pair_rbk,
        phi_pair_avg_rb,
    )
    print_header(f"Done. Saved: {PAIR_ORIENTATION_FILE}")
    print(f"Total time = {(time.time()-t_start)/60:.2f} min")


def save_pair_orientation_map(
    r_grid,
    beta_grid,
    U_single_k,
    U_single_analytic_k,
    U_pair_rbk,
    phi_pair_rbk,
    phi_pair_avg_rb,
) -> None:
    np.savez(
        PAIR_ORIENTATION_FILE,
        r_over_d=np.asarray(r_grid) / PARAMS.d,
        r=np.asarray(r_grid),
        beta_deg=np.asarray(beta_grid),
        lmax=int(LMAX_MAIN),
        U_single_k=U_single_k,
        U_single_analytic_k=U_single_analytic_k,
        U_pair_rbk=U_pair_rbk,
        phi_pair_rbk=phi_pair_rbk,
        phi_pair_avg_rb=phi_pair_avg_rb,
        n_orient=PARAMS.n_orient,
        n_quad=PARAMS.n_quad,
        a=PARAMS.a,
        d=PARAMS.d,
        E0=PARAMS.E0,
        eps0=PARAMS.eps0,
        eps1_r=PARAMS.eps1_r,
        eps2_r=PARAMS.eps2_r,
        timestamp=timestamp(),
    )


if __name__ == "__main__":
    run_pair_orientation_map()
