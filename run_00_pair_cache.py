"""
run_00_pair_cache.py

Purpose
-------
Build a cached table of phase-averaged pair excess energies phi_pair(r,lmax)
needed for the triplet nonadditivity analysis.

This script must be run before the triplet scripts:
    python run_00_pair_cache.py

Why this is needed
------------------
The non-pairwise three-body contribution is defined as

    Delta3 = Phi3 - sum_{i<j} phi_pair(r_ij),

where

    Phi3    = U3 - 3 U1,
    phi_pair = U2 - 2 U1.

For the linear triplet, pair distances are r and 2r. Since the main triplet grid
uses r/d up to 6, the pair cache must contain distances up to 12d.
For the angular triplet, an additional distance is

    r23 = 2 r sin(gamma/2).

The script automatically constructs all pair distances required by the selected
linear, equilateral, and angular triplet grids.

Output
------
results_triplet_scm/matrix_scm_pair_cache_lmax1_6.npz

Important arrays saved in the output file
-----------------------------------------
r_pair_list              physical distances [m]
r_pair_over_d            distances in units of d
lmax_list                [1,2,3,4,5,6]
U_single_lk              single-sphere energy, shape (n_lmax, n_orient)
U_single_analytic_k      analytic single-sphere energy, shape (n_orient,)
U_pair_lrk               full pair energy, shape (n_lmax, n_r_pair, n_orient)
U_pair_parts_lrkp        local particle energies in pair, shape (n_lmax,n_r_pair,n_orient,2)
phi_pair_lrk             excess pair energy by orientation, U2-2U1
phi_pair_avg_lr          phase-averaged pair excess energy
phi_pair_particle_avg_lr half of the phase-averaged pair excess energy
"""

from __future__ import annotations

import os
import time
import numpy as np

from scm_config import (
    OUT_DIR,
    PAIR_CACHE_FILE,
    PARAMS,
    LMAX_LIST,
    R_OVER_D_MAIN,
    R_OVER_D_ANGLE,
    GAMMA_DEG,
)
from scm_core import (
    MatrixSCMSystem,
    analytic_single_sphere_bem_like_energy,
    centers_pair,
    fibonacci_sphere_points,
    print_header,
    rotating_field_k,
    timestamp,
    unique_sorted,
)


def build_required_pair_distances() -> np.ndarray:
    """Return all physical pair distances required by all planned triplets."""
    d = PARAMS.d
    distances = []

    # Linear triplet: r and 2r.
    for x in R_OVER_D_MAIN:
        r = float(x) * d
        distances.append(r)
        distances.append(2.0 * r)

    # Equilateral triplet: edge r.
    for x in R_OVER_D_MAIN:
        distances.append(float(x) * d)

    # Angular triplet: r and r23 = 2r sin(gamma/2).
    for x in R_OVER_D_ANGLE:
        r = float(x) * d
        distances.append(r)
        for gamma in GAMMA_DEG:
            distances.append(2.0 * r * np.sin(0.5 * np.deg2rad(float(gamma))))

    return unique_sorted(distances, ndigits=15)


def main() -> None:
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print_header("00 | SCM pair cache for triplet nonadditivity")
    print("Output file:", PAIR_CACHE_FILE)
    print("Parameters:")
    print(PARAMS.to_json())
    print("lmax_list:", LMAX_LIST)
    print("n_orient:", PARAMS.n_orient)
    print("n_quad:", PARAMS.n_quad)

    normals = fibonacci_sphere_points(PARAMS.n_quad)
    r_pair_list = build_required_pair_distances()
    r_pair_over_d = r_pair_list / PARAMS.d

    print("\nRequired pair distances r/d:")
    print(r_pair_over_d)

    n_l = len(LMAX_LIST)
    n_r = len(r_pair_list)
    n_k = PARAMS.n_orient

    U_single_lk = np.full((n_l, n_k), np.nan, dtype=float)
    U_single_analytic_k = np.full(n_k, np.nan, dtype=float)
    U_pair_lrk = np.full((n_l, n_r, n_k), np.nan, dtype=float)
    U_pair_parts_lrkp = np.full((n_l, n_r, n_k, 2), np.nan, dtype=float)

    # Single sphere for each lmax. For a single sphere, the result should not depend
    # on lmax beyond l=1 for a uniform field; we still calculate it for consistency.
    print_header("Single sphere")
    for il, lmax in enumerate(LMAX_LIST):
        print(f"\nlmax={lmax}")
        sys1 = MatrixSCMSystem(
            centers=np.array([[0.0, 0.0, 0.0]], dtype=float),
            lmax=int(lmax),
            normals=normals,
            params=PARAMS,
        )
        for k in range(n_k):
            E = rotating_field_k(k, PARAMS)
            U_total, U_parts = sys1.energy_parts(E)
            U_single_lk[il, k] = U_total
            if il == 0:
                U_single_analytic_k[k] = analytic_single_sphere_bem_like_energy(E, PARAMS)
            print(
                f"  k={k:02d}: U1={U_total:+.8e} J, "
                f"U1/analytic={U_total / U_single_analytic_k[k]:+.8e}"
            )

    print_header("Pair sweep")
    for il, lmax in enumerate(LMAX_LIST):
        print("\n" + "-" * 88)
        print(f"lmax={lmax}")
        print("-" * 88)
        for ir, r in enumerate(r_pair_list):
            print(f"\nPair distance {ir+1}/{n_r}: r/d={r/PARAMS.d:.8f}")
            t_geom = time.time()
            sys_pair = MatrixSCMSystem(
                centers=centers_pair(float(r)),
                lmax=int(lmax),
                normals=normals,
                params=PARAMS,
            )
            print(f"  assembled geometry in {time.time() - t_geom:.2f} s")

            for k in range(n_k):
                E = rotating_field_k(k, PARAMS)
                t0 = time.time()
                U_total, U_parts = sys_pair.energy_parts(E)
                U_pair_lrk[il, ir, k] = U_total
                U_pair_parts_lrkp[il, ir, k, :] = U_parts
                phi = U_total - 2.0 * U_single_lk[il, k]
                print(
                    f"  k={k:02d}: U2={U_total:+.8e} J, "
                    f"phi_pair={phi:+.8e} J, time={time.time()-t0:.2f} s"
                )

            save(
                r_pair_list,
                U_single_lk,
                U_single_analytic_k,
                U_pair_lrk,
                U_pair_parts_lrkp,
            )
            print("  saved intermediate cache")

    save(r_pair_list, U_single_lk, U_single_analytic_k, U_pair_lrk, U_pair_parts_lrkp)
    print_header("Done")
    print(f"Saved: {PAIR_CACHE_FILE}")
    print(f"Total time: {(time.time() - t_start)/60.0:.2f} min")


def save(r_pair_list, U_single_lk, U_single_analytic_k, U_pair_lrk, U_pair_parts_lrkp) -> None:
    phi_pair_lrk = U_pair_lrk - 2.0 * U_single_lk[:, None, :]
    phi_pair_avg_lr = np.nanmean(phi_pair_lrk, axis=2)
    phi_pair_particle_avg_lr = 0.5 * phi_pair_avg_lr

    np.savez(
        PAIR_CACHE_FILE,
        r_pair_list=r_pair_list,
        r_pair_over_d=r_pair_list / PARAMS.d,
        lmax_list=LMAX_LIST,
        U_single_lk=U_single_lk,
        U_single_analytic_k=U_single_analytic_k,
        U_pair_lrk=U_pair_lrk,
        U_pair_parts_lrkp=U_pair_parts_lrkp,
        phi_pair_lrk=phi_pair_lrk,
        phi_pair_avg_lr=phi_pair_avg_lr,
        phi_pair_particle_avg_lr=phi_pair_particle_avg_lr,
        n_orient=PARAMS.n_orient,
        n_quad=PARAMS.n_quad,
        E0=PARAMS.E0,
        a=PARAMS.a,
        d=PARAMS.d,
        eps0=PARAMS.eps0,
        eps1_r=PARAMS.eps1_r,
        eps2_r=PARAMS.eps2_r,
        params_json=PARAMS.to_json(),
        timestamp=timestamp(),
    )


if __name__ == "__main__":
    main()
