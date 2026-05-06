"""
run_02_triplet_equilateral.py

Purpose
-------
Calculate the equilateral triplet by matrix SCM.

Geometry
--------
Three particles are placed at vertices of an equilateral triangle with edge r and
center of mass at the origin:

    r1 = (-r/2, -sqrt(3)r/6, 0)
    r2 = (+r/2, -sqrt(3)r/6, 0)
    r3 = (0,    +sqrt(3)r/3, 0)

There is no distinguished central particle. After phase averaging over a rotating
field, the three local diagnostic contributions should agree within numerical error.

Calculated quantities
---------------------
    Phi3 = U3 - 3 U1,
    Phi_pairwise = 3 phi_pair(r),
    Delta3 = Phi3 - Phi_pairwise.

Required input
--------------
Run first:
    python run_00_pair_cache.py

Output
------
results_triplet_scm/matrix_scm_triplet_equilateral_lmax1_6.npz
"""

from __future__ import annotations

import os
import time
import numpy as np

from scm_config import OUT_DIR, PAIR_CACHE_FILE, EQUILATERAL_FILE, PARAMS, LMAX_LIST, R_OVER_D_MAIN
from scm_core import (
    MatrixSCMSystem,
    centers_equilateral,
    fibonacci_sphere_points,
    lookup_pair_energy,
    pair_distances,
    print_header,
    rotating_field_k,
    timestamp,
)


def main() -> None:
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(PAIR_CACHE_FILE):
        raise FileNotFoundError(f"Pair cache not found. Run run_00_pair_cache.py first: {PAIR_CACHE_FILE}")

    pair_cache = np.load(PAIR_CACHE_FILE)
    r_pair_list = pair_cache["r_pair_list"]
    phi_pair_avg_lr = pair_cache["phi_pair_avg_lr"]
    U_single_lk = pair_cache["U_single_lk"]

    normals = fibonacci_sphere_points(PARAMS.n_quad)
    r_list = R_OVER_D_MAIN * PARAMS.d

    n_l = len(LMAX_LIST)
    n_r = len(r_list)
    n_k = PARAMS.n_orient

    U_total_lrk = np.full((n_l, n_r, n_k), np.nan, dtype=float)
    U_parts_lrkp = np.full((n_l, n_r, n_k, 3), np.nan, dtype=float)

    print_header("02 | Equilateral triplet SCM")
    print("Output file:", EQUILATERAL_FILE)
    print("r/d:", R_OVER_D_MAIN)
    print("lmax_list:", LMAX_LIST)

    for il, lmax in enumerate(LMAX_LIST):
        print("\n" + "-" * 88)
        print(f"lmax={lmax}")
        print("-" * 88)

        for ir, r in enumerate(r_list):
            centers = centers_equilateral(float(r))
            distances = pair_distances(centers)
            print(f"\nEquilateral {ir+1}/{n_r}: edge r/d={r/PARAMS.d:.8f}, pair distances/d={distances/PARAMS.d}")
            t_geom = time.time()
            system = MatrixSCMSystem(centers=centers, lmax=int(lmax), normals=normals, params=PARAMS)
            print(f"  assembled geometry in {time.time() - t_geom:.2f} s")

            for k in range(n_k):
                E = rotating_field_k(k, PARAMS)
                t0 = time.time()
                U_total, U_parts = system.energy_parts(E)
                U_total_lrk[il, ir, k] = U_total
                U_parts_lrkp[il, ir, k, :] = U_parts
                Phi3_k = U_total - 3.0 * U_single_lk[il, k]
                print(
                    f"  k={k:02d}: U3={U_total:+.8e} J, Phi3={Phi3_k:+.8e} J, "
                    f"parts={U_parts}, time={time.time()-t0:.2f} s"
                )

            save(r_list, U_single_lk, U_total_lrk, U_parts_lrkp, r_pair_list, phi_pair_avg_lr)
            print("  saved intermediate equilateral results")

    save(r_list, U_single_lk, U_total_lrk, U_parts_lrkp, r_pair_list, phi_pair_avg_lr)
    print_header("Done")
    print(f"Saved: {EQUILATERAL_FILE}")
    print(f"Total time: {(time.time()-t_start)/60.0:.2f} min")


def save(r_list, U_single_lk, U_total_lrk, U_parts_lrkp, r_pair_list, phi_pair_avg_lr) -> None:
    Phi3_lrk = U_total_lrk - 3.0 * U_single_lk[:, None, :]
    Phi3_lr = np.nanmean(Phi3_lrk, axis=2)

    Phi_particles_lrkp = U_parts_lrkp - U_single_lk[:, None, :, None]
    Phi_particles_lrp = np.nanmean(Phi_particles_lrkp, axis=2)

    # Symmetry diagnostic: after phase averaging, three particles should be equivalent.
    Phi_particles_mean_lr = np.nanmean(Phi_particles_lrp, axis=2)
    Phi_particles_maxdev_lr = np.nanmax(np.abs(Phi_particles_lrp - Phi_particles_mean_lr[:, :, None]), axis=2)

    Phi_pairwise_lr = np.full((len(LMAX_LIST), len(r_list)), np.nan, dtype=float)
    for il in range(len(LMAX_LIST)):
        for ir, r in enumerate(r_list):
            phi_r = lookup_pair_energy(float(r), r_pair_list, phi_pair_avg_lr, il)
            Phi_pairwise_lr[il, ir] = 3.0 * phi_r

    Delta3_lr = Phi3_lr - Phi_pairwise_lr
    eta3_pair_lr = np.abs(Delta3_lr) / (np.abs(Phi_pairwise_lr) + 1e-300)
    eta3_sym_lr = 2.0 * np.abs(Delta3_lr) / (np.abs(Phi3_lr) + np.abs(Phi_pairwise_lr) + 1e-300)

    np.savez(
        EQUILATERAL_FILE,
        r_list=r_list,
        r_over_d=r_list / PARAMS.d,
        lmax_list=LMAX_LIST,
        U_single_lk=U_single_lk,
        U_total_lrk=U_total_lrk,
        U_parts_lrkp=U_parts_lrkp,
        Phi3_lrk=Phi3_lrk,
        Phi3_lr=Phi3_lr,
        Phi_particles_lrkp=Phi_particles_lrkp,
        Phi_particles_lrp=Phi_particles_lrp,
        Phi_particles_maxdev_lr=Phi_particles_maxdev_lr,
        Phi_pairwise_lr=Phi_pairwise_lr,
        Delta3_lr=Delta3_lr,
        eta3_pair_lr=eta3_pair_lr,
        eta3_sym_lr=eta3_sym_lr,
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
