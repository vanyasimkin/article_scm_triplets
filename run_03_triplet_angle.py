"""
run_03_triplet_angle.py

Purpose
-------
Calculate the angular triplet by matrix SCM.

Geometry
--------
One central particle and two neighbors at equal distance r:

    r1 = (0, 0, 0)                         central particle
    r2 = (r, 0, 0)
    r3 = (r cos gamma, r sin gamma, 0)

The neighbor-neighbor distance is

    r23 = 2 r sin(gamma/2).

The scan uses:
    r/d = 1.05, 1.20, 2.00
    gamma = 60, 90, 120, 150, 180 deg

The endpoints are also code-validation checks:
    gamma=60 deg  -> equilateral triangle, same total energy as run_02
    gamma=180 deg -> linear triplet, same total energy as run_01 for the same r

Calculated quantities
---------------------
    Phi3 = U3 - 3 U1,
    Phi_center = U_center - U1,
    Phi_pairwise = 2 phi_pair(r) + phi_pair(r23),
    Delta3 = Phi3 - Phi_pairwise.

Required input
--------------
Run first:
    python run_00_pair_cache.py

Output
------
results_triplet_scm/matrix_scm_triplet_angle_lmax1_6.npz
"""

from __future__ import annotations

import os
import time
import numpy as np

from scm_config import OUT_DIR, PAIR_CACHE_FILE, ANGLE_FILE, PARAMS, LMAX_LIST, R_OVER_D_ANGLE, GAMMA_DEG
from scm_core import (
    MatrixSCMSystem,
    centers_angle_triplet,
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
    r_list = R_OVER_D_ANGLE * PARAMS.d

    n_l = len(LMAX_LIST)
    n_r = len(r_list)
    n_g = len(GAMMA_DEG)
    n_k = PARAMS.n_orient

    U_total_lrgk = np.full((n_l, n_r, n_g, n_k), np.nan, dtype=float)
    U_parts_lrgkp = np.full((n_l, n_r, n_g, n_k, 3), np.nan, dtype=float)
    r23_over_d_rg = np.full((n_r, n_g), np.nan, dtype=float)

    print_header("03 | Angular triplet SCM")
    print("Output file:", ANGLE_FILE)
    print("r/d:", R_OVER_D_ANGLE)
    print("gamma_deg:", GAMMA_DEG)
    print("lmax_list:", LMAX_LIST)

    for il, lmax in enumerate(LMAX_LIST):
        print("\n" + "-" * 88)
        print(f"lmax={lmax}")
        print("-" * 88)

        for ir, r in enumerate(r_list):
            for ig, gamma in enumerate(GAMMA_DEG):
                centers = centers_angle_triplet(float(r), float(gamma))
                distances = pair_distances(centers)
                r23_over_d_rg[ir, ig] = distances[2] / PARAMS.d  # pair order: 12, 13, 23
                print(
                    f"\nAngle r-index {ir+1}/{n_r}, gamma-index {ig+1}/{n_g}: "
                    f"r/d={r/PARAMS.d:.8f}, gamma={gamma:.1f}, pair distances/d={distances/PARAMS.d}"
                )
                t_geom = time.time()
                system = MatrixSCMSystem(centers=centers, lmax=int(lmax), normals=normals, params=PARAMS)
                print(f"  assembled geometry in {time.time() - t_geom:.2f} s")

                for k in range(n_k):
                    E = rotating_field_k(k, PARAMS)
                    t0 = time.time()
                    U_total, U_parts = system.energy_parts(E)
                    U_total_lrgk[il, ir, ig, k] = U_total
                    U_parts_lrgkp[il, ir, ig, k, :] = U_parts
                    Phi3_k = U_total - 3.0 * U_single_lk[il, k]
                    Phi_c_k = U_parts[0] - U_single_lk[il, k]
                    print(
                        f"  k={k:02d}: U3={U_total:+.8e} J, "
                        f"Phi3={Phi3_k:+.8e} J, Phi_center={Phi_c_k:+.8e} J, "
                        f"time={time.time()-t0:.2f} s"
                    )

                save(r_list, U_single_lk, U_total_lrgk, U_parts_lrgkp, r_pair_list, phi_pair_avg_lr, r23_over_d_rg)
                print("  saved intermediate angular results")

    save(r_list, U_single_lk, U_total_lrgk, U_parts_lrgkp, r_pair_list, phi_pair_avg_lr, r23_over_d_rg)
    print_header("Done")
    print(f"Saved: {ANGLE_FILE}")
    print(f"Total time: {(time.time()-t_start)/60.0:.2f} min")


def save(r_list, U_single_lk, U_total_lrgk, U_parts_lrgkp, r_pair_list, phi_pair_avg_lr, r23_over_d_rg) -> None:
    Phi3_lrgk = U_total_lrgk - 3.0 * U_single_lk[:, None, None, :]
    Phi3_lrg = np.nanmean(Phi3_lrgk, axis=3)

    Phi_particles_lrgkp = U_parts_lrgkp - U_single_lk[:, None, None, :, None]
    Phi_particles_lrgp = np.nanmean(Phi_particles_lrgkp, axis=3)
    Phi_center_lrgk = U_parts_lrgkp[:, :, :, :, 0] - U_single_lk[:, None, None, :]
    Phi_center_lrg = np.nanmean(Phi_center_lrgk, axis=3)

    Phi_pairwise_lrg = np.full((len(LMAX_LIST), len(r_list), len(GAMMA_DEG)), np.nan, dtype=float)
    Phi_center_pairwise_lrg = np.full_like(Phi_pairwise_lrg, np.nan)
    Phi_neighbor_pairwise_lrg = np.full_like(Phi_pairwise_lrg, np.nan)

    for il in range(len(LMAX_LIST)):
        for ir, r in enumerate(r_list):
            phi_r = lookup_pair_energy(float(r), r_pair_list, phi_pair_avg_lr, il)
            for ig, gamma in enumerate(GAMMA_DEG):
                r23 = 2.0 * float(r) * np.sin(0.5 * np.deg2rad(float(gamma)))
                phi_r23 = lookup_pair_energy(float(r23), r_pair_list, phi_pair_avg_lr, il)
                Phi_pairwise_lrg[il, ir, ig] = 2.0 * phi_r + phi_r23
                # Diagnostic local pairwise split.
                Phi_center_pairwise_lrg[il, ir, ig] = phi_r
                Phi_neighbor_pairwise_lrg[il, ir, ig] = 0.5 * phi_r + 0.5 * phi_r23

    Delta3_lrg = Phi3_lrg - Phi_pairwise_lrg
    eta3_pair_lrg = np.abs(Delta3_lrg) / (np.abs(Phi_pairwise_lrg) + 1e-300)
    eta3_sym_lrg = 2.0 * np.abs(Delta3_lrg) / (np.abs(Phi3_lrg) + np.abs(Phi_pairwise_lrg) + 1e-300)

    np.savez(
        ANGLE_FILE,
        r_list=r_list,
        r_over_d_values=r_list / PARAMS.d,
        gamma_deg=GAMMA_DEG,
        lmax_list=LMAX_LIST,
        U_single_lk=U_single_lk,
        U_total_lrgk=U_total_lrgk,
        U_parts_lrgkp=U_parts_lrgkp,
        Phi3_lrgk=Phi3_lrgk,
        Phi3_lrg=Phi3_lrg,
        Phi_particles_lrgkp=Phi_particles_lrgkp,
        Phi_particles_lrgp=Phi_particles_lrgp,
        Phi_center_lrgk=Phi_center_lrgk,
        Phi_center_lrg=Phi_center_lrg,
        Phi_pairwise_lrg=Phi_pairwise_lrg,
        Phi_center_pairwise_lrg=Phi_center_pairwise_lrg,
        Phi_neighbor_pairwise_lrg=Phi_neighbor_pairwise_lrg,
        Delta3_lrg=Delta3_lrg,
        eta3_pair_lrg=eta3_pair_lrg,
        eta3_sym_lrg=eta3_sym_lrg,
        r23_over_d_rg=r23_over_d_rg,
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
