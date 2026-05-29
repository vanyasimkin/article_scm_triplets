"""
scm3d_triangle_map.py

Compute SCM maps for central-particle triangular triplets.

The central particle is particle 1. Particles 2 and 3 are placed at distances
r12 and r13 from particle 1, with opening angle gamma. The triangle plane is
then tilted by psi relative to the field-rotation plane and rotated in-plane by
alpha.

Main output quantities:
    Phi3      = <U3 - 3 U1>_k,
    Pairwise  = sum_edges phi_pair(r_ij, beta_ij),
    Delta3    = Phi3 - Pairwise,
    eta_pair  = |Delta3| / |Pairwise|,
    eta_sym   = 2|Delta3| / (|Phi3| + |Pairwise|).
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from scm_core import fibonacci_sphere_points, timestamp

from .scm3d_config import (
    ALPHA_DEG,
    ASYM_ALPHA_DEG,
    ASYM_GAMMA_DEG,
    ASYM_PSI_DEG,
    ASYM_R12_OVER_D,
    ASYM_R13_OVER_D,
    GAMMA_DEG,
    LMAX_MAIN,
    PAIR_ORIENTATION_FILE,
    PARAMS,
    PSI_DEG,
    R_TRI_OVER_D,
    TRIANGLE_ASYMMETRIC_FILE,
    TRIANGLE_ISOSCELES_FILE,
)
from .scm3d_geometry import edge_info, min_surface_gap_over_d, triangle_centers
from .scm3d_utils import (
    compute_cluster_energy_k,
    compute_single_energy_k,
    eta_pair,
    eta_sym,
    load_pair_map,
    pairwise_energy_from_edges,
    print_header,
)


def run_triangle_isosceles_map() -> None:
    """Run the main triangle map with r12=r13=r."""
    t_start = time.time()
    print_header("SCM 3D Stage 2: isosceles triangle map, r12=r13")

    if not PAIR_ORIENTATION_FILE.exists():
        raise FileNotFoundError(
            f"Pair orientation map not found: {PAIR_ORIENTATION_FILE}. "
            "Run: python -m scm_3d_maps.scripts.run_00_pair_orientation_map"
        )

    pair_map = load_pair_map(PAIR_ORIENTATION_FILE)
    normals = fibonacci_sphere_points(PARAMS.n_quad)
    U_single_k, U_single_analytic_k = compute_single_energy_k(LMAX_MAIN, normals, PARAMS)

    r_grid = R_TRI_OVER_D * PARAMS.d
    n_r, n_g, n_p, n_a, n_k = len(r_grid), len(GAMMA_DEG), len(PSI_DEG), len(ALPHA_DEG), PARAMS.n_orient

    U_triplet_rgpak = np.full((n_r, n_g, n_p, n_a, n_k), np.nan, dtype=float)
    U_parts_rgpakp = np.full((n_r, n_g, n_p, n_a, n_k, 3), np.nan, dtype=float)
    Phi3_rgpak = np.full((n_r, n_g, n_p, n_a, n_k), np.nan, dtype=float)
    Phi3_avg_rgpa = np.full((n_r, n_g, n_p, n_a), np.nan, dtype=float)
    Phi_pairwise_rgpa = np.full((n_r, n_g, n_p, n_a), np.nan, dtype=float)
    Delta3_rgpa = np.full((n_r, n_g, n_p, n_a), np.nan, dtype=float)
    eta3_pair_rgpa = np.full((n_r, n_g, n_p, n_a), np.nan, dtype=float)
    eta3_sym_rgpa = np.full((n_r, n_g, n_p, n_a), np.nan, dtype=float)
    edge_dist_rgpae = np.full((n_r, n_g, n_p, n_a, 3), np.nan, dtype=float)
    edge_beta_rgpae = np.full((n_r, n_g, n_p, n_a, 3), np.nan, dtype=float)
    min_gap_rgpa = np.full((n_r, n_g, n_p, n_a), np.nan, dtype=float)

    total = n_r * n_g * n_p * n_a
    counter = 0
    for ir, r in enumerate(r_grid):
        for ig, gamma in enumerate(GAMMA_DEG):
            for ip, psi in enumerate(PSI_DEG):
                for ia, alpha in enumerate(ALPHA_DEG):
                    counter += 1
                    print(
                        f"\n[{counter}/{total}] r/d={r/PARAMS.d:.3f}, "
                        f"gamma={gamma:.1f}, psi={psi:.1f}, alpha={alpha:.1f}"
                    )
                    centers = triangle_centers(r, r, gamma, psi, alpha)
                    info = edge_info(centers)
                    edge_dist_rgpae[ir, ig, ip, ia] = info["distances"]
                    edge_beta_rgpae[ir, ig, ip, ia] = info["beta_deg"]
                    min_gap_rgpa[ir, ig, ip, ia] = min_surface_gap_over_d(centers, PARAMS.d)

                    pairwise = pairwise_energy_from_edges(info["distances"], info["beta_deg"], pair_map)
                    Phi_pairwise_rgpa[ir, ig, ip, ia] = pairwise

                    t0 = time.time()
                    U_triplet_k, U_parts_kp = compute_cluster_energy_k(centers, LMAX_MAIN, normals, PARAMS)
                    U_triplet_rgpak[ir, ig, ip, ia] = U_triplet_k
                    U_parts_rgpakp[ir, ig, ip, ia] = U_parts_kp
                    Phi3_rgpak[ir, ig, ip, ia] = U_triplet_k - 3.0 * U_single_k
                    phi3 = float(np.mean(Phi3_rgpak[ir, ig, ip, ia]))
                    Phi3_avg_rgpa[ir, ig, ip, ia] = phi3
                    delta = phi3 - pairwise
                    Delta3_rgpa[ir, ig, ip, ia] = delta
                    eta3_pair_rgpa[ir, ig, ip, ia] = eta_pair(delta, pairwise)
                    eta3_sym_rgpa[ir, ig, ip, ia] = eta_sym(delta, phi3, pairwise)
                    print(
                        f"  Phi3={phi3:+.8e} J, Pairwise={pairwise:+.8e} J, "
                        f"Delta3={delta:+.8e} J, eta={eta3_pair_rgpa[ir,ig,ip,ia]:.3e}, "
                        f"time={time.time()-t0:.2f} s"
                    )

        save_triangle_isosceles(
            r_grid,
            U_single_k,
            U_single_analytic_k,
            U_triplet_rgpak,
            U_parts_rgpakp,
            Phi3_rgpak,
            Phi3_avg_rgpa,
            Phi_pairwise_rgpa,
            Delta3_rgpa,
            eta3_pair_rgpa,
            eta3_sym_rgpa,
            edge_dist_rgpae,
            edge_beta_rgpae,
            min_gap_rgpa,
        )
        print("Saved intermediate:", TRIANGLE_ISOSCELES_FILE)

    save_triangle_isosceles(
        r_grid,
        U_single_k,
        U_single_analytic_k,
        U_triplet_rgpak,
        U_parts_rgpakp,
        Phi3_rgpak,
        Phi3_avg_rgpa,
        Phi_pairwise_rgpa,
        Delta3_rgpa,
        eta3_pair_rgpa,
        eta3_sym_rgpa,
        edge_dist_rgpae,
        edge_beta_rgpae,
        min_gap_rgpa,
    )
    print_header(f"Done. Saved: {TRIANGLE_ISOSCELES_FILE}")
    print(f"Total time = {(time.time()-t_start)/60:.2f} min")


def save_triangle_isosceles(
    r_grid,
    U_single_k,
    U_single_analytic_k,
    U_triplet_rgpak,
    U_parts_rgpakp,
    Phi3_rgpak,
    Phi3_avg_rgpa,
    Phi_pairwise_rgpa,
    Delta3_rgpa,
    eta3_pair_rgpa,
    eta3_sym_rgpa,
    edge_dist_rgpae,
    edge_beta_rgpae,
    min_gap_rgpa,
) -> None:
    np.savez(
        TRIANGLE_ISOSCELES_FILE,
        r_over_d=np.asarray(r_grid) / PARAMS.d,
        r=np.asarray(r_grid),
        gamma_deg=GAMMA_DEG,
        psi_deg=PSI_DEG,
        alpha_deg=ALPHA_DEG,
        lmax=int(LMAX_MAIN),
        U_single_k=U_single_k,
        U_single_analytic_k=U_single_analytic_k,
        U_triplet_rgpak=U_triplet_rgpak,
        U_parts_rgpakp=U_parts_rgpakp,
        Phi3_rgpak=Phi3_rgpak,
        Phi3_avg_rgpa=Phi3_avg_rgpa,
        Phi_pairwise_rgpa=Phi_pairwise_rgpa,
        Delta3_rgpa=Delta3_rgpa,
        eta3_pair_rgpa=eta3_pair_rgpa,
        eta3_sym_rgpa=eta3_sym_rgpa,
        edge_dist_rgpae=edge_dist_rgpae,
        edge_beta_rgpae=edge_beta_rgpae,
        min_gap_rgpa=min_gap_rgpa,
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


def run_triangle_asymmetric_map() -> None:
    """Run the full general-triangle map over r12 and r13.

    Historical note: the public script is still named
    run_02_triangle_asymmetric_map.py, but this function now performs the full
    square scan over r12 and r13 rather than the older compact q-scan.
    """
    t_start = time.time()
    print_header("SCM 3D Stage 3: full general triangle map")

    if not PAIR_ORIENTATION_FILE.exists():
        raise FileNotFoundError(f"Pair orientation map not found: {PAIR_ORIENTATION_FILE}")

    pair_map = load_pair_map(PAIR_ORIENTATION_FILE)
    normals = fibonacci_sphere_points(PARAMS.n_quad)
    U_single_k, U_single_analytic_k = compute_single_energy_k(LMAX_MAIN, normals, PARAMS)

    r12_grid = ASYM_R12_OVER_D * PARAMS.d
    r13_grid = ASYM_R13_OVER_D * PARAMS.d
    n_r12, n_r13, n_g, n_p, n_a, n_k = (
        len(r12_grid),
        len(r13_grid),
        len(ASYM_GAMMA_DEG),
        len(ASYM_PSI_DEG),
        len(ASYM_ALPHA_DEG),
        PARAMS.n_orient,
    )

    U_triplet_rrgpak = np.full((n_r12, n_r13, n_g, n_p, n_a, n_k), np.nan, dtype=float)
    Phi3_avg_rrgpa = np.full((n_r12, n_r13, n_g, n_p, n_a), np.nan, dtype=float)
    Phi_pairwise_rrgpa = np.full((n_r12, n_r13, n_g, n_p, n_a), np.nan, dtype=float)
    Delta3_rrgpa = np.full((n_r12, n_r13, n_g, n_p, n_a), np.nan, dtype=float)
    eta3_pair_rrgpa = np.full((n_r12, n_r13, n_g, n_p, n_a), np.nan, dtype=float)
    eta3_sym_rrgpa = np.full((n_r12, n_r13, n_g, n_p, n_a), np.nan, dtype=float)
    edge_dist_rrgpae = np.full((n_r12, n_r13, n_g, n_p, n_a, 3), np.nan, dtype=float)
    edge_beta_rrgpae = np.full((n_r12, n_r13, n_g, n_p, n_a, 3), np.nan, dtype=float)
    min_gap_rrgpa = np.full((n_r12, n_r13, n_g, n_p, n_a), np.nan, dtype=float)

    total = n_r12 * n_r13 * n_g * n_p * n_a
    counter = 0
    for ir12, r12 in enumerate(r12_grid):
        for ir13, r13 in enumerate(r13_grid):
            for ig, gamma in enumerate(ASYM_GAMMA_DEG):
                for ip, psi in enumerate(ASYM_PSI_DEG):
                    for ia, alpha in enumerate(ASYM_ALPHA_DEG):
                        counter += 1
                        print(
                            f"\n[{counter}/{total}] r12/d={r12/PARAMS.d:.3f}, r13/d={r13/PARAMS.d:.3f}, "
                            f"gamma={gamma:.1f}, psi={psi:.1f}, alpha={alpha:.1f}"
                        )
                        centers = triangle_centers(r12, r13, gamma, psi, alpha)
                        info = edge_info(centers)
                        edge_dist_rrgpae[ir12, ir13, ig, ip, ia] = info["distances"]
                        edge_beta_rrgpae[ir12, ir13, ig, ip, ia] = info["beta_deg"]
                        min_gap_rrgpa[ir12, ir13, ig, ip, ia] = min_surface_gap_over_d(centers, PARAMS.d)

                        pairwise = pairwise_energy_from_edges(info["distances"], info["beta_deg"], pair_map)
                        Phi_pairwise_rrgpa[ir12, ir13, ig, ip, ia] = pairwise

                        t0 = time.time()
                        U_triplet_k, _ = compute_cluster_energy_k(centers, LMAX_MAIN, normals, PARAMS)
                        U_triplet_rrgpak[ir12, ir13, ig, ip, ia] = U_triplet_k
                        phi3 = float(np.mean(U_triplet_k - 3.0 * U_single_k))
                        Phi3_avg_rrgpa[ir12, ir13, ig, ip, ia] = phi3
                        delta = phi3 - pairwise
                        Delta3_rrgpa[ir12, ir13, ig, ip, ia] = delta
                        eta3_pair_rrgpa[ir12, ir13, ig, ip, ia] = eta_pair(delta, pairwise)
                        eta3_sym_rrgpa[ir12, ir13, ig, ip, ia] = eta_sym(delta, phi3, pairwise)
                        print(
                            f"  Phi3={phi3:+.8e} J, Pairwise={pairwise:+.8e} J, "
                            f"Delta3={delta:+.8e} J, eta={eta3_pair_rrgpa[ir12,ir13,ig,ip,ia]:.3e}, "
                            f"min_gap/d={min_gap_rrgpa[ir12,ir13,ig,ip,ia]:.3f}, "
                            f"time={time.time()-t0:.2f} s"
                        )
        save_triangle_asymmetric(
            r12_grid,
            r13_grid,
            U_single_k,
            U_single_analytic_k,
            U_triplet_rrgpak,
            Phi3_avg_rrgpa,
            Phi_pairwise_rrgpa,
            Delta3_rrgpa,
            eta3_pair_rrgpa,
            eta3_sym_rrgpa,
            edge_dist_rrgpae,
            edge_beta_rrgpae,
            min_gap_rrgpa,
        )
        print("Saved intermediate:", TRIANGLE_ASYMMETRIC_FILE)

    save_triangle_asymmetric(
        r12_grid,
        r13_grid,
        U_single_k,
        U_single_analytic_k,
        U_triplet_rrgpak,
        Phi3_avg_rrgpa,
        Phi_pairwise_rrgpa,
        Delta3_rrgpa,
        eta3_pair_rrgpa,
        eta3_sym_rrgpa,
        edge_dist_rrgpae,
        edge_beta_rrgpae,
        min_gap_rrgpa,
    )
    print_header(f"Done. Saved: {TRIANGLE_ASYMMETRIC_FILE}")
    print(f"Total time = {(time.time()-t_start)/60:.2f} min")

def save_triangle_asymmetric(
    r12_grid,
    r13_grid,
    U_single_k,
    U_single_analytic_k,
    U_triplet_rrgpak,
    Phi3_avg_rrgpa,
    Phi_pairwise_rrgpa,
    Delta3_rrgpa,
    eta3_pair_rrgpa,
    eta3_sym_rrgpa,
    edge_dist_rrgpae,
    edge_beta_rrgpae,
    min_gap_rrgpa,
) -> None:
    np.savez(
        TRIANGLE_ASYMMETRIC_FILE,
        r12_over_d=np.asarray(r12_grid) / PARAMS.d,
        r13_over_d=np.asarray(r13_grid) / PARAMS.d,
        r12=np.asarray(r12_grid),
        r13=np.asarray(r13_grid),
        gamma_deg=ASYM_GAMMA_DEG,
        psi_deg=ASYM_PSI_DEG,
        alpha_deg=ASYM_ALPHA_DEG,
        lmax=int(LMAX_MAIN),
        U_single_k=U_single_k,
        U_single_analytic_k=U_single_analytic_k,
        U_triplet_rrgpak=U_triplet_rrgpak,
        Phi3_avg_rrgpa=Phi3_avg_rrgpa,
        Phi_pairwise_rrgpa=Phi_pairwise_rrgpa,
        Delta3_rrgpa=Delta3_rrgpa,
        eta3_pair_rrgpa=eta3_pair_rrgpa,
        eta3_sym_rrgpa=eta3_sym_rrgpa,
        edge_dist_rrgpae=edge_dist_rrgpae,
        edge_beta_rrgpae=edge_beta_rrgpae,
        min_gap_rrgpa=min_gap_rrgpa,
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
