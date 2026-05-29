"""
scm3d_convergence.py

Selected lmax convergence checks for 3D pair and triangular configurations.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np

from scm_core import fibonacci_sphere_points, timestamp

from .scm3d_config import CONV_PAIR_CASES, CONV_TRIANGLE_CASES, LMAX_CONVERGENCE_FILE, LMAX_LIST, PARAMS
from .scm3d_geometry import edge_info, pair_oriented_centers, triangle_centers
from .scm3d_utils import compute_cluster_energy_k, compute_single_energy_k, eta_pair, eta_sym, print_header


def run_lmax_convergence_checks() -> None:
    t_start = time.time()
    print_header("SCM 3D Stage 4: lmax convergence checks")

    normals = fibonacci_sphere_points(PARAMS.n_quad)
    n_l = len(LMAX_LIST)
    n_pk = len(CONV_PAIR_CASES)
    n_tc = len(CONV_TRIANGLE_CASES)
    n_k = PARAMS.n_orient

    U_single_lk = np.full((n_l, n_k), np.nan, dtype=float)
    U_single_analytic_k = np.full(n_k, np.nan, dtype=float)

    pair_phi_lc = np.full((n_l, n_pk), np.nan, dtype=float)
    pair_phi_lck = np.full((n_l, n_pk, n_k), np.nan, dtype=float)

    tri_Phi3_lc = np.full((n_l, n_tc), np.nan, dtype=float)
    tri_pairwise_lc = np.full((n_l, n_tc), np.nan, dtype=float)
    tri_Delta3_lc = np.full((n_l, n_tc), np.nan, dtype=float)
    tri_eta_pair_lc = np.full((n_l, n_tc), np.nan, dtype=float)
    tri_eta_sym_lc = np.full((n_l, n_tc), np.nan, dtype=float)

    for il, lmax in enumerate(LMAX_LIST):
        print_header(f"lmax={lmax}")
        U_single_k, U_single_analytic = compute_single_energy_k(int(lmax), normals, PARAMS)
        U_single_lk[il] = U_single_k
        if il == 0:
            U_single_analytic_k[:] = U_single_analytic

        # Pair convergence cases.
        pair_cache: Dict[Tuple[float, float], float] = {}
        for ic, (r_over_d, beta) in enumerate(CONV_PAIR_CASES):
            r = float(r_over_d) * PARAMS.d
            centers = pair_oriented_centers(r, beta)
            U_pair_k, _ = compute_cluster_energy_k(centers, int(lmax), normals, PARAMS)
            phi_k = U_pair_k - 2.0 * U_single_k
            pair_phi_lck[il, ic] = phi_k
            pair_phi_lc[il, ic] = float(np.mean(phi_k))
            pair_cache[(round(r_over_d, 12), round(beta, 12))] = pair_phi_lc[il, ic]
            print(f"Pair case {ic}: r/d={r_over_d}, beta={beta}, phi={pair_phi_lc[il,ic]:+.8e}")

        # Triangle convergence cases. Pairwise baseline is computed on the fly
        # at the same lmax for the exact edge distances and betas of each triangle.
        local_pair_cache: Dict[Tuple[float, float], float] = {}

        def get_pair_phi(distance: float, beta_deg: float) -> float:
            key = (round(distance / PARAMS.d, 10), round(float(beta_deg), 10))
            if key not in local_pair_cache:
                centers_p = pair_oriented_centers(float(distance), float(beta_deg))
                U2_k, _ = compute_cluster_energy_k(centers_p, int(lmax), normals, PARAMS)
                local_pair_cache[key] = float(np.mean(U2_k - 2.0 * U_single_k))
            return local_pair_cache[key]

        for ic, (r12_od, r13_od, gamma, psi, alpha) in enumerate(CONV_TRIANGLE_CASES):
            centers = triangle_centers(r12_od * PARAMS.d, r13_od * PARAMS.d, gamma, psi, alpha)
            info = edge_info(centers)
            pairwise = 0.0
            for dist, beta in zip(info["distances"], info["beta_deg"]):
                pairwise += get_pair_phi(float(dist), float(beta))
            U3_k, _ = compute_cluster_energy_k(centers, int(lmax), normals, PARAMS)
            phi3 = float(np.mean(U3_k - 3.0 * U_single_k))
            delta = phi3 - pairwise
            tri_Phi3_lc[il, ic] = phi3
            tri_pairwise_lc[il, ic] = pairwise
            tri_Delta3_lc[il, ic] = delta
            tri_eta_pair_lc[il, ic] = eta_pair(delta, pairwise)
            tri_eta_sym_lc[il, ic] = eta_sym(delta, phi3, pairwise)
            print(
                f"Tri case {ic}: r12/d={r12_od}, r13/d={r13_od}, gamma={gamma}, psi={psi}, alpha={alpha}, "
                f"Phi3={phi3:+.8e}, Delta3={delta:+.8e}"
            )

        save_lmax_convergence(
            U_single_lk,
            U_single_analytic_k,
            pair_phi_lck,
            pair_phi_lc,
            tri_Phi3_lc,
            tri_pairwise_lc,
            tri_Delta3_lc,
            tri_eta_pair_lc,
            tri_eta_sym_lc,
        )
        print("Saved intermediate:", LMAX_CONVERGENCE_FILE)

    save_lmax_convergence(
        U_single_lk,
        U_single_analytic_k,
        pair_phi_lck,
        pair_phi_lc,
        tri_Phi3_lc,
        tri_pairwise_lc,
        tri_Delta3_lc,
        tri_eta_pair_lc,
        tri_eta_sym_lc,
    )
    print_header(f"Done. Saved: {LMAX_CONVERGENCE_FILE}")
    print(f"Total time = {(time.time()-t_start)/60:.2f} min")


def save_lmax_convergence(
    U_single_lk,
    U_single_analytic_k,
    pair_phi_lck,
    pair_phi_lc,
    tri_Phi3_lc,
    tri_pairwise_lc,
    tri_Delta3_lc,
    tri_eta_pair_lc,
    tri_eta_sym_lc,
) -> None:
    np.savez(
        LMAX_CONVERGENCE_FILE,
        lmax_list=LMAX_LIST,
        pair_cases=np.array(CONV_PAIR_CASES, dtype=float),
        triangle_cases=np.array(CONV_TRIANGLE_CASES, dtype=float),
        U_single_lk=U_single_lk,
        U_single_analytic_k=U_single_analytic_k,
        pair_phi_lck=pair_phi_lck,
        pair_phi_lc=pair_phi_lc,
        tri_Phi3_lc=tri_Phi3_lc,
        tri_pairwise_lc=tri_pairwise_lc,
        tri_Delta3_lc=tri_Delta3_lc,
        tri_eta_pair_lc=tri_eta_pair_lc,
        tri_eta_sym_lc=tri_eta_sym_lc,
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
    run_lmax_convergence_checks()
