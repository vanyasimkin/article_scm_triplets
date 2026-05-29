"""
scm3d_analysis.py

Create compact CSV summaries and publication-style diagnostic figures from the
3D SCM maps.

Run from repository root:

    python -m scm_3d_maps.scripts.analyze_3d_maps
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt

from .scm3d_config import (
    FIG_DIR,
    LMAX_CONVERGENCE_FILE,
    PAIR_ORIENTATION_FILE,
    TABLE_DIR,
    TRIANGLE_ASYMMETRIC_FILE,
    TRIANGLE_ISOSCELES_FILE,
)
from .scm3d_utils import print_header, save_csv


def _nearest_index(arr: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(arr, dtype=float) - float(value))))


def _heatmap(x, y, z, xlabel, ylabel, title, path: Path):
    plt.figure(figsize=(8, 5.5))
    extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]
    # z is expected as (len(y), len(x)) for visual axes y,x.
    plt.imshow(z, origin="lower", aspect="auto", extent=extent)
    plt.colorbar(label=title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def analyze_pair_map() -> None:
    if not PAIR_ORIENTATION_FILE.exists():
        print("Skip pair analysis: file not found", PAIR_ORIENTATION_FILE)
        return

    data = np.load(PAIR_ORIENTATION_FILE)
    r_over_d = data["r_over_d"]
    beta_deg = data["beta_deg"]
    phi = data["phi_pair_avg_rb"]

    rows: List[Dict[str, object]] = []
    for ir, r in enumerate(r_over_d):
        for ib, beta in enumerate(beta_deg):
            rows.append({"r_over_d": float(r), "beta_deg": float(beta), "phi_pair_avg_J": float(phi[ir, ib])})
    save_csv(TABLE_DIR / "pair_orientation_summary.csv", rows, ["r_over_d", "beta_deg", "phi_pair_avg_J"])

    _heatmap(
        x=r_over_d,
        y=beta_deg,
        z=phi.T,
        xlabel="r/d",
        ylabel="beta, deg",
        title="phi_pair(r,beta), J",
        path=FIG_DIR / "fig_pair_phi_map_r_beta.png",
    )

    # Far-tail diagnostic r^3 phi for selected beta values.
    plt.figure(figsize=(8, 5))
    for beta_target in [0.0, 30.0, 60.0, 90.0]:
        ib = _nearest_index(beta_deg, beta_target)
        plt.plot(r_over_d, (r_over_d ** 3) * phi[:, ib], "o-", label=f"beta={beta_deg[ib]:.0f} deg")
    plt.xlabel("r/d")
    plt.ylabel("(r/d)^3 * phi_pair [J]")
    plt.title("Pair far-tail diagnostic")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_pair_far_tail_r3_phi.png", dpi=300)
    plt.close()


def analyze_isosceles_map() -> None:
    if not TRIANGLE_ISOSCELES_FILE.exists():
        print("Skip isosceles analysis: file not found", TRIANGLE_ISOSCELES_FILE)
        return

    data = np.load(TRIANGLE_ISOSCELES_FILE)
    r = data["r_over_d"]
    gamma = data["gamma_deg"]
    psi = data["psi_deg"]
    alpha = data["alpha_deg"]
    phi3 = data["Phi3_avg_rgpa"]
    pairwise = data["Phi_pairwise_rgpa"]
    delta = data["Delta3_rgpa"]
    eta = data["eta3_pair_rgpa"]
    eta_sym = data["eta3_sym_rgpa"]
    min_gap = data["min_gap_rgpa"]

    rows: List[Dict[str, object]] = []
    for ir, rv in enumerate(r):
        for ig, gv in enumerate(gamma):
            for ip, pv in enumerate(psi):
                for ia, av in enumerate(alpha):
                    rows.append(
                        {
                            "r_over_d": float(rv),
                            "gamma_deg": float(gv),
                            "psi_deg": float(pv),
                            "alpha_deg": float(av),
                            "Phi3_avg_J": float(phi3[ir, ig, ip, ia]),
                            "Phi_pairwise_J": float(pairwise[ir, ig, ip, ia]),
                            "Delta3_J": float(delta[ir, ig, ip, ia]),
                            "eta3_pair": float(eta[ir, ig, ip, ia]),
                            "eta3_sym": float(eta_sym[ir, ig, ip, ia]),
                            "min_gap_over_d": float(min_gap[ir, ig, ip, ia]),
                        }
                    )
    save_csv(
        TABLE_DIR / "triangle_isosceles_summary.csv",
        rows,
        [
            "r_over_d",
            "gamma_deg",
            "psi_deg",
            "alpha_deg",
            "Phi3_avg_J",
            "Phi_pairwise_J",
            "Delta3_J",
            "eta3_pair",
            "eta3_sym",
            "min_gap_over_d",
        ],
    )

    # Main maps at r/d=1.20 for selected alpha values.
    ir = _nearest_index(r, 1.20)
    for alpha_target in [0.0, 45.0, 90.0]:
        ia = _nearest_index(alpha, alpha_target)
        _heatmap(
            x=gamma,
            y=psi,
            z=delta[ir, :, :, ia].T,
            xlabel="gamma, deg",
            ylabel="psi, deg",
            title=f"Delta3, J; r/d={r[ir]:.2f}, alpha={alpha[ia]:.0f} deg",
            path=FIG_DIR / f"fig_isosceles_Delta3_gamma_psi_r{r[ir]:.2f}_alpha{alpha[ia]:.0f}.png",
        )
        _heatmap(
            x=gamma,
            y=psi,
            z=eta[ir, :, :, ia].T,
            xlabel="gamma, deg",
            ylabel="psi, deg",
            title=f"eta3_pair; r/d={r[ir]:.2f}, alpha={alpha[ia]:.0f} deg",
            path=FIG_DIR / f"fig_isosceles_eta3_gamma_psi_r{r[ir]:.2f}_alpha{alpha[ia]:.0f}.png",
        )

    # Distance decay slices at alpha=0, psi=0/45/90, gamma=60/90/120/180.
    ia0 = _nearest_index(alpha, 0.0)
    plt.figure(figsize=(8, 5))
    for gv in [60.0, 90.0, 120.0, 180.0]:
        ig = _nearest_index(gamma, gv)
        ip = _nearest_index(psi, 0.0)
        plt.plot(r, delta[:, ig, ip, ia0], "o-", label=f"gamma={gamma[ig]:.0f}, psi=0")
    plt.axhline(0.0, linewidth=0.8)
    plt.xlabel("r/d")
    plt.ylabel("Delta3 [J]")
    plt.title("Isosceles triangle: Delta3 distance slices, alpha=0")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_isosceles_Delta3_vs_r_alpha0_psi0.png", dpi=300)
    plt.close()


def analyze_asymmetric_map() -> None:
    if not TRIANGLE_ASYMMETRIC_FILE.exists():
        print("Skip asymmetric analysis: file not found", TRIANGLE_ASYMMETRIC_FILE)
        return

    data = np.load(TRIANGLE_ASYMMETRIC_FILE)
    r12 = data["r12_over_d"]
    q = data["q"]
    r13 = data["r13_over_d_rq"]
    gamma = data["gamma_deg"]
    psi = data["psi_deg"]
    alpha = data["alpha_deg"]
    valid = data["valid_rq"]
    phi3 = data["Phi3_avg_rqgpa"]
    pairwise = data["Phi_pairwise_rqgpa"]
    delta = data["Delta3_rqgpa"]
    eta = data["eta3_pair_rqgpa"]
    eta_sym = data["eta3_sym_rqgpa"]

    rows: List[Dict[str, object]] = []
    for ir, r12v in enumerate(r12):
        for iq, qv in enumerate(q):
            if not bool(valid[ir, iq]):
                continue
            for ig, gv in enumerate(gamma):
                for ip, pv in enumerate(psi):
                    for ia, av in enumerate(alpha):
                        val = delta[ir, iq, ig, ip, ia]
                        if not np.isfinite(val):
                            continue
                        rows.append(
                            {
                                "r12_over_d": float(r12v),
                                "q_r13_over_r12": float(qv),
                                "r13_over_d": float(r13[ir, iq]),
                                "gamma_deg": float(gv),
                                "psi_deg": float(pv),
                                "alpha_deg": float(av),
                                "Phi3_avg_J": float(phi3[ir, iq, ig, ip, ia]),
                                "Phi_pairwise_J": float(pairwise[ir, iq, ig, ip, ia]),
                                "Delta3_J": float(delta[ir, iq, ig, ip, ia]),
                                "eta3_pair": float(eta[ir, iq, ig, ip, ia]),
                                "eta3_sym": float(eta_sym[ir, iq, ig, ip, ia]),
                            }
                        )
    save_csv(
        TABLE_DIR / "triangle_asymmetric_summary.csv",
        rows,
        [
            "r12_over_d",
            "q_r13_over_r12",
            "r13_over_d",
            "gamma_deg",
            "psi_deg",
            "alpha_deg",
            "Phi3_avg_J",
            "Phi_pairwise_J",
            "Delta3_J",
            "eta3_pair",
            "eta3_sym",
        ],
    )

    # Example map at r12/d=1.2, q=1.5, alpha=0.
    ir = _nearest_index(r12, 1.20)
    iq = _nearest_index(q, 1.50)
    ia = _nearest_index(alpha, 0.0)
    if bool(valid[ir, iq]):
        _heatmap(
            x=gamma,
            y=psi,
            z=delta[ir, iq, :, :, ia].T,
            xlabel="gamma, deg",
            ylabel="psi, deg",
            title=f"Asymmetric Delta3, J; r12/d={r12[ir]:.2f}, q={q[iq]:.2f}, alpha={alpha[ia]:.0f}",
            path=FIG_DIR / f"fig_asymmetric_Delta3_gamma_psi_r12_{r12[ir]:.2f}_q{q[iq]:.2f}.png",
        )


def analyze_lmax_convergence() -> None:
    if not LMAX_CONVERGENCE_FILE.exists():
        print("Skip lmax analysis: file not found", LMAX_CONVERGENCE_FILE)
        return

    data = np.load(LMAX_CONVERGENCE_FILE)
    lmax = data["lmax_list"]
    pair_cases = data["pair_cases"]
    tri_cases = data["triangle_cases"]
    pair_phi = data["pair_phi_lc"]
    tri_phi3 = data["tri_Phi3_lc"]
    tri_delta = data["tri_Delta3_lc"]

    rows: List[Dict[str, object]] = []
    for il, lm in enumerate(lmax):
        for ic, case in enumerate(pair_cases):
            rows.append({"kind": "pair", "lmax": int(lm), "case_index": ic, "r_over_d": float(case[0]), "beta_deg": float(case[1]), "value_J": float(pair_phi[il, ic])})
        for ic, case in enumerate(tri_cases):
            rows.append(
                {
                    "kind": "triangle_Delta3",
                    "lmax": int(lm),
                    "case_index": ic,
                    "r12_over_d": float(case[0]),
                    "r13_over_d": float(case[1]),
                    "gamma_deg": float(case[2]),
                    "psi_deg": float(case[3]),
                    "alpha_deg": float(case[4]),
                    "value_J": float(tri_delta[il, ic]),
                }
            )
    save_csv(TABLE_DIR / "lmax_convergence_summary.csv", rows, sorted(set().union(*(r.keys() for r in rows))))

    # Plot first several triangle convergence cases.
    plt.figure(figsize=(8, 5))
    for ic in range(min(6, tri_delta.shape[1])):
        plt.plot(lmax, tri_delta[:, ic], "o-", label=f"case {ic}")
    plt.xlabel("l_max")
    plt.ylabel("Delta3 [J]")
    plt.title("Selected triangle Delta3 convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_lmax_convergence_triangle_Delta3.png", dpi=300)
    plt.close()


def run_analysis() -> None:
    print_header("Analyze SCM 3D maps")
    analyze_pair_map()
    analyze_isosceles_map()
    analyze_asymmetric_map()
    analyze_lmax_convergence()
    print_header("Analysis complete")
    print("Figures:", FIG_DIR)
    print("Tables :", TABLE_DIR)


if __name__ == "__main__":
    run_analysis()
