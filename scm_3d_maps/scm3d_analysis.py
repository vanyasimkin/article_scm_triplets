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
    """Analyze the full general-triangle map.

    The current output format uses arrays with dimensions
        (r12, r13, gamma, psi, alpha).
    A fallback for the older compact q-scan format is kept for compatibility.
    """
    if not TRIANGLE_ASYMMETRIC_FILE.exists():
        print("Skip full-triangle analysis: file not found", TRIANGLE_ASYMMETRIC_FILE)
        return

    data = np.load(TRIANGLE_ASYMMETRIC_FILE)

    # New full-scan format.
    if "r13_over_d" in data.files and "Delta3_rrgpa" in data.files:
        r12 = data["r12_over_d"]
        r13 = data["r13_over_d"]
        gamma = data["gamma_deg"]
        psi = data["psi_deg"]
        alpha = data["alpha_deg"]
        phi3 = data["Phi3_avg_rrgpa"]
        pairwise = data["Phi_pairwise_rrgpa"]
        delta = data["Delta3_rrgpa"]
        eta = data["eta3_pair_rrgpa"]
        eta_sym = data["eta3_sym_rrgpa"]
        min_gap = data["min_gap_rrgpa"] if "min_gap_rrgpa" in data.files else np.full_like(delta, np.nan)

        rows: List[Dict[str, object]] = []
        for i12, r12v in enumerate(r12):
            for i13, r13v in enumerate(r13):
                for ig, gv in enumerate(gamma):
                    for ip, pv in enumerate(psi):
                        for ia, av in enumerate(alpha):
                            val = delta[i12, i13, ig, ip, ia]
                            if not np.isfinite(val):
                                continue
                            rows.append(
                                {
                                    "r12_over_d": float(r12v),
                                    "r13_over_d": float(r13v),
                                    "gamma_deg": float(gv),
                                    "psi_deg": float(pv),
                                    "alpha_deg": float(av),
                                    "Phi3_avg_J": float(phi3[i12, i13, ig, ip, ia]),
                                    "Phi_pairwise_J": float(pairwise[i12, i13, ig, ip, ia]),
                                    "Delta3_J": float(delta[i12, i13, ig, ip, ia]),
                                    "eta3_pair": float(eta[i12, i13, ig, ip, ia]),
                                    "eta3_sym": float(eta_sym[i12, i13, ig, ip, ia]),
                                    "min_gap_over_d": float(min_gap[i12, i13, ig, ip, ia]),
                                }
                            )
        save_csv(
            TABLE_DIR / "triangle_full_summary.csv",
            rows,
            [
                "r12_over_d",
                "r13_over_d",
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

        # Example maps for representative r12/r13/alpha combinations.
        examples = [
            (1.20, 1.20, 0.0),
            (1.20, 2.00, 0.0),
            (1.20, 2.00, 45.0),
            (2.00, 1.20, 0.0),
        ]
        for r12_target, r13_target, alpha_target in examples:
            i12 = _nearest_index(r12, r12_target)
            i13 = _nearest_index(r13, r13_target)
            ia = _nearest_index(alpha, alpha_target)
            _heatmap(
                x=gamma,
                y=psi,
                z=delta[i12, i13, :, :, ia].T,
                xlabel="gamma, deg",
                ylabel="psi, deg",
                title=(
                    f"Full triangle Delta3, J; r12/d={r12[i12]:.2f}, "
                    f"r13/d={r13[i13]:.2f}, alpha={alpha[ia]:.0f}"
                ),
                path=(
                    FIG_DIR
                    / f"fig_full_triangle_Delta3_gamma_psi_r12_{r12[i12]:.2f}_r13_{r13[i13]:.2f}_alpha{alpha[ia]:.0f}.png"
                ),
            )

        # Symmetry diagnostic: compare Delta3(r12,r13) and Delta3(r13,r12)
        # at the same gamma/psi/alpha index. This is not the exact relabelling
        # operation for arbitrary alpha, but it is a useful quick diagnostic.
        if len(r12) == len(r13) and np.allclose(r12, r13):
            diff = delta - np.swapaxes(delta, 0, 1)
            denom = np.maximum(np.abs(delta) + np.abs(np.swapaxes(delta, 0, 1)), 1e-300)
            sym = 2.0 * np.abs(diff) / denom
            rows_sym: List[Dict[str, object]] = [
                {
                    "symmetry_metric_mean": float(np.nanmean(sym)),
                    "symmetry_metric_max": float(np.nanmax(sym)),
                    "note": "Quick r12/r13 swap diagnostic at fixed gamma, psi, alpha; exact particle relabelling may also change alpha.",
                }
            ]
            save_csv(TABLE_DIR / "triangle_full_r12_r13_symmetry_diagnostic.csv", rows_sym, ["symmetry_metric_mean", "symmetry_metric_max", "note"])
        return

    # Older compact q-scan fallback.
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
