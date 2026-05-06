"""
analyze_triplet_results.py

Purpose
-------
Read NPZ files produced by the SCM triplet diagnostic scripts and generate summary
figures and CSV tables.

Run order
---------
1. python run_00_pair_cache.py
2. python run_01_triplet_linear.py
3. python run_02_triplet_equilateral.py
4. python run_03_triplet_angle.py
5. python analyze_triplet_results.py

Outputs
-------
Figures and CSV files are saved to:
    results_triplet_scm/analysis_figures/

What to look at first
---------------------
1. fig_linear_Delta3_vs_r.png
   Shows where the linear triplet is nonadditive.

2. fig_equilateral_Delta3_vs_r.png
   Shows where the compact triangular triplet is nonadditive.

3. fig_angle_Delta3_vs_gamma_lmax6.png
   Shows the geometry dependence of the three-body correction.

4. fig_convergence_lmax_linear_equilateral.png
   Shows whether lmax=6 is converged relative to lower lmax values.

5. summary_cross_checks.txt
   Checks whether:
       angle gamma=60 deg matches equilateral,
       angle gamma=180 deg matches linear.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

from scm_config import ANALYSIS_DIR, LINEAR_FILE, EQUILATERAL_FILE, ANGLE_FILE, PAIR_CACHE_FILE


def ensure_inputs() -> None:
    for f in [PAIR_CACHE_FILE, LINEAR_FILE, EQUILATERAL_FILE, ANGLE_FILE]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing input file: {f}")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)


def idx_lmax(lmax_arr, value: int) -> int:
    hits = np.where(lmax_arr.astype(int) == int(value))[0]
    if len(hits) == 0:
        raise KeyError(f"lmax={value} not found in {lmax_arr}")
    return int(hits[0])


def save_csv(path: str, header: str, arr: np.ndarray) -> None:
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


def plot_family_vs_r(x, y_lm, lmax_arr, title, ylabel, path, logy=False):
    plt.figure(figsize=(8, 5))
    for il, lm in enumerate(lmax_arr):
        if logy:
            plt.semilogy(x, np.abs(y_lm[il]), "o-", label=f"lmax={lm}")
        else:
            plt.plot(x, y_lm[il], "o-", label=f"lmax={lm}")
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.xlabel("r/d")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    ensure_inputs()

    pair = np.load(PAIR_CACHE_FILE)
    lin = np.load(LINEAR_FILE)
    equi = np.load(EQUILATERAL_FILE)
    ang = np.load(ANGLE_FILE)

    lmax = lin["lmax_list"].astype(int)
    il6 = idx_lmax(lmax, 6)

    # ------------------------------------------------------------------
    # Pair baseline
    # ------------------------------------------------------------------
    x_pair = pair["r_pair_over_d"]
    phi_pair = pair["phi_pair_avg_lr"]
    plot_family_vs_r(
        x_pair,
        phi_pair,
        lmax,
        "Pair baseline: phase-averaged excess energy",
        "phi_pair [J]",
        os.path.join(ANALYSIS_DIR, "fig_pair_phi_vs_r.png"),
    )
    save_csv(
        os.path.join(ANALYSIS_DIR, "table_pair_phi_avg.csv"),
        "r_over_d," + ",".join([f"phi_lmax{lm}" for lm in lmax]),
        np.column_stack([x_pair, phi_pair.T]),
    )

    # ------------------------------------------------------------------
    # Linear triplet
    # ------------------------------------------------------------------
    x_lin = lin["r_over_d"]
    plot_family_vs_r(
        x_lin,
        lin["Phi3_lr"],
        lmax,
        "Linear triplet: total excess energy Phi3",
        "Phi3 [J]",
        os.path.join(ANALYSIS_DIR, "fig_linear_Phi3_vs_r.png"),
    )
    plot_family_vs_r(
        x_lin,
        lin["Phi_pairwise_lr"],
        lmax,
        "Linear triplet: pairwise estimate",
        "Phi_pairwise [J]",
        os.path.join(ANALYSIS_DIR, "fig_linear_pairwise_vs_r.png"),
    )
    plot_family_vs_r(
        x_lin,
        lin["Delta3_lr"],
        lmax,
        "Linear triplet: non-pairwise contribution Delta3",
        "Delta3 [J]",
        os.path.join(ANALYSIS_DIR, "fig_linear_Delta3_vs_r.png"),
    )
    plot_family_vs_r(
        x_lin,
        lin["eta3_pair_lr"],
        lmax,
        "Linear triplet: relative nonadditivity |Delta3|/|pairwise|",
        "eta3_pair",
        os.path.join(ANALYSIS_DIR, "fig_linear_eta3_pair_vs_r.png"),
        logy=True,
    )

    # Local diagnostic for linear at lmax=6.
    plt.figure(figsize=(8, 5))
    Phi_particles = lin["Phi_particles_lrp"][il6]
    for p in range(3):
        label = "left edge" if p == 0 else ("center" if p == 1 else "right edge")
        plt.plot(x_lin, Phi_particles[:, p], "o-", label=label)
    plt.xlabel("r/d")
    plt.ylabel("Phi_i [J]")
    plt.title("Linear triplet: local diagnostic contributions, lmax=6")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "fig_linear_local_parts_lmax6.png"), dpi=300)
    plt.close()

    save_csv(
        os.path.join(ANALYSIS_DIR, "table_linear_summary_lmax6.csv"),
        "r_over_d,Phi3,Phi_pairwise,Delta3,eta3_pair,eta3_sym,Phi_center",
        np.column_stack([
            x_lin,
            lin["Phi3_lr"][il6],
            lin["Phi_pairwise_lr"][il6],
            lin["Delta3_lr"][il6],
            lin["eta3_pair_lr"][il6],
            lin["eta3_sym_lr"][il6],
            lin["Phi_center_lr"][il6],
        ]),
    )

    # ------------------------------------------------------------------
    # Equilateral triplet
    # ------------------------------------------------------------------
    x_eq = equi["r_over_d"]
    plot_family_vs_r(
        x_eq,
        equi["Phi3_lr"],
        lmax,
        "Equilateral triplet: total excess energy Phi3",
        "Phi3 [J]",
        os.path.join(ANALYSIS_DIR, "fig_equilateral_Phi3_vs_r.png"),
    )
    plot_family_vs_r(
        x_eq,
        equi["Delta3_lr"],
        lmax,
        "Equilateral triplet: non-pairwise contribution Delta3",
        "Delta3 [J]",
        os.path.join(ANALYSIS_DIR, "fig_equilateral_Delta3_vs_r.png"),
    )
    plot_family_vs_r(
        x_eq,
        equi["eta3_pair_lr"],
        lmax,
        "Equilateral triplet: relative nonadditivity |Delta3|/|pairwise|",
        "eta3_pair",
        os.path.join(ANALYSIS_DIR, "fig_equilateral_eta3_pair_vs_r.png"),
        logy=True,
    )

    plt.figure(figsize=(8, 5))
    Phi_particles_eq = equi["Phi_particles_lrp"][il6]
    for p in range(3):
        plt.plot(x_eq, Phi_particles_eq[:, p], "o-", label=f"particle {p+1}")
    plt.xlabel("r/d")
    plt.ylabel("Phi_i [J]")
    plt.title("Equilateral triplet: local symmetry diagnostic, lmax=6")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "fig_equilateral_local_parts_lmax6.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    for il, lm in enumerate(lmax):
        plt.semilogy(x_eq, equi["Phi_particles_maxdev_lr"][il], "o-", label=f"lmax={lm}")
    plt.xlabel("r/d")
    plt.ylabel("max_i |Phi_i - mean(Phi_i)| [J]")
    plt.title("Equilateral triplet: local symmetry error after phase averaging")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "fig_equilateral_symmetry_error.png"), dpi=300)
    plt.close()

    save_csv(
        os.path.join(ANALYSIS_DIR, "table_equilateral_summary_lmax6.csv"),
        "r_over_d,Phi3,Phi_pairwise,Delta3,eta3_pair,eta3_sym,symmetry_maxdev",
        np.column_stack([
            x_eq,
            equi["Phi3_lr"][il6],
            equi["Phi_pairwise_lr"][il6],
            equi["Delta3_lr"][il6],
            equi["eta3_pair_lr"][il6],
            equi["eta3_sym_lr"][il6],
            equi["Phi_particles_maxdev_lr"][il6],
        ]),
    )

    # ------------------------------------------------------------------
    # Angle scan
    # ------------------------------------------------------------------
    x_ang = ang["r_over_d_values"]
    gamma = ang["gamma_deg"]

    # lmax=6 geometry scans.
    for quantity, ylabel, fname, title in [
        ("Phi3_lrg", "Phi3 [J]", "fig_angle_Phi3_vs_gamma_lmax6.png", "Angular triplet: Phi3 vs gamma, lmax=6"),
        ("Phi_center_lrg", "Phi_center [J]", "fig_angle_center_vs_gamma_lmax6.png", "Angular triplet: central diagnostic vs gamma, lmax=6"),
        ("Delta3_lrg", "Delta3 [J]", "fig_angle_Delta3_vs_gamma_lmax6.png", "Angular triplet: Delta3 vs gamma, lmax=6"),
        ("eta3_pair_lrg", "eta3_pair", "fig_angle_eta3_pair_vs_gamma_lmax6.png", "Angular triplet: relative nonadditivity vs gamma, lmax=6"),
    ]:
        plt.figure(figsize=(8, 5))
        for ir, xr in enumerate(x_ang):
            y = ang[quantity][il6, ir, :]
            if "eta" in quantity:
                plt.semilogy(gamma, y, "o-", label=f"r/d={xr:g}")
            else:
                plt.plot(gamma, y, "o-", label=f"r/d={xr:g}")
        plt.axhline(0.0, color="k", linewidth=0.8)
        plt.xlabel("gamma [deg]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, fname), dpi=300)
        plt.close()

    # lmax convergence for angle scan at each r.
    for ir, xr in enumerate(x_ang):
        plt.figure(figsize=(8, 5))
        for il, lm in enumerate(lmax):
            plt.plot(gamma, ang["Delta3_lrg"][il, ir, :], "o-", label=f"lmax={lm}")
        plt.axhline(0.0, color="k", linewidth=0.8)
        plt.xlabel("gamma [deg]")
        plt.ylabel("Delta3 [J]")
        plt.title(f"Angular triplet: Delta3 lmax convergence, r/d={xr:g}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, f"fig_angle_Delta3_lmax_convergence_r{xr:g}.png"), dpi=300)
        plt.close()

    # Save angle tables for lmax=6.
    for ir, xr in enumerate(x_ang):
        save_csv(
            os.path.join(ANALYSIS_DIR, f"table_angle_summary_lmax6_r{xr:g}.csv"),
            "gamma_deg,r23_over_d,Phi3,Phi_center,Phi_pairwise,Delta3,eta3_pair,eta3_sym",
            np.column_stack([
                gamma,
                ang["r23_over_d_rg"][ir],
                ang["Phi3_lrg"][il6, ir],
                ang["Phi_center_lrg"][il6, ir],
                ang["Phi_pairwise_lrg"][il6, ir],
                ang["Delta3_lrg"][il6, ir],
                ang["eta3_pair_lrg"][il6, ir],
                ang["eta3_sym_lrg"][il6, ir],
            ]),
        )

    # ------------------------------------------------------------------
    # Convergence summary: difference between consecutive lmax values.
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for name, data, x in [
        ("linear", lin["Phi3_lr"], x_lin),
        ("equilateral", equi["Phi3_lr"], x_eq),
    ]:
        d65 = np.abs(data[idx_lmax(lmax, 6)] - data[idx_lmax(lmax, 5)])
        d54 = np.abs(data[idx_lmax(lmax, 5)] - data[idx_lmax(lmax, 4)])
        plt.semilogy(x, d65, "o-", label=f"{name}: |Phi6-Phi5|")
        plt.semilogy(x, d54, "s--", label=f"{name}: |Phi5-Phi4|")
    plt.xlabel("r/d")
    plt.ylabel("absolute lmax increment [J]")
    plt.title("SCM lmax convergence diagnostic for Phi3")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "fig_convergence_lmax_linear_equilateral.png"), dpi=300)
    plt.close()

    # ------------------------------------------------------------------
    # Cross-checks: angular endpoints vs separately calculated families.
    # ------------------------------------------------------------------
    lines = []
    lines.append("Cross-checks for SCM triplet geometries")
    lines.append("========================================")
    lines.append("Expected:")
    lines.append("  angle gamma=60 deg  should match equilateral for same r/d.")
    lines.append("  angle gamma=180 deg should match linear for same r/d.")
    lines.append("")

    ig60 = int(np.where(np.isclose(gamma, 60.0))[0][0])
    ig180 = int(np.where(np.isclose(gamma, 180.0))[0][0])

    for xr in x_ang:
        ir_ang = int(np.where(np.isclose(x_ang, xr))[0][0])
        ir_lin = int(np.where(np.isclose(x_lin, xr))[0][0])
        ir_eq = int(np.where(np.isclose(x_eq, xr))[0][0])
        diff_eq = ang["Phi3_lrg"][:, ir_ang, ig60] - equi["Phi3_lr"][:, ir_eq]
        diff_lin = ang["Phi3_lrg"][:, ir_ang, ig180] - lin["Phi3_lr"][:, ir_lin]
        lines.append(f"r/d={xr:g}")
        for il, lm in enumerate(lmax):
            lines.append(
                f"  lmax={lm}: angle60-equilateral={diff_eq[il]:+.6e} J, "
                f"angle180-linear={diff_lin[il]:+.6e} J"
            )
        lines.append("")

    with open(os.path.join(ANALYSIS_DIR, "summary_cross_checks.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Analysis completed.")
    print("Figures and tables saved to:", ANALYSIS_DIR)


if __name__ == "__main__":
    main()
