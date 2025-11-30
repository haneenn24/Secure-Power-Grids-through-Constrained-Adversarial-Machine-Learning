# ===============================================================
# visualize_fdia_results.py  --  Fig. 8-style boxplots (YOUR DATA)
# ===============================================================

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

CSV = "results/fdia_meter_placement.csv"
OUT = "results/plots"


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def main():
    ensure_dir(OUT)

    # ---------------------------
    # Load CSV
    # ---------------------------
    df = pd.read_csv(CSV)

    # Normalize column names
    df = df.rename(columns={
        "meter_distribution": "dist",
        "compromised_fraction": "frac",
        "percent_delta_J": "deltaJ",
        "perceived_load_drop_percent": "drop_percent",
    })

    df["dist"] = df["dist"].astype(str).str.strip().str.lower()
    df["frac"] = df["frac"].astype(float)
    df["deltaJ"] = pd.to_numeric(df["deltaJ"], errors="coerce")
    df["drop_percent"] = pd.to_numeric(df["drop_percent"], errors="coerce")

    # Drop NaN rows
    df = df[np.isfinite(df.deltaJ) & np.isfinite(df.drop_percent)]

    # ---------------------------
    # PAPER distribution only
    # ---------------------------
    paper = df[df["dist"] == "paper"].copy()
    if paper.empty:
        print("[ERROR] No rows for dist='paper'.")
        return

    # Create categorical fractions sorted (0.1...1.0)
    frac_order = sorted(paper["frac"].unique())
    paper["frac_cat"] = pd.Categorical(
        [f"{x:.1f}" for x in paper["frac"]],
        categories=[f"{x:.1f}" for x in frac_order],
        ordered=True
    )

    # ---------------------------
    # Plot (NO forced y-limits)
    # ---------------------------
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

    # --- TOP (ΔJ) ---
    sns.boxplot(data=paper, x="frac_cat", y="deltaJ", ax=ax1)
    ax1.set_title("Attack Detectability", fontsize=15)
    ax1.set_ylabel("% Change in J-Value from baseline")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    # --- BOTTOM (% drop) ---
    sns.boxplot(data=paper, x="frac_cat", y="drop_percent", ax=ax2)
    ax2.set_title("Attack Impact", fontsize=15)
    ax2.set_ylabel("% Change in Operator Perceived Demand")
    ax2.set_xlabel("Proportion of Compromised Measurement Devices")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Use clean tick labels: "0.1 0.2 ... 1.0"
    ax2.set_xticklabels([f"{x:.1f}" for x in frac_order])

    plt.tight_layout()
    out_file = f"{OUT}/fig8_paper_autoY.png"
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"[OK] Saved: {out_file}")


if __name__ == "__main__":
    main()
