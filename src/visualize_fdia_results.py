# ===============================================================
# visualize_fdia_results.py — FIXED FOR YOUR CSV FORMAT
# ===============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUT = "results/plots"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_all(df):

    ensure_dir(OUT)

    # Rename to consistent internal names
    df = df.rename(columns={
        "meter_distribution": "dist",
        "compromised_fraction": "frac",
        "percent_delta_J": "deltaJ",
        "perceived_load_drop_percent": "drop_percent"
    })

    # Convert types
    df["frac"] = df["frac"].astype(float)
    df["deltaJ"] = df["deltaJ"].astype(float)
    df["drop_percent"] = df["drop_percent"].astype(float)

    # ==========================
    # FIGURE 8 — BOX PLOTS
    # ==========================
    for dist in df["dist"].unique():

        sub = df[df["dist"] == dist].sort_values("frac")

        plt.figure(figsize=(10,8))

        # Detectability (ΔJ)
        plt.subplot(2,1,1)
        sns.boxplot(data=sub, x="frac", y="deltaJ")
        plt.title(f"Fig 8 (Top) Detectability (ΔJ) — {dist}")
        plt.grid(True, axis='y', linestyle='--', alpha=0.4)

        # Impact (% load drop)
        plt.subplot(2,1,2)
        sns.boxplot(data=sub, x="frac", y="drop_percent")
        plt.title(f"Fig 8 (Bottom) Impact (% drop) — {dist}")
        plt.grid(True, axis='y', linestyle='--', alpha=0.4)

        plt.tight_layout()
        plt.savefig(f"{OUT}/fig8_{dist}.png", dpi=300)
        plt.close()

    print("[OK] All fig8 plots generated.")


def main():
    df = pd.read_csv("results/fdia_meter_placement.csv")
    plot_all(df)

if __name__ == "__main__":
    main()
