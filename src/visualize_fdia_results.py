import matplotlib
matplotlib.use("Agg")   # headless backend

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import time

CSV_PATH = "results/fdia_meter_placement.csv"
OUT_DIR = "results/plots"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
def load_data():
    df = pd.read_csv(CSV_PATH)
    print("Loaded", len(df), "rows.")

    # ORDER distributions (+ sort)
    order = ["paper", "uniform", "generator_heavy", "load_heavy", "sparse", "dense"]
    df["meter_distribution"] = pd.Categorical(df["meter_distribution"],
                                              categories=order,
                                              ordered=True)
    df = df.sort_values(["meter_distribution", "compromised_fraction"])

    # Compute the %ΔJ = ((J_attack - J_baseline) / J_baseline) * 100
    df["percent_delta_J"] = (df["delta_J"] / df["J_baseline"]) * 100

    # Already have perceived_load_drop_percent from experiment
    # (Make sure it's numeric)
    df["perceived_load_drop_percent"] = pd.to_numeric(df["perceived_load_drop_percent"],
                                                      errors="coerce")

    return df


# ------------------------------------------------------------
# 1. Figure 8 – J vs perceived load drop (scatter)
# ------------------------------------------------------------
def plot_fig8(df):
    ensure_dir(OUT_DIR)
    for dist in df["meter_distribution"].unique():

        sub = df[df["meter_distribution"] == dist]
        if sub.empty:
            continue

        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=sub,
            x="perceived_load_drop_percent",
            y="delta_J",
            hue="compromised_fraction",
            palette="viridis",
            alpha=0.7
        )
        plt.title(f"Figure 8-style scatter – {dist}")
        plt.xlabel("Perceived Load Drop (%)")
        plt.ylabel("ΔJ (Detectability)")
        plt.grid(True)
        plt.savefig(f"{OUT_DIR}/fig8_scatter_{dist}.png", dpi=300)
        plt.close()


# ------------------------------------------------------------
# 2. Histograms (Load Drop + ΔJ)
# ------------------------------------------------------------
def plot_histograms(df):
    ensure_dir(OUT_DIR)

    for dist in df["meter_distribution"].unique():
        sub = df[df["meter_distribution"] == dist]
        if sub.empty:
            continue

        # Load drop histogram
        plt.figure(figsize=(7, 5))
        sns.histplot(data=sub, x="perceived_load_drop_percent",
                     bins=30, kde=True)
        plt.title(f"Histogram – Load Drop – {dist}")
        plt.savefig(f"{OUT_DIR}/hist_load_{dist}.png", dpi=300)
        plt.close()

        # ΔJ histogram
        plt.figure(figsize=(7, 5))
        sns.histplot(data=sub, x="delta_J", bins=30, kde=True)
        plt.title(f"Histogram – ΔJ – {dist}")
        plt.savefig(f"{OUT_DIR}/hist_J_{dist}.png", dpi=300)
        plt.close()


# ------------------------------------------------------------
# 3. SUCCESS RATE
# ------------------------------------------------------------
def plot_success_rate(df, drop_threshold=15, deltaJ_threshold=None):
    ensure_dir(OUT_DIR)

    if deltaJ_threshold is None:
        deltaJ_threshold = df["delta_J"].median()

    df["success"] = (
        (df["perceived_load_drop_percent"] >= drop_threshold) &
        (df["delta_J"] <= deltaJ_threshold)
    )

    success_rates = (
        df.groupby(["meter_distribution", "compromised_fraction"])["success"]
            .mean()
            .reset_index()
    )

    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=success_rates,
        x="compromised_fraction",
        y="success",
        hue="meter_distribution",
        marker="o"
    )
    plt.title(f"Success Rate vs Attacker Strength\n(drop≥{drop_threshold}%, ΔJ≤{deltaJ_threshold:.2f})")
    plt.ylabel("Success Probability")
    plt.grid(True)
    plt.savefig(f"{OUT_DIR}/success_rate.png", dpi=300)
    plt.close()


# ------------------------------------------------------------
# 4. HEATMAP (median load drop)
# ------------------------------------------------------------
def plot_heatmap(df):
    ensure_dir(OUT_DIR)

    pivot = df.pivot_table(
        index="meter_distribution",
        columns="compromised_fraction",
        values="perceived_load_drop_percent",
        aggfunc="median"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap="coolwarm", fmt=".1f")
    plt.title("Median Load Drop (%)")
    plt.savefig(f"{OUT_DIR}/heatmap_load_drop.png", dpi=300)
    plt.close()


# ------------------------------------------------------------
# 5. FIGURE 8 – BOX PLOTS (detectability + impact)
#     ***This is the main new addition***
# ------------------------------------------------------------
def plot_figure8_boxplots(df):
    ensure_dir(OUT_DIR)

    # axis ranges matched to paper
    detectability_ylim = (-50, 150)
    impact_ylim = (-25, 5)

    for dist in df["meter_distribution"].unique():
        sub = df[df["meter_distribution"] == dist]
        if sub.empty:
            continue

        plt.figure(figsize=(10, 8))

        # -------------------------
        # TOP PANEL — Detectability
        # -------------------------
        plt.subplot(2, 1, 1)
        sns.boxplot(
            data=sub,
            x="compromised_fraction",
            y="percent_delta_J",
            color="skyblue"
        )
        plt.title(f"Attack Detectability (N trials) – {dist}")
        plt.ylabel("% Change in J-Value from baseline")
        plt.ylim(detectability_ylim)
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)

        # -------------------------
        # BOTTOM PANEL — Impact
        # -------------------------
        plt.subplot(2, 1, 2)
        sns.boxplot(
            data=sub,
            x="compromised_fraction",
            y="perceived_load_drop_percent",
            color="lightgreen"
        )
        plt.title(f"Attack Impact – {dist}")
        plt.ylabel("% Change in Operator Perceived Demand")
        plt.ylim(impact_ylim)
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/fig8_boxplots_{dist}.png", dpi=300)
        plt.close()


# ------------------------------------------------------------
# 6. RUNTIME PLOT
# ------------------------------------------------------------
def plot_runtime(df):
    ensure_dir(OUT_DIR)

    # We assume each row is one trial → simulate runtime per trial
    # (If your experiment logged timestamps, we would use those,
    #  but here we approximate by ordering.)
    df = df.copy()
    df["trial_index"] = np.arange(len(df))

    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=df,
        x="trial_index",
        y="delta_J",
        hue="meter_distribution",
        alpha=0.7
    )
    plt.title("Runtime-like Plot (delta_J vs trial order)\nProxy for computational behavior")
    plt.ylabel("ΔJ")
    plt.grid(True)
    plt.savefig(f"{OUT_DIR}/runtime_proxy.png", dpi=300)
    plt.close()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    start = time.time()

    df = load_data()

    plot_fig8(df)
    plot_histograms(df)
    plot_success_rate(df)
    plot_heatmap(df)
    plot_figure8_boxplots(df)
    plot_runtime(df)

    end = time.time()
    print(f"All plots generated in: {OUT_DIR}")
    print(f"Total runtime of plotting script: {end - start:.2f} seconds")
