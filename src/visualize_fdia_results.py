#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# Ensure output directory exists
# ---------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------
CSV_PATH = "results/fdia_meter_placement.csv"
OUT_DIR = "results/plots"

df = pd.read_csv(CSV_PATH)
ensure_dir(OUT_DIR)

print(f"Loaded {len(df)} rows.")


# ---------------------------------------------------------
# Precompute fixed columns
# ---------------------------------------------------------
df["abs_detectability"] = df["J_attack"].abs()
df["xcat"] = df["compromised_fraction"].astype(str)   # categorical axis for perfect boxplots


# ---------------------------------------------------------
# PLOT 1 + 2: Boxplots for Detectability + Impact (Figure 8)
# ---------------------------------------------------------
def plot_boxplots(sub, dist_name):

    out_dir = f"{OUT_DIR}/{dist_name}"
    ensure_dir(out_dir)

    plt.figure(figsize=(12, 10))

    # ---------------------- TOP: Detectability ----------------------
    plt.subplot(2, 1, 1)
    sns.boxplot(
        data=sub,
        x="xcat",
        y="abs_detectability",
        color="skyblue"
    )
    plt.title(f"Attack Detectability – {dist_name}")
    plt.xlabel("compromised_fraction")
    plt.ylabel("Absolute Detectability (|ΔJ|)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # ---------------------- BOTTOM: Impact ----------------------
    plt.subplot(2, 1, 2)
    sns.boxplot(
        data=sub,
        x="xcat",
        y="perceived_load_drop_percent",
        color="lightgreen"
    )
    plt.title(f"Attack Impact – {dist_name}")
    plt.xlabel("compromised_fraction")
    plt.ylabel("% Load Drop")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/figure8_boxplots.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# PLOT 3: Success-Rate Heatmap
# ---------------------------------------------------------
def plot_success_heatmap(sub, dist_name):

    out_dir = f"{OUT_DIR}/{dist_name}"
    ensure_dir(out_dir)

    pivot = sub.pivot_table(
        index="compromised_fraction",
        values="success_flag",
        aggfunc="mean"
    )

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        pivot,
        annot=True,
        cmap="Blues",
        cbar=True,
        fmt=".2f"
    )
    plt.title(f"Success Rate Heatmap – {dist_name}")
    plt.ylabel("compromised_fraction")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/success_heatmap.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# PLOT 4: Line Plot — Avg Detectability vs Fraction
# ---------------------------------------------------------
def plot_detectability_curve(sub, dist_name):
    out_dir = f"{OUT_DIR}/{dist_name}"
    ensure_dir(out_dir)

    means = sub.groupby("compromised_fraction")["abs_detectability"].mean()

    plt.figure(figsize=(7, 4))
    plt.plot(means.index, means.values, marker="o", linewidth=2)
    plt.title(f"Average Detectability vs Compromised Fraction – {dist_name}")
    plt.xlabel("compromised_fraction")
    plt.ylabel("Mean |ΔJ|")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/detectability_curve.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# PLOT 5: Line Plot — Avg Impact vs Fraction
# ---------------------------------------------------------
def plot_impact_curve(sub, dist_name):
    out_dir = f"{OUT_DIR}/{dist_name}"
    ensure_dir(out_dir)

    means = sub.groupby("compromised_fraction")["perceived_load_drop_percent"].mean()

    plt.figure(figsize=(7, 4))
    plt.plot(means.index, means.values, marker="o", color="green", linewidth=2)
    plt.title(f"Average Load Drop vs Comp Fraction – {dist_name}")
    plt.xlabel("compromised_fraction")
    plt.ylabel("Mean % Load Drop")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/impact_curve.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# PLOT 6: Scatter – Detectability vs Impact
# ---------------------------------------------------------
def plot_scatter_detectability_vs_impact(sub, dist_name):

    out_dir = f"{OUT_DIR}/{dist_name}"
    ensure_dir(out_dir)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=sub,
        x="abs_detectability",
        y="perceived_load_drop_percent",
        hue="compromised_fraction",
        palette="viridis",
        s=60
    )
    plt.title(f"Detectability vs Impact – {dist_name}")
    plt.xlabel("|ΔJ|")
    plt.ylabel("% Load Drop")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/detectability_vs_impact_scatter.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# MAIN LOOP FOR ALL DISTRIBUTIONS
# ---------------------------------------------------------
for dist in df["meter_distribution"].unique():

    print(f"[INFO] Plotting all figures for distribution = {dist}")
    sub = df[df["meter_distribution"] == dist]

    if sub.empty:
        continue

    # 1 + 2
    plot_boxplots(sub, dist)

    # 3
    plot_success_heatmap(sub, dist)

    # 4
    plot_detectability_curve(sub, dist)

    # 5
    plot_impact_curve(sub, dist)

    # 6
    plot_scatter_detectability_vs_impact(sub, dist)

print(f"All plots saved in: {OUT_DIR}")
