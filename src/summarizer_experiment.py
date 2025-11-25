"""
summarize_experiment.py

Offline diagnostic tool for the FDIA meter-placement experiment.

It does three main things:

1. Load the *results CSV* produced by run_experiment.py.
2. Reload the *topology* (SC-500 / ACTIVSg500) through pandapower_backend.
3. Rebuild the *meter distributions* and print a structured summary:

   - Topology summary:
       * number of buses / lines / load buses / generator buses
       * basic load and generation totals

   - Meter distribution summary:
       * for each distribution (paper, uniform, generator_heavy, load_heavy, sparse, dense):
           - number of meters used
           - sample of meter bus indices
           - how that compares across distributions

   - Compromised-meter & results summary:
       * for each (distribution, compromised_fraction):
           - expected number of compromised meters
           - how many trials exist in the CSV
           - average / min / max load drop
           - average / min / max ΔJ
           - success rate (mean(success_flag))

The goal is to help you "see" the structure of the experiment:
  - What topology was used?
  - How many meters per distribution?
  - How strong is the attacker (how many meters compromised)?
  - What behavior do we see in terms of load drop and J?
"""

import os
from typing import Dict, Any

import numpy as np
import pandas as pd

from pandapower_backend import load_real_topology
from meter_placement import get_meter_list


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path)


def _write_and_print(fh, text: str = ""):
    """Write a line to both stdout and the summary file."""
    print(text)
    fh.write(text + "\n")


# ---------------------------------------------------------------------
# 1. Topology summary
# ---------------------------------------------------------------------


def summarize_topology(topology_name: str, topo: Dict[str, Any], fh):
    """
    Print basic information about the grid topology:
      - number of buses, lines, load buses, generator buses
      - simple load & generation totals if available
    """
    net = topo["net"]
    buses = topo["buses"]
    lines = topo["lines"]
    load_buses = topo["load_buses"]
    gen_buses = topo["gen_buses"]

    _write_and_print(fh, "====================================================")
    _write_and_print(fh, f"TOPOLOGY SUMMARY: {topology_name}")
    _write_and_print(fh, "====================================================")
    _write_and_print(fh, f"# Buses           : {len(buses)}")
    _write_and_print(fh, f"# Lines           : {len(lines)}")
    _write_and_print(fh, f"# Load buses      : {len(load_buses)}")
    _write_and_print(fh, f"# Generator buses : {len(gen_buses)}")

    # Totals (if available)
    total_load = float(net.load["p_mw"].sum()) if len(net.load) > 0 else 0.0
    total_gen = float(net.gen["p_mw"].sum()) if len(net.gen) > 0 else 0.0

    _write_and_print(fh, f"Total load (MW)   : {total_load:.3f}")
    _write_and_print(fh, f"Total gen  (MW)   : {total_gen:.3f}")
    _write_and_print(fh, "")


# ---------------------------------------------------------------------
# 2. Meter distributions summary
# ---------------------------------------------------------------------


def summarize_meter_distributions(topo: Dict[str, Any], df: pd.DataFrame, fh):
    """
    For each meter_distribution present in the CSV, rebuild the meter list
    using meter_placement.get_meter_list (with fixed target count),
    and summarize:

      - number of meters
      - first few bus indices (sample)
    """
    _write_and_print(fh, "====================================================")
    _write_and_print(fh, "METER DISTRIBUTION SUMMARY")
    _write_and_print(fh, "====================================================")

    # Get set of distributions actually used in the experiment
    dists = list(df["meter_distribution"].unique())
    dists = [d for d in dists if isinstance(d, str)]

    _write_and_print(fh, f"Distributions found in CSV: {dists}")
    _write_and_print(fh, "")

    rng = np.random.default_rng(42)

    # First, compute "paper" meter count as baseline (if available)
    try:
        paper_meters = get_meter_list("paper", topo)
        target_count = len(paper_meters)
        _write_and_print(fh, f"Paper distribution has {target_count} meters.")
    except Exception as e:
        target_count = None
        _write_and_print(fh, f"Warning: could not build 'paper' distribution: {e}")
        _write_and_print(fh, "         Other distributions will be shown with raw meter counts.")
    _write_and_print(fh, "")

    # Summarize each distribution
    for dist in dists:
        try:
            if target_count is not None:
                meter_list = get_meter_list(dist, topo, target_meter_count=target_count, rng=rng)
            else:
                meter_list = get_meter_list(dist, topo)
        except Exception as e:
            _write_and_print(fh, f"[ERROR] Could not build meter list for '{dist}': {e}")
            _write_and_print(fh, "")
            continue

        meter_list = list(meter_list)
        n_meters = len(meter_list)

        _write_and_print(fh, f"--- Distribution: {dist} ---")
        _write_and_print(fh, f"  # meters            : {n_meters}")

        # Show a small sample of meter indices
        sample = meter_list[:10]
        _write_and_print(fh, f"  Sample meter buses  : {sample}")

        # Range of bus indices (just to see spread)
        if len(meter_list) > 0:
            _write_and_print(
                fh,
                f"  Bus index range     : {min(meter_list)} .. {max(meter_list)}",
            )

        _write_and_print(fh, "")


# ---------------------------------------------------------------------
# 3. Results summary: per distribution & compromised fraction
# ---------------------------------------------------------------------


def summarize_results_structure(df: pd.DataFrame, topo: Dict[str, Any], fh):
    """
    For each combination of (meter_distribution, compromised_fraction),
    print:

      - expected #compromised meters
      - #trials in CSV
      - mean/min/max perceived_load_drop_percent
      - mean/min/max delta_J
      - success rate
    """
    _write_and_print(fh, "====================================================")
    _write_and_print(fh, "EXPERIMENT RESULT SUMMARY (by distribution & fraction)")
    _write_and_print(fh, "====================================================")

    # We need the same meter counts as in the experiment
    rng = np.random.default_rng(123)

    # Pre-compute meter counts per distribution
    dist_to_meter_count = {}
    dists = [d for d in df["meter_distribution"].unique() if isinstance(d, str)]

    # Try to use "paper" count as target (if defined)
    try:
        paper_meters = get_meter_list("paper", topo)
        target_count = len(paper_meters)
    except Exception:
        target_count = None

    for dist in dists:
        try:
            if target_count is not None:
                meters = get_meter_list(dist, topo, target_meter_count=target_count, rng=rng)
            else:
                meters = get_meter_list(dist, topo)
            dist_to_meter_count[dist] = len(meters)
        except Exception:
            # If fails, skip; we'll handle gracefully below
            continue

    # Group by distribution and compromised_fraction
    grouped = df.groupby(["meter_distribution", "compromised_fraction"])

    for (dist, frac), sub in grouped:
        if not isinstance(dist, str):
            continue

        _write_and_print(fh, f"--- Distribution: {dist}, Compromised fraction: {frac:.2f} ---")

        # Expected #compromised meters (from reconstructed meter count)
        if dist in dist_to_meter_count:
            total_meters = dist_to_meter_count[dist]
            expected_comp = int(total_meters * frac)
            _write_and_print(fh, f"  Total meters (reconstructed)          : {total_meters}")
            _write_and_print(fh, f"  Expected compromised meters           : {expected_comp}")
        else:
            _write_and_print(fh, "  [WARNING] Could not reconstruct meter count for this distribution.")

        # Trials
        num_trials = len(sub)
        _write_and_print(fh, f"  # trials in CSV                       : {num_trials}")

        # Load drop stats
        ld = sub["perceived_load_drop_percent"].astype(float)
        _write_and_print(
            fh,
            f"  Load drop (%): mean={ld.mean():.2f}, min={ld.min():.2f}, max={ld.max():.2f}",
        )

        # ΔJ stats
        dJ = sub["delta_J"].astype(float)
        _write_and_print(
            fh,
            f"  ΔJ:            mean={dJ.mean():.3e}, min={dJ.min():.3e}, max={dJ.max():.3e}",
        )

        # Success rate
        if "success_flag" in sub.columns:
            succ = sub["success_flag"].mean()
            _write_and_print(fh, f"  Success rate (mean(success_flag))     : {succ:.3f}")
        else:
            _write_and_print(fh, "  Success rate: column 'success_flag' not found.")

        _write_and_print(fh, "")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():
    # 1) Load CSV
    csv_path = os.path.join("results", "fdia_meter_placement.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find results CSV at: {csv_path}\n"
            "Run `python src/run_experiment.py` first."
        )

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows from {csv_path}")

    if "topology" not in df.columns:
        raise ValueError("CSV does not contain 'topology' column.")

    topology_name = str(df["topology"].iloc[0])

    # 2) Reload topology
    topo = load_real_topology(None, topology_name)

    # 3) Prepare summary output file
    out_dir = os.path.join("results")
    ensure_dir(out_dir)
    summary_path = os.path.join(out_dir, "experiment_summary.txt")

    with open(summary_path, "w") as fh:
        # Topology level
        summarize_topology(topology_name, topo, fh)

        # Meter distributions (structure)
        summarize_meter_distributions(topo, df, fh)

        # Results structure & statistics
        summarize_results_structure(df, topo, fh)

    print("\n[INFO] Experiment structure summary saved to:")
    print("   ", summary_path)


if __name__ == "__main__":
    main()
