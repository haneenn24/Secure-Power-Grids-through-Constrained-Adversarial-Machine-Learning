"""
run_fdia_experiment.py — Core FDIA Experiment Driver
----------------------------------------------------

This file is the *central engine* of the entire research experiment.
It performs the FULL experimental loop described in the paper-style
workflow:

    1. Pick a meter-placement distribution
    2. Build a list of meters for that distribution
    3. Compute the baseline (no-attack) measurements
    4. For each attacker strength (20%, 40%, 60%, 80%):
           - randomly choose which meters the attacker compromises
           - run a full FDIA attack simulation
           - compute perceived load after attack
           - compute J-value (residual) after attack
           - measure “impact vs detectability”
    5. Store *every trial* into a CSV file

This file glues together all core components of the project:

  • topology loading (from pandapower_backend.load_real_topology)
  • meter placement logic (meter_placement.get_meter_list)
  • attacker selection (attacker_selection.pick_random_compromised)
  • baseline state computation (pandapower_backend.compute_baseline)
  • FDIA attack execution (pandapower_backend.run_fdia_attack)

It does NOT perform physics or optimization itself — it only manages:
     - looping structure
     - orchestration
     - consistency across setups
     - writing results to CSV

Outputs:
    A CSV where each row corresponds to a single FDIA trial containing:
        - topology name
        - meter distribution
        - attacker fraction
        - trial_id
        - true_load (MW)
        - perceived load baseline
        - J_baseline
        - attacked perceived load
        - J_attack
        - delta_J
        - perceived_load_drop_percent
        - success_flag

This CSV is then consumed by `visualize_fdia_results.py` to generate:
    - Figure 8–style scatter plots
    - histograms
    - heatmaps
    - success-rate curves
    - boxplots

In short:
    run_experiment.py  → loads YAML → calls THIS FILE
    THIS FILE         → produces the experimental dataset
    visualization.py  → turns dataset into final research figures
"""


import csv
import os
from typing import List, Dict, Any, Optional
import numpy as np

from meter_placement import get_meter_list
from attacker_selection import pick_random_compromised
# from matlab_interface import (
#     start_matlab,
#     load_real_topology,
#     compute_baseline,
#     run_fdia_attack,
# )
from pandapower_backend import (
    start_matlab,
    load_real_topology,
    compute_baseline,
    run_fdia_attack,
)




# ------------------------------------------------------------
# Helper: ensure directory exists
# ------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------
# Main experiment function
# ------------------------------------------------------------

def run_fdia_meter_distribution_experiment(
    results_csv_path: str,
    topology_name: str,
    meter_distributions: List[str],
    compromised_fractions: List[float],
    num_trials_per_setting: int,
    target_load_drop: float,
    config: Optional[Dict[str, Any]] = None,
):

    """
    This function runs the FULL experiment, looping over:

        - meter distributions
        - compromised fractions
        - N random trials for each setting

    For each trial:
        - Generate meter list for that distribution
        - Compute baseline (true load, perceived load, J_baseline)
        - Pick compromised meters randomly
        - Run FDIA attack through MATLAB backend
        - Log results into CSV

    Args:
        results_csv_path: where to save the CSV
        topology_name: case name, e.g. "ACTIVSg500_real"
        meter_distributions: list of names, e.g. ["uniform", "dense"]
        compromised_fractions: list of floats, e.g. [0.2, 0.4]
        num_trials_per_setting: N random trials
        target_load_drop: 0.20 means 20% lower perceived load target
        config: optional extra config dict passed to MATLAB backend
    """

    print("============================================================")
    print("Running FDIA meter-placement experiment")
    print("============================================================")
    print(f"Topology: {topology_name}")
    print(f"Meter distributions: {meter_distributions}")
    print(f"Attacker fractions: {compromised_fractions}")
    print(f"Trials per setting: {num_trials_per_setting}")
    print(f"Target load drop: {target_load_drop*100:.1f}%")
    print("Results file:", results_csv_path)
    print("============================================================\n")

    # Ensure results folder exists
    ensure_dir(os.path.dirname(results_csv_path))

    # Start MATLAB engine (or fail cleanly)
    eng = start_matlab()

    # Load topology (MATLAB should return bus/line lists)
    topology = load_real_topology(eng, topology_name)

    # Prepare CSV writer
    with open(results_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "topology",
                "meter_distribution",
                "compromised_fraction",
                "trial_id",
                "true_load",
                "perceived_load_baseline",
                "J_baseline",
                "perceived_load_attack",
                "J_attack",
                "delta_J",
                "perceived_load_drop_percent",
                "success_flag",
            ],
        )
        writer.writeheader()

        global_trial_id = 0

        # Loop over meter distributions
        for dist in meter_distributions:

            print(f"\n=== Distribution: {dist} ===")

            # Compute meter list for this distribution
            try:
                # meter_list = get_meter_list(dist, topology)
                rng = np.random.default_rng(42)

                # 1) Always compute the paper distribution meter count first
                paper_meters = get_meter_list("paper", topology)
                TARGET = len(paper_meters)

                # 2) Then produce normalized meters
                meter_list = get_meter_list(dist, topology, target_meter_count=TARGET, rng=rng)

            except Exception as e:
                print(f"ERROR in meter placement for '{dist}': {e}")
                continue

            # Compute baseline metrics (no attack)
            print("Computing baseline (no attack)...")
            try:
                baseline = compute_baseline(
                    eng, topology_name, meter_list, config=config
                )
            except NotImplementedError:
                print("compute_baseline() not implemented yet.")
                raise
            except Exception as e:
                print("ERROR computing baseline:", e)
                continue

            true_load = baseline["true_load"]
            perceived_load_baseline = baseline["perceived_load"]
            J_baseline = baseline["J_baseline"]

            # Store baseline metrics for printing
            print(f"Baseline true load        = {true_load:.3f}")
            print(f"Baseline perceived load   = {perceived_load_baseline:.3f}")
            print(f"Baseline J-value          = {J_baseline:.6f}")

            # Loop over compromised fractions
            for frac in compromised_fractions:

                print(f"\n  -> Attacker controls {frac*100:.0f}% of meters")

                # Run N random trials
                for trial in range(num_trials_per_setting):
                    rng = np.random.default_rng(seed=global_trial_id)

                    compromised_meters = pick_random_compromised(
                        meter_list, frac, rng
                    )

                    # Run FDIA attack via MATLAB
                    try:
                        attack_result = run_fdia_attack(
                            eng=eng,
                            topology_name=topology_name,
                            meter_list=meter_list,
                            compromised_meters=compromised_meters,
                            target_load_drop=target_load_drop,
                            rng_seed=global_trial_id,
                            config=config,
                        )
                    except NotImplementedError:
                        print("run_fdia_attack() not implemented yet.")
                        raise
                    except Exception as e:
                        print("ERROR running FDIA attack:", e)
                        continue

                    # Determine "success"
                    drop_percent = attack_result["perceived_load_drop_percent"]
                    delta_J = attack_result["delta_J"]
                    success_flag = 1 if drop_percent >= target_load_drop*100 else 0

                    # Write CSV row
                    row = {
                        "topology": topology_name,
                        "meter_distribution": dist,
                        "compromised_fraction": frac,
                        "trial_id": trial,
                        "true_load": true_load,
                        "perceived_load_baseline": perceived_load_baseline,
                        "J_baseline": J_baseline,
                        "perceived_load_attack": attack_result["perceived_load_attack"],
                        "J_attack": attack_result["J_attack"],
                        "delta_J": delta_J,
                        "perceived_load_drop_percent": drop_percent,
                        "success_flag": success_flag,
                    }
                    writer.writerow(row)

                    global_trial_id += 1

    print("\nExperiment completed (skeleton).")
