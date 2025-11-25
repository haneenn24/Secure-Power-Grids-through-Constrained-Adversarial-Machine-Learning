"""
run_experiment.py — Master entry point for the FDIA experiment pipeline
-----------------------------------------------------------------------

This script is the *top-level orchestrator* of the entire FDIA experiment.
It does not run physics, attacks, or state estimation itself. Instead, it:

  1. Loads the experiment configuration from configs/config.yaml
     The YAML file specifies:
         - which topology to load (e.g., SC500)
         - which meter placement distributions to test
         - attacker compromised fractions (20%, 40%, etc.)
         - number of trials per setting (e.g., N = 100 or 1000)
         - target load-drop percentage (e.g., 20%)
         - output CSV file

  2. Prints the loaded configuration so the user can verify it.

  3. Calls:
        run_fdia_meter_distribution_experiment(...)
     which performs the full experiment:
         - loads topology once
         - loops over all meter distributions
         - generates meters for each distribution
         - computes baseline measurements
         - loops over attacker fractions
         - runs many random FDIA trials (N)
         - collects true_load, perceived_load, ΔJ, etc.
         - writes every result row into a CSV file

This file is therefore the **master controller** of the pipeline.
It glues together:
    - YAML configuration
    - experiment driver (run_fdia_experiment.py)
    - backend physics engine (pandapower_backend.py)
    - meter placement strategy (meter_placement.py)
    - attacker selection logic (attacker_selection.py)

You run the entire research experiment simply with:

    python run_experiment.py

which produces:
    results/*.csv

These CSV files are then consumed by:
    visualize_fdia_results.py

to generate:
    - Figure 8 style plots
    - histograms
    - heatmaps
    - boxplots
    - success-rate curves
    - KDE plots
"""


import yaml
import os
from run_fdia_experiment import run_fdia_meter_distribution_experiment


def main():
    # Path to config
    config_path = os.path.join("configs", "config.yaml")

    # Load YAML
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    topology_name = cfg["topology"]
    meter_distributions = cfg["meter_distributions"]
    compromised_fractions = cfg["compromised_fractions"]
    num_trials = cfg["num_trials_per_setting"]
    target_load_drop = cfg["target_load_drop"]
    results_csv = cfg["results_csv"]

    matlab_config = cfg.get("matlab_config", {})

    print("Loaded config:")
    print(f"  Topology              : {topology_name}")
    print(f"  Distributions         : {meter_distributions}")
    print(f"  Compromised fractions : {compromised_fractions}")
    print(f"  Trials per setting    : {num_trials}")
    print(f"  Target load drop      : {target_load_drop}")
    print(f"  Output CSV            : {results_csv}\n")

    # Run experiment
    run_fdia_meter_distribution_experiment(
        results_csv_path=results_csv,
        topology_name=topology_name,
        meter_distributions=meter_distributions,
        compromised_fractions=compromised_fractions,
        num_trials_per_setting=num_trials,
        target_load_drop=target_load_drop,
        config=matlab_config,
    )


if __name__ == "__main__":
    main()
