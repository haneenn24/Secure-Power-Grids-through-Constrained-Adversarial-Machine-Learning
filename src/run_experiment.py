"""
run_experiment.py

Loads experiment configuration from YAML and runs the full FDIA experiment.
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
