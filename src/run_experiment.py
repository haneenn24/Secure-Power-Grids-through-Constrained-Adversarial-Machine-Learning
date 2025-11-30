#!/usr/bin/env python3
# ============================================================
# run_experiment.py  --  FDIA meter placement (Fig. 8 style)
# ============================================================

import os
import sys
import yaml
import logging
import traceback
import copy
import time

from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")

from utils_power import load_matpower_file
from meter_placement import select_meter_distribution
from fdia_attack import run_fdia_attack
from utils_io import ensure_dir


CONFIG = "configs/config.yaml"
OUT_CSV = "results/fdia_meter_placement.csv"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "fdia_experiment.log")


# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
def setup_logging():
    ensure_dir(LOG_DIR)

    logger = logging.getLogger("fdia_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ------------------------------------------------------------
# CSV appender
# ------------------------------------------------------------
def append_result(dist: str,
                  frac: float,
                  trial: int,
                  J0: float,
                  J1: float,
                  L0: float,
                  L1: float):
    row = {
        "meter_distribution": dist,
        "compromised_fraction": frac,
        "trial": trial,
        "J0": J0,
        "J1": J1,
        "L0": L0,
        "L1": L1,
    }

    # Percent change in J
    denom_J = max(abs(J0), abs(J1), 1e-9)
    row["percent_delta_J"] = ((J1 - J0) / denom_J) * 100.0

    # Percent change in perceived demand
    denom_L = max(abs(L0), abs(L1), 1e-6)
    row["perceived_load_drop_percent"] = ((L0 - L1) / denom_L) * 100.0

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df = pd.DataFrame([row])
    if not os.path.exists(OUT_CSV):
        df.to_csv(OUT_CSV, index=False)
    else:
        df.to_csv(OUT_CSV, mode="a", header=False, index=False)


# ------------------------------------------------------------
# Select compromised buses: subset of load buses that have meters
# ------------------------------------------------------------
def select_compromised_buses(
    meter_bus_list: List[int],
    load_buses: List[int],
    frac: float,
) -> List[int]:
    meter_set = set(meter_bus_list)
    load_set = set(load_buses)

    # Only buses that are both load and metered
    candidates = sorted(meter_set & load_set)
    if not candidates:
        # fallback: compromise among all metered buses
        candidates = sorted(meter_set)

    k = max(1, int(round(frac * len(candidates))))
    k = min(k, len(candidates))

    return list(np.random.choice(candidates, size=k, replace=False))


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------
def main():
    logger = setup_logging()
    logger.info("Loading config...")

    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    topo_name = config["topology"]
    logger.info(f"Loading MATPOWER topology: {topo_name}")

    topology = load_matpower_file(topo_name)
    net = topology["net"]
    buses = topology["buses"]
    lines = topology["lines"]
    load_buses = topology["load_buses"]
    gen_buses = topology["gen_buses"]

    logger.info("Topology loaded successfully.")
    logger.info("--------------------------------------------------------")
    logger.info(f"Buses: {len(buses)}")
    logger.info(f"Lines: {len(lines)}")
    logger.info(f"Loads: {len(load_buses)}")
    logger.info(f"Gens : {len(gen_buses)}")
    logger.info("--------------------------------------------------------")

    distributions = config["distributions"]
    compromised_fracs = config["compromised_fractions"]
    trials = config["trials"]
    M = config["meters"]  # number of meters to place (non-paper dists)

    ensure_dir(os.path.dirname(OUT_CSV))

    seed = config.get("seed", 42)
    np.random.seed(seed)
    logger.info(f"Using random seed = {seed}")

    # --------------------------------------------------------
    # Loop over meter distributions
    # --------------------------------------------------------
    for dist in distributions:
        logger.info("")
        logger.info("====================================================")
        logger.info(f"[RUN] Distribution = {dist}")
        logger.info("====================================================")

        meter_bus_list = select_meter_distribution(topology, dist, M)

        logger.info(
            f"[INFO] Selected {len(meter_bus_list)} meters "
            f"for distribution '{dist}'"
        )

        total_iters = len(compromised_fracs) * trials
        progress = tqdm(
            total=total_iters,
            desc=f"{dist}",
            unit="run",
            leave=False,
        )

        for frac in compromised_fracs:
            for t in range(trials):
                logger.info(
                    f"[RUN] dist={dist} | fraction={frac} | "
                    f"trial={t + 1}/{trials}"
                )
                start_time = time.time()   # <--- ADDED
                compromised_bus_list = select_compromised_buses(
                    meter_bus_list,
                    load_buses,
                    frac,
                )

                try:
                    net_copy = copy.deepcopy(net)

                    J0, J1, L0, L1 = run_fdia_attack(
                        net_copy,
                        meter_bus_list,
                        compromised_bus_list,
                        load_buses,
                    )

                    append_result(dist, frac, t, J0, J1, L0, L1)

                    elapsed = time.time() - start_time   # <--- ADDED

                    logger.info(
                        f"[OK]  time={elapsed:.2f}s | "
                        f"J0={J0:.4e}, J1={J1:.4e}, "
                        f"ΔJ%={((J1 - J0) / max(abs(J0), 1e-9)) * 100.0:.2f}, "
                        f"L0={L0:.4f}, L1={L1:.4f}, "
                        f"drop%={((L0 - L1) / max(abs(L0), 1e-6)) * 100.0:.2f}"
                    )

                except Exception:
                    tb = traceback.format_exc()
                    logger.error(
                        f"[ERROR] Attack failed for dist={dist}, "
                        f"frac={frac}, trial={t}:\n{tb}"
                    )

                progress.update(1)

        progress.close()

    logger.info("")
    logger.info("====================================================")
    logger.info("Experiment COMPLETED. Results saved to:")
    logger.info(OUT_CSV)
    logger.info("====================================================")


if __name__ == "__main__":
    main()
