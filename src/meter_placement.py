"""
meter_placement.py — Final version
----------------------------------

Defines all SCADA meter-placement distributions for FDIA experiments.
All distributions return bus indices only (int), consistent with the
pandapower backend and the FDIA formulation used in this project.

This file supports normalization so each distribution can be scaled to
the same meter count.
"""

from typing import List, Dict
import numpy as np


# ------------------------------------------------------------
# PAPER-STYLE DISTRIBUTION
# ------------------------------------------------------------

def place_meters_paper_style(topology: Dict) -> List[int]:
    """
    SCADA-like configuration used in the paper:
        - injection meter on each load bus
        - injection meter on each generator bus
        - voltage meter on every bus
        - flow meters approximated by adding line endpoints
    Returns: list of bus indices
    """
    buses = topology["buses"]
    load_buses = topology["load_buses"]
    gen_buses = topology["gen_buses"]
    net = topology["net"]

    meter_set = set()

    # injection meters
    for b in load_buses:
        meter_set.add(b)

    for b in gen_buses:
        meter_set.add(b)

    # voltage magnitude on all buses
    for b in buses:
        meter_set.add(b)

    # flow meters -> endpoints
    for line_id in topology["lines"]:
        fb = net.line.at[line_id, "from_bus"]
        tb = net.line.at[line_id, "to_bus"]
        meter_set.add(int(fb))
        meter_set.add(int(tb))

    # Convert to sorted python ints
    return sorted(int(m) for m in meter_set)


# ------------------------------------------------------------
# SYNTHETIC DISTRIBUTIONS
# ------------------------------------------------------------

def place_meters_uniform(topology: Dict) -> List[int]:
    """
    For now identical to paper-style (the difference is only attacker selection).
    """
    return place_meters_paper_style(topology)


def place_meters_generator_heavy(topology: Dict) -> List[int]:
    return place_meters_paper_style(topology)


def place_meters_load_heavy(topology: Dict) -> List[int]:
    return place_meters_paper_style(topology)


def place_meters_sparse(topology: Dict) -> List[int]:
    """
    Sparse but still observable:
        - voltage meter at every bus
        - injection only on half of the buses
        - endpoints of every line
    """
    buses = topology["buses"]
    net = topology["net"]

    meters = set()

    # voltage everywhere
    for b in buses:
        meters.add(b)

    # injection on half buses
    for i, b in enumerate(buses):
        if i % 2 == 0:
            meters.add(b)

    # line endpoints
    for l in topology["lines"]:
        fb = net.line.at[l, "from_bus"]
        tb = net.line.at[l, "to_bus"]
        meters.add(int(fb))
        meters.add(int(tb))

    return sorted(int(m) for m in meters)


def place_meters_dense(topology: Dict) -> List[int]:
    """
    Dense = more redundancy:
        - all buses
        - all line endpoints
        (bus-index based so redundancy is conceptual)
    """
    buses = topology["buses"]
    net = topology["net"]

    meters = set(buses)

    for l in topology["lines"]:
        fb = net.line.at[l, "from_bus"]
        tb = net.line.at[l, "to_bus"]
        meters.add(int(fb))
        meters.add(int(tb))

    return sorted(int(m) for m in meters)


# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------

def normalize_meter_count(meter_list: List[int],
                          target_count: int,
                          rng: np.random.Generator) -> List[int]:
    """
    Make all distributions comparable by matching the same meter count:
        - if too many meters → drop randomly
        - if too few meters → sample with replacement
    """
    meter_list = [int(m) for m in meter_list]  # ensure ints
    n = len(meter_list)

    if n == target_count:
        return sorted(meter_list)

    if n > target_count:
        # randomly drop
        keep = rng.choice(meter_list, size=target_count, replace=False)
        return sorted(int(m) for m in keep)

    # n < target_count → add more by sampling existing ones
    needed = target_count - n
    extra = rng.choice(meter_list, size=needed, replace=True)
    full = meter_list + extra.tolist()
    return sorted(int(m) for m in full)


# ------------------------------------------------------------
# DISPATCHER
# ------------------------------------------------------------

def get_meter_list(distribution_name: str,
                   topology: Dict,
                   target_meter_count: int = None,
                   rng: np.random.Generator = None) -> List[int]:
    """
    Public API used by run_fdia_experiment.
    Always returns a list of python ints for bus indices.
    """

    if rng is None:
        rng = np.random.default_rng()

    if distribution_name == "paper":
        meters = place_meters_paper_style(topology)

    elif distribution_name == "uniform":
        meters = place_meters_uniform(topology)

    elif distribution_name == "generator_heavy":
        meters = place_meters_generator_heavy(topology)

    elif distribution_name == "load_heavy":
        meters = place_meters_load_heavy(topology)

    elif distribution_name == "sparse":
        meters = place_meters_sparse(topology)

    elif distribution_name == "dense":
        meters = place_meters_dense(topology)

    else:
        raise ValueError(f"Unknown distribution: {distribution_name}")

    # ensure pure python ints
    meters = [int(m) for m in meters]

    # normalize if needed
    if target_meter_count is not None:
        meters = normalize_meter_count(meters, target_meter_count, rng)

    return meters
