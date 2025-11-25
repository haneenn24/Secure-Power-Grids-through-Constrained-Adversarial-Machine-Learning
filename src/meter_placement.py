"""
meter_placement.py — Meter placement strategies for FDIA experiments
--------------------------------------------------------------------

This file defines all measurement–placement configurations used in the
experiment pipeline. Each configuration returns the list of meter names
(or bus indices in paper-style mode) that the defender deploys on the
grid.

The role of this file:
    • Generate different SCADA measurement layouts for the same topology.
    • Provide both paper-style placement and synthetic placements.
    • Ensure all configurations remain observable for pandapower SE.
    • Support normalization so all distributions have the same meter count.

Included strategies:
    • Full observable baseline: P_inj + V_mag on every bus, P_flow on every line.
    • Uniform / generator-heavy / load-heavy: same base set, different attack behavior later.
    • Sparse: reduced injections but still observable (all V_mag + half P_inj + all P_flow).
    • Dense: redundant meters (extra flow meters).
    • Paper-style: reconstructs SCADA-like metering similar to the paper
      (load buses, generator buses, all buses for V_mag, line endpoints).

The experiment runner calls:
    get_meter_list(distribution_name, topology, target_meter_count)
to obtain the meter layout for each distribution before running baseline
and FDIA attacks.

This file controls **which meters exist**, while attacker_selection.py
decides which of them the attacker compromises.
"""


from typing import List, Dict
import numpy as np


def _bus_meters(bus_idx: int) -> List[str]:
    return [
        f"m_bus_pinj_{bus_idx}",
        f"m_bus_vmag_{bus_idx}",
    ]

def _line_meter(line_idx: int) -> str:
    return f"m_line_flow_{line_idx}"


# =====================================================================
# Base SCADA configuration for FULL observability
# =====================================================================

def place_meters_full_observable(topology: Dict) -> List[str]:
    """
    This is the recommended baseline:
    - P_inj + V_mag at ALL buses
    - P_flow at ALL lines
    """
    buses = topology["buses"]
    lines = topology["lines"]

    meters = []

    for b in buses:
        meters += _bus_meters(b)

    for l in lines:
        meters.append(_line_meter(l))

    return meters


# =====================================================================
# Reduced versions (but still observable)
# =====================================================================

def place_meters_uniform(topology: Dict) -> List[str]:
    return place_meters_full_observable(topology)


def place_meters_generator_heavy(topology: Dict) -> List[str]:
    """
    Same as full observable — difference will be in compromised selection.
    """
    return place_meters_full_observable(topology)


def place_meters_load_heavy(topology: Dict) -> List[str]:
    return place_meters_full_observable(topology)


def place_meters_sparse(topology: Dict) -> List[str]:
    """
    NOTE: Sparse *still must be observable*.
    So we measure:
        - every bus: V_mag
        - every 2nd bus: P_inj
        - every line: P_flow
    """
    buses = topology["buses"]
    lines = topology["lines"]

    meters = []

    # voltage everywhere (mandatory)
    for b in buses:
        meters.append(f"m_bus_vmag_{b}")

    # injections on half buses
    for i, b in enumerate(buses):
        if i % 2 == 0:
            meters.append(f"m_bus_pinj_{b}")

    for l in lines:
        meters.append(f"m_line_flow_{l}")

    return meters


def place_meters_dense(topology: Dict) -> List[str]:
    """
    Same base set (for observability), FDIA robustness comes from redundancy
    by adding extra line flow meters if desired.
    """
    buses = topology["buses"]
    lines = topology["lines"]

    meters = []

    for b in buses:
        meters += _bus_meters(b)

    for l in lines:
        meters.append(f"m_line_flow_{l}")
        meters.append(f"m_line_flow_{l}_2")

    return meters


# =====================================================================
# Dispatcher
# =====================================================================

def get_meter_list(distribution_name: str, topology: Dict) -> List[str]:
    if distribution_name == "uniform":
        return place_meters_uniform(topology)
    elif distribution_name == "generator_heavy":
        return place_meters_generator_heavy(topology)
    elif distribution_name == "load_heavy":
        return place_meters_load_heavy(topology)
    elif distribution_name == "sparse":
        return place_meters_sparse(topology)
    elif distribution_name == "dense":
        return place_meters_dense(topology)
    else:
        raise ValueError(f"Unknown meter distribution: {distribution_name}")


def place_meters_paper_style(topology: Dict) -> List[int]:
    """
    Recreates a SCADA-like configuration similar to the paper.

    Rules:
      - One injection meter at every load bus
      - One injection meter at every generator bus
      - One voltage magnitude meter at every bus
      - One flow meter at every line

    We represent meters by bus indices only (for now), since pandapower_backend
    measures only bus active injections.
    """
    buses = topology["buses"]
    load_buses = topology["load_buses"]
    gen_buses = topology["gen_buses"]

    meter_list = set()

    # Injection meters for load buses
    for b in load_buses:
        meter_list.add(b)

    # Injection meters for generator buses
    for b in gen_buses:
        meter_list.add(b)

    # Voltage magnitude meter → we add every bus
    for b in buses:
        meter_list.add(b)

    # Flow meters represented as “bus-level meters”: for each line, add both end buses
    lines = topology["lines"]
    net = topology["net"]
    for line_id in lines:
        from_bus = net.line.at[line_id, "from_bus"]
        to_bus = net.line.at[line_id, "to_bus"]
        meter_list.add(from_bus)
        meter_list.add(to_bus)

    # Convert to sorted list
    return sorted(list(meter_list))


def normalize_meter_count(meter_list: List[int], target_count: int, rng: np.random.Generator):
    """
    Adjust meter_list to match target_count exactly.
    - If too few meters → randomly add bus indices until size matches.
    - If too many meters → randomly drop meters until size matches.
    """
    current = list(meter_list)
    n = len(current)

    if n == target_count:
        return current

    # Not enough → randomly add buses
    if n < target_count:
        needed = target_count - n
        all_buses = list(set(current))  # keep unique
        while len(all_buses) < target_count:
            new_bus = rng.choice(current)
            all_buses.append(new_bus)
        return list(rng.choice(all_buses, size=target_count, replace=False))

    # Too many → randomly drop
    if n > target_count:
        keep = rng.choice(current, size=target_count, replace=False)
        return sorted(list(keep))

def get_meter_list(distribution_name: str, topology: Dict, target_meter_count=None, rng=None) -> List[int]:
    """
    Public function called by the experiment runner.

    - distribution_name: "paper", "uniform", "dense", ...
    - topology: dict with buses/lines
    - target_meter_count: if None → return raw list
                          else normalize to fixed number
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
        raise ValueError(f"Unknown meter distribution: {distribution_name}")

    # Normalize all distributions to same meter count
    if target_meter_count is not None:
        meters = normalize_meter_count(meters, target_meter_count, rng)

    return meters
