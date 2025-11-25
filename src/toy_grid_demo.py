"""
Toy example and skeleton for ConAML power grid project.

Part 1: Tiny toy grid to understand
- power grid topology (buses, lines)
- meters (measurement devices)
- load vs generation
- a simple FDI attack
- a simple "J-like" residual as a detection signal

Part 2: Skeleton placeholders for integrating:
- real topologies from ./Topologies/
- real FDIA implementation from ./FDIA_Attacks/
- big experiment with many random attacks and different meter distributions

NOTE:
- Part 1 is runnable now.
- Part 2 is just a skeleton (NotImplementedError) and will be filled in later.
"""

import math
import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any


# ============================================================
# 1. Basic data structures (toy world)
# ============================================================

@dataclass
class Bus:
    name: str
    generation: float  # MW, positive
    load: float        # MW, positive
    # later you can add voltage, phase, etc.


@dataclass
class Line:
    name: str
    from_bus: str
    to_bus: str
    loss: float = 0.0  # MW losses on the line (toy)


@dataclass
class Meter:
    name: str
    meter_type: str  # "bus_injection" or "line_flow"
    location: str    # bus name or line name
    true_value: float
    reported_value: float = field(init=False)

    def __post_init__(self):
        # Initially, reported value = true value (no attack)
        self.reported_value = self.true_value


# ============================================================
# 2. Build a tiny toy grid (3-bus system)
# ============================================================

def build_toy_grid():
    """
    Toy grid:

        [GenBus] --(Line1)--> [MiddleBus] --(Line2)--> [LoadBus]

    - GenBus: generates 150 MW, no load.
    - MiddleBus: no generation, small load.
    - LoadBus: big load.
    """
    buses = {
        "GenBus": Bus(name="GenBus", generation=150.0, load=0.0),
        "MiddleBus": Bus(name="MiddleBus", generation=0.0, load=20.0),
        "LoadBus": Bus(name="LoadBus", generation=0.0, load=120.0),
    }

    lines = {
        "Line1": Line(name="Line1", from_bus="GenBus", to_bus="MiddleBus", loss=2.0),
        "Line2": Line(name="Line2", from_bus="MiddleBus", to_bus="LoadBus", loss=3.0),
    }

    return buses, lines


# ============================================================
# 3. Define meters for the toy grid
# ============================================================

def build_meters(buses: Dict[str, Bus], lines: Dict[str, Line]) -> Dict[str, Meter]:
    """
    We create:
    - one bus_injection meter per bus (net injection = generation - load)
    - one line_flow meter per line (power flow from from_bus to to_bus)

    This is NOT physically exact AC power flow, just a toy to see numbers change.
    """
    meters: Dict[str, Meter] = {}

    # Bus injection meters
    for bus_name, bus in buses.items():
        injection = bus.generation - bus.load
        m_name = f"m_bus_{bus_name}"
        meters[m_name] = Meter(
            name=m_name,
            meter_type="bus_injection",
            location=bus_name,
            true_value=injection,
        )

    # Line flow meters: approximate flows from simple balance
    # For the toy, we say:
    # - Line1 carries MiddleBus.load + LoadBus.load + losses
    # - Line2 carries LoadBus.load + its own loss
    total_middle_and_load = buses["MiddleBus"].load + buses["LoadBus"].load
    line1_flow = total_middle_and_load + lines["Line1"].loss + lines["Line2"].loss
    line2_flow = buses["LoadBus"].load + lines["Line2"].loss

    meters["m_line_Line1"] = Meter(
        name="m_line_Line1",
        meter_type="line_flow",
        location="Line1",
        true_value=line1_flow,
    )
    meters["m_line_Line2"] = Meter(
        name="m_line_Line2",
        meter_type="line_flow",
        location="Line2",
        true_value=line2_flow,
    )

    return meters


# ============================================================
# 4. Simple "J-like" residual and load calculations (toy)
# ============================================================

def compute_residual_J(buses: Dict[str, Bus], meters: Dict[str, Meter]) -> float:
    """
    Toy version of a J-value:

    For each bus:
    - True injection = generation - load.
    - Estimated injection from meters = bus_injection meter's reported value.
    - J = sum of squared mismatches over all buses.

    This mirrors the idea from the paper:
    if measurements are inconsistent with physical balance,
    the J-value (residual) grows.
    """

    J = 0.0

    for bus_name, bus in buses.items():
        true_injection = bus.generation - bus.load

        m_name = f"m_bus_{bus_name}"
        reported_injection = meters[m_name].reported_value

        mismatch = reported_injection - true_injection
        J += mismatch * mismatch

    return J


def total_true_load(buses: Dict[str, Bus]) -> float:
    return sum(bus.load for bus in buses.values())


def total_perceived_load_from_meters(buses: Dict[str, Bus], meters: Dict[str, Meter]) -> float:
    """
    Toy idea of "perceived load":

    The operator looks at bus injection meters (reported). For each bus:

        injection_reported = generation - load_estimated
        => load_estimated = generation - injection_reported

    We sum estimated loads across all buses.
    """
    total_load_est = 0.0
    for bus_name, bus in buses.items():
        m_name = f"m_bus_{bus_name}"
        reported_injection = meters[m_name].reported_value
        est_load = bus.generation - reported_injection
        total_load_est += est_load
    return total_load_est


# ============================================================
# 5. Simple FDIA for the toy grid
# ============================================================

def apply_fdia_attack(meters: Dict[str, Meter], target_drop_percent: float, compromised_meter_names: List[str]):
    """
    Simple FDI attack (toy):

    - Attacker wants the operator to perceive lower load by 'target_drop_percent'.
    - Attacker can only change the meters in 'compromised_meter_names'.
    - Here we just scale the reported injection values for these meters.
      (Not physically perfect, just to show the intuition.)
    """
    factor = 1.0 + target_drop_percent  # e.g., +0.2 if we want 20% lower load

    for name in compromised_meter_names:
        m = meters[name]
        if m.meter_type == "bus_injection":
            # To reduce perceived load, the attacker increases injection
            # (generation - load_est). That makes load_est smaller.
            m.reported_value = m.true_value * factor
        else:
            # For line meters, we keep it simple and don't modify them here
            m.reported_value = m.true_value


# ============================================================
# 6. Part 1: Toy demo run
# ============================================================

def run_toy_demo():
    # High-level explanation printed to logs
    print("===================================================")
    print("TOY DEMO: basic concepts (topology, meters, FDIA, J)")
    print("===================================================\n")
    print("You will see:\n")
    print("1) Topology (grid)")
    print("   - A tiny 3-bus grid:")
    print("       GenBus (generator)")
    print("       MiddleBus")
    print("       LoadBus (big consumer)")
    print("   - Lines: Line1, Line2.\n")
    print("2) Meters")
    print("   - Meters on each bus: 'bus injection' = generation − load.")
    print("   - Meters on lines: 'line flow'.\n")
    print("3) True load vs perceived load")
    print("   - true_load = sum of real loads (from Bus objects).")
    print("   - perceived_load = what the operator reconstructs from meters.\n")
    print("4) FDIA attack")
    print("   - We simulate an attacker changing only:")
    print("       m_bus_MiddleBus")
    print("       m_bus_LoadBus")
    print("   - Target: make perceived load 20% lower.\n")
    print("5) J-like residual")
    print("   - J is computed as sum of squared mismatches between:")
    print("       true injection (generation − load)")
    print("       reported injection (from meters)")
    print("   - If the attack changes meters too much → mismatches grow")
    print("     → J grows → easier to detect.\n")
    print("This is exactly the same intuition as Figure 8 in the paper,")
    print("but with a tiny toy example you can read and modify.\n")

    # Build grid and meters
    buses, lines = build_toy_grid()
    meters = build_meters(buses, lines)

    print("=== TOPOLOGY (TOY GRID) ===")
    print("Buses:")
    for b in buses.values():
        print(f"  {b.name}: generation={b.generation} MW, load={b.load} MW")

    print("\nLines:")
    for l in lines.values():
        print(f"  {l.name}: {l.from_bus} -> {l.to_bus}, loss={l.loss} MW")

    print("\nMeters (TRUE values, before any attack):")
    for m in meters.values():
        print(f"  {m.name}: type={m.meter_type}, location={m.location}, true_value={m.true_value:.2f}")

    # Baseline (no attack)
    J_baseline = compute_residual_J(buses, meters)
    true_load = total_true_load(buses)
    perceived_load_baseline = total_perceived_load_from_meters(buses, meters)

    print("\n=== BASELINE (NO ATTACK) ===")
    print(f"True total load: {true_load:.2f} MW")
    print(f"Perceived load (from meters): {perceived_load_baseline:.2f} MW")
    print(f"J (residual): {J_baseline:.6f}")

    # -------------------------
    # Apply a toy FDIA
    # -------------------------
    # Suppose attacker can only change bus injection meters at MiddleBus and LoadBus
    compromised = ["m_bus_MiddleBus", "m_bus_LoadBus"]

    # Reset reported to true first (just to be safe)
    for m in meters.values():
        m.reported_value = m.true_value

    # Attacker wants perceived load to be 20% lower
    target_drop = 0.2  # 20%
    apply_fdia_attack(meters, target_drop, compromised)

    # After attack
    J_attack = compute_residual_J(buses, meters)
    perceived_load_attack = total_perceived_load_from_meters(buses, meters)

    print("\n=== UNDER FDIA ATTACK (TOY) ===")
    print(f"Perceived load (from meters): {perceived_load_attack:.2f} MW")
    drop_percent = (true_load - perceived_load_attack) / true_load * 100.0
    print(f"Perceived load drop: {drop_percent:.2f}% (target was 20%)")
    print(f"J (residual): {J_attack:.6f}")

    print("\n=== TOY SUMMARY ===")
    print("In this toy grid, the attacker changes only a couple of meters.")
    print("You can see how much they manage to change perceived load")
    print("and how the residual J changes (detection signal).")
    print("This mirrors the same logic as Figure 8: partial control")
    print("makes it harder to stay stealthy and impactful at the same time.")
    print("===================================================\n")


# ============================================================
# 7. Part 2: Skeleton for REAL experiment with CyberGridSim
# ============================================================

# NOTE: These are placeholders to show structure only.
# They will be filled with real logic later, using:
#   - Topologies/           (your copied folder)
#   - FDIA_Attacks/        (your copied folder, MATLAB/MATPOWER)
#   - configs/             (your own small experiment configs)


def load_real_topology(topology_name: str) -> Any:
    """
    Placeholder: load a real topology from ./Topologies/

    Example:
        topology_name = "ACTIVSg500_real"

    In the future, this would:
    - Parse the RAW / MATPOWER file.
    - Build an internal representation (buses, lines, meters, etc.)
    - Or delegate to CyberGridSim's helper loaders.

    For now: we just mark this as not implemented.
    """
    raise NotImplementedError(
        "load_real_topology() should load a real grid from ./Topologies/. "
        "We will implement this after agreeing on the data format."
    )


def run_real_fdia_attack(
    topology: Any,
    meter_distribution: str,
    compromised_fraction: float,
    target_load_drop: float,
    rng_seed: int,
) -> Dict[str, float]:
    """
    Placeholder: run a SINGLE FDIA attack on a real topology.

    Inputs:
    - topology: object returned from load_real_topology(...)
    - meter_distribution: e.g. 'uniform', 'generator_heavy', 'load_heavy', 'dense', 'sparse'
    - compromised_fraction: e.g. 0.2 (20% of meters compromised)
    - target_load_drop: e.g. 0.2 (20% target drop)
    - rng_seed: for reproducibility when randomly picking compromised meters

    Output dictionary (example keys):
    - 'true_load'
    - 'perceived_load'
    - 'perceived_load_drop_percent'
    - 'J_baseline'
    - 'J_attack'
    - 'delta_J'
    - 'success_flag' (1 if attack is both stealthy & impactful, else 0)

    Inside (later):
    - Choose a set of meters according to meter_distribution.
    - Randomly pick compromised_fraction of them.
    - Call MATLAB / FDIA_Attacks to:
        * generate FDIA
        * run state estimation
        * compute J and perceived load
    - Return summary metrics.
    """
    raise NotImplementedError(
        "run_real_fdia_attack() will interface with FDIA_Attacks/ (MATLAB/MATPOWER) "
        "to compute real J and perceived load on real topologies."
    )


def run_fdia_meter_distribution_experiment(
    results_csv_path: str,
    topology_name: str,
    meter_distributions: List[str],
    compromised_fractions: List[float],
    num_trials_per_setting: int,
    target_load_drop: float,
):
    """
    Skeleton for the BIG experiment:
    - many random attacks
    - different meter distributions
    - logs results into a CSV in ./results/

    This mirrors the structure we described in the project plan.
    """
    print("===================================================")
    print("SKELETON: FDIA meter placement experiment (REAL GRID)")
    print("===================================================")
    print(f"Topology: {topology_name}")
    print(f"Meter distributions: {meter_distributions}")
    print(f"Compromised fractions: {compromised_fractions}")
    print(f"Trials per setting: {num_trials_per_setting}")
    print(f"Target load drop: {target_load_drop * 100:.1f}%")
    print("Results will be written to:", results_csv_path)
    print("NOTE: This is just a skeleton; functions are NotImplemented yet.\n")

    # 1. Load real topology (later)
    topology = load_real_topology(topology_name)

    # 2. Prepare CSV logging
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
    with open(results_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "topology",
                "meter_distribution",
                "compromised_fraction",
                "trial_id",
                "true_load",
                "perceived_load",
                "perceived_load_drop_percent",
                "J_baseline",
                "J_attack",
                "delta_J",
                "success_flag",
            ],
        )
        writer.writeheader()

        # 3. Main loops over distributions and attacker strength
        trial_global_id = 0
        for md in meter_distributions:
            for cf in compromised_fractions:
                for trial in range(num_trials_per_setting):
                    rng_seed = trial_global_id  # or more complex later
                    trial_global_id += 1

                    # 4. Run one FDIA attack (placeholder)
                    result = run_real_fdia_attack(
                        topology=topology,
                        meter_distribution=md,
                        compromised_fraction=cf,
                        target_load_drop=target_load_drop,
                        rng_seed=rng_seed,
                    )

                    # 5. Append metadata and write to CSV
                    row = {
                        "topology": topology_name,
                        "meter_distribution": md,
                        "compromised_fraction": cf,
                        "trial_id": trial,
                    }
                    row.update(result)
                    writer.writerow(row)

    print("Experiment finished (skeleton). Once the NotImplemented parts")
    print("are filled in, this will generate a CSV for plotting and analysis.\n")


# ============================================================
# 8. Main
# ============================================================

def main():
    # First, run the toy demo to build intuition.
    run_toy_demo()

    # Then, just show how you WOULD call the real experiment.
    # This will currently raise NotImplementedError (by design),
    # because we haven't filled in the real integration yet.
    #
    # Uncomment this block later when we implement load_real_topology()
    # and run_real_fdia_attack().

    # results_path = os.path.join("results", "fdia_meter_placement_experiment.csv")
    # run_fdia_meter_distribution_experiment(
    #     results_csv_path=results_path,
    #     topology_name="ACTIVSg500_real",  # example; must match a file under Topologies/
    #     meter_distributions=["uniform", "generator_heavy", "load_heavy", "dense", "sparse"],
    #     compromised_fractions=[0.2, 0.4, 0.6, 0.8],
    #     num_trials_per_setting=10,
    #     target_load_drop=0.2,  # 20%
    # )


if __name__ == "__main__":
    main()
