"""
pandapower_backend.py

Backend that mimics the paper's setup using pandapower only (no MATLAB):

- Loads the Synthetic South Carolina grid (ACTIVSg500).
- Runs AC power flow as the "true" physical state.
- Defines measurements as bus active power injections at selected buses.
- Computes a J-like residual (squared error between baseline and attacked
  measurements).
- Implements a simple FDIA-style attack:
    * attacker scales loads at compromised buses to move perceived total
      load toward a target drop (e.g., 20%).
    * we recompute AC power flow with attacked loads
    * we compute J from the change in measurements.

This is not bit-for-bit identical to MATPOWER WLS SE, but it follows the
same spirit as Figure 8:
- impact on perceived load
- detectability via a residual J
- dependence on which meters are compromised.
"""

import os
from typing import Dict, Any, List, Optional

import numpy as np
import pandapower as pp

from topology_loader import load_sc500_grid


# ---------------------------------------------------------------------
# 0. "start_matlab" stub (for compatibility with run_fdia_experiment.py)
# ---------------------------------------------------------------------

def start_matlab():
    """
    Kept only so run_fdia_experiment.py can call it.
    We don't actually use MATLAB here.
    """
    print("[INFO] Using pandapower backend (no MATLAB).")
    return None


# ---------------------------------------------------------------------
# 1. Topology loader wrapper
# ---------------------------------------------------------------------

def load_real_topology(eng, topology_name: str) -> Dict[str, Any]:
    """
    Load the real grid topology for experiments.

    Currently supports the Synthetic South Carolina grid (ACTIVSg500),
    as in the CyberGridSim paper.

    Returns:
        {
          "net": pandapowerNet,
          "buses": [...],
          "lines": [...],
          "load_buses": [...],
          "gen_buses": [...]
        }
    """

    name = topology_name.lower()

    if name in ["sc500", "activsg500", "synthetic_sc", "synthetic_south_carolina",
                "activsg500_real", "sc-grid", "sc_grid"]:
        # Default path relative to project root
        default_mat_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "Topologies",
            "Original",
            "MATPOWER",
            "ACTIVSg500.mat",
        )

        # Allow overriding via env var if needed
        mat_path = os.environ.get("SC500_MAT_PATH", default_mat_path)

        net, topo = load_sc500_grid(mat_path)
        topo_dict = dict(topo)
        topo_dict["net"] = net
        return topo_dict

    else:
        raise ValueError(
            f"Unknown topology_name '{topology_name}'. "
            "For now use 'SC500' / 'ACTIVSg500_real'."
        )


# ---------------------------------------------------------------------
# 2. Helpers: baseline state + measurements
# ---------------------------------------------------------------------

def _compute_baseline_state(topology_name: str, meter_list: List[int]) -> Dict[str, Any]:
    """
    Compute the "true" grid state and baseline measurements (no attack).

    Steps:
      - load SC500 grid
      - run AC power flow
      - compute:
          * true total load (MW)
          * bus active power injections at all buses
          * measurement vector z_baseline at the meter buses
    """

    topo = load_real_topology(None, topology_name)
    net = topo["net"]

    # Run AC power flow (true physical state)
    pp.runpp(net)

    # True total load in MW
    true_load = float(net.load.p_mw.sum())

    # Bus active power injections (net injection at each bus)
    # res_bus.p_mw: positive = net injection into network
    bus_p_inj = net.res_bus.p_mw.values  # NumPy array of length n_buses

    # Measurements at meter buses
    meter_indices = np.array(meter_list, dtype=int)
    z_baseline = bus_p_inj[meter_indices]

    return {
        "topology": topo,
        "net_true": net,
        "true_load": true_load,
        "bus_p_inj_true": bus_p_inj,
        "z_baseline": z_baseline,
        "meter_indices": meter_indices,
    }


# ---------------------------------------------------------------------
# 3. Public API: compute_baseline (used by run_fdia_experiment.py)
# ---------------------------------------------------------------------

def compute_baseline(
    eng,
    topology_name: str,
    meter_list: List[int],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute baseline quantities (no attack), similar to the paper:

      - true_load:      physical total load (MW)
      - perceived_load: operator's estimate (same as true in no-attack case)
      - J_baseline:     residual value (0 in our construction)

    The full baseline state is computed internally, but we only return
    the values that the experiment driver needs.
    """

    base = _compute_baseline_state(topology_name, meter_list)

    true_load = base["true_load"]
    perceived_load = true_load   # no attack => perfect estimate
    J_baseline = 0.0             # residual is zero with no attack

    return {
        "true_load": true_load,
        "perceived_load": perceived_load,
        "J_baseline": J_baseline,
    }


# ---------------------------------------------------------------------
# 4. FDIA-style attack + J-value
# ---------------------------------------------------------------------

def run_fdia_attack(
    eng,
    topology_name: str,
    meter_list: List[int],
    compromised_meters: List[int],
    target_load_drop: float,
    rng_seed: int,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Simulate one FDIA attempt.

    Conceptual model:
      - Physical grid stays as in the baseline true state.
      - Attacker can alter the *reported loads* at compromised buses
        (like an FDIA on load meters).
      - Attacker tries to move the total perceived load toward:
            (1 - target_load_drop) * true_load
      - We recompute AC power flow with these fake loads.
      - We compute a J-like residual as the squared difference between
        baseline measurements and attacked measurements.

    Returns:
        {
          "perceived_load_attack": float (MW),
          "J_attack": float,
          "delta_J": float,
          "perceived_load_drop_percent": float (%)
        }
    """

    # 1) Baseline true state + measurements
    base = _compute_baseline_state(topology_name, meter_list)
    topo = base["topology"]
    net_true = base["net_true"]
    true_load = base["true_load"]
    bus_p_inj_true = base["bus_p_inj_true"]
    z_baseline = base["z_baseline"]
    meter_indices = base["meter_indices"]

    J_baseline = 0.0  # by construction

    # 2) Build an attacked copy of the network
    # attacker changes loads at compromised buses in SCADA,
    # which affects how the operator reconstructs the state.
    if hasattr(net_true, "deepcopy"):
        net_attacked = net_true.deepcopy()
    else:
        import copy
        net_attacked = copy.deepcopy(net_true)

    # 3) Load per bus (true)
    load_df = net_true.load.copy()   # columns: ['bus', 'p_mw', ...]
    load_per_bus = load_df.groupby("bus")["p_mw"].sum()

    # Buses that (a) have load and (b) are compromised
    compromised_bus_ids = [
        b for b in compromised_meters
        if b in load_per_bus.index
    ]

    # If no load at compromised buses → attacker can't change total load in this model
    if len(compromised_bus_ids) == 0:
        # Nothing changes
        pp.runpp(net_attacked)
        attacked_load_per_bus = net_attacked.load.groupby("bus")["p_mw"].sum()
        perceived_load_attack = float(attacked_load_per_bus.sum())
        drop_percent = 100.0 * (true_load - perceived_load_attack) / true_load

        # Recompute attacked measurements & J
        bus_p_inj_attack = net_attacked.res_bus.p_mw.values
        h_attack = bus_p_inj_attack[meter_indices]
        residual = z_baseline - h_attack
        J_attack = float(np.sum(residual ** 2))
        delta_J = J_attack - J_baseline

        return {
            "perceived_load_attack": perceived_load_attack,
            "J_attack": J_attack,
            "delta_J": delta_J,
            "perceived_load_drop_percent": drop_percent,
        }

    # Total (true) load
    P_true = float(load_per_bus.sum())

    # Load on compromised buses and uncompromised buses
    P_c = float(load_per_bus.loc[compromised_bus_ids].sum())
    P_u = P_true - P_c

    # Desired total perceived load after attack
    target_total = (1.0 - target_load_drop) * P_true

    # We want:  P_u + s * P_c ≈ target_total  =>  s = (target_total - P_u) / P_c
    s = (target_total - P_u) / P_c
    if s < 0.0:
        s = 0.0  # cannot go below zero load

    # 4) Apply scaling s to loads at compromised buses in the attacked net
    for idx, row in net_attacked.load.iterrows():
        if row["bus"] in compromised_bus_ids:
            net_attacked.load.at[idx, "p_mw"] = row["p_mw"] * s

    # 5) Run AC power flow on attacked network
    pp.runpp(net_attacked)

    # Perceived load = sum of attacked loads
    attacked_load_per_bus = net_attacked.load.groupby("bus")["p_mw"].sum()
    perceived_load_attack = float(attacked_load_per_bus.sum())

    # Achieved drop (%)
    drop_percent = 100.0 * (P_true - perceived_load_attack) / P_true

    # 6) Attacked measurements and J
    bus_p_inj_attack = net_attacked.res_bus.p_mw.values
    h_attack = bus_p_inj_attack[meter_indices]

    residual = z_baseline - h_attack
    J_attack = float(np.sum(residual ** 2))
    delta_J = J_attack - J_baseline

    return {
        "perceived_load_attack": perceived_load_attack,
        "J_attack": J_attack,
        "delta_J": delta_J,
        "perceived_load_drop_percent": drop_percent,
    }
