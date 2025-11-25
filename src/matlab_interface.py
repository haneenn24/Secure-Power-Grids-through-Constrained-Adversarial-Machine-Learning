"""
matlab_interface.py  — Pandapower backend for FDIA experiments
--------------------------------------------------------------

This file replaces the MATLAB/Matpower backend from the paper and provides
a fully Python-based implementation of the experiment pipeline using
pandapower.

It is responsible for all FDIA "backend" logic:
    - Loading a real power grid (currently IEEE-118).
    - Running AC power flow.
    - Building realistic SCADA measurements (P_inj, P_flow, V_mag).
    - Running nonlinear weighted-least-squares state estimation.
    - Computing the J-value (residual) exactly like a bad-data detector.
    - Applying a simplified but physically meaningful FDIA by modifying
      the attacker-controlled injection meters.
    - Returning all metrics used by the experiment loop:
          true load, perceived load (after attack), J_attack, delta_J.

This file does NOT use MATLAB. It keeps the same interface so that the
experiment runner (run_fdia_experiment.py) does not need to change.

In summary:
    • load_real_topology() → loads the grid and caches it
    • compute_baseline() → baseline (no-attack) state estimation + J
    • run_fdia_attack() → apply attack + compute J_attack + perceived load
    • run_state_estimation_for_meters() → real AC state estimation
    • build_measurements() → produce the SCADA measurement vectors

The experiment runner calls these functions to generate the CSV results
for all distributions, attacker fractions, and trials.
"""


from typing import Any, Dict, List, Optional
import numpy as np
import pandapower as pp
import pandapower.networks as pn

# Cache topology by name so other functions can access it
_TOPOLOGY_CACHE: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------
# 1. "start_matlab" stub (now: start backend)
# ---------------------------------------------------------

def start_matlab():
    """
    Kept for compatibility with existing code.

    For the pandapower backend, there is no external engine to start.
    We just print a message and return None.
    """
    print("[INFO] Using pandapower backend (no MATLAB).")
    return None


# ---------------------------------------------------------
# 2. Load real topology (pandapower IEEE case)
# ---------------------------------------------------------

def load_real_topology(eng, topology_name: str) -> Dict[str, Any]:
    """
    Load a real power grid topology using pandapower.

    For now we map everything to IEEE 118-bus test case,
    regardless of the string, so your experiments are consistent.

    Returns a dict with:
        - "name"
        - "net"              : pandapower net object
        - "buses"            : list of bus indices
        - "generator_buses"  : bus indices with generators
        - "load_buses"       : bus indices with loads
        - "lines"            : list of line indices
    """
    # Choose a pandapower test case.
    # You can later switch to another one (e.g., case14, case39, case118).
    net = pn.case118()

    # Run power flow once to have results
    pp.runpp(net)

    buses = list(net.bus.index)
    gen_buses = list(net.gen["bus"].unique()) if len(net.gen) > 0 else []
    load_buses = list(net.load["bus"].unique()) if len(net.load) > 0 else []
    lines = list(net.line.index)

    topo = {
        "name": topology_name,
        "net": net,
        "buses": buses,
        "generator_buses": gen_buses,
        "load_buses": load_buses,
        "lines": lines,
    }

    _TOPOLOGY_CACHE[topology_name] = topo
    return topo


def _get_topology(topology_name: str) -> Dict[str, Any]:
    if topology_name not in _TOPOLOGY_CACHE:
        raise RuntimeError(
            f"Topology '{topology_name}' not loaded yet. "
            f"Call load_real_topology() first."
        )
    return _TOPOLOGY_CACHE[topology_name]


# ---------------------------------------------------------
# 3. Helper: build bus injections from net
# ---------------------------------------------------------

def _compute_bus_injections(net) -> Dict[int, float]:
    """
    Compute active power injection at each bus:

        injection(bus) = total generation at bus - total load at bus

    Returns a dict: {bus_idx: injection_mw}
    """
    buses = list(net.bus.index)

    if len(net.gen) > 0:
        gen_p = net.gen.groupby("bus")["p_mw"].sum()
    else:
        gen_p = {}

    if len(net.load) > 0:
        load_p = net.load.groupby("bus")["p_mw"].sum()
    else:
        load_p = {}

    injections = {}
    for b in buses:
        g = float(gen_p[b]) if b in gen_p else 0.0
        l = float(load_p[b]) if b in load_p else 0.0
        injections[b] = g - l
    return injections


# ---------------------------------------------------------
# 4. Baseline (no attack)
# ---------------------------------------------------------

def compute_baseline(
    eng,
    topology_name: str,
    meter_list: List[str],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Baseline (no attack) using REAL state estimation.

    - Uses the pandapower net for this topology
    - Builds SCADA measurements (P_inj, P_flow, V_mag)
    - Runs WLS state estimation (estimate)
    - Computes real J_baseline from residuals
    """
    topo = _get_topology(topology_name)
    net = topo["net"]

    true_load, perceived_load, J_baseline = run_state_estimation_for_meters(
        net,
        meter_list=meter_list,
        sigma_inj=0.02,
        sigma_flow=0.02,
        sigma_v=0.01,
        init="flat",
    )

    return {
        "true_load": true_load,
        "perceived_load": perceived_load,
        "J_baseline": J_baseline,
    }



# ---------------------------------------------------------
# 5. Helper: parse meter names
# ---------------------------------------------------------

def _parse_meter_name(meter_name: str):
    """
    Meter names are of form:
        m_bus_<id> or m_bus_<id>_extra
        m_line_<id> or m_line_<id>_extra

    Returns:
        kind: "bus" or "line"
        idx : integer index of bus or line
    """
    if meter_name.startswith("m_bus_"):
        rest = meter_name[len("m_bus_"):]
        token = rest.split("_")[0]
        return "bus", int(token)
    elif meter_name.startswith("m_line_"):
        rest = meter_name[len("m_line_"):]
        token = rest.split("_")[0]
        return "line", int(token)
    else:
        raise ValueError(f"Unknown meter name format: {meter_name}")


# ---------------------------------------------------------
# 6. FDIA attack (simple but real, using pandapower net)
# ---------------------------------------------------------

def run_fdia_attack(
    eng,
    topology_name: str,
    meter_list: List[str],
    compromised_meters: List[str],
    target_load_drop: float,
    rng_seed: int,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Run a simple but real FDIA attack on a pandapower grid.

    Idea:
    - Use real AC power flow.
    - Define "measurements" = bus injections + line flows.
    - True measurements come from net.res_bus / net.res_line.
    - Attacker can change a subset of meters in compromised_meters.
    - Attacker scales bus injection meters at load buses by (1 - target_load_drop)
      to try and reduce perceived load.
    - J_attack = sum ( (z_attack - z_true)^2 ) over all meters.
    - Perceived load is estimated from attacked bus-injection meters at load buses:
          load_hat(bus) = - injection_attack(bus)   (approximation for pure loads)

    This is not a full optimal FDIA solver, but it uses:
        - real topology
        - real power flow
        - real injections and flows
    """

    topo = _get_topology(topology_name)
    net = topo["net"]
    load_buses = set(topo["load_buses"])

    # Re-run power flow
    pp.runpp(net)

    # True total load
    if len(net.load) > 0:
        true_load = float(net.load["p_mw"].sum())
    else:
        true_load = 0.0

    # Compute bus injections
    injections = _compute_bus_injections(net)

    # Build true measurement dict: meter_name -> value
    z_true_dict: Dict[str, float] = {}

    # Add bus injection meters
    for m in meter_list:
        kind, idx = _parse_meter_name(m)
        if kind == "bus":
            val = injections.get(idx, 0.0)
            z_true_dict[m] = float(val)
        elif kind == "line":
            # use p_from_mw as line flow
            if idx in net.line.index:
                val = float(net.res_line.at[idx, "p_from_mw"])
            else:
                val = 0.0
            z_true_dict[m] = val
        else:
            raise ValueError(f"Unknown meter kind: {kind}")

    # Start attacked measurements same as true
    z_attack_dict = dict(z_true_dict)

    # Apply attack: scale injection meters at load buses for compromised meters
    # to try to reduce the perceived load.
    scale = 1.0 - target_load_drop  # e.g. 0.8 for 20% drop

    rng = np.random.default_rng(seed=rng_seed)

    for m in compromised_meters:
        if m not in z_attack_dict:
            continue
        kind, idx = _parse_meter_name(m)
        if kind == "bus" and idx in load_buses:
            # scale this injection
            orig = z_attack_dict[m]
            z_attack_dict[m] = orig * scale
        else:
            # Optionally: leave line meters unchanged for simplicity
            # or add small random noise if desired
            # z_attack_dict[m] = z_attack_dict[m] + rng.normal(0, 0.01)
            pass

    # Build arrays in consistent order
    z_true = np.array([z_true_dict[m] for m in meter_list], dtype=float)
    z_attack = np.array([z_attack_dict[m] for m in meter_list], dtype=float)

    # J_baseline (no mismatch between true & expected model) ~ 0
    J_baseline = 0.0

    # J_attack: simple squared residual
    residual = z_attack - z_true
    J_attack = float(np.sum(residual**2))
    delta_J = J_attack - J_baseline

    # Perceived load from attacked bus injection meters at load buses
    # Approximation: for a pure load bus (no generation),
    # injection ~ -load, so load_hat = -injection_attack.
    perceived_load_attack = 0.0
    for m in meter_list:
        kind, idx = _parse_meter_name(m)
        if kind == "bus" and idx in load_buses:
            inj = z_attack_dict[m]
            load_hat = -inj
            perceived_load_attack += load_hat

    perceived_load_attack = float(perceived_load_attack)

    if true_load > 1e-6:
        perceived_load_drop_percent = (true_load - perceived_load_attack) / true_load * 100.0
    else:
        perceived_load_drop_percent = 0.0

    return {
        "true_load": true_load,
        "perceived_load_attack": perceived_load_attack,
        "J_baseline": J_baseline,
        "J_attack": J_attack,
        "delta_J": delta_J,
        "perceived_load_drop_percent": perceived_load_drop_percent,
    }


# ---------------------------------------------------------
# STEP 1: Real SCADA Measurement Construction
# ---------------------------------------------------------

def build_measurements(net, meter_list, sigma_inj=0.01, sigma_flow=0.01, sigma_v=0.001):
    """
    Build real SCADA measurements:
        - Bus injection measurements (P_inj)
        - Line flow measurements (P_flow)
        - Voltage magnitude measurements (V_mag)

    For each meter name in meter_list, we create:
        z_true[i]      = physical measurement from pandapower AC power flow
        z_measured[i]  = z_true[i] + noise
        R[i]           = variance of measurement noise

    Returns:
        z_true        (np.array)
        z_measured    (np.array)
        R             (np.array)   – diagonal covariance
        measurement_info: list of dicts describing each meter
    """
    import numpy as np

    # Ensure power flow is solved
    pp.runpp(net)

    measurement_info = []
    z_true_values = []
    variances = []

    for meter in meter_list:
        parts = meter.split("_")
        # meter format examples:
        # m_bus_pinj_12
        # m_bus_vmag_17
        # m_line_flow_4

        if len(parts) < 4:
            raise ValueError(f"Bad meter name format: {meter}")

        _, kind1, kind2, idx_str = parts
        idx = int(idx_str)

        if kind1 == "bus" and kind2 == "pinj":
            # P injection
            injections = _compute_bus_injections(net)
            if idx not in injections:
                val = 0.0
            else:
                val = float(injections[idx])
            z_true_values.append(val)
            measurement_info.append({"type": "P_inj", "bus": idx})
            variances.append(sigma_inj**2)

        elif kind1 == "bus" and kind2 == "vmag":
            # Voltage magnitude
            if idx in net.res_bus.index:
                val = float(net.res_bus.at[idx, "vm_pu"])
            else:
                val = 1.0
            z_true_values.append(val)
            measurement_info.append({"type": "V_mag", "bus": idx})
            variances.append(sigma_v**2)

        elif kind1 == "line" and kind2 == "flow":
            # Line flow (p_from_mw)
            if idx in net.res_line.index:
                val = float(net.res_line.at[idx, "p_from_mw"])
            else:
                val = 0.0
            z_true_values.append(val)
            measurement_info.append({"type": "P_flow", "line": idx})
            variances.append(sigma_flow**2)

        else:
            raise ValueError(f"Unsupported meter kind: {kind1}_{kind2}")

    z_true = np.array(z_true_values, dtype=float)

    # noise per measurement
    noise = np.random.normal(0.0, np.sqrt(variances), size=len(variances))
    z_measured = z_true + noise

    R = np.array(variances, dtype=float)

    return z_true, z_measured, R, measurement_info

# ---------------------------------------------------------
# STEP 2: REAL AC STATE ESTIMATION USING PANDAPOWER
# ---------------------------------------------------------

import pandapower as pp
from pandapower.estimation.state_estimation import estimate

def run_state_estimation_for_meters(
    net,
    meter_list,
    sigma_inj=0.02,
    sigma_flow=0.02,
    sigma_v=0.01,
    init="flat",
):
    """
    Run a REAL state estimation using pandapower:
      - Creates SCADA measurements for the given meter_list
      - Runs estimate(net)
      - Computes J residual as sum(((z_true - z_est)/sigma)**2)
    
    This replaces the fake J and gives a real benchmark baseline like the paper.
    """

    import numpy as np

    # 1. Ensure AC power flow solved
    pp.runpp(net)

    # 2. Generate true measurements (from STEP 1)
    z_true, _, R, measurement_info = build_measurements(
        net,
        meter_list,
        sigma_inj=sigma_inj,
        sigma_flow=sigma_flow,
        sigma_v=sigma_v,
    )

    sigma = np.sqrt(R)

    # 3. Clear old measurements
    if hasattr(net, "measurement"):
        net.measurement.drop(net.measurement.index, inplace=True)

    # 4. Register SCADA measurements in pandapower
    for idx, info in enumerate(measurement_info):
        val = z_true[idx]
        std = sigma[idx]

        if info["type"] == "P_inj":
            pp.create_measurement(
                net,
                meas_type="p",
                element_type="bus",
                value=val,
                std_dev=std,
                element=info["bus"],
            )

        elif info["type"] == "V_mag":
            pp.create_measurement(
                net,
                meas_type="v",
                element_type="bus",
                value=val,
                std_dev=std,
                element=info["bus"],
            )

        elif info["type"] == "P_flow":
            pp.create_measurement(
                net,
                meas_type="p",
                element_type="line",
                value=val,
                std_dev=std,
                element=info["line"],
                side="from",
            )

        else:
            raise ValueError(f"Unknown meter type: {info}")

    # 5. Run state estimation
    success = estimate(net, init=init)
    if not success:
        raise RuntimeError("State estimation failed.")

    # 6. Reconstruct estimated measurements
    z_est = np.zeros_like(z_true)

    for idx, info in enumerate(measurement_info):
        if info["type"] == "P_inj":
            z_est[idx] = float(net.res_bus_est.p_mw.at[info["bus"]])

        elif info["type"] == "V_mag":
            z_est[idx] = float(net.res_bus_est.vm_pu.at[info["bus"]])

        elif info["type"] == "P_flow":
            z_est[idx] = float(net.res_line_est.p_from_mw.at[info["line"]])

    # 7. Compute J residual
    residual_normalized = (z_true - z_est) / sigma
    J = float(np.sum(residual_normalized ** 2))

    # 8. Compute total true load
    true_load = float(net.load["p_mw"].sum())

    # 9. Baseline perceived = true (no attack)
    perceived_load = true_load

    return true_load, perceived_load, J
