# fdia_attack.py
import copy
import numpy as np
import pandapower as pp
import pandapower.estimation as est


# ==========================================================
# Noise model (Gaussian noise on all meters)
# ==========================================================
def add_noise(value, std):
    return float(value + np.random.normal(0.0, std))


# ==========================================================
# Build measurement vector
#   - V at all buses
#   - P,Q at metered buses
#   - line P,Q at all lines
# ==========================================================
def build_measurements(net, meter_buses):
    """
    Returns a list of measurements:
      ("v", "bus", vm_pu, bus_idx, sigma_v)
      ("p", "bus", p_mw, bus_idx, sigma_p)
      ("q", "bus", q_mvar, bus_idx, sigma_q)
      ("p", "line", p_from/to_mw, line_idx, sigma_p, side)
      ("q", "line", q_from/to_mvar, line_idx, sigma_q, side)
    """

    # Make sure PF results exist
    if net.res_bus.empty or net.res_line.empty:
        pp.runpp(net, calculate_voltage_angles=True)

    M = []

    # --- Voltage at all buses ---
    for b in net.bus.index:
        vm = float(net.res_bus.loc[b, "vm_pu"])
        M.append(("v", "bus", add_noise(vm, 0.01), b, 0.01))

    # --- P,Q injections at metered buses ---
    meter_set = set(meter_buses)
    for b in meter_set:
        p = float(net.res_bus.loc[b, "p_mw"])
        q = float(net.res_bus.loc[b, "q_mvar"])
        M.append(("p", "bus", add_noise(p, 0.02), b, 0.02))
        M.append(("q", "bus", add_noise(q, 0.02), b, 0.02))

    # --- Line flows at both ends (for all lines) ---
    for l in net.line.index:
        pF = float(net.res_line.loc[l, "p_from_mw"])
        pT = float(net.res_line.loc[l, "p_to_mw"])
        qF = float(net.res_line.loc[l, "q_from_mvar"])
        qT = float(net.res_line.loc[l, "q_to_mvar"])

        M.append(("p", "line", add_noise(pF, 0.02), l, 0.02, "from"))
        M.append(("p", "line", add_noise(pT, 0.02), l, 0.02, "to"))

        M.append(("q", "line", add_noise(qF, 0.02), l, 0.02, "from"))
        M.append(("q", "line", add_noise(qT, 0.02), l, 0.02, "to"))

    return M


# ==========================================================
# Strong FDIA (α = 0.8) on compromised buses / lines
# ==========================================================
def apply_attack(M, compromised_buses, load_buses, attacked_lines=None):
    """
    Strong FDIA:
      - scale P,Q at compromised LOAD buses: alpha * value
      - scale line P,Q only on lines touching compromised buses
    """
    alpha = 0.8
    comp = set(compromised_buses)
    loads = set(load_buses)
    attacked_lines = set(attacked_lines or [])

    out = []

    for entry in M:
        if len(entry) == 5:
            m, t, val, idx, std = entry
            side = None
        else:
            m, t, val, idx, std, side = entry

        new_val = val

        # Bus injections at compromised LOAD buses
        if t == "bus" and idx in comp and idx in loads:
            if m in ["p", "q"]:
                new_val = alpha * val

        # Line flows only on attacked_lines
        if t == "line" and idx in attacked_lines:
            if m in ["p", "q"]:
                new_val = alpha * val

        if side is None:
            out.append((m, t, new_val, idx, std))
        else:
            out.append((m, t, new_val, idx, std, side))

    return out


# ==========================================================
# Helper: compute perceived load L from bus injections
# ==========================================================
def compute_perceived_load(net):
    """
    Perceived load = - sum of bus active injections at load buses.
    Prefer SE results (res_bus_est) when available, otherwise PF.
    """
    load_bus_idx = net.load["bus"].values

    if hasattr(net, "res_bus_est") and len(net.res_bus_est) and "p_mw" in net.res_bus_est:
        p_bus = net.res_bus_est["p_mw"]
    else:
        p_bus = net.res_bus["p_mw"]

    # negative injection = load, so flip sign
    L = float((-p_bus.loc[load_bus_idx]).sum())
    return L


# ==========================================================
# WLS wrapper with safe J and L computation
# ==========================================================
def run_wls(net, M):
    """
    Run WLS state estimation and compute:
      J = sum (vm - 1)^2   (surrogate detectability)
      L = perceived load   (see compute_perceived_load)
    """

    # Clear old measurements
    if "measurement" in net and len(net.measurement):
        net.measurement.drop(net.measurement.index, inplace=True)

    # Add new measurements
    for entry in M:
        if len(entry) == 5:
            m, t, v, idx, std = entry
            pp.create_measurement(net, m, t, v, std, idx)
        else:
            m, t, v, idx, std, side = entry
            pp.create_measurement(net, m, t, v, std, idx, side=side)

    # Run SE
    ok = est.estimate(net, algorithm="wls", init="flat")

    if not ok:
        # Fallback: run PF and use PF voltages + PF load
        pp.runpp(net, calculate_voltage_angles=True)
        vm = net.res_bus["vm_pu"].values
        J = float(np.sum((vm - 1.0) ** 2))
        L = compute_perceived_load(net)
        return J, L

    # SE succeeded: use estimated voltages
    if hasattr(net, "res_bus_est") and len(net.res_bus_est):
        vm = net.res_bus_est["vm_pu"].values
    else:
        vm = net.res_bus["vm_pu"].values

    J = float(np.sum((vm - 1.0) ** 2))
    L = compute_perceived_load(net)
    return J, L


# ==========================================================
# Main FDIA routine
# ==========================================================
def run_fdia_attack(net, meter_buses, compromised_buses, load_buses):
    """
    Returns:
        J0 (pre-attack cost)
        J1 (post-attack cost)
        L0 (pre-attack perceived load)
        L1 (post-attack perceived load)
    """

    # Baseline PF
    pp.runpp(net, calculate_voltage_angles=True)

    # Build noisy measurements
    M0 = build_measurements(net, meter_buses)

    # Baseline SE
    net0 = copy.deepcopy(net)
    J0, L0 = run_wls(net0, M0)

    # Lines touching compromised buses (for line attack)
    comp_set = set(compromised_buses)
    attacked_lines = [
        l for l in net.line.index
        if (net.line.loc[l, "from_bus"] in comp_set) or
           (net.line.loc[l, "to_bus"] in comp_set)
    ]

    # Apply FDIA to measurements
    M1 = apply_attack(M0, compromised_buses, load_buses, attacked_lines=attacked_lines)

    # Attacked SE
    net1 = copy.deepcopy(net)
    J1, L1 = run_wls(net1, M1)

    return J0, J1, L0, L1
