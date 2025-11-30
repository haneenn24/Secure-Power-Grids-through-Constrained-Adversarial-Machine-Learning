# fdia_attack.py
import copy
import numpy as np
import pandapower as pp
import pandapower.estimation as est


# ==========================================================
# Noise model (paper used Gaussian noise on all meters)
# ==========================================================
def add_noise(value, std):
    return float(value + np.random.normal(0, std))


# ==========================================================
# Build measurement vector (full: V, P,Q, line P,Q)
# ==========================================================
def build_measurements(net, meter_buses):

    if net.res_bus.empty:
        pp.runpp(net, calculate_voltage_angles=True)

    M = []

    # Voltage (all buses)
    for b in net.bus.index:
        vm = float(net.res_bus.loc[b, "vm_pu"])
        M.append(("v", "bus", add_noise(vm, 0.01), b, 0.01))

    # P,Q injections only for METERED buses
    for b in meter_buses:
        p = float(net.res_bus.loc[b, "p_mw"])
        q = float(net.res_bus.loc[b, "q_mvar"])
        M.append(("p", "bus", add_noise(p, 0.02), b, 0.02))
        M.append(("q", "bus", add_noise(q, 0.02), b, 0.02))

    # Line flows (all lines)
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
# Strong FDIA (α=0.8)
# ==========================================================
def apply_attack(M, compromised_buses, load_buses):
    alpha = 0.8
    comp = set(compromised_buses)
    loads = set(load_buses)

    out = []

    for entry in M:

        if len(entry) == 5:
            m, t, val, idx, std = entry
            side = None
        else:
            m, t, val, idx, std, side = entry

        new = val

        # Attack P,Q at compromised load buses
        if t == "bus" and idx in comp and idx in loads:
            if m in ["p", "q"]:
                new = alpha * val

        # Attack ALL line measurements touching compromised buses
        if t == "line":
            new = alpha * val

        # Store
        if side is None:
            out.append((m, t, new, idx, std))
        else:
            out.append((m, t, new, idx, std, side))

    return out


# ==========================================================
# WLS wrapper with fallback (no crash)
# ==========================================================
def run_wls(net, M):

    if "measurement" in net and len(net.measurement):
        net.measurement.drop(net.measurement.index, inplace=True)

    for entry in M:
        if len(entry) == 5:
            m, t, v, idx, std = entry
            pp.create_measurement(net, m, t, v, std, idx)
        else:
            m, t, v, idx, std, side = entry
            pp.create_measurement(net, m, t, v, std, idx, side=side)

    ok = est.estimate(net, algorithm="wls", init="flat")

    if not ok:
        # Fallback PF
        pp.runpp(net, calculate_voltage_angles=True)
        vm = net.res_bus.vm_pu.values
        J = float(np.sum((vm - 1.0) ** 2))
        L = float((-net.res_bus.loc[net.load.bus.values, "p_mw"]).sum())
        return J, L

    if hasattr(net, "res_bus_est") and len(net.res_bus_est):
        vm = net.res_bus_est.vm_pu.values
        L = float((-net.res_bus_est.loc[net.load.bus.values, "p_mw"]).sum())
    else:
        vm = net.res_bus.vm_pu.values
        L = float((-net.res_bus.loc[net.load.bus.values, "p_mw"]).sum())

    J = float(np.sum((vm - 1.0) ** 2))
    return J, L


# ==========================================================
# Main FDIA routine
# ==========================================================
def run_fdia_attack(net, meter_buses, compromised_buses, load_buses):

    pp.runpp(net, calculate_voltage_angles=True)

    M0 = build_measurements(net, meter_buses)

    net0 = copy.deepcopy(net)
    J0, L0 = run_wls(net0, M0)

    M1 = apply_attack(M0, compromised_buses, load_buses)

    net1 = copy.deepcopy(net)
    J1, L1 = run_wls(net1, M1)

    return J0, J1, L0, L1
