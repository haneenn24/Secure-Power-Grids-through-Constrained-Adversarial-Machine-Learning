import numpy as np
from typing import List, Tuple
import pandapower as pp
import pandapower.estimation as est
import traceback


def ext2int_bus(net, bus_ext):
    matches = net.bus.index[ net.bus["bus"] == bus_ext ]
    if len(matches) == 0:
        raise KeyError(f"Bus {bus_ext} not found in net.bus['bus']")
    return matches[0]

# ============================================================
# Attack builder
# ============================================================
def apply_attack(M, compromised, load_buses):
    out = []
    alpha = 0.8
    comp = set(compromised)
    load = set(load_buses)

    for m in M:
        if len(m) == 5:
            mtype, etype, val, idx, std = m
            side = None
        else:
            mtype, etype, val, idx, std, side = m

        new_val = val
        if mtype == "p" and etype == "bus" and idx in comp and idx in load:
            new_val = val * alpha

        if side is None:
            out.append((mtype, etype, new_val, idx, std))
        else:
            out.append((mtype, etype, new_val, idx, std, side))

    return out

# ------------------------------------------------------------
# Build measurement vector
# ------------------------------------------------------------
def build_measurements(net, meter_buses):
    """
    Build measurement list M = [(type, element_type, value, bus, noise_std)]
    After running PF (pp.runpp) so net.res_bus is populated.
    """
    M = []

    # ---- Run PF if needed (CRITICAL FIX) ----
    if net.res_bus.empty:
        pp.runpp(net, calculate_voltage_angles=True)

    # Extract res_bus once
    rb = net.res_bus

    # int bus indices (pandapower uses 0..N-1)
    all_buses = net.bus.index.tolist()

    # build measurements
    for b in meter_buses:
        if b not in all_buses:
            continue

        # Voltage magnitude
        vm = float(rb.at[b, "vm_pu"])
        M.append(("v", "bus", vm, b, 0.01))

        # Voltage angle
        va = float(rb.at[b, "va_degree"])
        M.append(("va", "bus", va, b, 0.01))

    return M


# ------------------------------------------------------------
# FDIA attack main routine
# ------------------------------------------------------------
def run_fdia_attack(net, meter_buses, compromised_buses, load_buses):
    """
    Returns:
        J0 (pre-attack cost)
        J1 (post-attack cost)
        L0 (pre-attack total load)
        L1 (post-attack perceived load)
    """
    import pandapower as pp

    # ---- Must run PF before everything ----
    if net.res_bus.empty:
        pp.runpp(net, calculate_voltage_angles=True)

    # Original cost J0
    rb = net.res_bus
    J0 = float(np.sum(rb.vm_pu.values ** 2))

    # Original perceived load
    L0 = float(net.res_bus.p_mw[load_buses].sum())

    # ---- Build original measurements ----
    M0 = build_measurements(net, meter_buses)

    # ---- Create attacked measurement list ----
    M1 = []
    for (t, et, val, b, noise) in M0:
        if b in compromised_buses:
            val = val + 0.05   # small attack
        M1.append((t, et, val, b, noise))

    # ---- Inject modifications into the network ----
    # Example: modify p_mw at compromised buses
    for b in compromised_buses:
        if b in load_buses:
            idx = net.load.index[net.load.bus == b]
            net.load.loc[idx, "p_mw"] *= 0.9   # 10% reduction

    # ---- Re-run PF for attacked state ----
    pp.runpp(net, calculate_voltage_angles=True)

    rb2 = net.res_bus

    # Post-attack cost J1
    J1 = float(np.sum(rb2.vm_pu.values ** 2))

    # Post-attack perceived load
    L1 = float(rb2.p_mw[load_buses].sum())

    return J0, J1, L0, L1

