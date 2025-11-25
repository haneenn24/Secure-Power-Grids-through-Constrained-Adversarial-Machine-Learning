import pandapower as pp
import pandapower.converter as pc
import pandapower.estimation as est
import numpy as np
import os

###########################################################################
# STEP 1 — Load ACTIVSg500 (SC-grid) + add SCADA meters
###########################################################################

def load_sc_grid():
    """
    Loads ACTIVSg500 (Synthetic South Carolina topology) and converts it
    into a pandapower network.

    Place the MATPOWER file at:
        Topologies/ACTIVSg500/case_ACTIVSg500.m
    """

    case_path = "Topologies/ACTIVSg500/case_ACTIVSg500.m"
    if not os.path.exists(case_path):
        raise FileNotFoundError(
            "❌ Missing MATPOWER file: case_ACTIVSg500.m\n"
            "Download from TAMU ACTIVSg library and place it here:\n"
            "Topologies/ACTIVSg500/case_ACTIVSg500.m"
        )

    print("[INFO] Loading ACTIVSg500 topology...")
    net = pc.from_mpc(case_path, casename_mpc_file="case_ACTIVSg500")

    print("[INFO] ACTIVSg500 loaded successfully.")
    print(f"[INFO] Buses: {len(net.bus)}, Lines: {len(net.line)}, Loads: {len(net.load)}")

    return net


###########################################################################
# STEP 2 — Add SCADA-style measurements like the paper uses
###########################################################################

def add_scada_measurements(net, noise_std=0.01):
    """
    Adds SCADA measurements:
    - Voltage magnitude at every bus
    - P/Q injections at loads
    - P/Q flows at lines (both ends)
    """

    print("[INFO] Adding SCADA measurements (Vmag, P_inj, Q_inj, P_flow, Q_flow)...")

    # 1. Voltage magnitude at every bus
    for b in net.bus.index:
        est.create_measurement(net, "v", "bus", net.bus.vn_kv.at[b], b, noise_std)

    # 2. P/Q injections
    for idx, row in net.load.iterrows():
        bus = row.bus
        p = row.p_mw
        q = row.q_mvar if "q_mvar" in row else 0.0
        est.create_measurement(net, "p", "bus", p, bus, noise_std)
        est.create_measurement(net, "q", "bus", q, bus, noise_std)

    # 3. Line P/Q flows (both ends)
    for l in net.line.index:
        est.create_measurement(net, "p", "line", 0.0, l, noise_std)
        est.create_measurement(net, "q", "line", 0.0, l, noise_std)
        est.create_measurement(net, "p", "line", 0.0, l, noise_std, side=1)
        est.create_measurement(net, "q", "line", 0.0, l, noise_std, side=1)

    print(f"[INFO] Total SCADA measurements: {len(net.measurement)}")
    return net


###########################################################################
# Combined loader (Step 1 COMPLETE)
###########################################################################

def load_sc_grid_with_measurements():
    net = load_sc_grid()
    net = add_scada_measurements(net)
    return net


# Test
if __name__ == "__main__":
    net = load_sc_grid_with_measurements()
    print(net)
