"""
topology_loader.py — Load the Synthetic South Carolina Grid (ACTIVSg500)
------------------------------------------------------------------------

This module provides the project’s **official loader** for the real power
grid used throughout the experiments: the ACTIVSg500 synthetic South Carolina
system (SC-500), the same topology used in the CyberGridSim paper.

Its purpose is to give all other components (meter placement, FDIA attack,
baseline computation) access to a consistent, physics-correct pandapower
network.

Workflow:
    • Read the MATPOWER “mpc” struct from ACTIVSg500.mat
    • Convert it into a pandapower network using pandapower.converter.from_mpc
    • Extract basic topology information needed for the experiments:
          - list of all buses
          - list of all lines
          - list of generator buses
          - list of load buses

Why this file matters:
    - It is the ONLY place where the physical topology enters the pipeline.
    - All FDIA attacks, meter placements, and state estimation experiments
      depend on the same network returned here.
    - Ensures the experiment uses the *realistic, large-scale* SC-500 grid,
      not a toy IEEE system.

Used by:
    • pandapower_backend.load_real_topology()
    • meter_placement (needs buses/lines)
    • run_fdia_experiment.py (ensures consistent topology across trials)

Outputs:
    (net, topo)
        net  → the full pandapower network (buses, lines, loads, generators)
        topo → a lightweight dictionary of key topology sets used everywhere else

This file contains **no attack logic, no meter logic, no SE logic**.
Its only job is: **load the real grid once, correctly, and consistently**.
"""

import scipy.io as sio
import pandapower.converter as pc
import pandapower as pp
import os


def load_sc500_grid(mat_path: str):
    """
    Loads the ACTIVSg500 MATPOWER case used in the CyberGridSim paper.

    Args:
        mat_path: path to ACTIVSg500.mat, e.g.:
            Topologies/Original/MATPOWER/ACTIVSg500.mat

    Returns:
        net:  pandapower network
        topo: dict with basic topology info:
              {
                "buses": [...],
                "lines": [...],
                "load_buses": [...],
                "gen_buses": [...]
              }
    """

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MATPOWER file not found: {mat_path}")

    print(f"[INFO] Loading SC-500 MATPOWER case from: {mat_path}")

    # Load MATPOWER .mat file
    mat = sio.loadmat(mat_path, squeeze_me=True)
    if "mpc" in mat:
        mpc = mat["mpc"]
    else:
        raise KeyError("MATPOWER .mat file does not contain 'mpc' structure")

    # Convert MATPOWER mpc → pandapower
    print("[INFO] Converting MATPOWER → pandapower...")
    net = pc.from_mpc(mpc, casename_mpc_file="ACTIVSg500")

    # Build topology dictionary (needed for meter placement, FDIA)
    topo = {
        "buses": list(net.bus.index),
        "lines": list(net.line.index),
        "load_buses": list(net.load.bus.unique()),
        "gen_buses": list(net.gen.bus.unique()),
    }

    print("=== SC-500 Grid Loaded ===")
    print(f"  Buses   : {len(topo['buses'])}")
    print(f"  Lines   : {len(topo['lines'])}")
    print(f"  Loads   : {len(topo['load_buses'])}")
    print(f"  Gens    : {len(topo['gen_buses'])}")

    return net, topo
