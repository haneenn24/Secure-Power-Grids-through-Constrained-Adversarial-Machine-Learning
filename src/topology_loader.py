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
import numpy as np
import pandapower.converter as pc
import os
import tempfile


def _unwrap_any(x):
    """
    Removes MATLAB-style wrappers and returns a clean numpy array or scalar.

    Handles:
      - object arrays of shape ()
      - object arrays with 1 element
      - nested wrappers
      - numeric Python scalars (int/float)
      - numeric numpy arrays
    """

    import numpy as np

    # ----- CASE A: already a plain python scalar -----
    if isinstance(x, (int, float)):
        return x

    # ----- CASE B: scalar object array (array(obj), shape=()) -----
    if isinstance(x, np.ndarray) and x.shape == () and x.dtype == object:
        return _unwrap_any(x.item())

    # ----- CASE C: single-element object array: array([ arr ], dtype=object) -----
    if isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        return _unwrap_any(x.flat[0])

    # ----- CASE D: numeric numpy array (valid bus/gen/branch table) -----
    if isinstance(x, np.ndarray) and x.dtype != object:
        return x

    # ----- CASE E: python list of lists -----
    if isinstance(x, list):
        return np.array(x, dtype=float)

    raise TypeError(f"Cannot unwrap MPC field, got type {type(x)}, value={x}")



def load_sc500_grid(mat_path: str):
    """
    Loads ACTIVSg500 from .mat, unwraps all MPC fields,
    reconstructs a valid MATPOWER .m file, and loads via pandapower.
    """

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MATPOWER file not found: {mat_path}")

    print(f"[INFO] Loading ACTIVSg500 (.mat) from: {mat_path}")
    mat = sio.loadmat(mat_path, squeeze_me=True)

    if "mpc" not in mat:
        raise KeyError("The .mat file does not contain 'mpc'")

    mpc_raw = mat["mpc"]

    # Unwrap all fields to clean numeric matrices
    bus    = _unwrap_any(mpc_raw["bus"])
    gen    = _unwrap_any(mpc_raw["gen"])
    branch = _unwrap_any(mpc_raw["branch"])
    gencost = _unwrap_any(mpc_raw["gencost"])
    baseMVA = float(_unwrap_any(mpc_raw["baseMVA"]))

    # -------------------------------------------------------------
    # Write temporary MATPOWER .m file
    # -------------------------------------------------------------
    with tempfile.NamedTemporaryFile(mode="w", suffix=".m", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write("function mpc = temp_mpc()\n")
        tmp.write("mpc.version = '2';\n")
        tmp.write(f"mpc.baseMVA = {baseMVA};\n")

        def write_matrix(name, M):
            tmp.write(f"mpc.{name} = [\n")
            for r in M:
                tmp.write(" ".join(str(float(x)) for x in r) + ";\n")
            tmp.write("];\n")

        write_matrix("bus", bus)
        write_matrix("gen", gen)
        write_matrix("branch", branch)
        write_matrix("gencost", gencost)

        tmp.write("end\n")

    print("[INFO] Reconstructed MATPOWER file:", tmp_path)

    # -------------------------------------------------------------
    # Load through pandapower
    # -------------------------------------------------------------
    print("[INFO] Loading file through pandapower …")
    net = pc.from_mpc(tmp_path)

    topo = {
        "buses": list(net.bus.index),
        "lines": list(net.line.index),
        "load_buses": list(net.load.bus.unique()),
        "gen_buses": list(net.gen.bus.unique()),
    }

    print("=== ACTIVSg500 Loaded Successfully ===")
    print(f"  Buses : {len(topo['buses'])}")
    print(f"  Lines : {len(topo['lines'])}")
    print(f"  Loads : {len(topo['load_buses'])}")
    print(f"  Gens  : {len(topo['gen_buses'])}")

    return net, topo




