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



# ===============================================================
# topology_loader.py  —  fully robust MPC loader (paper-aligned)
# ===============================================================
import numpy as np
import os
import scipy.io as sio
import tempfile

def load_sc500_grid(mat_path):
    """
    Loads ACTIVSg500 MATPOWER case from .mat file (any MPC format):
        - handles nested numpy objects
        - handles dtype 'O'
        - unwraps 0-D arrays
        - reconstructs consistent MPC dict
        - rewrites temporary .m file
        - loads into pandapower via from_mpc()

    Returns:
        net   : pandapower network
        topo  : dict {buses, lines, loads, gens, net, ...}
    """
    print(f"[INFO] Loading ACTIVSg500 (.mat) from: {mat_path}")
    raw = sio.loadmat(mat_path)

    if "mpc" not in raw:
        raise RuntimeError("MAT file does not contain MPC structure.")

    mpc_raw = raw["mpc"]

    # Unwrapper for ANY shape
    def unwrap(x):
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return unwrap(x.item())
            return [unwrap(v) for v in x]
        return x

    # Extract fields
    names = mpc_raw.dtype.names
    mpc = {}
    for field in names:
        mpc[field] = np.array(unwrap(mpc_raw[field])).astype(object)

    # Write synthetic MPC file (.m)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".m")
    tmp_path = tmp.name

    with open(tmp_path, "w") as f:
        f.write("function mpc = case_ACTIVSg500\nmpc = struct();\n")
        for k, v in mpc.items():
            arr = np.array(v, dtype=float)
            f.write(f"mpc.{k} = [\n")
            for row in arr.reshape(-1, arr.shape[-1]):
                f.write(" ".join(map(str, row)) + ";\n")
            f.write("];\n")
        f.write("end\n")

    print(f"[INFO] Reconstructed MATPOWER file: {tmp_path}")

    # Convert to pandapower
    import pandapower.converter as pc
    print("[INFO] Loading file through pandapower …")
    net = pc.from_mpc(tmp_path)

    print("=== ACTIVSg500 Loaded Successfully ===")
    print(f"  Buses : {len(net.bus)}")
    print(f"  Lines : {len(net.line)}")
    print(f"  Loads : {len(net.load)}")
    print(f"  Gens  : {len(net.gen)}")

    topo = {
        "buses": list(net.bus.index),
        "lines": list(net.line.index),
        "loads": list(net.load.index),
        "gens": list(net.gen.index),
        "load_buses": list(net.load["bus"]),
        "gen_buses": list(net.gen["bus"]),
        "net": net
    }
    return net, topo
