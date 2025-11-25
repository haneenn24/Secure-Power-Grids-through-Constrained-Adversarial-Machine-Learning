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
