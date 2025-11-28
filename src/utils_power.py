import scipy.io as sio
import tempfile
import pandapower as pp
import numpy as np
import os

def _extract(mat_field):
    """
    Safely unwrap MATLAB cell/struct arrays:
    Handles cases like:
        array([[bus]], dtype=object)
        array(bus, dtype=object)
        bus
    Always returns a clean numpy array.
    """
    if isinstance(mat_field, np.ndarray):
        # object array from MATLAB?
        if mat_field.dtype == object:
            while isinstance(mat_field, np.ndarray) and mat_field.dtype == object:
                mat_field = mat_field.item()
        return np.array(mat_field)
    return np.array(mat_field)


def load_matpower_file(name):
    """Loads ACTIVSg MATPOWER .mat file and converts to pandapower."""
    base = os.path.dirname(os.path.dirname(__file__))
    matpath = os.path.join(base, "Topologies", "Original", "MATPOWER", f"{name}.mat")

    print(f"[INFO] Loading MATPOWER (.mat) from: {matpath}")

    data = sio.loadmat(matpath, squeeze_me=True)

    if "mpc" not in data:
        raise ValueError("Error: .mat file does not contain 'mpc' struct.")

    mpc = data["mpc"]

    # Extract all fields safely:
    baseMVA  = float(_extract(mpc["baseMVA"]))
    bus      = _extract(mpc["bus"])
    gen      = _extract(mpc["gen"])
    branch   = _extract(mpc["branch"])

    # Create a temporary .m file for pandapower
    tmp_fd, tmp_mfile = tempfile.mkstemp(suffix=".m")
    os.close(tmp_fd)
    print(f"[INFO] Reconstructed MATPOWER file: {tmp_mfile}")

    with open(tmp_mfile, "w") as f:
        f.write("function mpc = tmpcase()\n")
        f.write(f"mpc.baseMVA = {baseMVA};\n")

        # BUS
        f.write("mpc.bus = [\n")
        for row in bus:
            f.write(" ".join(str(float(x)) for x in row) + ";\n")
        f.write("];\n")

        # GEN
        f.write("mpc.gen = [\n")
        for row in gen:
            f.write(" ".join(str(float(x)) for x in row) + ";\n")
        f.write("];\n")

        # BRANCH
        f.write("mpc.branch = [\n")
        for row in branch:
            f.write(" ".join(str(float(x)) for x in row) + ";\n")
        f.write("];\n")

        f.write("end\n")

    print("[INFO] Loading MATPOWER file through pandapower …")
    net = pp.converter.from_mpc(tmp_mfile, f_hz=60)

    print("=== MATPOWER network loaded successfully ===")
    print(f"  Buses : {len(net.bus)}")
    print(f"  Lines : {len(net.line)}")
    print(f"  Loads : {len(net.load)}")
    print(f"  Gens  : {len(net.gen)}")

    return {
        "net": net,
        "buses": list(net.bus.index),
        "lines": list(net.line.index),
        "load_buses": list(net.load.bus.values),
        "gen_buses": list(net.gen.bus.values),
    }
