# meter_placement.py
import random
import numpy as np

def get_pandapower_bus_mapping(net):
    """
    Maps pandapower bus indices 1:1 to integer positions.
    Pandapower bus.index is ALREADY 0..N-1 for MATPOWER imports.
    """
    pp_ids = list(net.bus.index)  # e.g., [0,1,2,...]
    mapping = {bid: pos for pos, bid in enumerate(pp_ids)}
    return mapping, pp_ids


# ===============================================================
# PAPER DISTRIBUTION — EXACT PAPER BEHAVIOR (20 meters per bus)
# ===============================================================

def place_paper_distribution(topology):
    """
    PAPER = 20 independent meters per bus (SP23 Grid Paper).
    Always uses exactly nb * 20 meters.

    Example:
      ACTIVSg500 → 500 * 20 = 10,000
      case3120sp → 3120 * 20 = 62,400
    """

    net = topology["net"]
    mapping, pp_ids = get_pandapower_bus_mapping(net)

    meters = []
    for bid in pp_ids:
        pos = mapping[bid]       # same as bid
        for _ in range(20):
            meters.append(pos)

    return meters


# ===============================================================
# Other distributions (respect M)
# ===============================================================

def place_uniform(topology, M):
    net = topology["net"]
    nb = len(net.bus)
    return sorted(np.random.choice(range(nb), size=M, replace=False).tolist())


def place_dense(topology, M):
    net = topology["net"]
    nb = len(net.bus)
    w = np.linspace(2.0, 0.1, nb)
    w = w / w.sum()
    return sorted(np.random.choice(range(nb), size=M, replace=False, p=w).tolist())


def place_sparse(topology, M):
    net = topology["net"]
    nb = len(net.bus)
    w = np.linspace(0.1, 2.0, nb)
    w = w / w.sum()
    return sorted(np.random.choice(range(nb), size=M, replace=False, p=w).tolist())


def place_load_heavy(topology, M):
    net = topology["net"]
    nb = len(net.bus)

    load_sum = np.zeros(nb)
    for _, row in net.load.iterrows():
        pp_bus = row["bus"]
        load_sum[pp_bus] += abs(row["p_mw"])

    w = load_sum + 1e-6
    w = w / w.sum()
    return sorted(np.random.choice(range(nb), size=M, replace=False, p=w).tolist())


def place_generator_heavy(topology, M):
    net = topology["net"]
    nb = len(net.bus)

    gen_count = np.zeros(nb)
    for _, row in net.gen.iterrows():
        gen_count[row["bus"]] += 1

    w = gen_count + 1e-6
    w = w / w.sum()
    return sorted(np.random.choice(range(nb), size=M, replace=False, p=w).tolist())


# ===============================================================
# DISPATCHER
# ===============================================================

def select_meter_distribution(topology, dist_name, M):
    dist_name = dist_name.lower()
    net = topology["net"]
    nb = len(net.bus)

    if dist_name == "paper":
        print("[INFO] PAPER distribution → 20 meters per bus (fixed, SP23 mode)")
        return place_paper_distribution(topology)

    elif dist_name == "uniform":
        return place_uniform(topology, M)

    elif dist_name == "dense":
        return place_dense(topology, M)

    elif dist_name == "sparse":
        return place_sparse(topology, M)

    elif dist_name == "load_heavy":
        return place_load_heavy(topology, M)

    elif dist_name == "generator_heavy":
        return place_generator_heavy(topology, M)

    else:
        raise ValueError(f"Unknown meter distribution: {dist_name}")
