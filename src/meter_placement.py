# meter_placement.py
import numpy as np
import random


def get_pandapower_bus_mapping(net):
    """
    Pandapower's bus indices for MATPOWER imports are already 0..N-1.
    We still return an explicit mapping for clarity.
    """
    pp_ids = list(net.bus.index)  # typically [0, 1, ..., nb-1]
    mapping = {bid: pos for pos, bid in enumerate(pp_ids)}
    return mapping, pp_ids


# ===============================================================
# PAPER DISTRIBUTION — 20 meters per bus (SP23 mode)
# ===============================================================

def place_paper_distribution(topology):
    """
    PAPER = 20 independent meters per bus (as in the paper).
    The meter list contains bus indices; each bus appears 20 times,
    representing 20 separate meters on that bus.
    """
    net = topology["net"]
    mapping, pp_ids = get_pandapower_bus_mapping(net)

    meters = []
    for bid in pp_ids:
        pos = mapping[bid]       # same as bid, but explicit
        for _ in range(20):
            meters.append(pos)

    return meters  # NOT sorted, duplicates allowed on purpose


# ===============================================================
# Other distributions (respect M from YAML)
# ===============================================================

def place_uniform(topology, M):
    net = topology["net"]
    nb = len(net.bus)
    M = min(M, nb)
    return sorted(np.random.choice(range(nb), size=M, replace=False).tolist())


def place_dense(topology, M):
    net = topology["net"]
    nb = len(net.bus)
    M = min(M, nb)
    w = np.linspace(2.0, 0.1, nb)
    w = w / w.sum()
    return sorted(np.random.choice(range(nb), size=M, replace=False, p=w).tolist())


def place_sparse(topology, M):
    net = topology["net"]
    nb = len(net.bus)
    M = min(M, nb)
    w = np.linspace(0.1, 2.0, nb)
    w = w / w.sum()
    return sorted(np.random.choice(range(nb), size=M, replace=False, p=w).tolist())


def place_load_heavy(topology, M):
    net = topology["net"]
    nb = len(net.bus)
    M = min(M, nb)

    load_sum = np.zeros(nb)
    for _, row in net.load.iterrows():
        pp_bus = int(row["bus"])
        load_sum[pp_bus] += abs(row["p_mw"])

    w = load_sum + 1e-6
    w = w / w.sum()
    return sorted(np.random.choice(range(nb), size=M, replace=False, p=w).tolist())


def place_generator_heavy(topology, M):
    net = topology["net"]
    nb = len(net.bus)
    M = min(M, nb)

    gen_count = np.zeros(nb)
    for _, row in net.gen.iterrows():
        pp_bus = int(row["bus"])
        gen_count[pp_bus] += 1

    w = gen_count + 1e-6
    w = w / w.sum()
    return sorted(np.random.choice(range(nb), size=M, replace=False, p=w).tolist())


# ===============================================================
# Dispatcher
# ===============================================================

def select_meter_distribution(topology, dist_name, M):
    """
    Returns a list of bus indices where meters are placed.

    For "paper": we ignore M and use 20 meters per bus.
    For all other distributions: we respect M from YAML.
    """
    dist_name = dist_name.lower()

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
