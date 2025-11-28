# meter_placement.py
import random
import numpy as np

def get_pandapower_bus_mapping(net):
    """
    Maps MATPOWER bus IDs to integer row positions (0..n_bus-1).
    """
    pp_ids = list(net.bus.index)         # pandapower bus index (original MATPOWER labels)
    mapping = {bid: pos for pos, bid in enumerate(pp_ids)}
    return mapping, pp_ids


def place_paper_distribution(topology):
    """
    PAPER DISTRIBUTION = 20 measurements per bus (exactly like the original SP23 Grid paper).
    That is: on each bus, assume 20 independent meters.
    Final meter count = 500 buses * 20 = 10,000 measurements.
    """
    net = topology["net"]
    mapping, pp_ids = get_pandapower_bus_mapping(net)

    meters = []

    for bid in pp_ids:
        pos = mapping[bid]        # always valid 0..n_bus-1
        for k in range(20):       # 20 meters per bus
            meters.append(pos)    # store POSITION, not original ID

    return sorted(meters)



def place_uniform(topology, M):
    """
    Uniform sampling of M bus positions.
    """
    net = topology["net"]
    nb = len(net.bus)
    base_positions = list(range(nb))
    meters = random.sample(base_positions, min(M, nb))
    return sorted(meters)



def place_dense(topology, M):
    """
    Dense = biased to lower-index buses.
    """
    net = topology["net"]
    nb = len(net.bus)

    weights = np.linspace(2.0, 0.1, nb)
    weights = weights / weights.sum()
    chosen = np.random.choice(range(nb), size=min(M, nb), replace=False, p=weights)
    return sorted(chosen.tolist())



def place_sparse(topology, M):
    """
    Sparse = biased to higher-index buses.
    """
    net = topology["net"]
    nb = len(net.bus)

    weights = np.linspace(0.1, 2.0, nb)
    weights = weights / weights.sum()
    chosen = np.random.choice(range(nb), size=min(M, nb), replace=False, p=weights)
    return sorted(chosen.tolist())



def place_load_heavy(topology, M):
    """
    Sample buses weighted by load magnitude.
    """
    net = topology["net"]
    nb = len(net.bus)

    # compute load per bus
    load_sum = np.zeros(nb)
    for _, row in net.load.iterrows():
        pp_bus = row["bus"]
        if pp_bus in net.bus.index:
            pos = list(net.bus.index).index(pp_bus)
            load_sum[pos] += row["p_mw"]

    weights = load_sum + 1e-3
    weights = weights / weights.sum()
    chosen = np.random.choice(range(nb), size=min(M, nb), replace=False, p=weights)

    return sorted(chosen.tolist())



def place_generator_heavy(topology, M):
    """
    Sample buses weighted by generator presence.
    """
    net = topology["net"]
    nb = len(net.bus)

    gen_count = np.zeros(nb)
    for _, row in net.gen.iterrows():
        pp_bus = row["bus"]
        if pp_bus in net.bus.index:
            pos = list(net.bus.index).index(pp_bus)
            gen_count[pos] += 1

    weights = gen_count + 1e-3
    weights = weights / weights.sum()
    chosen = np.random.choice(range(nb), size=min(M, nb), replace=False, p=weights)

    return sorted(chosen.tolist())



# ===========================================================
# MAIN DISPATCH FUNCTION
# ===========================================================

def select_meter_distribution(topology, dist_name, M):
    """
    Returns a VALID list of meter positions (0..nb-1).
    """

    dist_name = dist_name.lower()

    if dist_name == "paper":
        print("[INFO] Using PAPER distribution (20 meters per bus = 10,000 total)")
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
