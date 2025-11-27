
from pandapower_backend import compute_baseline, load_real_topology
from meter_placement import get_meter_list
import numpy as np

topology_name = "ACTIVSg500_real"
eng = None  # we don't use MATLAB

# load topology
topo = load_real_topology(eng, topology_name)

# compute paper-style meters
meters = get_meter_list("paper", topo)

print("Num meters:", len(meters))
print("First 10 meters:", meters[:10])

# run baseline
baseline = compute_baseline(eng, topology_name, meters)
print("BASELINE OK:", baseline)

