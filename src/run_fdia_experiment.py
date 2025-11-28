# ===============================================================
# run_fdia_experiment.py
# ===============================================================

import csv
import numpy as np
from fdia_attack import run_fdia_attack
from meter_placement import get_meter_list
from wls_estimator import run_wls_estimator
from topology_loader import load_sc500_grid

def run_fdia_meter_experiment(mat_path, distributions, fractions, trials, out_csv):

    net, topo = load_sc500_grid(mat_path)

    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "topology","meter_distribution","compromised_fraction","trial_id",
            "true_load","perceived_load_baseline","J_baseline",
            "perceived_load_attack","J_attack","delta_J","perceived_load_drop_percent",
            "success_flag"
        ])

        for dist in distributions:
            meters = get_meter_list(dist, topo)

            for frac in fractions:
                k = int(len(meters) * frac)

                for t in range(trials):
                    compromised = np.random.choice(len(meters), size=k, replace=False)

                    J0, J1, L0, L1 = run_fdia_attack(
                        topo["net"], meters, compromised,
                        target_drop=0.20,
                        wls_fn=run_wls_estimator
                    )

                    perc_drop = 100 * (L0 - L1) / L0

                    wr.writerow([
                        "ACTIVSg500","paper",frac,t,
                        L0,L0,J0,
                        L1,J1,J1-J0,perc_drop,
                        int(perc_drop>=20 and (J1-J0<=1e-3))
                    ])
