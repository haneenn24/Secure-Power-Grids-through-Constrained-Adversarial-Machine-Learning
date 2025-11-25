FDIA Meter-Placement Experiment

Reproducing Figure 8 from CyberGridSim (FDIA Detection vs Load Misestimation)

This project implements a full experimental pipeline for evaluating False Data Injection Attacks (FDIA) on power grids, under different meter placement configurations.
The goal is to analyze how meter distribution affects attack impact (load drop) and detectability (ΔJ), similar to Figure 8 in the CyberGridSim paper.


run_experiment.py
        ↓
run_fdia_experiment.py  → loops over distributions & attacker fractions
        ↓
topology_loader.py      → loads SC-500 grid
        ↓
meter_placement.py      → builds the meter list
        ↓
attacker_selection.py   → picks compromised meters
        ↓
pandapower_backend.py  
    ↳ compute_baseline
    ↳ run_fdia_attack
        ↓
CSV written (fdia_meter_placement.csv)
        ↓
visualize_fdia_results.py → all Figures / plots

