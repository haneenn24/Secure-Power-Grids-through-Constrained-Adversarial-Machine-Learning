Configs:

    Purpose:
    Holds configuration files for running experiments.

    Contents:

    config.yaml → Main experiment configuration (topology, distributions, attacker fractions, N trials, output CSV path, MATLAB/pandapower settings).

Configs_base:

    Purpose:
    Serves as a base storage of template configs or backup configs.
    Used when you want to switch experiment presets easily.

    Role:
    Not used automatically unless you load it manually. It’s a repository of predefined settings.

Documentation:

    Purpose:
    Internal documentation folder.
    You will place the descriptions we generate here (Markdown, PDF, notes).

    Role:
    Does not affect execution; for writing the research documentation.

FDIA_Attacks:

    Purpose:
    Contains MATLAB scripts or Python code implementing FDIA (False Data Injection Attacks).

    Role:
    Used indirectly by pandapower_backend.py or matlab_interface.py when performing the attack step.

    Examples:

    Attack models

    Mask/glass attacks (your previous work)

    Optimization logic (if MATLAB is used)

Results

    Purpose:
    Stores experiment outputs:

    CSVs

    Plots

    Logs

    Summary statistics

Src

    for full detailed documentation see inside the file