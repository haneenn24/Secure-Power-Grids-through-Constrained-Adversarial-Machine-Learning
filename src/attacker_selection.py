"""
attacker_selection.py

Defines how the attacker chooses which meters to compromise.

Main function:
    pick_random_compromised(meter_list, fraction, rng)

Where:
    meter_list : list of all meter names placed by the defender
    fraction   : fraction of meters the attacker controls (0.2, 0.4, etc.)
    rng        : random generator (numpy or Python random)
"""

from typing import List


def pick_random_compromised(meter_list: List[str], fraction: float, rng) -> List[str]:
    """
    Randomly selects a subset of meters to compromise.

    Args:
        meter_list (List[str]):
            All meters currently deployed on the topology (after applying
            a meter placement distribution).
        fraction (float):
            Fraction of meters the attacker can control.
            Example: 0.2 means attacker controls 20% of meters.
        rng:
            Random generator with .choice or .sample methods.
            Can be numpy.random.Generator or Python's random.

    Returns:
        List[str]: meters that the attacker compromises.
    """
    total_meters = len(meter_list)
    num_compromised = int(total_meters * fraction)

    if num_compromised <= 0:
        return []

    # If rng is numpy generator → use choice
    if hasattr(rng, "choice"):
        compromised = rng.choice(
            meter_list, size=num_compromised, replace=False
        ).tolist()
    else:
        # fallback for Python `random`
        compromised = rng.sample(meter_list, num_compromised)

    return compromised
