"""
attacker_selection.py

Utility for selecting which SCADA meters the attacker compromises.
Used inside the FDIA experiment loop.
"""

from typing import List


def pick_random_compromised(meter_list: List[str], fraction: float, rng) -> List[str]:
    """
    Randomly choose a subset of meters for the attacker to compromise.

    Parameters
    ----------
    meter_list : List[str]
        All meters currently deployed on the grid after applying the
        chosen meter-placement distribution (e.g., uniform, sparse).
    fraction : float
        Fraction of meters the attacker controls.
        Example: fraction=0.2 → attacker compromises 20% of all meters.
    rng :
        Random number generator. Must support:
        - rng.choice(...) for numpy generators, OR
        - rng.sample(...) for Python's random module.
        Used to ensure reproducibility across experiment trials.

    Returns
    -------
    List[str]
        List of meter identifiers selected as compromised.
        Length = int(len(meter_list) * fraction)

    Notes
    -----
    - No replacement is used: attacker compromises *distinct* meters.
    - If fraction * total_meters < 1 → returns empty list.
    """

    total_meters = len(meter_list)
    num_compromised = int(total_meters * fraction)

    if num_compromised <= 0:
        return []

    # numpy.random.Generator
    if hasattr(rng, "choice"):
        compromised = rng.choice(
            meter_list, size=num_compromised, replace=False
        ).tolist()

    # python random.Random
    else:
        compromised = rng.sample(meter_list, num_compromised)

    return compromised
