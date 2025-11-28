# src/wls_estimator.py

import numpy as np
import pandapower as pp
import pandapower.estimation as est
from typing import Iterable, Tuple, Union, List


Measurement = Union[
    Tuple[str, str, float, int, float],
    Tuple[str, str, float, int, float, Union[str, None]],
]


def run_wls_estimator(net: pp.pandapowerNet,
                      measurements: Iterable[Measurement]):
    """
    Run WLS state estimation on a pandapower net using a set of measurements.

    Parameters
    ----------
    net : pandapowerNet
        The network object (already built from MATPOWER / ACTIVSg500).
    measurements : iterable
        Iterable of measurement specs. Each entry is either:
          (meas_type, element_type, value, element_index, std_dev)
        or
          (meas_type, element_type, value, element_index, std_dev, side)

        where:
          - meas_type      ∈ {"v", "p", "q", "i"} (we mainly use "v" and "p")
          - element_type   ∈ {"bus", "line", ...}
          - value          : measurement value (V in p.u., P in MW, etc.)
          - element_index  : index of bus / line in net
          - std_dev        : standard deviation (same units as value)
          - side           : only for line measurements ("from" / "to"), or None

    Returns
    -------
    net : pandapowerNet
        The same net, updated with state estimation results in net.res_*_est.
    J : float
        WLS objective J = Σ ((z_i - h_i)^2 / σ_i^2) over all measurements.
    perceived_load : float
        Total estimated load (MW) in load convention, summed over all load buses.
    """

    # ------------------------------------------------------------------
    # 1. Clear existing measurements
    # ------------------------------------------------------------------
    if "measurement" in net and net.measurement is not None and len(net.measurement):
        net.measurement.drop(net.measurement.index, inplace=True)

    # We’ll keep a normalized list of measurement specs so we can
    # compute J after the SE is run.
    normalized_meas: List[Tuple[str, str, float, int, float, Union[str, None]]] = []

    # ------------------------------------------------------------------
    # 2. Create measurements in the pandapower net
    # ------------------------------------------------------------------
    for entry in measurements:
        if len(entry) == 5:
            meas_type, element_type, value, element_idx, std_dev = entry
            side = None
        elif len(entry) == 6:
            meas_type, element_type, value, element_idx, std_dev, side = entry
        else:
            raise ValueError(
                f"Measurement entry has invalid length {len(entry)}: {entry}"
            )

        # Create the measurement in the net
        if element_type == "line":
            # For lines, pass side="from"/"to"
            pp.create_measurement(
                net,
                meas_type,
                "line",
                float(value),
                float(std_dev),
                int(element_idx),
                side=side,
            )
        else:
            pp.create_measurement(
                net,
                meas_type,
                element_type,
                float(value),
                float(std_dev),
                int(element_idx),
            )

        normalized_meas.append(
            (meas_type, element_type, float(value),
             int(element_idx), float(std_dev), side)
        )

    # ------------------------------------------------------------------
    # 3. Run WLS state estimation using the official wrapper
    # ------------------------------------------------------------------
    success = est.estimate(net, algorithm="wls", init="flat")
    if not success:
        raise RuntimeError("State estimation (WLS) did not converge")

    # After this call we should have:
    #   net.res_bus_est   : estimated bus voltages / injections
    #   net.res_line_est  : estimated line flows (if line measurements exist)

    # ------------------------------------------------------------------
    # 4. Compute WLS objective J from measurement residuals
    #    J = Σ ((z_i - h_i)^2 / σ_i^2)
    # ------------------------------------------------------------------
    J = 0.0

    # res tables
    res_bus = getattr(net, "res_bus_est", None)
    res_line = getattr(net, "res_line_est", None)

    for meas_type, element_type, value, element_idx, std_dev, side in normalized_meas:
        # Estimated value h_i depends on type / element
        if meas_type == "v" and element_type == "bus":
            # Voltage magnitude (p.u.)
            if res_bus is None:
                continue
            est_val = float(res_bus.loc[element_idx, "vm_pu"])

        elif meas_type == "p" and element_type == "bus":
            # Active power at bus.
            # NOTE: pandapower stores bus injections in res_bus_est.p_mw.
            # For consistency, we assume the measurement value is in the same
            # sign convention (injection). If you use load convention, make
            # sure you convert consistently when building 'measurements'.
            if res_bus is None:
                continue
            est_val = float(res_bus.loc[element_idx, "p_mw"])

        elif meas_type == "p" and element_type == "line":
            # Active power flow on a line, at from/to end.
            if res_line is None or side not in ("from", "to"):
                continue
            col = "p_from_mw" if side == "from" else "p_to_mw"
            est_val = float(res_line.loc[element_idx, col])

        else:
            # Other types (q, i, etc.) can be added similarly if needed
            continue

        # Normalized residual
        if std_dev <= 0:
            continue  # avoid division by zero
        r = (value - est_val) / std_dev
        J += float(r * r)

    # ------------------------------------------------------------------
    # 5. Compute perceived total load from estimated state
    # ------------------------------------------------------------------
    perceived_load = 0.0
    if res_bus is not None and len(net.load):
        load_buses = net.load.bus.values
        # res_bus_est.p_mw is injection; for loads, injection is negative.
        # Total load in MW (positive) is -sum(p_inj at load buses).
        p_inj = res_bus.loc[load_buses, "p_mw"]
        perceived_load = float((-p_inj).sum())

    return net, float(J), float(perceived_load)
