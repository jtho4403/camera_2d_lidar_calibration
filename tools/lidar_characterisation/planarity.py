"""Step 3: planarity / systematic-distortion analysis of aggregated residuals."""
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class PlanarityResult:
    r2_by_degree: dict         # degree -> R^2 (variance explained vs flat baseline)
    max_abs_deviation_mm: float  # max |best-fit trend curve - flat mean|, in mm
    n_points: int
    coeffs_best: np.ndarray    # coefficients of the highest-order fit, for plotting
    best_degree: int


def _r2_vs_flat(x: np.ndarray, y: np.ndarray, degree: int) -> tuple[float, np.ndarray]:
    baseline = y.mean()
    ss_tot = np.sum((y - baseline) ** 2)
    coeffs = np.polyfit(x, y, degree)
    pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - pred) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(r2), coeffs


def analyse_trend(x: np.ndarray, residuals_m: np.ndarray) -> PlanarityResult:
    """Fit low-order polynomials of residual vs x (angle or along-line position).

    Returns R^2 for each configured degree relative to a flat (mean) baseline,
    plus the max deviation of the highest-degree trend curve from that mean.
    """
    r2_by_degree = {}
    coeffs_by_degree = {}
    for d in config.PLANARITY_POLY_DEGREES:
        r2, coeffs = _r2_vs_flat(x, residuals_m, d)
        r2_by_degree[d] = r2
        coeffs_by_degree[d] = coeffs

    best_degree = max(r2_by_degree, key=r2_by_degree.get)
    coeffs_best = coeffs_by_degree[best_degree]
    baseline = residuals_m.mean()
    trend = np.polyval(coeffs_best, np.sort(x))
    max_dev_mm = float(np.max(np.abs(trend - baseline)) * 1000.0)

    return PlanarityResult(
        r2_by_degree=r2_by_degree,
        max_abs_deviation_mm=max_dev_mm,
        n_points=len(x),
        coeffs_best=coeffs_best,
        best_degree=best_degree,
    )
