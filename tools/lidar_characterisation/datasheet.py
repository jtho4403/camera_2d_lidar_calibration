"""Datasheet-band lookup and objective status classification."""
import config


def precision_band_mm(distance_m: float):
    """Return the datasheet precision STD (mm) band for a given distance, or None."""
    for lo, hi, spec_mm in config.DATASHEET_PRECISION_BANDS_MM:
        if lo <= distance_m <= hi:
            return spec_mm
    return None


def classify(measured_mm: float, spec_mm: float | None) -> str:
    """Objective banding: within spec (<0.9x), at spec (0.9-1.1x), worse (>1.1x).

    Returns "no datasheet spec at this distance" if spec_mm is None.
    """
    if spec_mm is None:
        return "no datasheet spec at this distance"
    if measured_mm < 0.9 * spec_mm:
        return "within spec"
    if measured_mm <= 1.1 * spec_mm:
        return "at spec"
    return "worse than spec"
