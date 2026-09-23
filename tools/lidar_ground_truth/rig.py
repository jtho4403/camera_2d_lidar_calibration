#!/usr/bin/env python3
"""Load and validate config/rig_<id>.json rig-parameter files (PLAN.md Sec 3.3).

Frame convention (PLAN.md Sec 1.1, 1.2): `delta_z_m` is the height of the
LiDAR scan plane above the rectified left camera optical centre, expressed in
camera "robot" frame R (right-handed, x forward / y left / z up). Positive
means the LiDAR sits higher than the camera. `pitch_deg` / `roll_deg` are the
LiDAR-to-camera relative-attitude errors from the zero-roll/zero-pitch
assumption that the SE(2) calibration otherwise makes, in degrees.

Two independent checks are applied to a rig file:

  1. Schema validation (`_validate_schema`, always enforced): required
     fields are present and well-formed. This is what "the rig file
     validates" means at the PLAN.md Sec 8 Phase 0 gate -- it says nothing
     about whether the *values* are real measurements.
  2. Measurement-readiness ("placeholder") gating (`load_rig`, enforced by
     default): a rig file whose measurement_status is not one of
     NON_PLACEHOLDER_STATUSES ("measured", or "estimated" for values derived
     from documented sources with stated tolerances rather than direct
     metrology), or whose measured_on still carries the PLACEHOLDER
     sentinel, or whose delta_z_chain still has any placeholder=true entry,
     refuses to load
     unless the caller explicitly passes allow_placeholder=True. Every call
     site that does so must also call `warn_if_placeholder` so the
     placeholder status is visible in every diagnostic that has to run
     before real metrology exists (PLAN.md Phase 0/1).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

PLACEHOLDER_MEASURED_ON_PREFIX = "PLACEHOLDER"

REQUIRED_FIELDS = [
    "rig_id",
    "measurement_status",
    "measured_on",
    "delta_z_m",
    "delta_z_tolerance_m",
    "delta_z_chain",
    "pitch_deg",
    "roll_deg",
    "attitude_tolerance_deg",
    "attitude_instrument",
    "board_standoff_m",
]

NON_PLACEHOLDER_STATUSES = {"MEASURED", "ESTIMATED"}


@dataclass
class RigConfig:
    rig_id: str
    measurement_status: str
    measured_on: str
    delta_z_m: float
    delta_z_tolerance_m: float
    delta_z_chain: list[dict]
    pitch_deg: float
    roll_deg: float
    attitude_tolerance_deg: float
    attitude_instrument: str
    board_standoff_m: float
    notes: str
    is_placeholder: bool
    source_path: Path


def _validate_schema(raw: dict, path: Path) -> None:
    """Hard, always-enforced structural validation.

    delta_z_chain is the provenance of delta_z_m: each term's value is added
    with its sign (+1 for LiDAR-side heights, -1 for camera-side heights), and
    the sum must equal delta_z_m unless the chain still has placeholder terms.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        raise ValueError(f"Rig file {path} is missing required fields: {missing}")

    if not isinstance(raw["delta_z_chain"], list) or not raw["delta_z_chain"]:
        raise ValueError(f"Rig file {path}: delta_z_chain must be a non-empty list")

    for entry in raw["delta_z_chain"]:
        for key in ("term", "sign", "value_m", "source", "tol_m"):
            if key not in entry:
                raise ValueError(
                    f"Rig file {path}: delta_z_chain entry missing '{key}': {entry}"
                )
        if entry["sign"] not in (1, -1):
            raise ValueError(
                f"Rig file {path}: delta_z_chain entry sign must be +1 or -1: {entry}"
            )

    if not any(entry.get("placeholder") for entry in raw["delta_z_chain"]):
        chain_sum = sum(e["sign"] * float(e["value_m"]) for e in raw["delta_z_chain"])
        if abs(chain_sum - float(raw["delta_z_m"])) > 1e-6:
            raise ValueError(
                f"Rig file {path}: delta_z_chain sums to {chain_sum:.6f} m but "
                f"delta_z_m is {raw['delta_z_m']}"
            )

    for name in ("delta_z_tolerance_m", "attitude_tolerance_deg"):
        if float(raw[name]) <= 0:
            raise ValueError(f"Rig file {path}: {name} must be positive, got {raw[name]}")

    if float(raw["board_standoff_m"]) < 0:
        raise ValueError(f"Rig file {path}: board_standoff_m must be >= 0")


def _is_placeholder(raw: dict) -> tuple[bool, list[str]]:
    placeholder_terms = [
        entry["term"] for entry in raw["delta_z_chain"] if entry.get("placeholder")
    ]
    status_is_placeholder = (
        str(raw["measurement_status"]).upper() not in NON_PLACEHOLDER_STATUSES
    )
    measured_on_is_placeholder = str(raw["measured_on"]).startswith(
        PLACEHOLDER_MEASURED_ON_PREFIX
    )
    is_placeholder = (
        status_is_placeholder or measured_on_is_placeholder or bool(placeholder_terms)
    )
    return is_placeholder, placeholder_terms


def load_rig(path: str | Path, allow_placeholder: bool = False) -> RigConfig:
    """Load and validate a rig parameter file.

    Schema validation always runs. The placeholder gate raises RuntimeError
    unless allow_placeholder=True -- pass that only from Phase 0/1
    diagnostic tools that must run before real rig metrology exists, and
    always follow with `warn_if_placeholder(cfg)`.
    """
    path = Path(path)
    raw = json.loads(path.read_text())

    _validate_schema(raw, path)

    is_placeholder, placeholder_terms = _is_placeholder(raw)
    if is_placeholder and not allow_placeholder:
        raise RuntimeError(
            f"Rig file {path} is still a PLACEHOLDER "
            f"(measurement_status={raw['measurement_status']!r}, "
            f"measured_on={raw['measured_on']!r}, "
            f"placeholder delta_z_chain terms={placeholder_terms}). "
            "Real rig metrology (PLAN.md Sec 3) must be recorded -- "
            "measurement_status set to \"measured\" or \"estimated\", measured_on set to a "
            "real date, and every delta_z_chain placeholder flag removed -- "
            "before this file may be used. Pass allow_placeholder=True only "
            "from Phase 0/1 development diagnostics, and warn loudly when "
            "you do."
        )

    return RigConfig(
        rig_id=raw["rig_id"],
        measurement_status=raw["measurement_status"],
        measured_on=raw["measured_on"],
        delta_z_m=float(raw["delta_z_m"]),
        delta_z_tolerance_m=float(raw["delta_z_tolerance_m"]),
        delta_z_chain=raw["delta_z_chain"],
        pitch_deg=float(raw["pitch_deg"]),
        roll_deg=float(raw["roll_deg"]),
        attitude_tolerance_deg=float(raw["attitude_tolerance_deg"]),
        attitude_instrument=raw["attitude_instrument"],
        board_standoff_m=float(raw["board_standoff_m"]),
        notes=raw.get("notes", ""),
        is_placeholder=is_placeholder,
        source_path=path,
    )


def warn_if_placeholder(cfg: RigConfig) -> None:
    """Print a loud, hard-to-miss banner if cfg came from a placeholder file.

    Call this from every diagnostic tool that loaded with
    allow_placeholder=True, right after loading.
    """
    if cfg.is_placeholder:
        print("!" * 80)
        print(f"WARNING: rig file {cfg.source_path} is a PLACEHOLDER "
              f"(measurement_status={cfg.measurement_status!r}).")
        print("delta_z_m / pitch_deg / roll_deg below are NOT real rig "
              "metrology -- do not trust row (v) predictions from this run.")
        print("!" * 80)


def predicted_row_offset_px(f_y: float, delta_z_m: float, x_r_m: float) -> float:
    """v - c_y = -f_y * delta_z_m / x_r_m (PLAN.md Sec 1.2/1.3).

    x_r_m is camera-frame depth (frame R), i.e. d_gt for the point in
    question. Only valid for x_r_m > 0 (in front of the camera).
    """
    return -f_y * delta_z_m / x_r_m


def row_geometry_report(
    f_y: float, c_y: float, delta_z_m: float, distances_m: list[float]
) -> list[dict]:
    """PLAN.md Sec 1.3 row-geometry sanity check across a list of depths."""
    rows = []
    for x_r in distances_m:
        offset_px = predicted_row_offset_px(f_y, delta_z_m, x_r)
        rows.append({
            "distance_m": float(x_r),
            "row_offset_px": float(offset_px),
            "predicted_row_px": float(c_y + offset_px),
        })
    return rows


def _load_fy_cy_height_from_manifest(manifest_path: Path) -> tuple[float, float, int]:
    with open(manifest_path) as f:
        manifest = json.load(f)
    left = manifest["calibration"]["rectified"]["left"]
    return float(left["fy"]), float(left["cy"]), int(left["image_size"][1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rig",
        default=str(Path(__file__).resolve().parents[2] / "config" / "rig_template.json"),
    )
    parser.add_argument(
        "--camera-manifest",
        required=True,
        help=(
            "a capture session's session_manifest.json to source the rectified "
            "f_y/c_y and image height from (PLAN.md Sec 1.3), e.g. "
            "data/<session>/captures/session_manifest.json"
        ),
    )
    parser.add_argument(
        "--distances-m", nargs="+", type=float,
        default=[0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    )
    args = parser.parse_args()

    rig_path = Path(args.rig)

    print("=" * 80)
    print(f"Schema-validating rig file: {rig_path}")
    raw = json.loads(rig_path.read_text())
    _validate_schema(raw, rig_path)
    print("Schema validation: PASSED (all required fields present and well-formed)")

    print()
    print("Strict (measurement-readiness) load, default allow_placeholder=False:")
    try:
        strict = load_rig(rig_path, allow_placeholder=False)
        print(
            "Strict load: PASSED (measurement_status="
            f"{strict.measurement_status!r}, not a placeholder)"
        )
    except RuntimeError as exc:
        print(f"Strict load: FAILED LOUDLY as expected for a placeholder rig file:\n  {exc}")

    print()
    cfg = load_rig(rig_path, allow_placeholder=True)
    warn_if_placeholder(cfg)

    f_y, c_y, image_height_px = _load_fy_cy_height_from_manifest(Path(args.camera_manifest))
    print()
    print(f"Using f_y={f_y:.4f} px, c_y={c_y:.4f} px from {args.camera_manifest}")
    print(f"delta_z_m={cfg.delta_z_m} (rig measurement_status={cfg.measurement_status})")
    print()
    header = f"{'x_R (m)':>10} | {'row offset (px)':>16} | {'predicted row (px)':>19} | inside image?"
    print(header)
    print("-" * len(header))
    for row in row_geometry_report(f_y, c_y, cfg.delta_z_m, args.distances_m):
        inside = 0 <= row["predicted_row_px"] <= image_height_px
        print(
            f"{row['distance_m']:>10.2f} | {row['row_offset_px']:>16.2f} | "
            f"{row['predicted_row_px']:>19.2f} | {'yes' if inside else 'NO'}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
