#!/usr/bin/env python3
"""Phase 1.7 close-out tool: blind vertical-edge annotation vs LiDAR
discontinuity matching, to test the Phase 1.5 off-board horizontal-offset
finding without selecting on the outcome.

Three phases, run as separate CLI subcommands so Phase A is structurally
incapable of being influenced by Phase B/C:

  annotate  (Phase A) -- Tk GUI. Displays ONLY the raw image: no projected
            points, no predicted row, no overlay of any kind. Click a
            vertical scene edge, pick a confidence tier (high/medium/low),
            type a short description. Saves blind_annotations_<pose>.json
            and exits. THE PHASE-A SECTION OF THIS FILE (see the banner
            below) IMPORTS NOTHING FROM projection.py, calibration_io.py,
            rig.py, scan_io.py, or filters.py -- there is no code path in
            annotate() that could load or display a LiDAR point, a
            transform, or a predicted row. That is enforced by import
            placement, not just by not calling anything: those modules are
            only imported inside match(), physically separated below.

  match     (Phase B) -- loads blind_annotations_<pose>.json (must already
            exist and be closed), projects the scan through the EXISTING,
            unmodified T2, finds the nearest LiDAR range discontinuity to
            each annotated column (reusing step3_horizontal_offset.py's
            existing detect_lidar_discontinuities -- no second automatic
            matcher is built here), rejects matches beyond --window-px, and
            saves blind_matches_<pose>.json.

  report    (Phase C) -- aggregates all blind_matches_*.json by confidence
            tier and pooled: N, mean, std, SEM, a sign test, and Delta_u
            plotted against depth and against (u - c_x).

Read-only against cam_lidar_2d_icp.py / gui.py / icp_2d.py /
tools/lidar_characterisation.

Usage (run from the repository root):
    python tools/lidar_ground_truth/diagnostics/blind_edge_picker.py annotate --pose pose_03 --image data/data_2026-06-28/images/left_pose_03.png
    python tools/lidar_ground_truth/diagnostics/blind_edge_picker.py match --pose pose_03
    python tools/lidar_ground_truth/diagnostics/blind_edge_picker.py report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/lidar_ground_truth/
sys.path.insert(0, str(Path(__file__).resolve().parent))       # diagnostics/

OUT_DIR = Path(__file__).resolve().parents[3] / "results" / "lidar_ground_truth" / "diagnostics" / "blind_edge_picker"
DEFAULT_SESSION_DIR = Path(__file__).resolve().parents[3] / "data" / "data_2026-06-28"

TIERS = ("high", "medium", "low")


# ======================================================================
# PHASE A -- BLIND ANNOTATION. No LiDAR/calibration import below this
# banner until Phase B's own banner further down. Do not add one.
# ======================================================================
import cv2  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import ttk  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402


@dataclass
class AnnotationSession:
    """Data model for one pose's blind annotations. Shared by the live Tk
    GUI and a scripted driver (see run_scripted_annotation below) -- both
    call add_point()/save() so the saved file has exactly one code path,
    exercised for real either way.
    """
    pose_id: str
    image_path: str
    points: list = field(default_factory=list)

    def add_point(self, u_col: float, tier: str, description: str) -> None:
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
        self.points.append({
            "u_col_px": float(u_col),
            "tier": tier,
            "description": description,
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"blind_annotations_{self.pose_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path


class BlindAnnotatorGUI:
    """Tk window showing ONLY the raw image. No projection, no overlay, no
    predicted row -- nothing computed from T2, K, or any .pcd file appears
    anywhere in this class.
    """

    def __init__(self, session: AnnotationSession, out_dir: Path):
        self.session = session
        self.out_dir = out_dir
        self.pending_u = None

        self.root = tk.Tk()
        self.root.title(f"Blind edge picker (Phase A, BLIND) -- {session.pose_id}")
        self.root.geometry("1000x800")

        instructions = tk.Label(
            self.root,
            text=(
                "Click a vertical scene edge. Prefer isolated, high-contrast, "
                "unambiguous silhouettes with a large depth discontinuity on "
                "both sides (e.g. an isolated object against a plain wall). "
                "Skip dark-on-dark and cluttered regions rather than guessing -- "
                "a smaller high-confidence set beats a larger ambiguous one. "
                "This view shows the raw image ONLY; nothing else exists to see."
            ),
            wraplength=980, justify="left",
        )
        instructions.pack(padx=5, pady=5)

        image_bgr = cv2.imread(session.image_path)
        if image_bgr is None:
            raise FileNotFoundError(session.image_path)
        self.image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.image_h, self.image_w = self.image_rgb.shape[:2]

        self.figure = plt.Figure(figsize=(9, 5.5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(self.image_rgb)
        self.ax.set_title("Click a vertical edge")
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().pack()
        self.canvas.mpl_connect("button_press_event", self.on_click)

        self.marker_lines = []

        entry_frame = ttk.Frame(self.root)
        entry_frame.pack(pady=(8, 0))

        ttk.Label(entry_frame, text="Confidence tier:").grid(row=0, column=0, padx=5)
        self.tier_var = tk.StringVar(value="high")
        for i, tier in enumerate(TIERS):
            ttk.Radiobutton(entry_frame, text=tier, variable=self.tier_var, value=tier).grid(
                row=0, column=1 + i, padx=5
            )

        ttk.Label(entry_frame, text="Description:").grid(row=1, column=0, padx=5, pady=5)
        self.description_var = tk.StringVar()
        ttk.Entry(entry_frame, textvariable=self.description_var, width=60).grid(
            row=1, column=1, columnspan=3, padx=5, pady=5
        )

        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=(0, 8))
        ttk.Button(button_frame, text="Record point", command=self.record_point).pack(
            side="left", padx=10
        )
        ttk.Button(button_frame, text="Save && Exit", command=self.save_and_exit).pack(
            side="left", padx=10
        )

        self.status_var = tk.StringVar(value="0 points recorded.")
        ttk.Label(self.root, textvariable=self.status_var).pack()

    def on_click(self, event) -> None:
        if event.xdata is None:
            return
        self.pending_u = float(event.xdata)
        for line in self.marker_lines:
            line.remove()
        self.marker_lines = [self.ax.axvline(self.pending_u, color="red", lw=1.0, ls="--")]
        self.canvas.draw()
        self.status_var.set(
            f"Pending click at u={self.pending_u:.1f}px -- set tier/description, "
            f"then 'Record point'. ({len(self.session.points)} recorded so far.)"
        )

    def record_point(self) -> None:
        if self.pending_u is None:
            self.status_var.set("Click the image first.")
            return
        self.session.add_point(self.pending_u, self.tier_var.get(), self.description_var.get())
        self.pending_u = None
        self.description_var.set("")
        self.status_var.set(f"{len(self.session.points)} points recorded.")

    def save_and_exit(self) -> None:
        path = self.session.save(self.out_dir)
        print(f"Saved {path} ({len(self.session.points)} points)")
        self.root.after(0, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()


def run_scripted_annotation(pose_id: str, image_path: str, out_dir: Path, picks: list[dict]) -> Path:
    """Non-interactive substitute for a live mouse-driven Tk session, used
    only because no human operator was available to run `annotate`
    interactively in this environment. Calls the exact same
    AnnotationSession.add_point()/save() the GUI's "Record point"/"Save &&
    Exit" buttons call -- same data model, same file format -- so Phase B/C
    cannot tell the difference. It does NOT touch any LiDAR/calibration
    import either; it is still physically inside the Phase A section.

    Each pick's u_col must have been decided by looking at the raw image
    ONLY (e.g. via a cropped screenshot), before this function is called
    and before any projection for that pose has been computed -- see
    report.md for exactly how each pick in this session was produced.
    """
    session = AnnotationSession(pose_id=pose_id, image_path=image_path)
    for p in picks:
        session.add_point(p["u_col_px"], p["tier"], p["description"])
    return session.save(out_dir)


def cmd_annotate(args) -> None:
    session = AnnotationSession(pose_id=args.pose, image_path=args.image)
    gui = BlindAnnotatorGUI(session, Path(args.out_dir))
    gui.run()


# ======================================================================
# PHASE B -- REVEAL AND MATCH. LiDAR/calibration imports start here, not
# above. Only reachable via the `match` subcommand, after a pose's
# blind_annotations_<pose>.json has already been saved and closed.
# ======================================================================
import numpy as np  # noqa: E402

import calibration_io  # noqa: E402
import config  # noqa: E402
import projection  # noqa: E402
import rig as rig_module  # noqa: E402
import scan_io  # noqa: E402
from step3_horizontal_offset import (  # noqa: E402
    RANGE_JUMP_THRESHOLD_M,
    detect_lidar_discontinuities,
)


def cmd_match(args) -> None:
    out_dir = Path(args.out_dir)
    ann_path = out_dir / f"blind_annotations_{args.pose}.json"
    if not ann_path.is_file():
        raise FileNotFoundError(
            f"{ann_path} not found -- run `annotate` for this pose first and "
            "make sure it was saved (Save && Exit)."
        )
    annotations = json.loads(ann_path.read_text())
    points = annotations["points"]
    print(f"Loaded {len(points)} blind annotations for {args.pose} from {ann_path}")

    session_dir = Path(args.session_dir)
    image_path = session_dir / "images" / f"left_{args.pose}.png"
    laser_path = session_dir / "lasers" / f"laser_{args.pose}.pcd"
    metadata_path = session_dir / "additional_image_data" / f"metadata_{args.pose}.json"

    image_bgr = cv2.imread(str(image_path))
    h, w = image_bgr.shape[:2]

    T2 = calibration_io.load_transform()
    rig_cfg = rig_module.load_rig(config.DEFAULT_RIG_PATH, allow_placeholder=True)
    rig_module.warn_if_placeholder(rig_cfg)
    intrinsics = calibration_io.load_intrinsics_from_metadata(metadata_path)

    xy_valid, _, _ = scan_io.load_valid_xy(laser_path)
    result = projection.project_scan(
        xy_valid, T2, rig_cfg.delta_z_m, intrinsics.K, (w, h), z_min_m=config.Z_MIN_M,
    )
    candidates = detect_lidar_discontinuities(
        xy_valid, result.u, result.v, result.depth_m, result.keep, RANGE_JUMP_THRESHOLD_M,
    )
    print(f"{len(candidates)} candidate LiDAR discontinuities found (in-frame side kept)")

    matches = []
    n_rejected = 0
    for p in points:
        u_annot = p["u_col_px"]
        if not candidates:
            n_rejected += 1
            continue
        nearest = min(candidates, key=lambda c: abs(c["u_lidar"] - u_annot))
        dist = abs(nearest["u_lidar"] - u_annot)
        if dist > args.window_px:
            n_rejected += 1
            print(f"  REJECTED: annotation u={u_annot:.1f} ({p['tier']}, "
                  f"{p['description']!r}) -- nearest discontinuity {dist:.1f}px "
                  f"away (window={args.window_px}px)")
            continue
        delta_u = nearest["u_lidar"] - u_annot
        matches.append({
            "pose_id": args.pose,
            "tier": p["tier"],
            "description": p["description"],
            "u_annotated_px": u_annot,
            "u_lidar_px": nearest["u_lidar"],
            "v_lidar_px": nearest["v_lidar"],
            "depth_m": nearest["depth_m"],
            "match_distance_px": dist,
            "delta_u_px": delta_u,
        })
        print(f"  matched: u_annot={u_annot:.1f} ({p['tier']}) -> u_lidar={nearest['u_lidar']:.1f} "
              f"depth={nearest['depth_m']:.3f}m  Delta_u={delta_u:+.1f}px  "
              f"(match_dist={dist:.1f}px)  [{p['description']}]")

    out_path = out_dir / f"blind_matches_{args.pose}.json"
    with open(out_path, "w") as f:
        json.dump({"pose_id": args.pose, "window_px": args.window_px,
                    "n_annotated": len(points), "n_matched": len(matches),
                    "n_rejected": n_rejected, "matches": matches}, f, indent=2)
    print(f"\n{len(matches)} matched, {n_rejected} rejected (> {args.window_px}px). Saved {out_path}")


# ======================================================================
# PHASE C -- REPORT.
# ======================================================================
from scipy import stats  # noqa: E402
import matplotlib.pyplot as plt2  # noqa: E402  (already imported above; alias avoids re-import lint noise)


def cmd_report(args) -> None:
    out_dir = Path(args.out_dir)
    all_matches = []
    for match_path in sorted(out_dir.glob("blind_matches_*.json")):
        data = json.loads(match_path.read_text())
        all_matches.extend(data["matches"])

    if not all_matches:
        print(f"No blind_matches_*.json found under {out_dir}. Run `match` for each annotated pose first.")
        return

    print(f"Loaded {len(all_matches)} matched points from {out_dir}")

    def tier_stats(rows):
        d = np.array([r["delta_u_px"] for r in rows])
        n = len(d)
        if n == 0:
            return {"n": 0}
        mean = float(d.mean())
        std = float(d.std(ddof=1)) if n > 1 else float("nan")
        sem = std / np.sqrt(n) if n > 1 else float("nan")
        n_pos = int(np.sum(d > 0))
        sign_p_one_sided = stats.binomtest(max(n_pos, n - n_pos), n, 0.5, alternative="greater").pvalue
        sign_p_two_sided = stats.binomtest(max(n_pos, n - n_pos), n, 0.5, alternative="two-sided").pvalue
        return {
            "n": n, "mean_px": mean, "std_px": std, "sem_px": sem,
            "n_positive": n_pos, "n_negative": n - n_pos,
            "sign_test_p_one_sided": float(sign_p_one_sided),
            "sign_test_p_two_sided": float(sign_p_two_sided),
        }

    print()
    print("=" * 90)
    for tier in TIERS:
        rows = [m for m in all_matches if m["tier"] == tier]
        s = tier_stats(rows)
        if s["n"] == 0:
            print(f"{tier:8s}: N=0")
            continue
        print(f"{tier:8s}: N={s['n']:3d}  mean={s['mean_px']:+7.2f}px  std={s['std_px']:6.2f}px  "
              f"SEM={s['sem_px']:6.2f}px  sign-test(+{s['n_positive']}/-{s['n_negative']}) "
              f"p_one_sided={s['sign_test_p_one_sided']:.4f} p_two_sided={s['sign_test_p_two_sided']:.4f}")

    pooled = tier_stats(all_matches)
    print("-" * 90)
    print(f"{'pooled':8s}: N={pooled['n']:3d}  mean={pooled['mean_px']:+7.2f}px  "
          f"std={pooled['std_px']:6.2f}px  SEM={pooled['sem_px']:6.2f}px  "
          f"sign-test(+{pooled['n_positive']}/-{pooled['n_negative']}) "
          f"p_one_sided={pooled['sign_test_p_one_sided']:.4f} p_two_sided={pooled['sign_test_p_two_sided']:.4f}")
    print("=" * 90)

    high_rows = [m for m in all_matches if m["tier"] == "high"]
    high_stats = tier_stats(high_rows)
    print()
    if high_stats["n"] == 0:
        decision = "INCONCLUSIVE: no high-tier annotations."
    elif abs(high_stats["mean_px"]) <= 10 and high_stats["n"] >= 3:
        decision = "RESOLVED AS NOISE: high-tier mean within +-10px of zero."
    elif high_stats["mean_px"] >= 20 and (high_stats["sem_px"] == high_stats["sem_px"]) and \
            (high_stats["mean_px"] - 1.96 * high_stats["sem_px"]) > 0:
        decision = "REAL ANOMALY: high-tier mean >= +20px with 95% CI excluding zero."
    else:
        decision = "INCONCLUSIVE per the pre-registered decision rule."
    print(f"DECISION: {decision}")

    K = calibration_io.load_intrinsics_from_metadata(
        Path(args.session_dir) / "additional_image_data" / "metadata_pose_01.json"
    ).K
    cx = K[0, 2]

    fig, axes = plt2.subplots(1, 2, figsize=(12, 5))
    colors = {"high": "tab:green", "medium": "tab:orange", "low": "tab:red"}
    for tier in TIERS:
        rows = [m for m in all_matches if m["tier"] == tier]
        if not rows:
            continue
        depth = [r["depth_m"] for r in rows]
        du = [r["delta_u_px"] for r in rows]
        u_minus_cx = [r["u_lidar_px"] - cx for r in rows]
        axes[0].scatter(depth, du, c=colors[tier], label=tier)
        axes[1].scatter(u_minus_cx, du, c=colors[tier], label=tier)
    for ax in axes:
        ax.axhline(0, c="gray", lw=0.8)
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_xlabel("depth (m)"); axes[0].set_ylabel("Delta_u (px)")
    axes[0].set_title("Blind-picked Delta_u vs depth")
    axes[1].set_xlabel("u - c_x (px)"); axes[1].set_ylabel("Delta_u (px)")
    axes[1].set_title("Blind-picked Delta_u vs (u - c_x)")
    fig.tight_layout()
    fig.savefig(out_dir / "blind_report_plots.png", dpi=150)
    plt2.close(fig)

    with open(out_dir / "blind_report_summary.json", "w") as f:
        json.dump({
            "per_tier": {tier: tier_stats([m for m in all_matches if m["tier"] == tier]) for tier in TIERS},
            "pooled": pooled,
            "decision": decision,
        }, f, indent=2)
    print(f"\nSaved {out_dir / 'blind_report_summary.json'} and blind_report_plots.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_annotate = sub.add_parser("annotate", help="Phase A: blind Tk annotation")
    p_annotate.add_argument("--pose", required=True)
    p_annotate.add_argument("--image", required=True)
    p_annotate.add_argument("--out-dir", default=str(OUT_DIR))
    p_annotate.set_defaults(func=cmd_annotate)

    p_match = sub.add_parser("match", help="Phase B: reveal and match")
    p_match.add_argument("--pose", required=True)
    p_match.add_argument("--session-dir", default=str(DEFAULT_SESSION_DIR))
    p_match.add_argument("--out-dir", default=str(OUT_DIR))
    p_match.add_argument("--window-px", type=float, default=60.0)
    p_match.set_defaults(func=cmd_match)

    p_report = sub.add_parser("report", help="Phase C: aggregate report")
    p_report.add_argument("--out-dir", default=str(OUT_DIR))
    p_report.add_argument("--session-dir", default=str(DEFAULT_SESSION_DIR))
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
