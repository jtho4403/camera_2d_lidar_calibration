from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


def read_ascii_pcd(path: Path) -> np.ndarray:
    with path.open("r") as f:
        lines = f.readlines()

    data_start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "data ascii":
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"No DATA ascii header found in {path}")

    pts = []
    for line in lines[data_start:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])

    return np.asarray(pts, dtype=float)


def choose_indices(n: int) -> list[int]:
    if n <= 5:
        return list(range(n))
    raw = [0, round(0.25 * (n - 1)), round(0.50 * (n - 1)), round(0.75 * (n - 1)), n - 1]
    return sorted(set(raw))


def summarise_points(xy: np.ndarray) -> dict:
    ranges = np.linalg.norm(xy, axis=1)
    return {
        "n": len(xy),
        "x_min": xy[:, 0].min(),
        "x_max": xy[:, 0].max(),
        "y_min": xy[:, 1].min(),
        "y_max": xy[:, 1].max(),
        "range_min": ranges.min(),
        "range_med": np.median(ranges),
        "range_max": ranges.max(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot early/25%/middle/75%/late LiDAR PCD candidates for each pose."
    )
    parser.add_argument(
        "--input-root",
        default="data/raw_lasers/extracted_pcd",
        help="Root containing pose_01, pose_02, ... folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/lidar_candidate_plots",
        help="Directory where diagnostic plots will be saved.",
    )
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        help="Optional fixed x-axis limits, e.g. --xlim 0 2.2",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        help="Optional fixed y-axis limits, e.g. --ylim -1.0 1.0",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_dirs = sorted([p for p in input_root.glob("pose_*") if p.is_dir()])
    if not pose_dirs:
        raise FileNotFoundError(f"No pose_* folders found in {input_root.resolve()}")

    print("=" * 80)
    print("LiDAR candidate PCD diagnostic plotting")
    print(f"Input root : {input_root.resolve()}")
    print(f"Output dir : {output_dir.resolve()}")
    print(f"Pose count : {len(pose_dirs)}")
    print("=" * 80)

    summary_lines = []

    for pose_dir in pose_dirs:
        pcd_files = sorted(pose_dir.glob("*.pcd"))
        print(f"\n[{pose_dir.name}]")
        print(f"  PCD files found: {len(pcd_files)}")

        if not pcd_files:
            print("  WARN: no PCD files found; skipping.")
            continue

        idxs = choose_indices(len(pcd_files))
        print(f"  Candidate indices: {idxs}")

        fig, ax = plt.subplots(figsize=(8, 8))

        pose_summary = []
        all_xy = []

        for rank, idx in enumerate(idxs):
            pcd_path = pcd_files[idx]
            pts = read_ascii_pcd(pcd_path)
            xy = pts[:, :2]
            all_xy.append(xy)

            stats = summarise_points(xy)
            label = f"{idx:03d}: {pcd_path.stem}"
            ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.55, label=label)

            print(
                f"  candidate {rank+1}/5 | idx={idx:03d} | file={pcd_path.name} | "
                f"points={stats['n']} | "
                f"x=[{stats['x_min']:.3f},{stats['x_max']:.3f}] | "
                f"y=[{stats['y_min']:.3f},{stats['y_max']:.3f}] | "
                f"range med={stats['range_med']:.3f} m"
            )

            pose_summary.append(
                f"{pose_dir.name},idx={idx:03d},file={pcd_path.name},"
                f"points={stats['n']},"
                f"x_min={stats['x_min']:.3f},x_max={stats['x_max']:.3f},"
                f"y_min={stats['y_min']:.3f},y_max={stats['y_max']:.3f},"
                f"range_med={stats['range_med']:.3f}"
            )

        all_xy_cat = np.vstack(all_xy)
        if args.xlim:
            ax.set_xlim(args.xlim)
        else:
            x_span = np.ptp(all_xy_cat[:, 0])
            pad_x = 0.10 * max(float(x_span), 1e-6)
            ax.set_xlim(
                all_xy_cat[:, 0].min() - pad_x,
                all_xy_cat[:, 0].max() + pad_x,
            )

        if args.ylim:
            ax.set_ylim(args.ylim)
        else:
            y_span = np.ptp(all_xy_cat[:, 1])
            pad_y = 0.10 * max(float(y_span), 1e-6)
            ax.set_ylim(
                all_xy_cat[:, 1].min() - pad_y,
                all_xy_cat[:, 1].max() + pad_y,
            )

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_xlabel("LiDAR x [m]")
        ax.set_ylabel("LiDAR y [m]")
        ax.set_title(f"{pose_dir.name}: early / 25% / middle / 75% / late PCD candidates")
        ax.legend(fontsize=7, loc="best")

        out_path = output_dir / f"{pose_dir.name}_candidate_scans.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved plot: {out_path}")
        summary_lines.extend(pose_summary)

    summary_path = output_dir / "candidate_scan_summary.csv"
    summary_path.write_text(
        "pose,index,file,points,x_min,x_max,y_min,y_max,range_med\n"
        + "\n".join(summary_lines)
        + "\n"
    )

    print("\n" + "=" * 80)
    print("DONE")
    print(f"Saved per-pose plots to: {output_dir.resolve()}")
    print(f"Saved CSV summary to : {summary_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
