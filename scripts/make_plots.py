"""Generate training curves from run logs"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METRIC_LABELS = {"train_loss": "Training loss (nats)","validation_loss": "Validation loss (nats)","learning_rate": "Learning rate","pre_clip_gradient_norm": "Pre-clipping gradient norm",}

X_AXIS_LABELS = {"step": "Optimizer step","tokens_processed": "Tokens processed","wall_clock_seconds": "Wall-clock time (seconds)",}

def load_jsonl(path: Path):
    # load a training log and validate its step ordering
    if not path.is_file():
        raise FileNotFoundError(path)

    records = []
    previous_step = 0

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(),start=1,):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
            step = int(record["step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid record at {path}:{line_number}") from error

        if step <= previous_step:
            raise ValueError(f"Steps must increase strictly in {path}")

        records.append(record)
        previous_step = step

    if not records:
        raise ValueError(f"Log is empty: {path}")

    return records

def plot_logs(log_paths: list[Path],output_path: Path,metric: str = "validation_loss",x_axis: str = "step",labels: list[str] | None = None,title: str | None = None,):
    # plot one metric from one or more run logs on common axes
    if metric not in METRIC_LABELS:
        raise ValueError(f"Unsupported metric: {metric}")

    if x_axis not in X_AXIS_LABELS:
        raise ValueError(f"Unsupported x-axis: {x_axis}")

    if not log_paths:
        raise ValueError("At least one log is required")

    if labels is not None and len(labels) != len(log_paths):
        raise ValueError("--labels must contain exactly one label per log")

    if output_path.suffix.lower() not in {".png", ".svg"}:
        raise ValueError("Output must use .png or .svg")

    cache_root = Path(tempfile.gettempdir()) / "csc3043s-plot-cache"
    matplotlib_cache = cache_root / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7.2, 4.6))

    for index, log_path in enumerate(log_paths):
        records = load_jsonl(log_path)
        points = [
            (record[x_axis], record[metric])
            for record in records
            if record.get(x_axis) is not None and record.get(metric) is not None
        ]

        if not points:
            raise ValueError(f"Metric {metric!r} is absent from {log_path}")

        x_values, y_values = zip(*points)
        label = labels[index] if labels is not None else log_path.stem
        axes.plot(x_values, y_values, marker="o", markersize=3, label=label)

    axes.set_xlabel(X_AXIS_LABELS[x_axis])
    axes.set_ylabel(METRIC_LABELS[metric])
    axes.set_title(title or METRIC_LABELS[metric])
    axes.set_yscale("log")
    axes.grid(True, alpha=0.25)
    axes.legend()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Plot training metrics directly from Assignment 1 JSONL logs.")
    parser.add_argument("--logs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric",choices=tuple(METRIC_LABELS),default="validation_loss",)
    parser.add_argument("--x-axis",choices=tuple(X_AXIS_LABELS),default="step",)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--title", default=None)
    return parser.parse_args(argv)

def main():
    args = parse_args()
    plot_logs(log_paths=args.logs,output_path=args.output,metric=args.metric,x_axis=args.x_axis,labels=args.labels,title=args.title,)

if __name__ == "__main__":
    main()