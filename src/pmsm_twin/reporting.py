import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def write_result(result, output, manifest):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {"time", "states", "voltages", "loads", "speed_reference"}
    metrics = {k: v for k, v in result.items() if k not in arrays}
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (path / "timeseries.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_s",
                "id_a",
                "iq_a",
                "speed_radps",
                "electrical_angle_rad",
                "vd_v",
                "vq_v",
                "load_nm",
                "speed_ref_radps",
            ]
        )
        writer.writerows(
            np.column_stack(
                (
                    result["time"],
                    result["states"],
                    result["voltages"],
                    result["loads"],
                    result["speed_reference"],
                )
            )
        )
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    axes[0].plot(result["time"], result["states"][:, 2], label="speed")
    axes[0].plot(result["time"], result["speed_reference"], "--", label="reference")
    axes[0].legend()
    axes[0].set_ylabel("rad/s")
    axes[1].plot(result["time"], result["states"][:, :2])
    axes[1].set_ylabel("current [A]")
    axes[2].plot(result["time"], result["voltages"])
    axes[2].set_ylabel("voltage [V]")
    axes[2].set_xlabel("time [s]")
    [ax.grid() for ax in axes]
    fig.tight_layout()
    fig.savefig(path / "response.png", dpi=150)
    plt.close(fig)
