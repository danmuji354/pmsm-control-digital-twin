"""Generate reproducible visual assets for the PMSM digital-twin showcase."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .core import PMSMParams, run_episode

INK = "#0f172a"
BLUE = "#2563eb"
ORANGE = "#f59e0b"
SLATE = "#64748b"
GRID = "#cbd5e1"
PAPER = "#f8fafc"
CONTROLLERS = ["foc-pi", "predictive"]
SCENARIOS = {
    "nominal": {},
    "hot winding": {"resistance_scale": 1.4},
    "flux drift": {"flux_scale": 0.9},
}


def _style(axis: plt.Axes) -> None:
    axis.set_facecolor(PAPER)
    axis.grid(color=GRID, linewidth=0.8, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=SLATE)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=125, facecolor=PAPER)
    plt.close(figure)


def benchmark_results() -> dict[str, dict[str, dict]]:
    return {
        controller: {
            scenario: run_episode(controller, **settings) for scenario, settings in SCENARIOS.items()
        }
        for controller in CONTROLLERS
    }


def _hero(results: dict[str, dict[str, dict]], output: Path) -> None:
    foc = results["foc-pi"]["nominal"]
    predictive = results["predictive"]["nominal"]
    figure, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), sharex=True)
    figure.patch.set_facecolor(PAPER)
    figure.suptitle("Voltage-constrained PMSM speed control", color=INK, fontsize=20)
    figure.text(
        0.5,
        0.925,
        "Matched outer speed loop; load torque steps to 1.8 N·m at 0.5 s.",
        ha="center",
        color=SLATE,
        fontsize=11,
    )
    axes[0].plot(
        predictive["time"],
        predictive["speed_reference"],
        color=INK,
        linestyle="--",
        linewidth=1.7,
        label="reference",
    )
    axes[0].plot(foc["time"], foc["states"][:, 2], color=ORANGE, linewidth=2, label="FOC PI")
    axes[0].plot(
        predictive["time"],
        predictive["states"][:, 2],
        color=BLUE,
        linewidth=2,
        linestyle="-.",
        label="predictive",
    )
    axes[0].axvline(0.5, color=SLATE, linestyle=":", linewidth=1.5, label="load step")
    axes[0].set_ylabel("mechanical speed [rad/s]")
    axes[0].legend(frameon=False, ncol=4, loc="lower right")
    axes[1].plot(foc["time"], foc["states"][:, 1], color=ORANGE, linewidth=1.7, label="FOC PI iₛ")
    axes[1].plot(
        predictive["time"],
        predictive["states"][:, 1],
        color=BLUE,
        linewidth=1.7,
        linestyle="-.",
        label="predictive iₛ",
    )
    axes[1].axhline(20.0, color=SLATE, linestyle="--", linewidth=1.2)
    axes[1].axhline(-20.0, color=SLATE, linestyle="--", linewidth=1.2)
    axes[1].set(xlabel="time [s]", ylabel="q-axis current [A]")
    axes[1].legend(frameon=False, loc="lower right")
    for axis in axes:
        _style(axis)
    figure.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.11, hspace=0.24)
    _save(figure, output / "hero.png")


def _benchmark(results: dict[str, dict[str, dict]], output: Path) -> None:
    scenarios = list(SCENARIOS)
    x = np.arange(len(scenarios))
    width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 7.2))
    figure.patch.set_facecolor(PAPER)
    figure.suptitle("Transient and efficiency comparison", color=INK, fontsize=20)
    figure.text(
        0.5,
        0.925,
        "Same references and constraints; parameter drift is applied only to the digital twin.",
        ha="center",
        color=SLATE,
        fontsize=11,
    )
    for offset, controller, color, hatch in [
        (-width / 2, "foc-pi", ORANGE, ""),
        (width / 2, "predictive", BLUE, "//"),
    ]:
        rmse = [results[controller][scenario]["speed_rmse_percent_rated"] for scenario in scenarios]
        loss = [results[controller][scenario]["copper_loss_j"] for scenario in scenarios]
        axes[0].bar(
            x + offset,
            rmse,
            width,
            color=color,
            edgecolor=INK,
            linewidth=0.6,
            hatch=hatch,
            label=controller,
        )
        axes[1].bar(
            x + offset,
            loss,
            width,
            color=color,
            edgecolor=INK,
            linewidth=0.6,
            hatch=hatch,
            label=controller,
        )
    axes[0].axhline(5.0, color=INK, linestyle="--", linewidth=1.7, label="5% target")
    axes[0].set(xticks=x, xticklabels=scenarios, ylabel="speed RMSE [% rated]")
    axes[1].set(xticks=x, xticklabels=scenarios, ylabel="copper loss [J]")
    for axis in axes:
        _style(axis)
        axis.tick_params(axis="x", rotation=18)
        axis.legend(frameon=False)
    figure.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.17, wspace=0.24)
    _save(figure, output / "benchmark.png")


def _animation(result: dict, output: Path) -> None:
    indices = np.linspace(0, len(result["time"]) - 1, 90, dtype=int)
    current_limit = PMSMParams().current_limit_a
    figure, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    figure.patch.set_facecolor(PAPER)
    angle = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(
        current_limit * np.cos(angle), current_limit * np.sin(angle), color=SLATE, linestyle="--"
    )
    axes[0].set(xlim=(-22, 22), ylim=(-22, 22), xlabel="i_d [A]", ylabel="i_q [A]", aspect="equal")
    (vector,) = axes[0].plot([], [], color=BLUE, linewidth=3, marker="o")
    (trail,) = axes[0].plot([], [], color=BLUE, alpha=0.20, linewidth=1.2)
    axes[1].plot(result["time"], result["speed_reference"], color=INK, linestyle="--", linewidth=1.4)
    axes[1].plot(result["time"], result["states"][:, 2], color=BLUE, linewidth=1.8)
    (marker,) = axes[1].plot([], [], marker="o", color=ORANGE, markersize=7)
    axes[1].set(xlabel="time [s]", ylabel="speed [rad/s]")
    status = axes[1].text(0.05, 0.92, "", transform=axes[1].transAxes, color=INK, fontsize=10)
    for axis in axes:
        _style(axis)

    def update(frame: int):
        index = indices[frame]
        current = result["states"][index, :2]
        vector.set_data([0, current[0]], [0, current[1]])
        trail.set_data(result["states"][: index + 1, 0], result["states"][: index + 1, 1])
        marker.set_data([result["time"][index]], [result["states"][index, 2]])
        status.set_text(f"t = {result['time'][index]:.2f} s   |i| = {np.linalg.norm(current):.1f} A")
        return vector, trail, marker, status

    animation = FuncAnimation(figure, update, frames=len(indices), interval=65, blit=True)
    animation.save(output / "demo.gif", writer=PillowWriter(fps=15), dpi=90)
    plt.close(figure)


def _architecture(output: Path) -> None:
    nodes = [
        (35, "Speed reference", "outer-loop target"),
        (275, "Speed PI", "i_q reference"),
        (515, "Current control", "FOC PI / predictive"),
        (755, "Voltage disk", "inverter constraint"),
        (995, "PMSM dq twin", "electrical + mechanical"),
    ]
    elements = []
    for index, (x, title, subtitle) in enumerate(nodes):
        elements.append(
            f'<rect x="{x}" y="84" width="190" height="88" rx="14" fill="white" stroke="{BLUE if index in {2, 3} else GRID}" stroke-width="2"/>'
        )
        elements.append(
            f'<text x="{x + 95}" y="119" text-anchor="middle" fill="{INK}" font-family="Arial" font-size="15">{html.escape(title)}</text>'
        )
        elements.append(
            f'<text x="{x + 95}" y="145" text-anchor="middle" fill="{SLATE}" font-family="Arial" font-size="12">{html.escape(subtitle)}</text>'
        )
        if index < len(nodes) - 1:
            elements.append(
                f'<line x1="{x + 190}" y1="128" x2="{nodes[index + 1][0] - 12}" y2="128" stroke="{INK}" stroke-width="2" marker-end="url(#arrow)"/>'
            )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1230" height="270" viewBox="0 0 1230 270"><rect width="1230" height="270" fill="{PAPER}"/><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="{INK}"/></marker></defs><text x="35" y="42" fill="{INK}" font-family="Arial" font-size="22" font-weight="700">Cascaded PMSM drive control</text>{"".join(elements)}<path d="M1090 184 C1090 234, 365 234, 365 184" fill="none" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="7 5" marker-end="url(#arrow)"/><text x="730" y="254" text-anchor="middle" fill="{SLATE}" font-family="Arial" font-size="13">current and speed feedback</text></svg>'''
    (output / "architecture.svg").write_text(svg)


def gallery_contract(results: dict[str, dict[str, dict]]) -> dict:
    foc = results["foc-pi"]["nominal"]
    predictive = results["predictive"]["nominal"]
    return {
        "schema_version": 1,
        "repository": "pmsm-control-digital-twin",
        "title": "PMSM Control Digital Twin",
        "tagline": "FOC PI and predictive current control under voltage and parameter limits.",
        "accent": BLUE,
        "highlights": [
            {
                "label": "predictive speed RMSE",
                "value": f"{predictive['speed_rmse_percent_rated']:.2f}%",
            },
            {"label": "peak current", "value": f"{predictive['peak_current_a']:.2f} A"},
            {
                "label": "copper loss delta",
                "value": f"{predictive['copper_loss_j'] - foc['copper_loss_j']:+.3f} J",
            },
        ],
        "assets": [
            {
                "path": "hero.png",
                "role": "hero",
                "width": 1600,
                "height": 900,
                "alt": "PMSM speed and q-axis current response for two controllers.",
            },
            {
                "path": "benchmark.png",
                "role": "analysis",
                "width": 1600,
                "height": 900,
                "alt": "Speed RMSE and copper loss across motor parameter drift.",
            },
            {
                "path": "demo.gif",
                "role": "animation",
                "width": 720,
                "height": 405,
                "alt": "Animated dq current vector and motor speed transient.",
            },
            {
                "path": "architecture.svg",
                "role": "diagram",
                "width": 1230,
                "height": 270,
                "alt": "Cascaded speed, current, inverter, and PMSM digital-twin loop.",
            },
        ],
        "reproduce": "python -m pmsm_twin.gallery --output artifacts/gallery",
    }


def generate_gallery(output: str | Path, animation: bool = True) -> dict:
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    results = benchmark_results()
    _hero(results, path)
    _benchmark(results, path)
    _architecture(path)
    if animation:
        _animation(results["predictive"]["nominal"], path)
    rows = [
        {
            "controller": controller,
            "scenario": scenario,
            "speed_rmse_percent_rated": result["speed_rmse_percent_rated"],
            "peak_current_a": result["peak_current_a"],
            "copper_loss_j": result["copper_loss_j"],
            "voltage_limit_violations": result["voltage_limit_violations"],
        }
        for controller, scenarios in results.items()
        for scenario, result in scenarios.items()
    ]
    with (path / "benchmark_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    contract = gallery_contract(results)
    (path / "showcase.json").write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/gallery")
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate_gallery(args.output, not args.no_animation)["highlights"], indent=2))


if __name__ == "__main__":
    main()
