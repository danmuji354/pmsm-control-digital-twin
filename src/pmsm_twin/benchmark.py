import argparse
import json
from pathlib import Path

from .core import run_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="standard", choices=["standard"])
    parser.add_argument("--output", default="artifacts/benchmark")
    args = parser.parse_args()
    scenarios = {
        "nominal": {},
        "hot-winding": {"resistance_scale": 1.4},
        "flux-drift": {"flux_scale": 0.9},
    }
    metrics = {
        controller: {
            scenario: {
                k: v for k, v in run_episode(controller, **settings).items() if not hasattr(v, "shape")
            }
            for scenario, settings in scenarios.items()
        }
        for controller in ["foc-pi", "predictive"]
    }
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (path / "manifest.json").write_text(
        json.dumps(
            {"suite": args.suite, "controllers": list(metrics), "scenarios": list(scenarios)}, indent=2
        )
        + "\n"
    )
    print(metrics)


if __name__ == "__main__":
    main()
