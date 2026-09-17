import argparse
import json
from pathlib import Path

from .config import Scenario
from .dataset import generate_dataset, validate_dataset
from .scenarios import expanded_suite, quick_suite, starter_suite


def main():
    parser = argparse.ArgumentParser(description="Synthetic physical thermal calibration targets")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "plan"):
        sub = commands.add_parser(name)
        sub.add_argument("--suite", choices=("starter", "quick", "expanded"), default="starter")
        sub.add_argument(
            "--output", required=True, help="New output directory, or JSON file for plan"
        )
        sub.add_argument(
            "--config", help="JSON with a scenarios list; takes precedence over suite flags"
        )
        sub.add_argument("--seed", type=int, default=20260916)
        sub.add_argument("--frames", type=int, default=16)
        sub.add_argument("--supersample", type=int, default=12)
        sub.add_argument("--boundary-samples", type=int, choices=(1, 2, 4, 8), default=4)
        sub.add_argument("--repetitions", type=int, default=3)
        if name == "generate":
            sub.add_argument("--resume", action="store_true", help="Reuse matching complete clips")
    validate = commands.add_parser("validate")
    validate.add_argument("path")
    args = parser.parse_args()
    if args.command == "validate":
        result = validate_dataset(args.path)
        print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
        return
    if args.config:
        value = json.loads(Path(args.config).read_text(encoding="utf-8"))
        scenarios = [Scenario.from_dict(s) for s in value["scenarios"]]
    else:
        kwargs = {
            "seed": args.seed,
            "frames": args.frames,
            "supersample": args.supersample,
            "boundary_samples": args.boundary_samples,
        }
        if args.suite == "expanded":
            if args.repetitions < 1:
                parser.error("repetitions must be positive")
            scenarios = expanded_suite(repetitions=args.repetitions, **kwargs)
        else:
            scenarios = (starter_suite if args.suite == "starter" else quick_suite)(**kwargs)
    if args.command == "plan":
        path = Path(args.output)
        if path.exists():
            raise FileExistsError("Plan output already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"scenarios": [s.to_dict() for s in scenarios]}, indent=2))
        print(f"Wrote {len(scenarios)} scenarios to {path}")
    else:
        result = generate_dataset(scenarios, args.output, resume=args.resume)
        print(
            json.dumps(
                {
                    "clips": len(result["clips"]),
                    "frames": result["total_frames"],
                    "seconds": round(result["generation_seconds"], 2),
                    "preview": str(Path(args.output).absolute() / "preview.html"),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
