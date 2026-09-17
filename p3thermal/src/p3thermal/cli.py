"""Repeatable diagnostics, capture, replay, and local-viewer commands."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .device import Camera
from .diagnostics import build_inspection_report
from .inference import RemoteLearnedPreview
from .recording import Recorder, Replay
from .service import serve
from .transport import P3Transport


def main() -> None:
    parser = argparse.ArgumentParser(prog="p3thermal")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser(
        "inspect", help="read P3 identifiers and USB descriptors"
    )
    inspect_parser.add_argument(
        "--out", type=Path, help="write the JSON report to this path"
    )
    record_parser = subparsers.add_parser("record", help="record a bounded P3 session")
    record_parser.add_argument("--duration", type=float, required=True)
    record_parser.add_argument("--out", type=Path, required=True)
    serve_parser = subparsers.add_parser("serve", help="run the local live viewer")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--inference-url")
    replay_parser = subparsers.add_parser("replay", help="view a recorded session")
    replay_parser.add_argument("session", type=Path)
    replay_parser.add_argument("--port", type=int, default=8765)
    replay_parser.add_argument("--inference-url")
    args = parser.parse_args()

    if args.command == "inspect":
        with P3Transport.open() as transport:
            report = build_inspection_report(transport)
        report_json = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.out is None:
            print(report_json, end="")
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(report_json, encoding="utf-8")
            print(args.out)
    elif args.command == "record":
        if args.duration <= 0:
            parser.error("--duration must be positive")
        try:
            with (
                Camera.open() as camera,
                Recorder(args.out, camera.profile) as recorder,
            ):
                deadline: float | None = None
                for frame in camera:
                    recorder.write(frame)
                    if deadline is None:
                        deadline = time.monotonic() + args.duration
                    elif time.monotonic() >= deadline:
                        break
        except FileExistsError:
            parser.error(
                f"recording directory already exists: {args.out}; "
                "choose a new --out path"
            )
        print(args.out)
    elif args.command == "serve":
        preview = _preview_runtime(args)
        serve(Camera.open(), port=args.port, preview=preview)
    elif args.command == "replay":
        preview = _preview_runtime(args)
        serve(Replay(args.session), port=args.port, preview=preview)


def _preview_runtime(
    args: argparse.Namespace,
) -> RemoteLearnedPreview | None:
    if args.inference_url is not None:
        return RemoteLearnedPreview(args.inference_url)
    return None


if __name__ == "__main__":
    main()
