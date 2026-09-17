"""Loopback-only TorchScript inference sidecar for the P3 viewer."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="serve a P3 TorchScript upscaler")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    metadata_path = args.artifact.with_suffix(args.artifact.suffix + ".json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read artifact metadata: {metadata_path}: {exc}")
    if metadata != {
        "format": "p3thermal-upscaler",
        "version": 1,
        "scales": [2, 3, 4],
    }:
        parser.error("unsupported p3thermal upscaler metadata")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(args.artifact, map_location=device).eval()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _handler(model, device))
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _handler(model: torch.jit.ScriptModule, device: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            query = parse_qs(urlsplit(self.path).query)
            try:
                scale = int(query["scale"][0])
                width = int(self.headers["X-P3-Width"])
                height = int(self.headers["X-P3-Height"])
                size = int(self.headers["Content-Length"])
                raw = np.frombuffer(self.rfile.read(size), dtype="<u2").reshape(height, width)
                with torch.inference_mode():
                    tensor = torch.from_numpy(raw.astype(np.float32, copy=True))[None, None]
                    output = model(tensor.to(device), scale).squeeze().cpu().numpy()
                values = np.rint(np.clip(output, 0, 65535)).astype("<u2")
            except (KeyError, ValueError) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(values.nbytes))
            self.send_header("X-P3-Width", str(values.shape[1]))
            self.send_header("X-P3-Height", str(values.shape[0]))
            self.end_headers()
            self.wfile.write(values.tobytes())

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    return Handler


if __name__ == "__main__":
    main()
