"""Command-line interface for ZUCE."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .api import ZUCE
from .config import load_config
from .errors import ZUCEError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zuce",
        description="ZUCE — Zero-Update Capability Extraction",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect model compatibility")
    inspect_parser.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    inspect_parser.add_argument("--device", default="auto")
    inspect_parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    inspect_parser.add_argument("--trust-remote-code", action="store_true")

    extract_parser = subparsers.add_parser("extract", help="Run extraction from a YAML config")
    extract_parser.add_argument("--config", required=True, help="Path to zuce.yaml")

    verify_parser = subparsers.add_parser("verify", help="Verify a saved ZUCE artifact")
    verify_parser.add_argument("output_dir", help="Extracted model directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            result = ZUCE.inspect(
                args.model,
                device=args.device,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
            ).to_dict()
        elif args.command == "extract":
            result = ZUCE.extract_config(load_config(args.config)).to_dict()
        else:
            result = ZUCE.verify(args.output_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except ZUCEError as exc:
        print(
            json.dumps(
                {"error": exc.code, "message": str(exc), "details": exc.details},
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(json.dumps({"error": "invalid_request", "message": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

