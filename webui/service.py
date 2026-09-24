#!/usr/bin/env python3
"""Run the single-worker H3 video gateway; see README.md for deployment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from server import serve


ROOT = Path(__file__).resolve().parent.parent


def _path_env(name: str, default: Path) -> Path:
    """Read a path option without making it depend on the launch directory."""
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default=os.getenv('H3_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.getenv('H3_PORT', '8200')))
    parser.add_argument('--data-dir', type=Path,
                        default=_path_env('H3_DATA_DIR', ROOT / 'h3-jobs'))
    parser.add_argument('--pipeline-python',
                        default=os.getenv('H3_PIPELINE_PYTHON', sys.executable))
    parser.add_argument('--pipeline-script', type=Path,
                        default=_path_env('H3_PIPELINE_SCRIPT', ROOT / 'scripts' / 'generate_video.py'))
    parser.add_argument('--queue-size', type=int, default=int(os.getenv('H3_QUEUE_SIZE', '8')))
    parser.add_argument('--public-prefix', default=os.getenv('H3_PUBLIC_PREFIX', '/h3-api'))
    parser.add_argument('--verified', action='store_true', default=os.getenv('H3_PIPELINE_VERIFIED') == '1',
                        help='Enable submissions after a successful real end-to-end generation')
    args = parser.parse_args()
    if args.queue_size < 1:
        parser.error('--queue-size must be positive')
    # All worker paths are absolute by default, so a systemd service may start
    # from any WorkingDirectory.  The worker itself remains single-process.
    args.data_dir = args.data_dir.expanduser().resolve()
    args.pipeline_script = args.pipeline_script.expanduser().resolve()
    serve(args)


if __name__ == '__main__':
    main()
