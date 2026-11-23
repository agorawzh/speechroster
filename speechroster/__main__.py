"""Command-line interface for SpeechRoster."""

import argparse
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import requests

from . import store, workflow


def main():
    parser = argparse.ArgumentParser(prog="speechroster", description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(".speechroster/roster.sqlite3"))
    parser.add_argument("--redimnet-dir", type=Path, default=Path(
        os.environ.get("SPEECHROSTER_REDIMNET_DIR", ".models/redimnet2-plus")))
    commands = parser.add_subparsers(dest="command", required=True)

    enrollment = commands.add_parser("enroll", help="Register clips of one known speaker")
    enrollment.add_argument("name")
    enrollment.add_argument("audio", type=Path, nargs="+")

    ingest = commands.add_parser("transcribe", help="Transcribe a recording and match registered speakers")
    ingest.add_argument("audio", type=Path)
    ingest.add_argument("--moss-url", default=workflow.DEFAULT_MOSS_URL)
    ingest.add_argument("--max-new-tokens", type=int, default=workflow.DEFAULT_MAX_NEW_TOKENS)
    ingest.add_argument("--threshold", type=float, default=workflow.DEFAULT_THRESHOLD)
    ingest.add_argument("--margin", type=float, default=workflow.DEFAULT_MARGIN)

    label = commands.add_parser("assign", help="Correct the name attached to a recording's speaker label")
    label.add_argument("recording")
    label.add_argument("label")
    label.add_argument("name")

    export = commands.add_parser("export", help="Export saved transcripts")
    export.add_argument("recording")
    export.add_argument("--format", choices=("json", "srt"), default="json")

    args = parser.parse_args()
    if hasattr(args, "name"):
        try:
            args.name = workflow.speaker_name(args.name)
        except ValueError as error:
            parser.error(str(error))
    if args.command == "transcribe" and not (
        args.max_new_tokens > 0 and 0 <= args.threshold <= 1 and 0 <= args.margin <= 2
    ):
        parser.error("Use a positive output budget, a threshold in [0, 1], and a margin in [0, 2].")

    try:
        with closing(store.connect(args.db)) as db:
            if args.command == "enroll":
                result = workflow.enroll(db, args.name, args.audio, args.redimnet_dir)
            elif args.command == "transcribe":
                result = workflow.transcribe(db, args.audio, args.redimnet_dir, args.moss_url,
                                             args.max_new_tokens, args.threshold, args.margin)
            elif args.command == "assign":
                store.assign(db, args.recording, args.label, args.name)
                result = store.read_recording(db, args.recording)
            else:
                result = store.read_recording(db, args.recording)
                if args.format == "srt":
                    if result["possibly_truncated"]:
                        print("Warning: the MOSS transcript may be incomplete.", file=sys.stderr)
                    print(workflow.export_srt(result))
                    return
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    except (OSError, ValueError, sqlite3.Error, requests.RequestException) as error:
        parser.exit(1, f"speechroster: {error}\n")


if __name__ == "__main__":
    main()
