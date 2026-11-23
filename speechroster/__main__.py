"""Command-line interface for SpeechRoster."""

import argparse
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import requests

from . import moss, store, voiceprint


def match_speakers(audio, segments, profiles, model_dir, threshold, margin):
    windows = voiceprint.voice_windows(segments)
    if not profiles or not windows:
        return {}
    samples = voiceprint.decode_audio(audio)
    rate = voiceprint.SAMPLE_RATE
    encoder = voiceprint.VoiceEncoder(model_dir)
    matches = {}
    for label, spans in windows.items():
        vectors = []
        for start, end in spans:
            span = min(end - start, 6)
            center = (start + end) / 2
            clip = samples[round((center - span / 2) * rate):round((center + span / 2) * rate)]
            if len(clip) >= 2 * rate:
                vectors.append(encoder.embed(clip))
        if vectors:
            matches[label] = voiceprint.identify(vectors, profiles, threshold, margin)
    return matches


def timestamp(seconds):
    total = round(seconds * 1000)
    hours, total = divmod(total, 3_600_000)
    minutes, total = divmod(total, 60_000)
    seconds, millis = divmod(total, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def export_srt(recording):
    cues = []
    for index, row in enumerate(recording["segments"], 1):
        label = row["person"] or row["label"]
        timing = f'{timestamp(row["start"])} --> {timestamp(row["end"])}'
        cues.append(f'{index}\n{timing}\n{label}: {row["text"]}')
    return "\n\n".join(cues)


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
    ingest.add_argument("--moss-url", default="http://127.0.0.1:7860")
    ingest.add_argument("--max-new-tokens", type=int, default=16384)
    ingest.add_argument("--threshold", type=float, default=0.75)
    ingest.add_argument("--margin", type=float, default=0.08)

    label = commands.add_parser("assign", help="Correct the name attached to a recording's speaker label")
    label.add_argument("recording")
    label.add_argument("label")
    label.add_argument("name")

    export = commands.add_parser("export", help="Export saved transcripts")
    export.add_argument("recording")
    export.add_argument("--format", choices=("json", "srt"), default="json")

    args = parser.parse_args()
    if hasattr(args, "name"):
        args.name = args.name.strip()
        if not args.name or "\n" in args.name or "\r" in args.name:
            parser.error("Use a nonempty, single-line speaker name.")
    if args.command == "transcribe" and not (
        args.max_new_tokens > 0 and 0 <= args.threshold <= 1 and 0 <= args.margin <= 2
    ):
        parser.error("Use a positive output budget, a threshold in [0, 1], and a margin in [0, 2].")

    try:
        with closing(store.connect(args.db)) as db:
            if args.command == "enroll":
                encoder = voiceprint.VoiceEncoder(args.redimnet_dir)
                vectors = [encoder.embed(voiceprint.decode_audio(path)) for path in args.audio]
                store.enroll(db, args.name, vectors)
                result = {"person": args.name, "clips_processed": len(vectors)}
            elif args.command == "transcribe":
                job_id, truncated, segments = moss.transcribe(args.audio, args.moss_url, args.max_new_tokens)
                matches = match_speakers(args.audio, segments, store.profiles(db),
                                         args.redimnet_dir, args.threshold, args.margin)
                store.save_recording(db, job_id, args.audio.name, truncated, segments, matches)
                result = store.read_recording(db, job_id)
            elif args.command == "assign":
                store.assign(db, args.recording, args.label, args.name)
                result = store.read_recording(db, args.recording)
            else:
                result = store.read_recording(db, args.recording)
                if args.format == "srt":
                    if result["possibly_truncated"]:
                        print("Warning: the MOSS transcript may be incomplete.", file=sys.stderr)
                    print(export_srt(result))
                    return
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    except (OSError, ValueError, sqlite3.Error, requests.RequestException) as error:
        parser.exit(1, f"speechroster: {error}\n")


if __name__ == "__main__":
    main()
