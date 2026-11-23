"""Generate synthetic enrollment and meeting audio, then run the offline demo."""

import argparse
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from speechroster.voiceprint import SAMPLE_RATE, decode_audio


def synthesize(root, voice, name, text):
    aiff = root / (name + ".aiff")
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    return decode_audio(aiff)


def write_wave(path, samples):
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redimnet-dir", type=Path)
    parser.add_argument("--moss-url", default="http://127.0.0.1:7860")
    parser.add_argument("--first-voice", default="Samantha")
    parser.add_argument("--second-voice", default="Daniel")
    args = parser.parse_args()

    parent = Path(".speechroster").resolve()
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="demo-", dir=parent))
    enrollment = root / "enrollment.wav"
    meeting = root / "meeting.wav"
    write_wave(enrollment, synthesize(
        root, args.first_voice, "enrollment",
        "This is my enrollment recording. I am leaving this message so the speaker roster can recognize my voice next time.",
    ))
    first = synthesize(
        root, args.first_voice, "first",
        "Welcome to the project meeting. I finished the transcript review, and the new example is ready for everyone to try.",
    )
    second = synthesize(
        root, args.second_voice, "second",
        "Thanks for the update. I will check the test results and bring the remaining questions to our next meeting.",
    )
    write_wave(meeting, np.concatenate((first, np.zeros(SAMPLE_RATE // 2), second)))

    tool = [sys.executable, "-m", "speechroster", "--db", str(root / "roster.sqlite3")]
    if args.redimnet_dir:
        tool += ["--redimnet-dir", str(args.redimnet_dir)]
    subprocess.run([*tool, "enroll", "Alice", str(enrollment)], check=True, stdout=subprocess.DEVNULL)
    result = json.loads(subprocess.check_output(
        [*tool, "transcribe", str(meeting), "--moss-url", args.moss_url], text=True,
    ))
    (root / "transcript.json").write_text(json.dumps(result, indent=2) + "\n")
    srt = subprocess.check_output([*tool, "export", result["id"], "--format", "srt"], text=True)
    (root / "transcript.srt").write_text(srt)
    speakers = {row["label"]: row["person"] for row in result["segments"]}
    print(f"Demo output: {root}")
    print(srt)
    if result["possibly_truncated"] or len(speakers) != 2 or set(speakers.values()) != {"Alice", None}:
        raise SystemExit(f"Expected Alice and an unknown speaker in a complete transcript: {speakers}")


if __name__ == "__main__":
    main()
