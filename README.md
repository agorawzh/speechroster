# SpeechRoster

Offline transcription, speaker diarization, and cross-recording voiceprint matching on Apple Silicon.

SpeechRoster turns one audio recording into a timestamped transcript via [MOSS](https://github.com/vanch007/mlx-MOSS-Transcribe-Diarize). It takes short clips from each detected speaker and compares their [ReDimNet2+](https://huggingface.co/lab260/redimnet2-plus) voiceprints against the people you enrolled. Uncertain and unenrolled speakers retain their recording-local labels, such as `S02`.

## Setup on Apple Silicon

With Python 3.12 and [uv](https://docs.astral.sh/uv/) installed, run these commands from the repository root. The setup downloads dependencies and weights once. Transcription, enrollment, and matching run locally after that.

```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[web]' 'moss-transcribe-diarize[mlx-runtime] @ git+https://github.com/vanch007/mlx-MOSS-Transcribe-Diarize.git@d2546316f76e93947e8d99a17fb88652f8ad6ab2'

hf download vanch007/mlx-MOSS-Transcribe-Diarize-8bit \
  --revision 7210aef739be6b9e068fba9b1d60369ec053655b --local-dir .models/moss
mkdir -p .models/redimnet2-plus
curl -fsSL https://codeload.github.com/lab260ru/redimnet2-plus/tar.gz/2a8d6dbd6d85e337a20b1d74ed3323c44bff86c5 \
  -o .models/redimnet2-plus.tar.gz
tar -xzf .models/redimnet2-plus.tar.gz --strip-components=1 -C .models/redimnet2-plus

HF_HUB_OFFLINE=1 mtd-subtitle-web --backend mlx --model .models/moss \
  --host 127.0.0.1 --port 7860 --runs-dir .speechroster/moss-runs
```

You can reuse a running local MOSS subtitle server. To reuse an existing Lab260 installation, set `SPEECHROSTER_REDIMNET_DIR` or pass `--redimnet-dir` before the subcommand. The model directory should contain the upstream `asv/` and `weights/` directories, either directly or inside `redimnet2-plus-main/`.

## Local web interface

With MOSS running, start SpeechRoster in a second terminal from the repository root:

```sh
source .venv/bin/activate
speechroster-web
```

Open [SpeechRoster](http://127.0.0.1:7861). For each speaker, enter a name and select one or more clean clips of that person. Use **添加说话人** to include additional speakers in the same submission. Then upload a meeting recording to transcribe and match its speakers. The page displays the transcript and offers SRT export. It uses the same SQLite database as the CLI when launched from the same directory.

## Reproducible demo

With the server running, open a second terminal in the repository root, activate the environment, and run:

```sh
source .venv/bin/activate
python demo.py
```

The demo uses macOS `say` to create a voice enrollment clip and a separate two-speaker recording. It checks that the enrolled speaker appears as **Alice** and a different speaker keeps their anonymous label. It writes its synthetic audio, SQLite database, JSON, and SRT to a fresh directory under `.speechroster/`. If a voice is unavailable on your Mac, use `say -v '?'` to list installed voices and pass `--first-voice` and `--second-voice`.

```text
00:00:00,070 --> 00:00:06,520
Alice: Welcome to the project meeting. I finished the transcript review, and the new example is ready for everyone to try.

00:00:07,020 --> 00:00:12,760
S02: Thanks for the update. I will check the test results and bring the remaining questions to our next meeting.
```

## Your own recordings

```sh
speechroster enroll "Alice" alice-1.m4a alice-2.wav
speechroster transcribe meeting.m4a
speechroster assign RECORDING_ID S02 "Bea"
speechroster export RECORDING_ID --format srt > meeting.srt
```

Enrollment clips should contain one speaker and be 2–60 seconds long. Give each person a few clean clips if possible. Match thresholds are starting values for experimentation with your own recordings. The import result includes its recording ID, each segment's speaker label, and any assigned name. A manual assignment applies to that recording without enrolling a new voiceprint.

The JSON records whether MOSS hit its output token limit. Use `transcribe --max-new-tokens` for longer recordings. SpeechRoster saves its local database under `.speechroster/` by default. The local MOSS service also keeps a copy of the submitted recording in its configured runs directory.

## Tests

```sh
python -m unittest discover -s tests -v
```
