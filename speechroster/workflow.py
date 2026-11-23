"""Enrollment, transcription, and subtitle output shared by the CLI and web app."""

from . import moss, store, voiceprint


DEFAULT_MOSS_URL = "http://127.0.0.1:7860"
DEFAULT_MAX_NEW_TOKENS = 16384
DEFAULT_THRESHOLD = 0.75
DEFAULT_MARGIN = 0.08


def speaker_name(value):
    name = value.strip()
    if not name or "\n" in name or "\r" in name:
        raise ValueError("Use a nonempty, single-line speaker name.")
    return name


def enroll(db, name, paths, model_dir):
    return enroll_many(db, [(name, paths)], model_dir)[0]


def enroll_many(db, groups, model_dir):
    groups = [(speaker_name(name), paths) for name, paths in groups]
    encoder = voiceprint.VoiceEncoder(model_dir)
    entries = []
    for name, paths in groups:
        try:
            vectors = [encoder.embed(voiceprint.decode_audio(path)) for path in paths]
        except (OSError, ValueError) as error:
            raise ValueError(f"{name} · {error}") from error
        entries.append((name, vectors))
    store.enroll_many(db, entries)
    return [{"person": name, "clips_processed": len(vectors)} for name, vectors in entries]


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


def transcribe(db, audio, model_dir, moss_url=DEFAULT_MOSS_URL,
               max_new_tokens=DEFAULT_MAX_NEW_TOKENS, threshold=DEFAULT_THRESHOLD, margin=DEFAULT_MARGIN):
    job_id, truncated, segments = moss.transcribe(audio, moss_url, max_new_tokens)
    matches = match_speakers(audio, segments, store.profiles(db), model_dir, threshold, margin)
    store.save_recording(db, job_id, audio.name, truncated, segments, matches)
    return store.read_recording(db, job_id)


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
