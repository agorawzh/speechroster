"""Store recordings, transcript segments, and voiceprints in SQLite."""

import json
import sqlite3
from collections import defaultdict


def connect(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS people (
            name TEXT PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS voiceprints (
            person TEXT NOT NULL REFERENCES people(name),
            vector TEXT NOT NULL,
            UNIQUE (person, vector)
        );
        CREATE TABLE IF NOT EXISTS recordings (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            possibly_truncated INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS recording_speakers (
            recording_id TEXT NOT NULL REFERENCES recordings(id),
            label TEXT NOT NULL,
            person TEXT REFERENCES people(name),
            score REAL,
            PRIMARY KEY (recording_id, label)
        );
        CREATE TABLE IF NOT EXISTS segments (
            recording_id TEXT NOT NULL,
            label TEXT NOT NULL,
            start REAL NOT NULL,
            end REAL NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY (recording_id, label) REFERENCES recording_speakers(recording_id, label)
        );
    """)
    return db


def profiles(db):
    result = defaultdict(list)
    for row in db.execute("SELECT person, vector FROM voiceprints"):
        result[row["person"]].append(json.loads(row["vector"]))
    return result


def list_profiles(db):
    rows = db.execute("""
        SELECT person AS name, COUNT(*) AS clips
        FROM voiceprints GROUP BY person ORDER BY person
    """)
    return [dict(row) for row in rows]


def list_recordings(db):
    rows = db.execute("SELECT id, filename FROM recordings ORDER BY rowid DESC")
    return [dict(row) for row in rows]


def enroll(db, name, vectors):
    with db:
        db.execute("INSERT OR IGNORE INTO people VALUES (?)", (name,))
        db.executemany("INSERT OR IGNORE INTO voiceprints VALUES (?, ?)",
                       [(name, json.dumps(vector, allow_nan=False)) for vector in vectors])


def save_recording(db, job_id, filename, truncated, segments, matches):
    with db:
        db.execute("INSERT INTO recordings VALUES (?, ?, ?)", (job_id, filename, truncated))
        db.executemany("INSERT INTO recording_speakers VALUES (?, ?, ?, ?)",
                       [(job_id, label, *matches.get(label, (None, None)))
                        for label in sorted({segment[2] for segment in segments})])
        db.executemany("INSERT INTO segments VALUES (?, ?, ?, ?, ?)",
                       [(job_id, label, start, end, text) for start, end, label, text in segments])


def assign(db, recording_id, label, person):
    with db:
        db.execute("INSERT OR IGNORE INTO people VALUES (?)", (person,))
        result = db.execute("""
            UPDATE recording_speakers SET person = ?, score = NULL
            WHERE recording_id = ? AND label = ?
        """, (person, recording_id, label))
        if result.rowcount != 1:
            raise ValueError(f"Speaker {label} not found in recording {recording_id}.")


def read_recording(db, job_id):
    recording = db.execute("SELECT * FROM recordings WHERE id = ?", (job_id,)).fetchone()
    if recording is None:
        raise ValueError(f"Recording {job_id} not found.")
    rows = db.execute("""
        SELECT segments.start, segments.end, segments.label,
               recording_speakers.person, recording_speakers.score, segments.text
        FROM segments JOIN recording_speakers USING (recording_id, label)
        WHERE segments.recording_id = ?
        ORDER BY segments.start, segments.end, segments.rowid
    """, (job_id,))
    return {"id": recording["id"], "filename": recording["filename"],
            "possibly_truncated": bool(recording["possibly_truncated"]),
            "segments": [dict(row) for row in rows]}
