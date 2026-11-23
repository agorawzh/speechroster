"""Submit recordings to a local MOSS subtitle server."""

import ipaddress
import math
import time
from urllib.parse import urlsplit

import requests


def transcribe(path, server, max_new_tokens):
    url = urlsplit(server)
    host = url.hostname
    try:
        is_local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_local = host == "localhost"
    if url.scheme != "http" or not is_local or url.username or url.password:
        raise ValueError("Use a local MOSS URL such as http://127.0.0.1:7860.")
    server = server.rstrip("/")

    with requests.Session() as http:
        http.trust_env = False

        def request(method, endpoint, **kwargs):
            response = http.request(method, server + endpoint, timeout=(5, 120),
                                    allow_redirects=False, **kwargs)
            if 300 <= response.status_code < 400:
                raise ValueError("The local MOSS endpoint redirected the request.")
            response.raise_for_status()
            return response.json()

        with path.open("rb") as audio:
            job = request("POST", "/api/jobs", files={"file": (path.name, audio)},
                          data={"max_new_tokens": max_new_tokens})
        endpoint = "/api/jobs/" + job["id"]
        deadline = time.monotonic() + 3600
        while job["status"] not in {"waiting_review", "done"}:
            if job["status"] in {"failed", "cancelled"}:
                raise ValueError(f"MOSS job {job['id']}: {job.get('error') or job['status']}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"MOSS job {job['id']} is still running.")
            time.sleep(1)
            job = request("GET", endpoint)
        rows = request("GET", endpoint + "/segments")["segments"]

    segments = []
    for row in rows:
        start, end = float(row["start"]), float(row["end"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            continue
        segments.append((start, end, row["speaker"] or "Unlabeled", row["text"]))
    return job["id"], bool(job["usage"]["possibly_truncated"]), sorted(segments)
