"""Run the SpeechRoster web interface on this computer."""

import argparse
import os
import shutil
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Annotated

import requests
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import store, workflow


@contextmanager
def staged_uploads(uploads):
    with tempfile.TemporaryDirectory(prefix="speechroster-upload-") as tmp:
        paths = []
        for index, upload in enumerate(uploads):
            directory = Path(tmp) / str(index)
            directory.mkdir()
            name = Path(upload.filename or "").name
            path = directory / (name if name not in {"", ".", ".."} else "recording")
            with path.open("wb") as file:
                shutil.copyfileobj(upload.file, file)
            paths.append(path)
        yield paths


def require_local_form(x_speechroster: Annotated[str | None, Header()] = None):
    # A non-simple header prevents writes from cross-origin HTML forms.
    if x_speechroster != "1":
        raise HTTPException(403, "Send uploads from the SpeechRoster page.")


def create_app(db_path, model_dir, moss_url=workflow.DEFAULT_MOSS_URL):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

    @app.exception_handler(ValueError)
    @app.exception_handler(OSError)
    async def invalid_input(request: Request, error: Exception):
        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.exception_handler(requests.RequestException)
    async def moss_error(request: Request, error: Exception):
        return JSONResponse({"detail": "MOSS 转写服务请求失败。请确认 MOSS 网页正在运行。"}, status_code=502)

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).with_name("web.html"), headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state():
        with closing(store.connect(db_path)) as db:
            return {"profiles": store.list_profiles(db), "recordings": store.list_recordings(db)}

    @app.post("/api/enroll", dependencies=[Depends(require_local_form)])
    def enroll(name: Annotated[str, Form()], audio: Annotated[list[UploadFile], File()]):
        with staged_uploads(audio) as paths, closing(store.connect(db_path)) as db:
            return workflow.enroll(db, workflow.speaker_name(name), paths, model_dir)

    @app.post("/api/transcribe", dependencies=[Depends(require_local_form)])
    def transcribe(audio: Annotated[UploadFile, File()]):
        with staged_uploads([audio]) as paths, closing(store.connect(db_path)) as db:
            return workflow.transcribe(db, paths[0], model_dir, moss_url)

    def load_recording(recording_id):
        with closing(store.connect(db_path)) as db:
            try:
                return store.read_recording(db, recording_id)
            except ValueError as error:
                raise HTTPException(404, str(error)) from error

    @app.get("/api/recordings/{recording_id}")
    def recording(recording_id: str):
        return load_recording(recording_id)

    @app.get("/api/recordings/{recording_id}/srt")
    def subtitle(recording_id: str):
        return PlainTextResponse(workflow.export_srt(load_recording(recording_id)),
                                 headers={"Content-Disposition": 'attachment; filename="speechroster.srt"'})

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(".speechroster/roster.sqlite3"))
    parser.add_argument("--redimnet-dir", type=Path, default=Path(
        os.environ.get("SPEECHROSTER_REDIMNET_DIR", ".models/redimnet2-plus")))
    parser.add_argument("--moss-url", default=workflow.DEFAULT_MOSS_URL)
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    uvicorn.run(create_app(args.db, args.redimnet_dir, args.moss_url), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
