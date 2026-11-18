"""Запуск панели: python -m dashboard  ->  http://127.0.0.1:8787

Сервер слушает только этот компьютер (127.0.0.1) и отдаёт данные только на чтение.
"""

import argparse
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from .data import MarketCache, build_state

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Панель робота", docs_url=None, redoc_url=None, openapi_url=None)
_cache = MarketCache()
_lock = threading.Lock()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/state")
def state() -> JSONResponse:
    with _lock:
        payload = build_state(cache=_cache)
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


def main() -> None:
    parser = argparse.ArgumentParser(prog="dashboard", description="Панель наблюдения за роботом")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
