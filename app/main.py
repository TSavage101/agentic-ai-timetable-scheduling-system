from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .agents import OrchestratorAgent
from .io_utils import load_problem_data, schedule_to_dataframe


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Agentic AI Timetable Scheduling System")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
orchestrator = OrchestratorAgent()
EXPORT_CACHE: Dict[str, object] = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "result": None,
            "download_id": None,
            "selected_strategy": "hybrid",
            "training_episodes": 0,
            "generations": 12,
        },
    )


@app.post("/generate", response_class=HTMLResponse)
async def generate_schedule(
    request: Request,
    courses: UploadFile | None = File(default=None),
    lecturers: UploadFile | None = File(default=None),
    rooms: UploadFile | None = File(default=None),
    timeslots: UploadFile | None = File(default=None),
    strategy: str = File(default="hybrid"),
    training_episodes: int = File(default=0),
    generations: int = File(default=12),
) -> HTMLResponse:
    problem = load_problem_data(courses, lecturers, rooms, timeslots)
    result = orchestrator.run(
        problem,
        strategy=strategy,
        training_episodes=max(0, training_episodes),
        generations=max(1, generations),
    )
    download_id = str(uuid.uuid4())
    EXPORT_CACHE[download_id] = result["schedule_rows"]
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "result": result,
            "download_id": download_id,
            "selected_strategy": strategy,
            "training_episodes": training_episodes,
            "generations": generations,
        },
    )


@app.get("/download/{download_id}")
async def download_schedule(download_id: str):
    rows = EXPORT_CACHE.get(download_id)
    if rows is None:
        return RedirectResponse("/")
    dataframe = schedule_to_dataframe(rows)
    buffer = io.StringIO()
    dataframe.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=timetable.csv"},
    )
