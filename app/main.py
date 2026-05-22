from __future__ import annotations

import io
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .agents import OrchestratorAgent
from .io_utils import clone_problem_data, load_problem_data, schedule_to_dataframe
from .models import Assignment, ConstraintConfig, FeedbackEntry, Lecturer, LecturerPreference, ProblemData, SoftConstraintWeights


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Agentic AI Timetable Scheduling System")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
orchestrator = OrchestratorAgent()
EXPORT_CACHE: Dict[str, object] = {}
RUN_CACHE: Dict[str, Dict[str, Any]] = {}


def _base_context(request: Request) -> Dict[str, Any]:
    return {
        "request": request,
        "result": None,
        "download_id": None,
        "run_id": None,
        "selected_strategy": "hybrid",
        "training_episodes": 10,
        "generations": 8,
        "constraint_defaults": ConstraintConfig(),
        "weight_defaults": SoftConstraintWeights(),
        "lecturer_options": [],
        "slot_options": [],
        "room_options": [],
        "message": "",
    }


def _constraint_config_from_form(
    lecturer_double_booking: bool,
    room_exclusivity: bool,
    room_capacity_enforced: bool,
    room_type_enforced: bool,
    fatigue_limit_enabled: bool,
    max_consecutive_sessions: int,
) -> ConstraintConfig:
    return ConstraintConfig(
        lecturer_double_booking=lecturer_double_booking,
        room_exclusivity=room_exclusivity,
        room_capacity_enforced=room_capacity_enforced,
        room_type_enforced=room_type_enforced,
        fatigue_limit_enabled=fatigue_limit_enabled,
        max_consecutive_sessions=max(1, max_consecutive_sessions),
    )


def _weights_from_form(
    preferred_slot_weight: float,
    idle_time_weight: float,
    geographic_weight: float,
    fatigue_weight: float,
    room_fit_weight: float,
) -> SoftConstraintWeights:
    return SoftConstraintWeights(
        preferred_slot=preferred_slot_weight,
        minimize_idle_time=idle_time_weight,
        geographic_grouping=geographic_weight,
        fatigue_balance=fatigue_weight,
        room_fit=room_fit_weight,
    )


def _serialize_preferences(lecturer_id: str, lecturer_request: str, preferred_slots: str, preferred_days: str) -> List[Dict[str, str]]:
    if not lecturer_id and not lecturer_request and not preferred_slots and not preferred_days:
        return []
    pref = LecturerPreference(
        lecturer_id=lecturer_id,
        request_text=lecturer_request,
        preferred_slots=[item.strip() for item in preferred_slots.split("|") if item.strip()],
        preferred_days=[item.strip() for item in preferred_days.split("|") if item.strip()],
    )
    return [asdict(pref)]


def _apply_natural_language(problem: ProblemData, prompt: str) -> ProblemData:
    if not prompt.strip():
        return problem
    updated = clone_problem_data(problem)
    lower_prompt = prompt.lower()
    for lecturer_id, lecturer in list(updated.lecturers.items()):
        lower_name = lecturer.name.lower()
        surname = lower_name.split()[-1]
        if surname in lower_prompt or lower_name in lower_prompt:
            allowed_days = []
            for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
                if day in lower_prompt:
                    allowed_days.append(day.title())
            morning_only = "morning" in lower_prompt
            if allowed_days:
                blocked = set(lecturer.unavailable_slots)
                for slot in updated.slots.values():
                    day_match = slot.day in allowed_days
                    morning_match = slot.start <= "12:00"
                    if not day_match or (morning_only and not morning_match):
                        blocked.add(slot.id)
                updated.lecturers[lecturer_id] = replace(lecturer, unavailable_slots=frozenset(blocked))
    return updated


def _apply_disruption(problem: ProblemData, disruption_type: str, target_id: str, slot_id: str) -> ProblemData:
    if not disruption_type or not target_id or not slot_id:
        return problem
    updated = clone_problem_data(problem)
    if disruption_type == "lecturer_unavailable" and target_id in updated.lecturers:
        lecturer = updated.lecturers[target_id]
        updated.lecturers[target_id] = replace(lecturer, unavailable_slots=frozenset(set(lecturer.unavailable_slots) | {slot_id}))
    elif disruption_type == "room_unavailable" and target_id in updated.rooms:
        room = updated.rooms[target_id]
        updated.rooms[target_id] = replace(room, equipment=frozenset(set(room.equipment) | {"TEMP_DISABLED"}))
    return updated


def _context_for_run(request: Request, run_id: str, message: str = "") -> Dict[str, Any]:
    run = RUN_CACHE[run_id]
    result = orchestrator.summarize(
        problem=run["problem"],
        assignments=run["assignments"],
        draft_assignments=run["draft_assignments"],
        strategy=run["strategy"],
        config=run["config"],
        weights=run["weights"],
        training_summary=run["training_summary"],
        optimization_summary=run["optimization_summary"],
        guidelines=run["guidelines"],
        lecturer_preferences=run["lecturer_preferences"],
        natural_language_request=run["natural_language_request"],
        disruptions=run["disruptions"],
        approved=run["approved"],
        feedback_log=run["feedback_log"],
    )
    download_id = str(uuid.uuid4())
    EXPORT_CACHE[download_id] = result["schedule_rows"]
    run["result"] = result
    return {
        "request": request,
        "result": result,
        "download_id": download_id,
        "run_id": run_id,
        "selected_strategy": run["strategy"],
        "training_episodes": run["training_episodes"],
        "generations": run["generations"],
        "constraint_defaults": run["config"],
        "weight_defaults": run["weights"],
        "lecturer_options": list(run["problem"].lecturers.values()),
        "slot_options": list(run["problem"].slots.values()),
        "room_options": list(run["problem"].rooms.values()),
        "message": message,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", _base_context(request))


@app.post("/generate", response_class=HTMLResponse)
async def generate_schedule(
    request: Request,
    courses: UploadFile | None = File(default=None),
    lecturers: UploadFile | None = File(default=None),
    rooms: UploadFile | None = File(default=None),
    timeslots: UploadFile | None = File(default=None),
    strategy: str = Form(default="hybrid"),
    training_episodes: int = Form(default=10),
    generations: int = Form(default=8),
    lecturer_double_booking: str | None = Form(default=None),
    room_exclusivity: str | None = Form(default=None),
    room_capacity_enforced: str | None = Form(default=None),
    room_type_enforced: str | None = Form(default=None),
    fatigue_limit_enabled: str | None = Form(default=None),
    max_consecutive_sessions: int = Form(default=3),
    preferred_slot_weight: float = Form(default=3.0),
    idle_time_weight: float = Form(default=2.0),
    geographic_weight: float = Form(default=1.5),
    fatigue_weight: float = Form(default=2.2),
    room_fit_weight: float = Form(default=1.4),
    lecturer_id: str = Form(default=""),
    lecturer_request: str = Form(default=""),
    preferred_slots: str = Form(default=""),
    preferred_days: str = Form(default=""),
    departmental_guidelines: str = Form(default=""),
    nl_request: str = Form(default=""),
    disruption_type: str = Form(default=""),
    disruption_target: str = Form(default=""),
    disruption_slot: str = Form(default=""),
) -> HTMLResponse:
    config = _constraint_config_from_form(
        lecturer_double_booking is not None,
        room_exclusivity is not None,
        room_capacity_enforced is not None,
        room_type_enforced is not None,
        fatigue_limit_enabled is not None,
        max_consecutive_sessions,
    )
    weights = _weights_from_form(
        preferred_slot_weight,
        idle_time_weight,
        geographic_weight,
        fatigue_weight,
        room_fit_weight,
    )
    preferences = _serialize_preferences(lecturer_id, lecturer_request, preferred_slots, preferred_days)
    disruptions = []
    base_problem = load_problem_data(courses, lecturers, rooms, timeslots)
    working_problem = _apply_natural_language(base_problem, nl_request)
    if disruption_type and disruption_target and disruption_slot:
        working_problem = _apply_disruption(working_problem, disruption_type, disruption_target, disruption_slot)
        disruptions.append({"disruption_type": disruption_type, "target_id": disruption_target, "slot_id": disruption_slot})

    result = orchestrator.run(
        problem=working_problem,
        strategy=strategy,
        training_episodes=max(0, training_episodes),
        generations=max(1, generations),
        config=config,
        weights=weights,
        guidelines=departmental_guidelines,
        lecturer_preferences=preferences,
        natural_language_request=nl_request,
        disruptions=disruptions,
    )
    run_id = str(uuid.uuid4())
    RUN_CACHE[run_id] = {
        "problem": working_problem,
        "base_problem": base_problem,
        "assignments": [Assignment(**assignment) if isinstance(assignment, dict) else assignment for assignment in []],
        "draft_assignments": [],
        "strategy": strategy,
        "training_episodes": max(0, training_episodes),
        "generations": max(1, generations),
        "config": config,
        "weights": weights,
        "training_summary": result["training_summary"],
        "optimization_summary": result["optimization_summary"],
        "guidelines": departmental_guidelines,
        "lecturer_preferences": preferences,
        "natural_language_request": nl_request,
        "disruptions": disruptions,
        "approved": False,
        "feedback_log": [],
        "result": result,
    }
    # persist structured assignments for later adjustments
    RUN_CACHE[run_id]["assignments"] = [
        Assignment(
            session_id=row["session_id"],
            course_id=next(course_id for course_id, course in working_problem.courses.items() if course.code == row["course_code"]),
            slot_id=next((slot.id for slot in working_problem.slots.values() if row["day"] == slot.day and row["time"] == f"{slot.start} - {slot.end}"), None) if row["day"] != "Unscheduled" else None,
            room_id=next((room.id for room in working_problem.rooms.values() if room.name == row["room"]), None) if row["room"] != "N/A" else None,
            lecturer_id=next(course.lecturer_id for course in working_problem.courses.values() if course.code == row["course_code"]),
            status=row["status"],
            notes=[row["notes"]] if row["notes"] else [],
        )
        for row in result["schedule_rows"]
    ]
    RUN_CACHE[run_id]["draft_assignments"] = RUN_CACHE[run_id]["assignments"][:]
    context = _context_for_run(request, run_id, message="Timetable generated with the latest client requirements enabled.")
    return templates.TemplateResponse(request, "index.html", context)


@app.post("/adjust", response_class=HTMLResponse)
async def adjust_schedule(
    request: Request,
    run_id: str = Form(...),
    session_id: str = Form(...),
    target_slot_id: str = Form(...),
) -> HTMLResponse:
    if run_id not in RUN_CACHE:
        return templates.TemplateResponse(request, "index.html", _base_context(request))
    run = RUN_CACHE[run_id]
    run["assignments"] = orchestrator.suggest_adjustment(run["problem"], run["assignments"], session_id, target_slot_id)
    context = _context_for_run(request, run_id, message=f"Manual what-if adjustment applied to {session_id}.")
    return templates.TemplateResponse(request, "index.html", context)


@app.post("/approve", response_class=HTMLResponse)
async def approve_schedule(
    request: Request,
    run_id: str = Form(...),
) -> HTMLResponse:
    if run_id not in RUN_CACHE:
        return templates.TemplateResponse(request, "index.html", _base_context(request))
    RUN_CACHE[run_id]["approved"] = True
    context = _context_for_run(request, run_id, message="Timetable approved by the administrator workflow.")
    return templates.TemplateResponse(request, "index.html", context)


@app.post("/feedback", response_class=HTMLResponse)
async def submit_feedback(
    request: Request,
    run_id: str = Form(...),
    feedback_rating: int = Form(...),
    feedback_comment: str = Form(default=""),
) -> HTMLResponse:
    if run_id not in RUN_CACHE:
        return templates.TemplateResponse(request, "index.html", _base_context(request))
    entry = FeedbackEntry(rating=feedback_rating, comment=feedback_comment)
    RUN_CACHE[run_id]["feedback_log"].append(asdict(entry))
    context = _context_for_run(request, run_id, message="Administrator feedback captured for iterative improvement.")
    return templates.TemplateResponse(request, "index.html", context)


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
