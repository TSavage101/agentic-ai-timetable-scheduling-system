from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .agents import OrchestratorAgent
from .io_utils import clone_problem_data, load_problem_data, schedule_to_dataframe
from .models import Assignment, ConstraintConfig, ProblemData, SoftConstraintWeights


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Agentic AI Timetable Scheduling System")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
orchestrator = OrchestratorAgent()
EXPORT_CACHE: Dict[str, object] = {}
RUN_CACHE: Dict[str, Dict[str, Any]] = {}


NAV_ITEMS = [
    ("dashboard", "Dashboard", "/dashboard"),
    ("generate", "Generate", "/generate"),
    ("timetable", "Timetable Grid", "/timetable"),
    ("scenarios", "Scenarios", "/scenarios"),
    ("resources", "Resources", "/resources"),
    ("feedback", "Feedback", "/feedback"),
    ("audit", "Audit Log", "/audit"),
    ("settings", "Settings", "/settings"),
]


def _latest_run_id() -> Optional[str]:
    if not RUN_CACHE:
        return None
    return list(RUN_CACHE.keys())[-1]


def _page_meta(page: str) -> tuple[str, str]:
    lookup = {
        "dashboard": ("Command Center", "High-level system overview, progress, and quality status."),
        "generate": ("Generate Schedule", "Configure constraints, upload data, and launch optimized timetable runs."),
        "timetable": ("Interactive Timetable", "Review, adjust, approve, and export the generated weekly grid."),
        "scenarios": ("What-If Scenarios", "Compare draft and optimized states to understand trade-offs."),
        "resources": ("Resource Manager", "Inspect rooms, lecturers, and uploaded operational resources."),
        "feedback": ("Feedback Loop", "Capture administrator feedback and improve future scheduling quality."),
        "audit": ("Decision Audit", "Understand why the agents made each important scheduling choice."),
        "settings": ("System Settings", "Review policy, weighting, and operational configuration values."),
    }
    return lookup[page]


def _nav_links(run_id: Optional[str]) -> List[Dict[str, str]]:
    items = []
    suffix = f"?run_id={run_id}" if run_id else ""
    for key, label, href in NAV_ITEMS:
        items.append({"key": key, "label": label, "href": f"{href}{suffix}"})
    return items


def _base_context(request: Request, active_page: str, run_id: Optional[str] = None, message: str = "") -> Dict[str, Any]:
    title, subtitle = _page_meta(active_page)
    try:
        sample_problem = load_problem_data()
        lecturer_options = list(sample_problem.lecturers.values())
        slot_options = list(sample_problem.slots.values())
        room_options = list(sample_problem.rooms.values())
    except Exception:
        lecturer_options = []
        slot_options = []
        room_options = []
    return {
        "request": request,
        "active_page": active_page,
        "page_title": title,
        "page_subtitle": subtitle,
        "nav_items": _nav_links(run_id),
        "result": None,
        "run_id": run_id,
        "download_id": None,
        "selected_strategy": "hybrid",
        "training_episodes": 10,
        "generations": 8,
        "constraint_defaults": ConstraintConfig(),
        "weight_defaults": SoftConstraintWeights(),
        "lecturer_options": lecturer_options,
        "slot_options": slot_options,
        "room_options": room_options,
        "message": message,
        "preference_rows": [{"lecturer_id": "", "request_text": "", "slot": "", "day": ""}],
        "disruption_rows": [{"disruption_type": "", "target_id": "", "slot_id": "", "note": ""}],
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


def _serialize_preferences(
    lecturer_ids: List[str],
    request_texts: List[str],
    slots: List[str],
    days: List[str],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    total = max(len(lecturer_ids), len(request_texts), len(slots), len(days), 1)
    for idx in range(total):
        lecturer_id = lecturer_ids[idx] if idx < len(lecturer_ids) else ""
        request_text = request_texts[idx] if idx < len(request_texts) else ""
        slot = slots[idx] if idx < len(slots) else ""
        day = days[idx] if idx < len(days) else ""
        if any([lecturer_id, request_text, slot, day]):
            rows.append(
                {
                    "lecturer_id": lecturer_id,
                    "request_text": request_text,
                    "slot": slot,
                    "day": day,
                }
            )
    return rows


def _serialize_disruptions(
    disruption_types: List[str],
    targets: List[str],
    slots: List[str],
    notes: List[str],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    total = max(len(disruption_types), len(targets), len(slots), len(notes), 1)
    for idx in range(total):
        disruption_type = disruption_types[idx] if idx < len(disruption_types) else ""
        target = targets[idx] if idx < len(targets) else ""
        slot = slots[idx] if idx < len(slots) else ""
        note = notes[idx] if idx < len(notes) else ""
        if any([disruption_type, target, slot, note]):
            rows.append(
                {
                    "disruption_type": disruption_type,
                    "target_id": target,
                    "slot_id": slot,
                    "note": note,
                }
            )
    return rows


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
                updated.lecturers[lecturer_id] = type(lecturer)(
                    id=lecturer.id,
                    name=lecturer.name,
                    unavailable_slots=frozenset(blocked),
                )
    return updated


def _apply_preference_rows(problem: ProblemData, preference_rows: List[Dict[str, str]]) -> ProblemData:
    if not preference_rows:
        return problem
    updated = clone_problem_data(problem)
    for row in preference_rows:
        lecturer_id = row["lecturer_id"]
        if lecturer_id and lecturer_id in updated.lecturers and row["slot"]:
            lecturer = updated.lecturers[lecturer_id]
            unavailable = set(lecturer.unavailable_slots)
            for slot in updated.slots.values():
                if row["day"] and slot.day != row["day"]:
                    unavailable.add(slot.id)
            updated.lecturers[lecturer_id] = type(lecturer)(
                id=lecturer.id,
                name=lecturer.name,
                unavailable_slots=frozenset(unavailable.difference({row["slot"]})),
            )
    return updated


def _apply_disruptions(problem: ProblemData, disruptions: List[Dict[str, str]]) -> ProblemData:
    updated = clone_problem_data(problem)
    for row in disruptions:
        disruption_type = row["disruption_type"]
        target_id = row["target_id"]
        slot_id = row["slot_id"]
        if not all([disruption_type, target_id, slot_id]):
            continue
        if disruption_type == "lecturer_unavailable" and target_id in updated.lecturers:
            lecturer = updated.lecturers[target_id]
            updated.lecturers[target_id] = type(lecturer)(
                id=lecturer.id,
                name=lecturer.name,
                unavailable_slots=frozenset(set(lecturer.unavailable_slots) | {slot_id}),
            )
    return updated


def _coerce_assignments(serialized: List[Dict[str, Any]]) -> List[Assignment]:
    return [Assignment(**dict(item)) for item in serialized]


def _run_context(request: Request, active_page: str, run_id: Optional[str], message: str = "") -> Dict[str, Any]:
    if not run_id or run_id not in RUN_CACHE:
        return _base_context(request, active_page, run_id, message)
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
    title, subtitle = _page_meta(active_page)
    return {
        "request": request,
        "active_page": active_page,
        "page_title": title,
        "page_subtitle": subtitle,
        "nav_items": _nav_links(run_id),
        "result": result,
        "run_id": run_id,
        "download_id": download_id,
        "selected_strategy": run["strategy"],
        "training_episodes": run["training_episodes"],
        "generations": run["generations"],
        "constraint_defaults": run["config"],
        "weight_defaults": run["weights"],
        "lecturer_options": list(run["problem"].lecturers.values()),
        "slot_options": list(run["problem"].slots.values()),
        "room_options": list(run["problem"].rooms.values()),
        "message": message,
        "preference_rows": run["preference_rows"] or [{"lecturer_id": "", "request_text": "", "slot": "", "day": ""}],
        "disruption_rows": run["disruption_rows"] or [{"disruption_type": "", "target_id": "", "slot_id": "", "note": ""}],
    }


def _render(request: Request, page: str, run_id: Optional[str] = None, message: str = "") -> HTMLResponse:
    active_run = run_id or _latest_run_id()
    context = _run_context(request, page, active_run, message)
    return templates.TemplateResponse(request, f"pages/{page}.html", context)


@app.get("/", response_class=HTMLResponse)
async def home() -> RedirectResponse:
    return RedirectResponse("/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, run_id: str | None = Query(default=None)) -> HTMLResponse:
    return _render(request, "dashboard", run_id)


@app.get("/generate", response_class=HTMLResponse)
async def generate_page(request: Request, run_id: str | None = Query(default=None)) -> HTMLResponse:
    return _render(request, "generate", run_id)


@app.get("/timetable", response_class=HTMLResponse)
async def timetable_page(request: Request, run_id: str | None = Query(default=None)) -> HTMLResponse:
    return _render(request, "timetable", run_id)


@app.get("/scenarios", response_class=HTMLResponse)
async def scenarios_page(request: Request, run_id: str | None = Query(default=None)) -> HTMLResponse:
    return _render(request, "scenarios", run_id)


@app.get("/resources", response_class=HTMLResponse)
async def resources_page(request: Request, run_id: str | None = Query(default=None)) -> HTMLResponse:
    return _render(request, "resources", run_id)


@app.get("/feedback", response_class=HTMLResponse)
async def feedback_page(request: Request, run_id: str | None = Query(default=None)) -> HTMLResponse:
    return _render(request, "feedback", run_id)


@app.get("/audit", response_class=HTMLResponse)
async def audit_page(request: Request, run_id: str | None = Query(default=None)) -> HTMLResponse:
    return _render(request, "audit", run_id)


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, run_id: str | None = Query(default=None)) -> HTMLResponse:
    return _render(request, "settings", run_id)


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
    lecturer_pref_lecturer_id: List[str] = Form(default=[]),
    lecturer_pref_request: List[str] = Form(default=[]),
    lecturer_pref_slot: List[str] = Form(default=[]),
    lecturer_pref_day: List[str] = Form(default=[]),
    departmental_guidelines: str = Form(default=""),
    nl_request: str = Form(default=""),
    disruption_type: List[str] = Form(default=[]),
    disruption_target: List[str] = Form(default=[]),
    disruption_slot: List[str] = Form(default=[]),
    disruption_note: List[str] = Form(default=[]),
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
    preference_rows = _serialize_preferences(
        lecturer_pref_lecturer_id,
        lecturer_pref_request,
        lecturer_pref_slot,
        lecturer_pref_day,
    )
    disruption_rows = _serialize_disruptions(
        disruption_type,
        disruption_target,
        disruption_slot,
        disruption_note,
    )

    base_problem = load_problem_data(courses, lecturers, rooms, timeslots)
    working_problem = _apply_natural_language(base_problem, nl_request)
    working_problem = _apply_preference_rows(working_problem, preference_rows)
    working_problem = _apply_disruptions(working_problem, disruption_rows)

    result = orchestrator.run(
        problem=working_problem,
        strategy=strategy,
        training_episodes=max(0, training_episodes),
        generations=max(1, generations),
        config=config,
        weights=weights,
        guidelines=departmental_guidelines,
        lecturer_preferences=preference_rows,
        natural_language_request=nl_request,
        disruptions=disruption_rows,
    )

    run_id = str(uuid.uuid4())
    RUN_CACHE[run_id] = {
        "problem": working_problem,
        "base_problem": base_problem,
        "assignments": _coerce_assignments(result["assignments"]),
        "draft_assignments": _coerce_assignments(result["draft_assignments"]),
        "strategy": strategy,
        "training_episodes": max(0, training_episodes),
        "generations": max(1, generations),
        "config": config,
        "weights": weights,
        "training_summary": result["training_summary"],
        "optimization_summary": result["optimization_summary"],
        "guidelines": departmental_guidelines,
        "lecturer_preferences": preference_rows,
        "natural_language_request": nl_request,
        "disruptions": disruption_rows,
        "approved": False,
        "feedback_log": [],
        "preference_rows": preference_rows,
        "disruption_rows": disruption_rows,
        "result": result,
    }
    return _render(request, "dashboard", run_id, "Fresh schedule generated. The dashboard now reflects the latest run.")


@app.post("/adjust", response_class=HTMLResponse)
async def adjust_schedule(
    request: Request,
    run_id: str = Form(...),
    session_id: str = Form(...),
    target_slot_id: str = Form(...),
) -> HTMLResponse:
    if run_id not in RUN_CACHE:
        return _render(request, "timetable", None)
    run = RUN_CACHE[run_id]
    run["assignments"] = orchestrator.suggest_adjustment(run["problem"], run["assignments"], session_id, target_slot_id)
    return _render(request, "timetable", run_id, f"Moved {session_id} into a new what-if slot for comparison.")


@app.post("/approve", response_class=HTMLResponse)
async def approve_schedule(request: Request, run_id: str = Form(...)) -> HTMLResponse:
    if run_id in RUN_CACHE:
        RUN_CACHE[run_id]["approved"] = True
    return _render(request, "timetable", run_id, "Timetable approved and marked ready for handoff.")


@app.post("/feedback", response_class=HTMLResponse)
async def submit_feedback(
    request: Request,
    run_id: str = Form(...),
    feedback_rating: int = Form(...),
    feedback_comment: str = Form(default=""),
) -> HTMLResponse:
    if run_id in RUN_CACHE:
        RUN_CACHE[run_id]["feedback_log"].append({"rating": feedback_rating, "comment": feedback_comment})
    return _render(request, "feedback", run_id, "Feedback saved to the improvement log.")


@app.get("/download/{download_id}")
async def download_schedule(download_id: str):
    rows = EXPORT_CACHE.get(download_id)
    if rows is None:
        return RedirectResponse("/dashboard")
    dataframe = schedule_to_dataframe(rows)
    buffer = io.StringIO()
    dataframe.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=timetable.csv"},
    )
