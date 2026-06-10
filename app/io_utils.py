from __future__ import annotations

import io
import re
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .models import Course, Lecturer, ProblemData, Room, SessionRequest, TimeSlot


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"

COURSE_ALIASES = {
    "id": ["id", "course_id"],
    "code": ["code", "course_code"],
    "title": ["title", "course_title", "name"],
    "lecturer_id": ["lecturer_id", "lecturer", "lecturerid"],
    "student_count": ["student_count", "students", "class_size", "enrollment"],
    "sessions_per_week": ["sessions_per_week", "sessions", "weekly_sessions", "contact_hours"],
    "room_type": ["room_type", "room", "venue_type"],
    "preferred_slots": ["preferred_slots", "preferred_slot", "preferred_times", "preferred_timeslots"],
    "blocked_slots": ["blocked_slots", "blocked_slot", "restricted_slots", "unavailable_slots"],
    "level": ["level", "year", "academic_level"],
    "department": ["department", "dept"],
    "equipment_needed": ["equipment_needed", "equipment", "required_equipment"],
}

LECTURER_ALIASES = {
    "id": ["id", "lecturer_id"],
    "name": ["name", "lecturer_name", "full_name"],
    "unavailable_slots": ["unavailable_slots", "unavailable", "blocked_slots", "restricted_slots"],
}

ROOM_ALIASES = {
    "id": ["id", "room_id"],
    "name": ["name", "room_name", "venue"],
    "capacity": ["capacity", "size", "room_capacity"],
    "room_type": ["room_type", "type", "venue_type"],
    "location": ["location", "block", "building"],
    "equipment": ["equipment", "facilities"],
}

SLOT_ALIASES = {
    "id": ["id", "slot_id", "timeslot_id"],
    "day": ["day", "weekday"],
    "start": ["start", "start_time", "from"],
    "end": ["end", "end_time", "to"],
}


def _split_values(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in re.split(r"[|,;\n]+", text) if item.strip()]


def _normalize_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _prepare_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [_normalize_column_name(column) for column in frame.columns]
    return frame


def _rename_columns(frame: pd.DataFrame, aliases: Dict[str, List[str]]) -> pd.DataFrame:
    normalized = _prepare_dataframe(frame)
    rename_map: Dict[str, str] = {}
    for target, names in aliases.items():
        for name in names:
            normalized_name = _normalize_column_name(name)
            if normalized_name in normalized.columns:
                rename_map[normalized_name] = target
                break
    return normalized.rename(columns=rename_map)


def _require_columns(frame: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} CSV is missing required columns: {', '.join(missing)}")


def _clean_scalar(value: object, default: str = "") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip()
    return text or default


def _clean_int(value: object, default: int = 0) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _normalize_room_type(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned in {"lab", "laboratory"}:
        return "lab"
    if cleaned in {"lecture", "lecture_hall", "hall", "classroom"}:
        return "lecture"
    return cleaned or "lecture"


def _load_dataframe(upload, fallback_name: str) -> pd.DataFrame:
    if upload is not None and getattr(upload, "filename", ""):
        raw = upload.file.read()
        upload.file.seek(0)
        return _prepare_dataframe(pd.read_csv(io.BytesIO(raw)))
    return _prepare_dataframe(pd.read_csv(DATA_DIR / fallback_name))


def load_problem_data(
    courses_upload=None,
    lecturers_upload=None,
    rooms_upload=None,
    slots_upload=None,
) -> ProblemData:
    courses_df = _rename_columns(_load_dataframe(courses_upload, "courses.csv"), COURSE_ALIASES)
    lecturers_df = _rename_columns(_load_dataframe(lecturers_upload, "lecturers.csv"), LECTURER_ALIASES)
    rooms_df = _rename_columns(_load_dataframe(rooms_upload, "rooms.csv"), ROOM_ALIASES)
    slots_df = _rename_columns(_load_dataframe(slots_upload, "timeslots.csv"), SLOT_ALIASES)

    _require_columns(courses_df, ["id", "code", "lecturer_id", "student_count", "sessions_per_week"], "Courses")
    _require_columns(lecturers_df, ["id", "name"], "Lecturers")
    _require_columns(rooms_df, ["id", "name", "capacity"], "Rooms")
    _require_columns(slots_df, ["id", "day", "start", "end"], "Timeslots")

    lecturers: Dict[str, Lecturer] = {}
    for row in lecturers_df.to_dict(orient="records"):
        lecturer_id = _clean_scalar(row.get("id"))
        lecturers[lecturer_id] = Lecturer(
            id=lecturer_id,
            name=_clean_scalar(row.get("name"), lecturer_id),
            unavailable_slots=frozenset(_split_values(row.get("unavailable_slots"))),
        )

    rooms: Dict[str, Room] = {}
    for row in rooms_df.to_dict(orient="records"):
        room_id = _clean_scalar(row.get("id"))
        rooms[room_id] = Room(
            id=room_id,
            name=_clean_scalar(row.get("name"), room_id),
            capacity=max(1, _clean_int(row.get("capacity"), 1)),
            room_type=_normalize_room_type(_clean_scalar(row.get("room_type"), "lecture")),
            location=_clean_scalar(row.get("location"), "Main Block"),
            equipment=frozenset(_split_values(row.get("equipment"))),
        )

    slots: Dict[str, TimeSlot] = {}
    for row in slots_df.to_dict(orient="records"):
        slot_id = _clean_scalar(row.get("id"))
        slots[slot_id] = TimeSlot(
            id=slot_id,
            day=_clean_scalar(row.get("day")),
            start=_clean_scalar(row.get("start")),
            end=_clean_scalar(row.get("end")),
        )

    courses: Dict[str, Course] = {}
    session_requests: List[SessionRequest] = []
    for row in courses_df.to_dict(orient="records"):
        course_id = _clean_scalar(row.get("id"))
        code = _clean_scalar(row.get("code"), course_id)
        title = _clean_scalar(row.get("title"), code)
        course = Course(
            id=course_id,
            code=code,
            title=title,
            lecturer_id=_clean_scalar(row.get("lecturer_id")),
            student_count=max(1, _clean_int(row.get("student_count"), 1)),
            sessions_per_week=max(1, _clean_int(row.get("sessions_per_week"), 1)),
            room_type=_normalize_room_type(_clean_scalar(row.get("room_type"), "lecture")),
            preferred_slots=frozenset(_split_values(row.get("preferred_slots"))),
            blocked_slots=frozenset(_split_values(row.get("blocked_slots"))),
            level=_clean_scalar(row.get("level")),
            department=_clean_scalar(row.get("department")),
            equipment_needed=frozenset(_split_values(row.get("equipment_needed"))),
        )
        courses[course_id] = course
        for index in range(1, course.sessions_per_week + 1):
            session_requests.append(
                SessionRequest(
                    session_id=f"{course.code}-S{index}",
                    course_id=course_id,
                    index=index,
                )
            )

    return ProblemData(
        courses=courses,
        lecturers=lecturers,
        rooms=rooms,
        slots=slots,
        session_requests=session_requests,
    )


def clone_problem_data(problem: ProblemData) -> ProblemData:
    return ProblemData(
        courses={key: replace(value) for key, value in problem.courses.items()},
        lecturers={key: replace(value) for key, value in problem.lecturers.items()},
        rooms={key: replace(value) for key, value in problem.rooms.items()},
        slots={key: replace(value) for key, value in problem.slots.items()},
        session_requests=list(problem.session_requests),
    )


def schedule_to_dataframe(assignments: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame(list(assignments))
