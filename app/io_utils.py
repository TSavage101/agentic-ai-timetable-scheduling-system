from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .models import Course, Lecturer, ProblemData, Room, SessionRequest, TimeSlot


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def _split_values(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def _load_dataframe(upload, fallback_name: str) -> pd.DataFrame:
    if upload is not None and getattr(upload, "filename", ""):
        raw = upload.file.read()
        upload.file.seek(0)
        return pd.read_csv(io.BytesIO(raw))
    return pd.read_csv(DATA_DIR / fallback_name)


def load_problem_data(
    courses_upload=None,
    lecturers_upload=None,
    rooms_upload=None,
    slots_upload=None,
) -> ProblemData:
    courses_df = _load_dataframe(courses_upload, "courses.csv")
    lecturers_df = _load_dataframe(lecturers_upload, "lecturers.csv")
    rooms_df = _load_dataframe(rooms_upload, "rooms.csv")
    slots_df = _load_dataframe(slots_upload, "timeslots.csv")

    lecturers: Dict[str, Lecturer] = {}
    for row in lecturers_df.to_dict(orient="records"):
        lecturers[str(row["id"])] = Lecturer(
            id=str(row["id"]),
            name=str(row["name"]),
            unavailable_slots=frozenset(_split_values(row.get("unavailable_slots"))),
        )

    rooms: Dict[str, Room] = {}
    for row in rooms_df.to_dict(orient="records"):
        rooms[str(row["id"])] = Room(
            id=str(row["id"]),
            name=str(row["name"]),
            capacity=int(row["capacity"]),
            room_type=str(row.get("room_type", "lecture")),
        )

    slots: Dict[str, TimeSlot] = {}
    for row in slots_df.to_dict(orient="records"):
        slots[str(row["id"])] = TimeSlot(
            id=str(row["id"]),
            day=str(row["day"]),
            start=str(row["start"]),
            end=str(row["end"]),
        )

    courses: Dict[str, Course] = {}
    session_requests: List[SessionRequest] = []
    for row in courses_df.to_dict(orient="records"):
        course_id = str(row["id"])
        course = Course(
            id=course_id,
            code=str(row["code"]),
            title=str(row["title"]),
            lecturer_id=str(row["lecturer_id"]),
            student_count=int(row["student_count"]),
            sessions_per_week=int(row["sessions_per_week"]),
            room_type=str(row.get("room_type", "lecture")),
            preferred_slots=frozenset(_split_values(row.get("preferred_slots"))),
            blocked_slots=frozenset(_split_values(row.get("blocked_slots"))),
            level=str(row.get("level", "")),
            department=str(row.get("department", "")),
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


def schedule_to_dataframe(assignments: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame(list(assignments))
