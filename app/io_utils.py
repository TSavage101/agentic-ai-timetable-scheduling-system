from __future__ import annotations

import io
import re
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .models import Course, Lecturer, ProblemData, Room, SessionRequest, TimeSlot, StudentGroup


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"

COURSE_ALIASES = {
    "id": ["id", "course_id"],
    "code": ["code", "course_code"],
    "title": ["title", "course_title", "name"],
    "lecturer_id": ["lecturer_id", "lecturer", "lecturerid"],
    "student_count": ["student_count", "students", "class_size", "enrollment"],
    "sessions_per_week": ["sessions_per_week", "sessions", "weekly_sessions", "contact_hours"],
    "room_type": ["room_type", "room", "venue_type"],
    "eligible_lecturer_ids": ["eligible_lecturer_ids", "eligible_lecturers", "allowed_lecturers", "qualified_lecturers"],
    "preferred_slots": ["preferred_slots", "preferred_slot", "preferred_times", "preferred_timeslots"],
    "blocked_slots": ["blocked_slots", "blocked_slot", "restricted_slots", "unavailable_slots"],
    "level": ["level", "year", "academic_level"],
    "department": ["department", "dept"],
    "location": ["location", "block", "building", "preferred_location"],
    "equipment_needed": ["equipment_needed", "equipment", "required_equipment"],
    "duration_hours": ["duration_hours", "duration", "hours", "duration_hour", "session_duration"],
    "student_group": ["student_group", "student_group_id", "group"],
}

LECTURER_ALIASES = {
    "id": ["id", "lecturer_id"],
    "name": ["name", "lecturer_name", "full_name"],
    "departments": ["departments", "department", "dept", "specialization", "specializations"],
    "qualified_course_codes": ["qualified_course_codes", "course_codes", "eligible_course_codes", "can_teach_courses"],
    "qualified_course_prefixes": ["qualified_course_prefixes", "course_prefixes", "eligible_course_prefixes", "can_teach_prefixes"],
    "unavailable_slots": ["unavailable_slots", "unavailable", "blocked_slots", "restricted_slots"],
    "preferred_slots": ["preferred_slots", "preferred_slot", "preferred_times", "preferred_timeslots"],
    "max_hours_per_day": ["max_hours_per_day", "max_hours_day", "daily_limit", "max_daily_hours"],
    "max_hours_per_week": ["max_hours_per_week", "max_hours_week", "weekly_limit", "max_weekly_hours"],
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

STUDENT_GROUP_ALIASES = {
    "id": ["group_id", "id", "student_group_id"],
    "department": ["department", "dept"],
    "level": ["level", "year", "academic_level"],
    "name": ["group_name", "name"],
    "student_count": ["student_count", "students", "size", "cohort_size"],
}

DEPARTMENT_LECTURER_POOLS: Dict[str, List[str]] = {
    "cst": [
        "Prof Oyelade",
        "Mr Osofuye",
        "Miss Tunde-Adeleke",
        "Dr Iheanatu",
        "Mr Franklyn",
        "Dr Jonathan",
        "Mrs Abiodun",
        "Mr Ogunyale",
        "Dr Elliot",
        "Mr Ogbu",
        "Mr Otavie",
        "Dr Odili",
        "Mr Ejiobih",
        "Miss Nathaniel",
        "Dr Azu",
    ],
    "cmss": [
        "Dr Akinlabi",
        "Mrs Ezenwa",
        "Mr Bamidele",
        "Dr Nwachukwu",
        "Mrs Balogun",
        "Mr Ibe",
        "Dr Afolayan",
        "Miss Chukwu",
        "Mr Salami",
        "Dr Okonkwo",
    ],
    "clds": [
        "Dr Aderinto",
        "Mrs Umeh",
        "Mr Fagbemi",
        "Dr Madu",
        "Mrs Oladipo",
        "Mr Ekanem",
        "Dr Lawal",
        "Miss Nnamani",
        "Mr Duru",
        "Dr Obiakor",
    ],
    "coe": [
        "Dr Akinsete",
        "Mr Olanrewaju",
        "Mrs Chidubem",
        "Dr Danjuma",
        "Mr Udo",
        "Dr Eromosele",
        "Mrs Akinola",
        "Mr Ndukwe",
        "Dr Bakare",
        "Mr Eyo",
    ],
    "aldc": [
        "Prof Adeniran",
        "Dr Chukwuma",
        "Mrs Adebayo",
        "Chaplain Michael Eze",
        "Mr Adesina",
        "Dr Kalu",
    ],
}

DEPARTMENT_GROUPS: Dict[str, List[str]] = {
    "cst": [
        "architecture",
        "building technology",
        "estate management",
        "biochemistry",
        "chemistry",
        "computer and information sciences",
        "computer science",
        "management and information sciences",
        "management information science",
        "mis",
        "microbiology",
        "industrial chemistry",
        "industrial mathematics",
        "industrial mathemactics",
        "industrial physics",
    ],
    "cmss": [
        "accounting",
        "banking/finance",
        "banking and finance",
        "business administration",
        "business addminstration",
        "economics",
        "mass communication",
        "sociology",
    ],
    "clds": [
        "english",
        "international relations",
        "policy and strategic studies",
        "political science",
        "psychology",
    ],
    "coe": [
        "chemical engineering",
        "civil engineering",
        "computer engineering",
        "electrical / electronics engineering",
        "electrical electronics engineering",
        "electrical and electronics engineering",
        "information and communication engineering",
        "mechanical engineering",
        "petroleum engineering",
    ],
}

GENERAL_COURSE_PREFIXES = ("GST", "TMC", "EDS", "DLD", "ALDC")


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


def _normalize_department_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _department_pool_key(department: str, course_code: str = "") -> str:
    normalized = _normalize_department_name(department)
    if any(course_code.upper().startswith(prefix) for prefix in GENERAL_COURSE_PREFIXES):
        return "aldc"
    for pool_key, entries in DEPARTMENT_GROUPS.items():
        if normalized in (_normalize_department_name(item) for item in entries):
            return pool_key
    return normalized or "general"


def _build_generated_lecturers(course_rows: List[dict], lecturer_rows: List[dict]) -> Dict[str, Lecturer]:
    generated: Dict[str, Lecturer] = {}
    inferred_departments: Dict[str, set[str]] = {}
    for row in course_rows:
        lecturer_id = _clean_scalar(row.get("lecturer_id"))
        department = _clean_scalar(row.get("department"))
        if lecturer_id and department:
            inferred_departments.setdefault(lecturer_id, set()).add(department)

    for row in lecturer_rows:
        lecturer_id = _clean_scalar(row.get("id"))
        if not lecturer_id:
            continue
        max_h_day = _clean_int(row.get("max_hours_per_day"), 4)
        max_h_week = _clean_int(row.get("max_hours_per_week"), 12)
        row_departments = set(_split_values(row.get("departments"))) or inferred_departments.get(lecturer_id, set())
        generated[lecturer_id] = Lecturer(
            id=lecturer_id,
            name=_clean_scalar(row.get("name"), lecturer_id),
            departments=frozenset(row_departments),
            qualified_course_codes=frozenset(item.upper() for item in _split_values(row.get("qualified_course_codes"))),
            qualified_course_prefixes=frozenset(item.upper() for item in _split_values(row.get("qualified_course_prefixes"))),
            unavailable_slots=frozenset(_split_values(row.get("unavailable_slots"))),
            preferred_slots=frozenset(_split_values(row.get("preferred_slots"))),
            max_hours_per_day=max_h_day if max_h_day > 0 else 4,
            max_hours_per_week=max_h_week if max_h_week > 0 else 12,
        )

    departments_needed = {
        (_clean_scalar(row.get("department")), _clean_scalar(row.get("code")))
        for row in course_rows
        if _clean_scalar(row.get("department")) or _clean_scalar(row.get("code"))
    }
    for department, course_code in departments_needed:
        pool_key = _department_pool_key(department, course_code)
        names = DEPARTMENT_LECTURER_POOLS.get(pool_key, [])
        for index, name in enumerate(names, start=1):
            lecturer_id = f"AUTO_{pool_key.upper()}_{index:02d}"
            if lecturer_id in generated:
                continue
            generated[lecturer_id] = Lecturer(
                id=lecturer_id,
                name=name,
                departments=frozenset({department or pool_key.upper()}),
            )
    return generated


def _eligible_lecturers_for_course(
    lecturers: Dict[str, Lecturer],
    department: str,
    course_code: str,
    explicit_lecturer_ids: List[str] | None = None,
) -> List[Lecturer]:
    pool_key = _department_pool_key(department, course_code)
    normalized_department = _normalize_department_name(department)
    explicit_set = {item for item in (explicit_lecturer_ids or []) if item in lecturers}
    if explicit_set:
        return sorted([lecturers[item] for item in explicit_set], key=lambda item: (item.id.startswith("AUTO_"), item.id))
    eligible: List[Lecturer] = []
    upper_code = course_code.upper()
    for lecturer in lecturers.values():
        if lecturer.qualified_course_codes and upper_code not in lecturer.qualified_course_codes:
            if not any(upper_code.startswith(prefix) for prefix in lecturer.qualified_course_prefixes):
                continue
        elif lecturer.qualified_course_prefixes and not any(upper_code.startswith(prefix) for prefix in lecturer.qualified_course_prefixes):
            if upper_code not in lecturer.qualified_course_codes:
                continue
        if not lecturer.departments:
            if pool_key in {"aldc", "general"}:
                eligible.append(lecturer)
            continue
        normalized_departments = {_normalize_department_name(item) for item in lecturer.departments}
        if normalized_department in normalized_departments:
            eligible.append(lecturer)
            continue
        if pool_key == "aldc" and any(
            marker in normalized_departments
            for marker in {
                _normalize_department_name("General Studies"),
                _normalize_department_name("ALDC"),
                _normalize_department_name("Mass Communication"),
                _normalize_department_name("Economics"),
                _normalize_department_name("Business Administration"),
            }
        ):
            eligible.append(lecturer)
            continue
        if pool_key != "general":
            pool_departments = {
                _normalize_department_name(item)
                for item in DEPARTMENT_GROUPS.get(pool_key, [])
            }
            if normalized_departments & pool_departments:
                eligible.append(lecturer)
    return sorted(eligible, key=lambda item: (item.id.startswith("AUTO_"), item.id))


def _load_dataframe(upload, fallback_name: str) -> pd.DataFrame:
    if upload is not None and getattr(upload, "filename", ""):
        raw = upload.file.read()
        upload.file.seek(0)
        return _prepare_dataframe(pd.read_csv(io.BytesIO(raw)))
    path = DATA_DIR / fallback_name
    if not path.exists() and fallback_name == "student_groups.csv":
        return pd.DataFrame()
    return _prepare_dataframe(pd.read_csv(path))


def load_problem_data(
    courses_upload=None,
    lecturers_upload=None,
    rooms_upload=None,
    slots_upload=None,
    student_groups_upload=None,
) -> ProblemData:
    courses_df = _rename_columns(_load_dataframe(courses_upload, "courses.csv"), COURSE_ALIASES)
    lecturers_df = _rename_columns(_load_dataframe(lecturers_upload, "lecturers.csv"), LECTURER_ALIASES)
    rooms_df = _rename_columns(_load_dataframe(rooms_upload, "rooms.csv"), ROOM_ALIASES)
    slots_df = _rename_columns(_load_dataframe(slots_upload, "timeslots.csv"), SLOT_ALIASES)
    
    try:
        student_groups_df = _rename_columns(_load_dataframe(student_groups_upload, "student_groups.csv"), STUDENT_GROUP_ALIASES)
    except Exception:
        student_groups_df = pd.DataFrame()

    _require_columns(courses_df, ["id", "code", "student_count", "sessions_per_week"], "Courses")
    _require_columns(lecturers_df, ["id", "name"], "Lecturers")
    _require_columns(rooms_df, ["id", "name", "capacity"], "Rooms")
    _require_columns(slots_df, ["id", "day", "start", "end"], "Timeslots")

    course_rows = courses_df.to_dict(orient="records")
    lecturer_rows = lecturers_df.to_dict(orient="records")
    lecturers = _build_generated_lecturers(course_rows, lecturer_rows)

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

    # Load or generate student groups
    student_groups: Dict[str, StudentGroup] = {}
    if not student_groups_df.empty:
        for row in student_groups_df.to_dict(orient="records"):
            g_id = _clean_scalar(row.get("id"))
            if g_id:
                student_groups[g_id] = StudentGroup(
                    id=g_id,
                    department=_clean_scalar(row.get("department")),
                    level=_clean_scalar(row.get("level")),
                    name=_clean_scalar(row.get("name"), g_id),
                    student_count=max(1, _clean_int(row.get("student_count"), 1))
                )

    courses: Dict[str, Course] = {}
    session_requests: List[SessionRequest] = []
    lecturer_assignment_counts: Dict[str, int] = {lecturer_id: 0 for lecturer_id in lecturers}
    for row in course_rows:
        course_id = _clean_scalar(row.get("id"))
        code = _clean_scalar(row.get("code"), course_id)
        title = _clean_scalar(row.get("title"), code)
        dept = _clean_scalar(row.get("department"))
        level = _clean_scalar(row.get("level"))
        student_count = max(1, _clean_int(row.get("student_count"), 1))

        # Find or generate student group for the course
        matched_group_id = None
        for g_id, sg in student_groups.items():
            if sg.department.strip().lower() == dept.strip().lower() and str(sg.level).strip() == str(level).strip():
                matched_group_id = g_id
                break
        
        if not matched_group_id:
            dept_clean = re.sub(r"[^a-zA-Z0-9]", "", dept)
            if not dept_clean:
                dept_clean = "Group"
            matched_group_id = f"{dept_clean}_{level}"
            if matched_group_id not in student_groups:
                student_groups[matched_group_id] = StudentGroup(
                    id=matched_group_id,
                    department=dept,
                    level=level,
                    name=f"{dept} - Level {level}",
                    student_count=student_count
                )
            else:
                if student_count > student_groups[matched_group_id].student_count:
                    student_groups[matched_group_id] = replace(
                        student_groups[matched_group_id],
                        student_count=student_count
                    )

        requested_lecturer_id = _clean_scalar(row.get("lecturer_id"))
        explicit_eligible_ids = _split_values(row.get("eligible_lecturer_ids"))
        eligible_lecturers = _eligible_lecturers_for_course(lecturers, dept, code, explicit_eligible_ids)
        lecturer_id = requested_lecturer_id
        eligible_ids = {item.id for item in eligible_lecturers}
        if lecturer_id not in lecturers or (eligible_lecturers and lecturer_id not in eligible_ids):
            if eligible_lecturers:
                lecturer_id = min(
                    eligible_lecturers,
                    key=lambda item: (lecturer_assignment_counts.get(item.id, 0), item.id.startswith("AUTO_"), item.id),
                ).id
            elif lecturers:
                lecturer_id = sorted(lecturers.keys())[0]
            else:
                raise ValueError(f"No lecturer roster is available for course {code}.")
        elif explicit_eligible_ids and lecturer_id not in explicit_eligible_ids and eligible_lecturers:
            lecturer_id = min(
                eligible_lecturers,
                key=lambda item: (lecturer_assignment_counts.get(item.id, 0), item.id.startswith("AUTO_"), item.id),
            ).id

        course = Course(
            id=course_id,
            code=code,
            title=title,
            lecturer_id=lecturer_id,
            student_count=student_count,
            sessions_per_week=max(1, _clean_int(row.get("sessions_per_week"), 1)),
            room_type=_normalize_room_type(_clean_scalar(row.get("room_type"), "lecture")),
            eligible_lecturer_ids=frozenset(explicit_eligible_ids or eligible_ids),
            preferred_slots=frozenset(_split_values(row.get("preferred_slots"))),
            blocked_slots=frozenset(_split_values(row.get("blocked_slots"))),
            level=level,
            department=dept,
            location=_clean_scalar(row.get("location")),
            equipment_needed=frozenset(_split_values(row.get("equipment_needed"))),
            duration_hours=max(1, _clean_int(row.get("duration_hours"), 1)),
            student_group=matched_group_id,
        )
        courses[course_id] = course
        lecturer_assignment_counts[lecturer_id] = lecturer_assignment_counts.get(lecturer_id, 0) + 1
        segment_plan = [(course.duration_hours, "")] if course.duration_hours != 3 else [(2, "A"), (1, "B")]
        for index in range(1, course.sessions_per_week + 1):
            for duration_hours, segment_label in segment_plan:
                suffix = f"{index}{segment_label}" if segment_label else str(index)
                session_requests.append(
                    SessionRequest(
                        session_id=f"{course.code}-S{suffix}",
                        course_id=course_id,
                        index=index,
                        duration_hours=duration_hours,
                        segment_label=segment_label,
                    )
                )

    return ProblemData(
        courses=courses,
        lecturers=lecturers,
        rooms=rooms,
        slots=slots,
        session_requests=session_requests,
        student_groups=student_groups,
    )


def clone_problem_data(problem: ProblemData) -> ProblemData:
    return ProblemData(
        courses={key: replace(value) for key, value in problem.courses.items()},
        lecturers={key: replace(value) for key, value in problem.lecturers.items()},
        rooms={key: replace(value) for key, value in problem.rooms.items()},
        slots={key: replace(value) for key, value in problem.slots.items()},
        session_requests=list(problem.session_requests),
        student_groups={key: replace(value) for key, value in problem.student_groups.items()} if hasattr(problem, "student_groups") else {},
    )


def schedule_to_dataframe(assignments: Iterable[dict]) -> pd.DataFrame:
    dataframe = pd.DataFrame(list(assignments))
    if dataframe.empty:
        return dataframe
    day_order = {
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
        "Sunday": 7,
        "Unscheduled": 99,
    }
    dataframe["_day_order"] = dataframe["day"].map(day_order).fillna(98)
    dataframe = dataframe.sort_values(
        by=["_day_order", "time", "department", "level", "course_code", "session_id"],
        kind="stable",
    ).drop(columns=["_day_order"])
    return dataframe


def schedule_to_export_bundle(assignments: Iterable[dict]) -> bytes:
    dataframe = schedule_to_dataframe(assignments)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("overview/timetable_overview.csv", dataframe.to_csv(index=False))
        if dataframe.empty:
            archive.writestr("README.txt", "No timetable rows were available for export.\n")
            return buffer.getvalue()

        summary = (
            dataframe.groupby(["department", "level"], dropna=False)
            .size()
            .reset_index(name="scheduled_sessions")
        )
        archive.writestr("overview/department_summary.csv", summary.to_csv(index=False))

        for department, frame in dataframe.groupby("department", dropna=False):
            safe_department = re.sub(r"[^A-Za-z0-9_-]+", "_", str(department or "Unspecified")).strip("_") or "Unspecified"
            archive.writestr(f"by_department/{safe_department}.csv", frame.to_csv(index=False))

        for lecturer, frame in dataframe.groupby("lecturer", dropna=False):
            safe_lecturer = re.sub(r"[^A-Za-z0-9_-]+", "_", str(lecturer or "Unassigned")).strip("_") or "Unassigned"
            archive.writestr(f"by_lecturer/{safe_lecturer}.csv", frame.to_csv(index=False))
    return buffer.getvalue()
