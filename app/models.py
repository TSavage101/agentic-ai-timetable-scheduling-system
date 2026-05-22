from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TimeSlot:
    id: str
    day: str
    start: str
    end: str


@dataclass(frozen=True)
class Room:
    id: str
    name: str
    capacity: int
    room_type: str = "lecture"


@dataclass(frozen=True)
class Lecturer:
    id: str
    name: str
    unavailable_slots: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Course:
    id: str
    code: str
    title: str
    lecturer_id: str
    student_count: int
    sessions_per_week: int
    room_type: str = "lecture"
    preferred_slots: frozenset[str] = frozenset()
    blocked_slots: frozenset[str] = frozenset()
    level: str = ""
    department: str = ""


@dataclass(frozen=True)
class SessionRequest:
    session_id: str
    course_id: str
    index: int


@dataclass
class Assignment:
    session_id: str
    course_id: str
    slot_id: Optional[str] = None
    room_id: Optional[str] = None
    lecturer_id: Optional[str] = None
    status: str = "scheduled"
    notes: List[str] = field(default_factory=list)


@dataclass
class Violation:
    kind: str
    message: str
    severity: str
    session_id: Optional[str] = None


@dataclass
class EvaluationResult:
    hard_violations: List[Violation]
    soft_violations: List[Violation]
    hard_score: int
    soft_score: int
    total_score: int
    metrics: Dict[str, float]


@dataclass
class ProblemData:
    courses: Dict[str, Course]
    lecturers: Dict[str, Lecturer]
    rooms: Dict[str, Room]
    slots: Dict[str, TimeSlot]
    session_requests: List[SessionRequest]
