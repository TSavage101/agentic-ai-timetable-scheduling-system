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
    location: str = "Main Block"
    equipment: frozenset[str] = frozenset()


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
    equipment_needed: frozenset[str] = frozenset()


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
    weight: float = 1.0


@dataclass
class EvaluationResult:
    hard_violations: List[Violation]
    soft_violations: List[Violation]
    hard_score: int
    soft_score: int
    total_score: int
    metrics: Dict[str, float]


@dataclass
class ConstraintConfig:
    lecturer_double_booking: bool = True
    room_exclusivity: bool = True
    room_capacity_enforced: bool = True
    room_type_enforced: bool = True
    fatigue_limit_enabled: bool = True
    max_consecutive_sessions: int = 3
    live_conflict_highlighting: bool = True


@dataclass
class SoftConstraintWeights:
    preferred_slot: float = 3.0
    minimize_idle_time: float = 2.0
    geographic_grouping: float = 1.5
    fatigue_balance: float = 2.2
    room_fit: float = 1.4


@dataclass
class LecturerPreference:
    lecturer_id: str
    request_text: str
    preferred_slots: List[str] = field(default_factory=list)
    preferred_days: List[str] = field(default_factory=list)


@dataclass
class DisruptionEvent:
    disruption_type: str
    target_id: str
    slot_id: str
    note: str = ""


@dataclass
class FeedbackEntry:
    rating: int
    comment: str


@dataclass
class ApprovalRecord:
    approved: bool = False
    approver: str = "Central Administrator"
    note: str = ""


@dataclass
class AgentEvent:
    progress: int
    agent: str
    action: str
    detail: str
    tone: str = "normal"


@dataclass
class AuditEntry:
    title: str
    explanation: str
    impact: str


@dataclass
class ResolutionSuggestion:
    session_id: str
    message: str
    alternatives: List[str] = field(default_factory=list)


@dataclass
class ProblemData:
    courses: Dict[str, Course]
    lecturers: Dict[str, Lecturer]
    rooms: Dict[str, Room]
    slots: Dict[str, TimeSlot]
    session_requests: List[SessionRequest]
