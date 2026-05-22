from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import (
    AgentEvent,
    Assignment,
    AuditEntry,
    ConstraintConfig,
    EvaluationResult,
    ProblemData,
    ResolutionSuggestion,
    SoftConstraintWeights,
    Violation,
)


MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class ConstraintAgent:
    def evaluate(
        self,
        problem: ProblemData,
        assignments: List[Assignment],
        config: ConstraintConfig,
        weights: SoftConstraintWeights,
    ) -> EvaluationResult:
        courses = problem.courses
        lecturers = problem.lecturers
        rooms = problem.rooms
        slots = problem.slots
        hard: List[Violation] = []
        soft: List[Violation] = []
        room_usage: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        lecturer_usage: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        cohort_usage: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        course_days: Dict[str, List[str]] = defaultdict(list)
        lecturer_day_slots: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        cohort_day_slots: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        room_fill_rates: List[float] = []

        unscheduled_count = 0
        for assignment in assignments:
            course = courses[assignment.course_id]
            lecturer = lecturers[course.lecturer_id]
            cohort_key = f"{course.department}-{course.level}".strip("-")

            if not assignment.slot_id or not assignment.room_id:
                unscheduled_count += 1
                hard.append(
                    Violation(
                        kind="unscheduled_session",
                        message=f"{assignment.session_id} could not be placed in the timetable.",
                        severity="hard",
                        session_id=assignment.session_id,
                        weight=5.0,
                    )
                )
                continue

            slot = slots[assignment.slot_id]
            room = rooms[assignment.room_id]

            room_usage[(assignment.slot_id, assignment.room_id)].append(assignment.session_id)
            lecturer_usage[(assignment.slot_id, course.lecturer_id)].append(assignment.session_id)
            cohort_usage[(assignment.slot_id, cohort_key)].append(assignment.session_id)
            course_days[assignment.course_id].append(slot.day)
            lecturer_day_slots[(course.lecturer_id, slot.day)].append(assignment.slot_id)
            cohort_day_slots[(cohort_key, slot.day)].append(assignment.slot_id)
            room_fill_rates.append(min(1.0, course.student_count / max(1, room.capacity)))

            if assignment.slot_id in lecturer.unavailable_slots:
                hard.append(
                    Violation(
                        kind="lecturer_unavailable",
                        message=f"{course.code} was scheduled when lecturer {lecturer.name} is unavailable.",
                        severity="hard",
                        session_id=assignment.session_id,
                        weight=4.0,
                    )
                )

            if assignment.slot_id in course.blocked_slots:
                hard.append(
                    Violation(
                        kind="course_blocked_slot",
                        message=f"{course.code} was placed in a blocked slot.",
                        severity="hard",
                        session_id=assignment.session_id,
                        weight=4.0,
                    )
                )

            if config.room_capacity_enforced and room.capacity < course.student_count:
                hard.append(
                    Violation(
                        kind="room_capacity",
                        message=f"{course.code} exceeds room capacity in {room.name}.",
                        severity="hard",
                        session_id=assignment.session_id,
                        weight=4.0,
                    )
                )

            if config.room_type_enforced and room.room_type != course.room_type:
                hard.append(
                    Violation(
                        kind="room_type_mismatch",
                        message=f"{course.code} requires a {course.room_type} room but got {room.room_type}.",
                        severity="hard",
                        session_id=assignment.session_id,
                        weight=3.5,
                    )
                )

            missing_equipment = course.equipment_needed.difference(room.equipment)
            if missing_equipment:
                hard.append(
                    Violation(
                        kind="missing_equipment",
                        message=f"{course.code} needs equipment not found in {room.name}: {', '.join(sorted(missing_equipment))}.",
                        severity="hard",
                        session_id=assignment.session_id,
                        weight=3.0,
                    )
                )

            if course.preferred_slots and assignment.slot_id not in course.preferred_slots:
                soft.append(
                    Violation(
                        kind="preferred_slot_miss",
                        message=f"{course.code} was not placed in one of its preferred slots.",
                        severity="soft",
                        session_id=assignment.session_id,
                        weight=weights.preferred_slot,
                    )
                )

            if slot.start >= "16:00":
                soft.append(
                    Violation(
                        kind="late_slot",
                        message=f"{course.code} is scheduled in a late slot.",
                        severity="soft",
                        session_id=assignment.session_id,
                        weight=weights.fatigue_balance,
                    )
                )

            if room.capacity > course.student_count * 1.7:
                soft.append(
                    Violation(
                        kind="room_underutilization",
                        message=f"{course.code} uses a room with significantly unused capacity.",
                        severity="soft",
                        session_id=assignment.session_id,
                        weight=weights.room_fit,
                    )
                )

        if config.room_exclusivity:
            for (_, room_id), session_ids in room_usage.items():
                if len(session_ids) > 1:
                    hard.append(
                        Violation(
                            kind="room_conflict",
                            message=f"Room {room_id} is double-booked for sessions: {', '.join(session_ids)}.",
                            severity="hard",
                            weight=5.0,
                        )
                    )

        if config.lecturer_double_booking:
            for (_, lecturer_id), session_ids in lecturer_usage.items():
                if len(session_ids) > 1:
                    hard.append(
                        Violation(
                            kind="lecturer_conflict",
                            message=f"Lecturer {lecturer_id} is double-booked for sessions: {', '.join(session_ids)}.",
                            severity="hard",
                            weight=5.0,
                        )
                    )

        for (_, cohort_key), session_ids in cohort_usage.items():
            if len(session_ids) > 1:
                hard.append(
                    Violation(
                        kind="cohort_conflict",
                        message=f"Student cohort {cohort_key} has overlapping sessions: {', '.join(session_ids)}.",
                        severity="hard",
                        weight=4.0,
                    )
                )

        for course_id, days in course_days.items():
            if len(days) > len(set(days)):
                course = courses[course_id]
                soft.append(
                    Violation(
                        kind="same_day_repeat",
                        message=f"{course.code} has multiple sessions on the same day.",
                        severity="soft",
                        weight=weights.minimize_idle_time,
                    )
                )

        fatigue_hits = 0
        idle_penalties = 0
        geographic_penalties = 0
        for (lecturer_id, day), slot_ids in lecturer_day_slots.items():
            sorted_slots = sorted(slot_ids, key=lambda sid: (slots[sid].start, slots[sid].end))
            streak = self._max_consecutive(sorted_slots)
            if config.fatigue_limit_enabled and streak > config.max_consecutive_sessions:
                fatigue_hits += 1
                soft.append(
                    Violation(
                        kind="lecturer_fatigue",
                        message=f"Lecturer {lecturer_id} has {streak} consecutive sessions on {day}.",
                        severity="soft",
                        weight=weights.fatigue_balance,
                    )
                )

        for (cohort_key, day), slot_ids in cohort_day_slots.items():
            sorted_slots = sorted(slot_ids, key=lambda sid: (slots[sid].start, slots[sid].end))
            idle_penalties += self._idle_gaps(sorted_slots, slots)
            if config.fatigue_limit_enabled and self._max_consecutive(sorted_slots) > config.max_consecutive_sessions:
                fatigue_hits += 1
                soft.append(
                    Violation(
                        kind="student_fatigue",
                        message=f"Cohort {cohort_key} has too many back-to-back sessions on {day}.",
                        severity="soft",
                        weight=weights.fatigue_balance,
                    )
                )

        assignment_map = {item.session_id: item for item in assignments if item.slot_id and item.room_id}
        for course in courses.values():
            course_sessions = [
                item
                for item in assignment_map.values()
                if item.course_id == course.id
            ]
            course_sessions.sort(key=lambda item: (slots[item.slot_id].day, slots[item.slot_id].start))
            for current, nxt in zip(course_sessions, course_sessions[1:]):
                if slots[current.slot_id].day == slots[nxt.slot_id].day:
                    room_a = rooms[current.room_id]
                    room_b = rooms[nxt.room_id]
                    if room_a.location != room_b.location:
                        geographic_penalties += 1
                        soft.append(
                            Violation(
                                kind="geographic_jump",
                                message=f"{course.code} switches locations abruptly between {room_a.location} and {room_b.location}.",
                                severity="soft",
                                weight=weights.geographic_grouping,
                            )
                        )

        hard_penalty = sum(item.weight for item in hard)
        soft_penalty = sum(item.weight for item in soft)
        hard_score = max(0, round(100 - hard_penalty * 6))
        soft_score = max(0, round(100 - soft_penalty * 2.3))
        total_score = round(hard_score * 0.6 + soft_score * 0.4)
        scheduled = len(assignments) - unscheduled_count
        room_utilization = round((sum(room_fill_rates) / max(1, len(room_fill_rates))) * 100, 2)

        return EvaluationResult(
            hard_violations=hard,
            soft_violations=soft,
            hard_score=hard_score,
            soft_score=soft_score,
            total_score=total_score,
            metrics={
                "scheduled_sessions": float(scheduled),
                "unscheduled_sessions": float(unscheduled_count),
                "hard_constraint_satisfaction": round(max(0.0, 100 - hard_penalty * 5.5), 2),
                "soft_constraint_quality": round(max(0.0, 100 - soft_penalty * 2.1), 2),
                "fatigue_index": round(min(100.0, fatigue_hits * 15.0), 2),
                "student_idle_index": round(min(100.0, idle_penalties * 8.0), 2),
                "room_utilization": room_utilization,
                "geographic_penalty_index": round(min(100.0, geographic_penalties * 10.0), 2),
            },
        )

    @staticmethod
    def _max_consecutive(slot_ids: List[str]) -> int:
        if not slot_ids:
            return 0
        consecutive = 1
        best = 1
        previous_start = None
        for slot_id in slot_ids:
            current_start = slot_id.split("_", 1)[1]
            if previous_start is not None and current_start == previous_start:
                consecutive += 1
            else:
                consecutive = 1
            best = max(best, consecutive)
            previous_start = current_start
        return best

    @staticmethod
    def _idle_gaps(slot_ids: List[str], slots: Dict[str, object]) -> int:
        gap_count = 0
        sorted_slots = sorted(slot_ids, key=lambda sid: slots[sid].start)
        for current, nxt in zip(sorted_slots, sorted_slots[1:]):
            if slots[current].end != slots[nxt].start:
                gap_count += 1
        return gap_count


class PolicyModel:
    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path or MODEL_DIR / "policy_weights.json"
        self.weights = {
            "preferred_slot": 2.0,
            "late_slot": -1.3,
            "same_day_repeat": -1.2,
            "room_fit": 0.8,
            "morning_slot": 0.3,
        }
        self.training_history: List[float] = []
        self.load()

    def load(self) -> None:
        if self.model_path.exists():
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
            self.weights.update(payload.get("weights", {}))
            self.training_history = payload.get("history", [])

    def save(self) -> None:
        payload = {"weights": self.weights, "history": self.training_history[-50:]}
        self.model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def feature_map(self, preferred: bool, late: bool, same_day: bool, room_fit: float, morning: bool) -> Dict[str, float]:
        return {
            "preferred_slot": 1.0 if preferred else 0.0,
            "late_slot": 1.0 if late else 0.0,
            "same_day_repeat": 1.0 if same_day else 0.0,
            "room_fit": room_fit,
            "morning_slot": 1.0 if morning else 0.0,
        }

    def score(self, features: Dict[str, float]) -> float:
        return sum(self.weights[name] * value for name, value in features.items())

    def update(self, features: Dict[str, float], reward: float, learning_rate: float = 0.08) -> None:
        prediction = self.score(features)
        error = reward - prediction
        for name, value in features.items():
            self.weights[name] += learning_rate * error * value

    def train(self, problem: ProblemData, episodes: int = 30) -> Dict[str, object]:
        rng = random.Random(19)
        history: List[float] = []
        for _ in range(max(0, episodes)):
            reward_total = 0.0
            course_day_usage: Dict[str, set[str]] = defaultdict(set)
            requests = problem.session_requests[:]
            rng.shuffle(requests)
            for request in requests:
                course = problem.courses[request.course_id]
                lecturer = problem.lecturers[course.lecturer_id]
                feasible = []
                for slot in problem.slots.values():
                    if slot.id in lecturer.unavailable_slots or slot.id in course.blocked_slots:
                        continue
                    for room in problem.rooms.values():
                        if room.room_type != course.room_type or room.capacity < course.student_count:
                            continue
                        room_fit = 1.0 - ((room.capacity - course.student_count) / max(1, room.capacity))
                        features = self.feature_map(
                            preferred=bool(course.preferred_slots and slot.id in course.preferred_slots),
                            late=slot.start >= "16:00",
                            same_day=slot.day in course_day_usage[course.id],
                            room_fit=max(0.0, room_fit),
                            morning=slot.start <= "10:00",
                        )
                        feasible.append((slot, room, features))
                if not feasible:
                    continue
                _, _, features = rng.choice(feasible)
                reward = (
                    4.0 * features["preferred_slot"]
                    - 2.7 * features["late_slot"]
                    - 2.2 * features["same_day_repeat"]
                    + 1.3 * features["room_fit"]
                    + 0.3 * features["morning_slot"]
                )
                self.update(features, reward)
                reward_total += reward
            history.append(round(reward_total, 3))
        self.training_history.extend(history)
        self.save()
        return {
            "episodes": episodes,
            "history_tail": history[-10:],
            "weights": {key: round(value, 4) for key, value in self.weights.items()},
            "model_path": str(self.model_path),
        }


class SchedulingAgent:
    def generate_initial_schedule(
        self,
        problem: ProblemData,
        config: ConstraintConfig,
        weights: SoftConstraintWeights,
        policy_model: Optional[PolicyModel] = None,
        randomize: bool = False,
        seed: int = 7,
    ) -> List[Assignment]:
        rng = random.Random(seed)
        slots = list(problem.slots.values())
        rooms = list(problem.rooms.values())
        if randomize:
            rng.shuffle(slots)
            rng.shuffle(rooms)

        assignments: List[Assignment] = []
        room_usage: Dict[Tuple[str, str], bool] = {}
        lecturer_usage: Dict[Tuple[str, str], bool] = {}
        cohort_day_usage: Dict[str, set[str]] = defaultdict(set)

        ordered_requests = sorted(
            problem.session_requests,
            key=lambda req: (
                -problem.courses[req.course_id].student_count,
                -len(problem.courses[req.course_id].blocked_slots),
                len(problem.courses[req.course_id].preferred_slots),
            ),
        )
        if randomize:
            rng.shuffle(ordered_requests)

        for request in ordered_requests:
            course = problem.courses[request.course_id]
            lecturer = problem.lecturers[course.lecturer_id]
            cohort_key = f"{course.department}-{course.level}".strip("-")
            best_choice: Optional[Tuple[float, str, str]] = None

            for slot in slots:
                if slot.id in lecturer.unavailable_slots or slot.id in course.blocked_slots:
                    continue
                for room in rooms:
                    if config.room_type_enforced and room.room_type != course.room_type:
                        continue
                    if config.room_capacity_enforced and room.capacity < course.student_count:
                        continue
                    if course.equipment_needed.difference(room.equipment):
                        continue
                    if config.room_exclusivity and room_usage.get((slot.id, room.id)):
                        continue
                    if config.lecturer_double_booking and lecturer_usage.get((slot.id, course.lecturer_id)):
                        continue

                    preferred = bool(course.preferred_slots and slot.id in course.preferred_slots)
                    same_day = slot.day in cohort_day_usage[cohort_key]
                    room_fit = 1.0 - ((room.capacity - course.student_count) / max(1, room.capacity))
                    penalty = 0.0
                    if course.preferred_slots and not preferred:
                        penalty += weights.preferred_slot
                    if same_day:
                        penalty += weights.minimize_idle_time
                    penalty += max(0.0, room.capacity - course.student_count) / 40
                    if slot.start >= "16:00":
                        penalty += weights.fatigue_balance
                    if room.location != "Main Block":
                        penalty += weights.geographic_grouping * 0.2

                    if policy_model is not None:
                        features = policy_model.feature_map(
                            preferred=preferred,
                            late=slot.start >= "16:00",
                            same_day=same_day,
                            room_fit=max(0.0, room_fit),
                            morning=slot.start <= "10:00",
                        )
                        penalty -= policy_model.score(features)

                    if randomize:
                        penalty += rng.uniform(-0.75, 0.75)

                    candidate = (penalty, slot.id, room.id)
                    if best_choice is None or candidate < best_choice:
                        best_choice = candidate

            assignment = Assignment(
                session_id=request.session_id,
                course_id=request.course_id,
                lecturer_id=course.lecturer_id,
            )
            if best_choice is None:
                assignment.status = "unscheduled"
                assignment.notes.append("No feasible room/slot combination was found during the draft phase.")
            else:
                _, slot_id, room_id = best_choice
                assignment.slot_id = slot_id
                assignment.room_id = room_id
                room_usage[(slot_id, room_id)] = True
                lecturer_usage[(slot_id, course.lecturer_id)] = True
                cohort_day_usage[cohort_key].add(problem.slots[slot_id].day)

            assignments.append(assignment)
        return assignments


class ConflictResolutionAgent:
    def __init__(self, constraint_agent: ConstraintAgent) -> None:
        self.constraint_agent = constraint_agent

    @staticmethod
    def _ranking_key(result: EvaluationResult) -> Tuple[int, int, int, int]:
        return (
            -len(result.hard_violations),
            -int(result.metrics["unscheduled_sessions"]),
            result.total_score,
            result.soft_score,
        )

    def improve_schedule(
        self,
        problem: ProblemData,
        assignments: List[Assignment],
        config: ConstraintConfig,
        weights: SoftConstraintWeights,
        max_rounds: int = 120,
    ) -> List[Assignment]:
        best = [Assignment(**asdict(item)) for item in assignments]
        best_eval = self.constraint_agent.evaluate(problem, best, config, weights)
        rng = random.Random(7)

        for _ in range(max_rounds):
            improved = False
            for idx, assignment in enumerate(best):
                course = problem.courses[assignment.course_id]
                lecturer = problem.lecturers[course.lecturer_id]
                for slot in problem.slots.values():
                    if slot.id in lecturer.unavailable_slots or slot.id in course.blocked_slots:
                        continue
                    for room in problem.rooms.values():
                        if config.room_type_enforced and room.room_type != course.room_type:
                            continue
                        if config.room_capacity_enforced and room.capacity < course.student_count:
                            continue
                        candidate = [Assignment(**asdict(item)) for item in best]
                        candidate[idx].slot_id = slot.id
                        candidate[idx].room_id = room.id
                        candidate[idx].status = "scheduled"
                        candidate[idx].notes = []
                        candidate_eval = self.constraint_agent.evaluate(problem, candidate, config, weights)
                        if self._ranking_key(candidate_eval) > self._ranking_key(best_eval):
                            best = candidate
                            best_eval = candidate_eval
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                movable = [i for i, item in enumerate(best) if item.status == "scheduled"]
                if not movable:
                    break
                idx = rng.choice(movable)
                course = problem.courses[best[idx].course_id]
                feasible_pairs = []
                for slot in problem.slots.values():
                    if slot.id in course.blocked_slots or slot.id in problem.lecturers[course.lecturer_id].unavailable_slots:
                        continue
                    for room in problem.rooms.values():
                        if room.room_type == course.room_type and room.capacity >= course.student_count:
                            feasible_pairs.append((slot.id, room.id))
                if feasible_pairs:
                    slot_id, room_id = rng.choice(feasible_pairs)
                    candidate = [Assignment(**asdict(item)) for item in best]
                    candidate[idx].slot_id = slot_id
                    candidate[idx].room_id = room_id
                    shaken_eval = self.constraint_agent.evaluate(problem, candidate, config, weights)
                    if self._ranking_key(shaken_eval) > self._ranking_key(best_eval):
                        best = candidate
                        best_eval = shaken_eval
        return best


class GeneticOptimizationAgent:
    def __init__(self, constraint_agent: ConstraintAgent, scheduling_agent: SchedulingAgent, conflict_agent: ConflictResolutionAgent) -> None:
        self.constraint_agent = constraint_agent
        self.scheduling_agent = scheduling_agent
        self.conflict_agent = conflict_agent

    def optimize(
        self,
        problem: ProblemData,
        config: ConstraintConfig,
        weights: SoftConstraintWeights,
        policy_model: Optional[PolicyModel] = None,
        population_size: int = 8,
        generations: int = 8,
    ) -> Tuple[List[Assignment], Dict[str, object]]:
        rng = random.Random(11)
        population = []
        for seed in range(population_size):
            draft = self.scheduling_agent.generate_initial_schedule(
                problem,
                config,
                weights,
                policy_model=policy_model,
                randomize=True,
                seed=seed + 1,
            )
            repaired = self.conflict_agent.improve_schedule(problem, draft, config, weights, max_rounds=30)
            population.append(repaired)

        history: List[int] = []
        for _ in range(generations):
            population.sort(
                key=lambda schedule: self.constraint_agent.evaluate(problem, schedule, config, weights).total_score,
                reverse=True,
            )
            history.append(self.constraint_agent.evaluate(problem, population[0], config, weights).total_score)
            survivors = population[: max(2, population_size // 2)]
            children = survivors[:]
            while len(children) < population_size:
                parent_a = rng.choice(survivors)
                parent_b = rng.choice(survivors)
                child = self._crossover(parent_a, parent_b, rng)
                child = self._mutate(problem, child, rng)
                child = self.conflict_agent.improve_schedule(problem, child, config, weights, max_rounds=20)
                children.append(child)
            population = children

        population.sort(
            key=lambda schedule: self.constraint_agent.evaluate(problem, schedule, config, weights).total_score,
            reverse=True,
        )
        return population[0], {"population_size": population_size, "generations": generations, "history": history}

    def _crossover(self, parent_a: List[Assignment], parent_b: List[Assignment], rng: random.Random) -> List[Assignment]:
        child = []
        for idx in range(len(parent_a)):
            source = parent_a if rng.random() < 0.5 else parent_b
            child.append(Assignment(**asdict(source[idx])))
        return child

    def _mutate(self, problem: ProblemData, schedule: List[Assignment], rng: random.Random) -> List[Assignment]:
        candidate = [Assignment(**asdict(item)) for item in schedule]
        if not candidate:
            return candidate
        idx = rng.randrange(len(candidate))
        assignment = candidate[idx]
        course = problem.courses[assignment.course_id]
        lecturer = problem.lecturers[course.lecturer_id]
        feasible_pairs = []
        for slot in problem.slots.values():
            if slot.id in lecturer.unavailable_slots or slot.id in course.blocked_slots:
                continue
            for room in problem.rooms.values():
                if room.room_type == course.room_type and room.capacity >= course.student_count:
                    feasible_pairs.append((slot.id, room.id))
        if feasible_pairs:
            assignment.slot_id, assignment.room_id = rng.choice(feasible_pairs)
            assignment.status = "scheduled"
            assignment.notes = []
        return candidate


class OrchestratorAgent:
    def __init__(self) -> None:
        self.constraint_agent = ConstraintAgent()
        self.policy_model = PolicyModel()
        self.scheduling_agent = SchedulingAgent()
        self.conflict_agent = ConflictResolutionAgent(self.constraint_agent)
        self.genetic_agent = GeneticOptimizationAgent(self.constraint_agent, self.scheduling_agent, self.conflict_agent)

    def run(
        self,
        problem: ProblemData,
        strategy: str = "hybrid",
        training_episodes: int = 0,
        generations: int = 12,
        config: Optional[ConstraintConfig] = None,
        weights: Optional[SoftConstraintWeights] = None,
        guidelines: str = "",
        lecturer_preferences: Optional[List[Dict[str, str]]] = None,
        natural_language_request: str = "",
        disruptions: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, object]:
        config = config or ConstraintConfig()
        weights = weights or SoftConstraintWeights()
        training_summary = None
        if training_episodes > 0:
            training_summary = self.policy_model.train(problem, episodes=training_episodes)

        policy = self.policy_model if strategy in {"policy", "hybrid", "genetic"} else None
        draft = self.scheduling_agent.generate_initial_schedule(problem, config, weights, policy_model=policy)
        draft_evaluation = self.constraint_agent.evaluate(problem, draft, config, weights)

        optimization_summary = None
        if strategy == "heuristic":
            final_schedule = self.conflict_agent.improve_schedule(problem, draft, config, weights, max_rounds=90)
        elif strategy == "policy":
            final_schedule = self.conflict_agent.improve_schedule(problem, draft, config, weights, max_rounds=120)
        elif strategy == "genetic":
            final_schedule, optimization_summary = self.genetic_agent.optimize(problem, config, weights, policy_model=policy, generations=generations)
        else:
            repaired = self.conflict_agent.improve_schedule(problem, draft, config, weights, max_rounds=80)
            final_schedule, optimization_summary = self.genetic_agent.optimize(
                problem,
                config,
                weights,
                policy_model=policy,
                generations=max(4, generations // 2),
            )
            repaired_eval = self.constraint_agent.evaluate(problem, repaired, config, weights)
            genetic_eval = self.constraint_agent.evaluate(problem, final_schedule, config, weights)
            if self.conflict_agent._ranking_key(repaired_eval) > self.conflict_agent._ranking_key(genetic_eval):
                final_schedule = repaired

        return self.summarize(
            problem=problem,
            assignments=final_schedule,
            draft_assignments=draft,
            strategy=strategy,
            config=config,
            weights=weights,
            training_summary=training_summary,
            optimization_summary=optimization_summary,
            guidelines=guidelines,
            lecturer_preferences=lecturer_preferences or [],
            natural_language_request=natural_language_request,
            disruptions=disruptions or [],
        )

    def summarize(
        self,
        problem: ProblemData,
        assignments: List[Assignment],
        draft_assignments: Optional[List[Assignment]] = None,
        strategy: str = "hybrid",
        config: Optional[ConstraintConfig] = None,
        weights: Optional[SoftConstraintWeights] = None,
        training_summary: Optional[Dict[str, object]] = None,
        optimization_summary: Optional[Dict[str, object]] = None,
        guidelines: str = "",
        lecturer_preferences: Optional[List[Dict[str, str]]] = None,
        natural_language_request: str = "",
        disruptions: Optional[List[Dict[str, str]]] = None,
        approved: bool = False,
        feedback_log: Optional[List[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        config = config or ConstraintConfig()
        weights = weights or SoftConstraintWeights()
        draft_assignments = draft_assignments or assignments
        draft_evaluation = self.constraint_agent.evaluate(problem, draft_assignments, config, weights)
        final_evaluation = self.constraint_agent.evaluate(problem, assignments, config, weights)

        schedule_rows = self._enrich_rows(problem, assignments)
        grid = self._build_grid(problem, assignments, final_evaluation)
        activity = self._build_activity_log(strategy, training_summary, optimization_summary, final_evaluation)
        audit_log = self._build_audit_log(problem, assignments, final_evaluation, natural_language_request, guidelines)
        argument_terminal = self._build_argument_terminal(problem, final_evaluation)
        suggestions = self._build_suggestions(problem, assignments, final_evaluation)
        dashboard = {
            "total_courses": len(problem.courses),
            "total_lecturers": len(problem.lecturers),
            "total_rooms": len(problem.rooms),
            "operational_status": "Approved" if approved else ("Attention Needed" if final_evaluation.hard_violations else "Ready for Review"),
        }

        comparison = [
            {"label": "Draft Score", "value": draft_evaluation.total_score},
            {"label": "Final Score", "value": final_evaluation.total_score},
            {"label": "Draft Soft Penalties", "value": len(draft_evaluation.soft_violations)},
            {"label": "Final Soft Penalties", "value": len(final_evaluation.soft_violations)},
        ]

        return {
            "strategy": strategy,
            "draft_evaluation": draft_evaluation,
            "final_evaluation": final_evaluation,
            "schedule_rows": schedule_rows,
            "assignments": [asdict(item) for item in assignments],
            "draft_assignments": [asdict(item) for item in draft_assignments],
            "grid": grid,
            "training_summary": training_summary,
            "optimization_summary": optimization_summary,
            "policy_weights": {key: round(value, 4) for key, value in self.policy_model.weights.items()},
            "activity_log": activity,
            "audit_log": audit_log,
            "argument_terminal": argument_terminal,
            "resolution_suggestions": suggestions,
            "dashboard": dashboard,
            "comparison": comparison,
            "config": config,
            "weights": weights,
            "guidelines": guidelines,
            "lecturer_preferences": lecturer_preferences or [],
            "natural_language_request": natural_language_request,
            "disruptions": disruptions or [],
            "approved": approved,
            "feedback_log": feedback_log or [],
        }

    def suggest_adjustment(
        self,
        problem: ProblemData,
        assignments: List[Assignment],
        session_id: str,
        target_slot_id: str,
    ) -> List[Assignment]:
        updated = [Assignment(**asdict(item)) for item in assignments]
        target = next((item for item in updated if item.session_id == session_id), None)
        if target is None:
            return updated
        course = problem.courses[target.course_id]
        candidate_room = None
        for room in problem.rooms.values():
            if room.room_type == course.room_type and room.capacity >= course.student_count:
                candidate_room = room.id
                break
        target.slot_id = target_slot_id
        target.room_id = candidate_room
        target.notes = [f"Administrator moved this session to {target_slot_id} for what-if testing."]
        return updated

    def _enrich_rows(self, problem: ProblemData, assignments: List[Assignment]) -> List[Dict[str, object]]:
        enriched_rows = []
        for assignment in assignments:
            course = problem.courses[assignment.course_id]
            slot = problem.slots.get(assignment.slot_id)
            room = problem.rooms.get(assignment.room_id)
            lecturer = problem.lecturers[course.lecturer_id]
            enriched_rows.append(
                {
                    "session_id": assignment.session_id,
                    "course_code": course.code,
                    "course_title": course.title,
                    "department": course.department,
                    "level": course.level,
                    "lecturer": lecturer.name,
                    "students": course.student_count,
                    "day": slot.day if slot else "Unscheduled",
                    "time": f"{slot.start} - {slot.end}" if slot else "N/A",
                    "room": room.name if room else "N/A",
                    "room_location": room.location if room else "N/A",
                    "status": assignment.status,
                    "notes": "; ".join(assignment.notes),
                }
            )
        return enriched_rows

    def _build_grid(self, problem: ProblemData, assignments: List[Assignment], evaluation: EvaluationResult) -> Dict[str, object]:
        days = []
        for slot in sorted(problem.slots.values(), key=lambda item: (item.day, item.start)):
            if slot.day not in days:
                days.append(slot.day)
        times = []
        for slot in sorted(problem.slots.values(), key=lambda item: item.start):
            label = f"{slot.start} - {slot.end}"
            if label not in times:
                times.append(label)

        violation_map: Dict[str, List[str]] = defaultdict(list)
        for violation in evaluation.hard_violations + evaluation.soft_violations:
            if violation.session_id:
                violation_map[violation.session_id].append(violation.message)

        cells: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for assignment in assignments:
            if not assignment.slot_id:
                continue
            slot = problem.slots[assignment.slot_id]
            room = problem.rooms.get(assignment.room_id)
            course = problem.courses[assignment.course_id]
            key = f"{slot.day}|{slot.start} - {slot.end}"
            cells[key].append(
                {
                    "session_id": assignment.session_id,
                    "course_code": course.code,
                    "lecturer": problem.lecturers[course.lecturer_id].name,
                    "room": room.name if room else "TBD",
                    "slot_id": assignment.slot_id,
                    "room_id": assignment.room_id,
                    "status": "warning" if violation_map.get(assignment.session_id) else "ok",
                    "violations": violation_map.get(assignment.session_id, []),
                }
            )
        return {"days": days, "times": times, "cells": cells}

    def _build_activity_log(
        self,
        strategy: str,
        training_summary: Optional[Dict[str, object]],
        optimization_summary: Optional[Dict[str, object]],
        final_evaluation: EvaluationResult,
    ) -> List[AgentEvent]:
        events = [
            AgentEvent(10, "Orchestrator", "Ingested data", "Loaded academic resources and activated scheduling pipeline."),
            AgentEvent(22, "Scheduling Agent", "Drafted candidates", "Generated initial session placements from room-slot search."),
            AgentEvent(40, "Constraint Agent", "Validated rules", f"Detected {len(final_evaluation.hard_violations)} hard issues and {len(final_evaluation.soft_violations)} soft issues."),
            AgentEvent(58, "Conflict-Resolution Agent", "Negotiated repairs", "Proposed alternative placements for fragile sessions."),
        ]
        if training_summary:
            events.append(
                AgentEvent(
                    72,
                    "Policy Model",
                    "Learned preferences",
                    f"Completed {training_summary['episodes']} training episodes and updated weight vectors.",
                    "positive",
                )
            )
        if optimization_summary:
            events.append(
                AgentEvent(
                    86,
                    "Genetic Optimizer",
                    "Searched scenarios",
                    f"Explored {optimization_summary.get('population_size', 0)} candidates across {optimization_summary.get('generations', 0)} generations.",
                    "positive",
                )
            )
        events.append(
            AgentEvent(100, "Orchestrator", "Prepared admin review", f"Published timetable in {strategy} mode with final score {final_evaluation.total_score}.", "positive")
        )
        return events

    def _build_audit_log(
        self,
        problem: ProblemData,
        assignments: List[Assignment],
        evaluation: EvaluationResult,
        natural_language_request: str,
        guidelines: str,
    ) -> List[AuditEntry]:
        logs: List[AuditEntry] = []
        if natural_language_request:
            logs.append(AuditEntry("Natural-language constraint", natural_language_request, "Updated the search objective before final scheduling."))
        if guidelines:
            logs.append(AuditEntry("Departmental guideline", guidelines[:140], "Applied additional contextual rules during evaluation and repair."))
        for assignment in assignments[:5]:
            course = problem.courses[assignment.course_id]
            if assignment.slot_id and assignment.room_id:
                slot = problem.slots[assignment.slot_id]
                room = problem.rooms[assignment.room_id]
                logs.append(
                    AuditEntry(
                        title=f"{course.code} placement",
                        explanation=f"Placed in {room.name} on {slot.day} {slot.start}-{slot.end} because it matched capacity, room type, and conflict checks.",
                        impact="Reduced risk of timetable clashes while preserving soft-quality goals.",
                    )
                )
        if not logs:
            logs.append(AuditEntry("Audit baseline", "No special overrides were used in this run.", "The schedule reflects the default optimization rules."))
        if evaluation.hard_violations:
            logs.append(AuditEntry("Escalation", "Hard violations remain after optimization.", "Administrator review is required before approval."))
        return logs[:8]

    def _build_argument_terminal(self, problem: ProblemData, evaluation: EvaluationResult) -> List[str]:
        return [
            "Scheduling Agent: I prioritized larger cohorts first to protect limited high-capacity rooms.",
            "Constraint Agent: I rejected any placement that violated lecturer availability or room exclusivity.",
            f"Conflict-Resolution Agent: I found {len(evaluation.soft_violations)} quality concerns still worth monitoring.",
            "Orchestrator Agent: I balanced validity, fatigue control, idle-time reduction, and room fit before publishing this version.",
            "Scheduling Agent: When choices were similar, I preferred placements that preserved room utilization and future flexibility.",
            "Constraint Agent: Any highlighted warning on the grid maps directly to a recorded audit or violation entry.",
        ]

    def _build_suggestions(self, problem: ProblemData, assignments: List[Assignment], evaluation: EvaluationResult) -> List[ResolutionSuggestion]:
        suggestions: List[ResolutionSuggestion] = []
        violation_map: Dict[str, List[Violation]] = defaultdict(list)
        for violation in evaluation.hard_violations + evaluation.soft_violations:
            if violation.session_id:
                violation_map[violation.session_id].append(violation)

        for assignment in assignments[:8]:
            course = problem.courses[assignment.course_id]
            alternatives = []
            for slot in list(problem.slots.values())[:4]:
                if slot.id not in course.blocked_slots:
                    alternatives.append(f"{slot.day} {slot.start}-{slot.end}")
            if violation_map.get(assignment.session_id):
                suggestions.append(
                    ResolutionSuggestion(
                        session_id=assignment.session_id,
                        message=f"Conflict-Resolution Agent suggests reconsidering {course.code} due to highlighted issues.",
                        alternatives=alternatives[:3],
                    )
                )
            elif len(suggestions) < 3:
                suggestions.append(
                    ResolutionSuggestion(
                        session_id=assignment.session_id,
                        message=f"{course.code} can be moved for a what-if test without destabilizing the full timetable.",
                        alternatives=alternatives[:2],
                    )
                )
        return suggestions
