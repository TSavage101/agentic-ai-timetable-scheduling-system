from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import Assignment, EvaluationResult, ProblemData, Violation


MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class ConstraintAgent:
    def evaluate(self, problem: ProblemData, assignments: List[Assignment]) -> EvaluationResult:
        courses = problem.courses
        lecturers = problem.lecturers
        rooms = problem.rooms
        slots = problem.slots
        hard: List[Violation] = []
        soft: List[Violation] = []
        room_usage: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        lecturer_usage: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        course_days: Dict[str, List[str]] = defaultdict(list)
        lecturer_day_slots: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        unscheduled_count = 0
        for assignment in assignments:
            course = courses[assignment.course_id]
            lecturer = lecturers[course.lecturer_id]

            if not assignment.slot_id or not assignment.room_id:
                unscheduled_count += 1
                hard.append(
                    Violation(
                        kind="unscheduled_session",
                        message=f"{assignment.session_id} could not be placed in the timetable.",
                        severity="hard",
                        session_id=assignment.session_id,
                    )
                )
                continue

            slot = slots[assignment.slot_id]
            room = rooms[assignment.room_id]

            room_usage[(assignment.slot_id, assignment.room_id)].append(assignment.session_id)
            lecturer_usage[(assignment.slot_id, course.lecturer_id)].append(assignment.session_id)
            course_days[assignment.course_id].append(slot.day)
            lecturer_day_slots[(course.lecturer_id, slot.day)].append(assignment.slot_id)

            if assignment.slot_id in lecturer.unavailable_slots:
                hard.append(
                    Violation(
                        kind="lecturer_unavailable",
                        message=f"{course.code} was scheduled when lecturer {lecturer.name} is unavailable.",
                        severity="hard",
                        session_id=assignment.session_id,
                    )
                )

            if assignment.slot_id in course.blocked_slots:
                hard.append(
                    Violation(
                        kind="course_blocked_slot",
                        message=f"{course.code} was placed in a blocked slot.",
                        severity="hard",
                        session_id=assignment.session_id,
                    )
                )

            if room.capacity < course.student_count:
                hard.append(
                    Violation(
                        kind="room_capacity",
                        message=f"{course.code} exceeds room capacity in {room.name}.",
                        severity="hard",
                        session_id=assignment.session_id,
                    )
                )

            if room.room_type != course.room_type:
                hard.append(
                    Violation(
                        kind="room_type_mismatch",
                        message=f"{course.code} requires a {course.room_type} room but got {room.room_type}.",
                        severity="hard",
                        session_id=assignment.session_id,
                    )
                )

            if course.preferred_slots and assignment.slot_id not in course.preferred_slots:
                soft.append(
                    Violation(
                        kind="preferred_slot_miss",
                        message=f"{course.code} was not placed in one of its preferred slots.",
                        severity="soft",
                        session_id=assignment.session_id,
                    )
                )

            if slot.start >= "16:00":
                soft.append(
                    Violation(
                        kind="late_slot",
                        message=f"{course.code} is scheduled in a late slot.",
                        severity="soft",
                        session_id=assignment.session_id,
                    )
                )

        for (_, room_id), session_ids in room_usage.items():
            if len(session_ids) > 1:
                hard.append(
                    Violation(
                        kind="room_conflict",
                        message=f"Room {room_id} is double-booked for sessions: {', '.join(session_ids)}.",
                        severity="hard",
                    )
                )

        for (_, lecturer_id), session_ids in lecturer_usage.items():
            if len(session_ids) > 1:
                hard.append(
                    Violation(
                        kind="lecturer_conflict",
                        message=f"Lecturer {lecturer_id} is double-booked for sessions: {', '.join(session_ids)}.",
                        severity="hard",
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
                    )
                )

        for (lecturer_id, day), slot_ids in lecturer_day_slots.items():
            if len(slot_ids) > 3:
                soft.append(
                    Violation(
                        kind="lecturer_overload",
                        message=f"Lecturer {lecturer_id} has a heavy load on {day}.",
                        severity="soft",
                    )
                )

        hard_score = max(0, 100 - len(hard) * 10)
        soft_score = max(0, 100 - len(soft) * 3)
        total_score = int(hard_score * 0.7 + soft_score * 0.3)
        scheduled = len(assignments) - unscheduled_count
        hard_satisfaction = max(0.0, min(100.0, round((1 - (len(hard) / max(1, len(assignments)))) * 100, 2)))
        soft_quality = max(0.0, min(100.0, round((1 - (len(soft) / max(1, len(assignments)))) * 100, 2)))

        return EvaluationResult(
            hard_violations=hard,
            soft_violations=soft,
            hard_score=hard_score,
            soft_score=soft_score,
            total_score=total_score,
            metrics={
                "scheduled_sessions": scheduled,
                "unscheduled_sessions": unscheduled_count,
                "hard_constraint_satisfaction": hard_satisfaction,
                "soft_constraint_quality": soft_quality,
            },
        )


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

    def train(self, problem: ProblemData, episodes: int = 50) -> Dict[str, object]:
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
                slot, room, features = rng.choice(feasible)
                reward = (
                    4.0 * features["preferred_slot"]
                    - 2.5 * features["late_slot"]
                    - 2.0 * features["same_day_repeat"]
                    + 1.5 * features["room_fit"]
                    + 0.3 * features["morning_slot"]
                )
                self.update(features, reward)
                reward_total += reward
                course_day_usage[course.id].add(slot.day)
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
        course_day_usage: Dict[str, set[str]] = defaultdict(set)

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
            best_choice: Optional[Tuple[float, str, str]] = None

            for slot in slots:
                if slot.id in lecturer.unavailable_slots or slot.id in course.blocked_slots:
                    continue
                for room in rooms:
                    if room.room_type != course.room_type or room.capacity < course.student_count:
                        continue
                    if room_usage.get((slot.id, room.id)):
                        continue
                    if lecturer_usage.get((slot.id, course.lecturer_id)):
                        continue

                    penalty = 0.0
                    preferred = bool(course.preferred_slots and slot.id in course.preferred_slots)
                    same_day = slot.day in course_day_usage[course.id]
                    room_fit = 1.0 - ((room.capacity - course.student_count) / max(1, room.capacity))

                    if course.preferred_slots and not preferred:
                        penalty += 3
                    if same_day:
                        penalty += 2
                    penalty += max(0, room.capacity - course.student_count) / 25
                    if slot.start >= "16:00":
                        penalty += 2

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
                course_day_usage[course.id].add(problem.slots[slot_id].day)

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

    def _is_better(self, candidate: EvaluationResult, baseline: EvaluationResult) -> bool:
        return self._ranking_key(candidate) > self._ranking_key(baseline)

    def improve_schedule(self, problem: ProblemData, assignments: List[Assignment], max_rounds: int = 120) -> List[Assignment]:
        best = [Assignment(**asdict(item)) for item in assignments]
        best_eval = self.constraint_agent.evaluate(problem, best)
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
                        if room.room_type != course.room_type or room.capacity < course.student_count:
                            continue
                        candidate = [Assignment(**asdict(item)) for item in best]
                        candidate[idx].slot_id = slot.id
                        candidate[idx].room_id = room.id
                        candidate[idx].status = "scheduled"
                        candidate[idx].notes = []
                        candidate_eval = self.constraint_agent.evaluate(problem, candidate)
                        if self._is_better(candidate_eval, best_eval):
                            best = candidate
                            best_eval = candidate_eval
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            if improved:
                continue

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
                shaken_eval = self.constraint_agent.evaluate(problem, candidate)
                if self._is_better(shaken_eval, best_eval):
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
        policy_model: Optional[PolicyModel] = None,
        population_size: int = 10,
        generations: int = 12,
    ) -> Tuple[List[Assignment], Dict[str, object]]:
        rng = random.Random(11)
        population = []
        for seed in range(population_size):
            draft = self.scheduling_agent.generate_initial_schedule(problem, policy_model=policy_model, randomize=True, seed=seed + 1)
            repaired = self.conflict_agent.improve_schedule(problem, draft, max_rounds=30)
            population.append(repaired)

        history: List[int] = []
        for _ in range(generations):
            population.sort(key=lambda schedule: self.constraint_agent.evaluate(problem, schedule).total_score, reverse=True)
            history.append(self.constraint_agent.evaluate(problem, population[0]).total_score)
            survivors = population[: max(2, population_size // 2)]
            children = survivors[:]

            while len(children) < population_size:
                parent_a = rng.choice(survivors)
                parent_b = rng.choice(survivors)
                child = self._crossover(parent_a, parent_b, rng)
                child = self._mutate(problem, child, rng)
                child = self.conflict_agent.improve_schedule(problem, child, max_rounds=20)
                children.append(child)
            population = children

        population.sort(key=lambda schedule: self.constraint_agent.evaluate(problem, schedule).total_score, reverse=True)
        best = population[0]
        return best, {"population_size": population_size, "generations": generations, "history": history}

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
    ) -> Dict[str, object]:
        training_summary = None
        if training_episodes > 0:
            training_summary = self.policy_model.train(problem, episodes=training_episodes)

        policy = self.policy_model if strategy in {"policy", "hybrid", "genetic"} else None
        draft = self.scheduling_agent.generate_initial_schedule(problem, policy_model=policy)
        draft_evaluation = self.constraint_agent.evaluate(problem, draft)

        optimization_summary = None
        if strategy == "heuristic":
            final_schedule = self.conflict_agent.improve_schedule(problem, draft)
        elif strategy == "policy":
            final_schedule = self.conflict_agent.improve_schedule(problem, draft, max_rounds=140)
        elif strategy == "genetic":
            final_schedule, optimization_summary = self.genetic_agent.optimize(
                problem,
                policy_model=policy,
                population_size=10,
                generations=generations,
            )
        else:
            repaired = self.conflict_agent.improve_schedule(problem, draft, max_rounds=100)
            final_schedule, optimization_summary = self.genetic_agent.optimize(
                problem,
                policy_model=policy,
                population_size=8,
                generations=max(6, generations // 2),
            )
            repaired_eval = self.constraint_agent.evaluate(problem, repaired)
            genetic_eval = self.constraint_agent.evaluate(problem, final_schedule)
            if self.conflict_agent._ranking_key(repaired_eval) > self.conflict_agent._ranking_key(genetic_eval):
                final_schedule = repaired

        final_evaluation = self.constraint_agent.evaluate(problem, final_schedule)

        enriched_rows = []
        for assignment in final_schedule:
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
                    "status": assignment.status,
                    "notes": "; ".join(assignment.notes),
                }
            )

        return {
            "strategy": strategy,
            "draft_evaluation": draft_evaluation,
            "final_evaluation": final_evaluation,
            "schedule_rows": enriched_rows,
            "training_summary": training_summary,
            "optimization_summary": optimization_summary,
            "policy_weights": {key: round(value, 4) for key, value in self.policy_model.weights.items()},
        }
