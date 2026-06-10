from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

from app.agents import ConstraintAgent, OrchestratorAgent, PolicyModel
from app.main import JOB_CACHE, app
from app.io_utils import load_problem_data
from app.models import ConstraintConfig, SoftConstraintWeights


def test_sample_dataset_generates_schedule_without_hard_violations():
    problem = load_problem_data()
    result = OrchestratorAgent().run(problem)
    assert len(result["final_evaluation"].hard_violations) == 0
    assert result["final_evaluation"].metrics["scheduled_sessions"] == len(problem.session_requests)


def test_constraint_agent_flags_room_capacity_violation():
    problem = load_problem_data()
    config = ConstraintConfig()
    weights = SoftConstraintWeights()
    assignments = OrchestratorAgent().scheduling_agent.generate_initial_schedule(problem, config, weights)
    assignments[0].room_id = "R3"
    evaluation = ConstraintAgent().evaluate(problem, assignments, config, weights)
    assert any(item.kind == "room_capacity" for item in evaluation.hard_violations)


def test_policy_training_produces_summary_and_persists_weights(tmp_path):
    problem = load_problem_data()
    model = PolicyModel(model_path=tmp_path / "policy.json")
    summary = model.train(problem, episodes=5)
    assert summary["episodes"] == 5
    assert model.model_path.exists()
    assert "preferred_slot" in summary["weights"]


def test_genetic_strategy_runs_without_hard_violations():
    problem = load_problem_data()
    result = OrchestratorAgent().run(problem, strategy="genetic", training_episodes=3, generations=4)
    assert len(result["final_evaluation"].hard_violations) == 0


def test_large_dataset_switches_to_adaptive_mode_and_preserves_hard_constraints():
    courses_bytes = Path(r"C:\Users\HP\Downloads\Telegram Desktop\courses(6).csv").read_bytes()
    lecturers_bytes = Path(r"C:\Users\HP\Downloads\Telegram Desktop\lecturers(5).csv").read_bytes()
    rooms_bytes = Path(r"C:\Users\HP\Downloads\Telegram Desktop\rooms(5).csv").read_bytes()
    slots_bytes = Path(r"C:\Users\HP\Downloads\Telegram Desktop\timeslots(5).csv").read_bytes()
    problem = load_problem_data(
        type("Upload", (), {"filename": "courses.csv", "file": BytesIO(courses_bytes)})(),
        type("Upload", (), {"filename": "lecturers.csv", "file": BytesIO(lecturers_bytes)})(),
        type("Upload", (), {"filename": "rooms.csv", "file": BytesIO(rooms_bytes)})(),
        type("Upload", (), {"filename": "timeslots.csv", "file": BytesIO(slots_bytes)})(),
    )
    result = OrchestratorAgent().run(problem, strategy="hybrid", training_episodes=0, generations=4)
    assert result["adaptive_summary"]["very_large_problem"] is True
    assert result["adaptive_summary"]["effective_strategy"] == "heuristic"
    assert len(result["final_evaluation"].hard_violations) == 0


def test_generate_endpoint_returns_quickly_and_creates_background_job():
    client = TestClient(app)
    courses_bytes = Path(r"C:\Users\HP\Downloads\Telegram Desktop\courses(6).csv").read_bytes()
    lecturers_bytes = Path(r"C:\Users\HP\Downloads\Telegram Desktop\lecturers(5).csv").read_bytes()
    rooms_bytes = Path(r"C:\Users\HP\Downloads\Telegram Desktop\rooms(5).csv").read_bytes()
    slots_bytes = Path(r"C:\Users\HP\Downloads\Telegram Desktop\timeslots(5).csv").read_bytes()
    files = {
        "courses": ("courses.csv", BytesIO(courses_bytes), "text/csv"),
        "lecturers": ("lecturers.csv", BytesIO(lecturers_bytes), "text/csv"),
        "rooms": ("rooms.csv", BytesIO(rooms_bytes), "text/csv"),
        "timeslots": ("timeslots.csv", BytesIO(slots_bytes), "text/csv"),
    }
    response = client.post("/generate", files=files, data={"strategy": "hybrid", "training_episodes": "0", "generations": "4"})
    assert response.status_code == 200
    assert "Generation Job" in response.text
    latest_job = JOB_CACHE[next(reversed(JOB_CACHE))]
    assert latest_job["status"] in {"queued", "running", "completed"}
