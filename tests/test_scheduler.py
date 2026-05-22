from app.agents import ConstraintAgent, OrchestratorAgent, PolicyModel
from app.io_utils import load_problem_data


def test_sample_dataset_generates_schedule_without_hard_violations():
    problem = load_problem_data()
    result = OrchestratorAgent().run(problem)
    assert result["final_evaluation"].hard_violations == []
    assert result["final_evaluation"].metrics["scheduled_sessions"] == len(problem.session_requests)


def test_constraint_agent_flags_room_capacity_violation():
    problem = load_problem_data()
    assignments = OrchestratorAgent().scheduling_agent.generate_initial_schedule(problem)
    assignments[0].room_id = "R3"
    evaluation = ConstraintAgent().evaluate(problem, assignments)
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
    assert result["final_evaluation"].hard_violations == []
