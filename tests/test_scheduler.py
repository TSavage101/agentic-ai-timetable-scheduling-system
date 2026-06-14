from dataclasses import replace
from io import BytesIO

from fastapi.testclient import TestClient

from app.agents import ConstraintAgent, OrchestratorAgent, PolicyModel
from app.main import ADMIN_EMAIL, ADMIN_PASSWORD, JOB_CACHE, app
from app.io_utils import load_problem_data
from app.models import ConstraintConfig, SoftConstraintWeights


def _make_large_problem_uploads():
    course_rows = [
        "id,code,title,department,level,lecturer_id,student_count,sessions_per_week,room_type"
    ]
    for i in range(60):
        course_rows.append(
            f"C{i},CSC{300 + i},Course {i},Computer Science,{300 + (i % 4) * 100},L{i % 6},{50 + (i % 25)},2,lecture"
        )
    lecturer_rows = ["id,name,departments,max_hours_per_day,max_hours_per_week"] + [
        f"L{i},Lecturer {i},Computer Science,10,40" for i in range(6)
    ]
    room_rows = ["id,name,capacity,room_type"] + [
        f"R{i},Room {i},{120 + i},lecture" for i in range(60)
    ]
    slot_rows = ["id,day,start,end"]
    days = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday", "Weekday8", "Weekday9", "Weekday10",
    ]
    time_pairs = [(f"{hour:02d}:00", f"{hour + 1:02d}:00") for hour in range(8, 18)]
    day_codes = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN", "D08", "D09", "D10"]
    for day_code, day_name in zip(day_codes, days):
        for start, end in time_pairs:
            slot_rows.append(f"{day_code}_{start[:2]}{start[3:5]},{day_name},{start},{end}")
    return (
        BytesIO("\n".join(course_rows).encode()),
        BytesIO("\n".join(lecturer_rows).encode()),
        BytesIO("\n".join(room_rows).encode()),
        BytesIO("\n".join(slot_rows).encode()),
    )


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
    courses_bytes, lecturers_bytes, rooms_bytes, slots_bytes = _make_large_problem_uploads()
    problem = load_problem_data(
        type("Upload", (), {"filename": "courses.csv", "file": courses_bytes})(),
        type("Upload", (), {"filename": "lecturers.csv", "file": lecturers_bytes})(),
        type("Upload", (), {"filename": "rooms.csv", "file": rooms_bytes})(),
        type("Upload", (), {"filename": "timeslots.csv", "file": slots_bytes})(),
    )
    result = OrchestratorAgent().run(problem, strategy="hybrid", training_episodes=0, generations=4)
    assert result["adaptive_summary"]["large_problem"] is True
    assert result["adaptive_summary"]["effective_strategy"] == "heuristic"
    assert len(result["final_evaluation"].hard_violations) == 0


def test_generate_endpoint_returns_quickly_and_creates_background_job():
    client = TestClient(app)
    login_response = client.post("/login", data={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}, follow_redirects=False)
    assert login_response.status_code == 303
    courses_bytes, lecturers_bytes, rooms_bytes, slots_bytes = _make_large_problem_uploads()
    files = {
        "courses": ("courses.csv", courses_bytes, "text/csv"),
        "lecturers": ("lecturers.csv", lecturers_bytes, "text/csv"),
        "rooms": ("rooms.csv", rooms_bytes, "text/csv"),
        "timeslots": ("timeslots.csv", slots_bytes, "text/csv"),
    }
    response = client.post("/generate", files=files, data={"strategy": "hybrid", "training_episodes": "0", "generations": "4"})
    assert response.status_code == 200
    assert "Generation Job" in response.text
    latest_job = JOB_CACHE[next(reversed(JOB_CACHE))]
    assert latest_job["status"] in {"queued", "running", "completed"}


def test_student_group_overlap_conflict():
    problem = load_problem_data()
    # Force two sessions of different courses but same department and level (e.g. CSC401 and CSC405 are both Computer Science Level 400)
    # into the same timeslot
    config = ConstraintConfig()
    weights = SoftConstraintWeights()
    assignments = OrchestratorAgent().scheduling_agent.generate_initial_schedule(problem, config, weights)
    
    # Locate two sessions of CSC401 and CSC405
    csc401_sess = next(a for a in assignments if a.course_id == "C1")
    csc405_sess = next(a for a in assignments if a.course_id == "C2")
    
    # Place them in the same slot (MON_08) but different rooms so room conflict is avoided
    csc401_sess.slot_id = "MON_08"
    csc401_sess.room_id = "R1"
    
    csc405_sess.slot_id = "MON_08"
    csc405_sess.room_id = "R2"
    
    evaluation = ConstraintAgent().evaluate(problem, assignments, config, weights)
    assert any(item.kind == "cohort_conflict" for item in evaluation.hard_violations)


def test_lecturer_daily_hours_limit():
    problem = load_problem_data()
    config = ConstraintConfig()
    weights = SoftConstraintWeights()
    assignments = OrchestratorAgent().scheduling_agent.generate_initial_schedule(problem, config, weights)
    
    # Set lecturer L1's max_hours_per_day to 1 hour
    # Course C1 duration is 1 hour (default), sessions_per_week is 2.
    # If we schedule two sessions of C1 on Monday, L1 will teach 2 hours on Monday, which exceeds 1 hour.
    l1 = problem.lecturers["L1"]
    problem.lecturers["L1"] = replace(l1, max_hours_per_day=1)
    
    c1_sessions = [a for a in assignments if a.course_id == "C1"]
    # Schedule both in MON_08 and MON_10 (both are Monday slots)
    c1_sessions[0].slot_id = "MON_08"
    c1_sessions[0].room_id = "R1"
    c1_sessions[1].slot_id = "MON_10"
    c1_sessions[1].room_id = "R2"
    
    evaluation = ConstraintAgent().evaluate(problem, assignments, config, weights)
    assert any(item.kind == "lecturer_daily_hours_exceeded" for item in evaluation.hard_violations)


def test_missing_lecturer_ids_are_auto_assigned_by_department():
    courses = BytesIO(
        b"id,code,title,department,level,student_count,sessions_per_week,duration_hours,room_type\n"
        b"C1,CSC401,Artificial Intelligence,Computer Science,400,80,1,2,lecture\n"
        b"C2,CVE201,Engineering Drawing,Civil Engineering,200,65,1,2,lecture\n"
    )
    lecturers = BytesIO(
        b"id,name,departments\n"
        b"L1,Prof Oyelade,Computer Science|Management and Information Sciences\n"
    )
    rooms = BytesIO(
        b"id,name,capacity,room_type\n"
        b"R1,Hall A,120,lecture\n"
    )
    slots = BytesIO(
        b"id,day,start,end\n"
        b"MON_08,Monday,08:00,10:00\n"
        b"TUE_08,Tuesday,08:00,10:00\n"
    )

    problem = load_problem_data(
        type("Upload", (), {"filename": "courses.csv", "file": courses})(),
        type("Upload", (), {"filename": "lecturers.csv", "file": lecturers})(),
        type("Upload", (), {"filename": "rooms.csv", "file": rooms})(),
        type("Upload", (), {"filename": "timeslots.csv", "file": slots})(),
    )

    assert problem.courses["C1"].lecturer_id == "L1"
    assert problem.lecturers[problem.courses["C2"].lecturer_id].name == "Dr Akinsete"


def test_three_hour_courses_are_split_into_two_sessions():
    courses = BytesIO(
        b"id,code,title,department,level,lecturer_id,student_count,sessions_per_week,duration_hours,room_type\n"
        b"C1,CSC401,Artificial Intelligence,Computer Science,400,L1,80,1,3,lecture\n"
    )
    lecturers = BytesIO(b"id,name,departments\nL1,Prof Oyelade,Computer Science\n")
    rooms = BytesIO(b"id,name,capacity,room_type\nR1,Hall A,120,lecture\n")
    slots = BytesIO(
        b"id,day,start,end\n"
        b"MON_08,Monday,08:00,10:00\n"
        b"MON_10,Monday,10:00,12:00\n"
    )

    problem = load_problem_data(
        type("Upload", (), {"filename": "courses.csv", "file": courses})(),
        type("Upload", (), {"filename": "lecturers.csv", "file": lecturers})(),
        type("Upload", (), {"filename": "rooms.csv", "file": rooms})(),
        type("Upload", (), {"filename": "timeslots.csv", "file": slots})(),
    )

    durations = sorted(request.duration_hours for request in problem.session_requests)
    assert durations == [1, 2]


def test_course_location_guides_room_assignment():
    courses = BytesIO(
        b"id,code,title,department,level,lecturer_id,student_count,sessions_per_week,duration_hours,room_type,location\n"
        b"C1,CSC401,Artificial Intelligence,Computer Science,400,L1,80,1,2,lecture,CST\n"
    )
    lecturers = BytesIO(b"id,name,departments\nL1,Prof Oyelade,Computer Science\n")
    rooms = BytesIO(
        b"id,name,capacity,room_type,location\n"
        b"R1,CLDS Hall,120,lecture,CLDS\n"
        b"R2,CST Hall,120,lecture,CST\n"
    )
    slots = BytesIO(b"id,day,start,end\nMON_08,Monday,08:00,10:00\n")

    problem = load_problem_data(
        type("Upload", (), {"filename": "courses.csv", "file": courses})(),
        type("Upload", (), {"filename": "lecturers.csv", "file": lecturers})(),
        type("Upload", (), {"filename": "rooms.csv", "file": rooms})(),
        type("Upload", (), {"filename": "timeslots.csv", "file": slots})(),
    )

    result = OrchestratorAgent().run(problem, strategy="hybrid", training_episodes=0, generations=2)
    scheduled = [item for item in result["assignments"] if item["status"] == "scheduled"]
    assert scheduled
    assert scheduled[0]["room_id"] == "R2"
