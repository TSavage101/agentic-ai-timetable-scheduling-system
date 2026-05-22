# Agentic AI-Driven Timetable Scheduling System

This project is a full implementation of the proposed final year project on an **Agentic AI-driven timetabling scheduling system** for universities. It generates timetables automatically, validates them against institutional constraints, improves them through optimization, and presents the result through a web interface that an administrator can use during review and approval.

This version goes beyond a basic rule engine. It includes:

- a multi-agent scheduling architecture
- a trainable policy model
- a genetic optimization path
- a heuristic repair layer
- a web interface for experimentation and demonstration

## What the system does

- Generates university course timetables from CSV input data
- Satisfies hard constraints such as lecturer availability, room capacity, room compatibility, and no double-booking
- Improves soft constraints such as preferred slot usage, day spread, and reduced late classes
- Allows different scheduling modes:
  - `Pure Heuristic`
  - `Policy-Guided Agent`
  - `Genetic Optimizer`
  - `Hybrid Agentic + Genetic`
- Supports optional training episodes before schedule generation
- Shows scores, constraint results, learned weights, and optimization history in the UI
- Exports the generated timetable as CSV

## System architecture

The project follows the proposal's agentic AI idea by assigning responsibilities to specialized agents:

- `OrchestratorAgent`: controls the full workflow
- `SchedulingAgent`: generates the first timetable draft
- `ConstraintAgent`: checks hard and soft constraints
- `ConflictResolutionAgent`: repairs and improves schedules iteratively
- `GeneticOptimizationAgent`: evolves candidate schedules across generations
- `PolicyModel`: learns feature weights that influence slot and room selection

This means the project is not just "one algorithm." It is a collaborative decision pipeline.

## Where the model is

The main model-driven logic lives in [app/agents.py](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/app/agents.py).

The trainable component is `PolicyModel`. It learns weights for features such as:

- preferred slot usage
- late-slot penalty
- same-day repeat penalty
- room-fit quality
- morning-slot preference

These learned weights are saved to:

- [data/models/policy_weights.json](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/data/models/policy_weights.json)

if training has been run at least once.

## What kind of model this is

This is not a deep neural network. It is a lightweight trainable policy model used inside an agentic scheduling pipeline.

That is actually a strong academic choice because:

- it is explainable
- it trains fast
- it runs on a normal laptop
- it can be demonstrated live
- it supports the proposal's "reinforcement learning and/or heuristic optimization" direction

The project combines:

- heuristic scheduling
- reinforcement-inspired policy learning
- genetic optimization
- constraint-based repair

## Tech stack

- `Python`
- `FastAPI`
- `Jinja2`
- `Pandas`
- `NumPy`
- `Pytest`
- plain `HTML/CSS`

## Project structure

```text
Timetable-Scheduling-System/
├── app/
│   ├── __init__.py
│   ├── agents.py
│   ├── io_utils.py
│   ├── main.py
│   ├── models.py
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
├── data/
│   ├── models/
│   └── sample/
│       ├── courses.csv
│       ├── lecturers.csv
│       ├── rooms.csv
│       └── timeslots.csv
├── tests/
│   └── test_scheduler.py
├── requirements.txt
└── README.md
```

## Input files

The system expects four CSV files.

### `courses.csv`

Fields:

- `id`
- `code`
- `title`
- `lecturer_id`
- `student_count`
- `sessions_per_week`
- `room_type`
- `preferred_slots`
- `blocked_slots`
- `level`
- `department`

### `lecturers.csv`

Fields:

- `id`
- `name`
- `unavailable_slots`

### `rooms.csv`

Fields:

- `id`
- `name`
- `capacity`
- `room_type`

### `timeslots.csv`

Fields:

- `id`
- `day`
- `start`
- `end`

## Constraints implemented

### Hard constraints

- no lecturer double-booking
- no room double-booking
- room capacity must be enough
- room type must match course type
- blocked lecturer slots must not be used
- blocked course slots must not be used
- sessions should be scheduled

### Soft constraints

- preferred slots should be used where possible
- repeated sessions of a course should be spread across days
- late slots should be minimized
- lecturer daily overload should be reduced

## Optimization strategies

### 1. Pure Heuristic

The scheduler places sessions greedily using penalties and then uses the repair agent to improve the result.

Simple meaning:

- fastest option
- easiest to explain
- uses fixed rules instead of learned behavior

### 2. Policy-Guided Agent

The scheduler uses the learned policy weights to bias decisions toward better slot-room assignments before repair.

Simple meaning:

- this is the mode that uses the trained model most directly
- it learns what kinds of placements are usually better
- good for showing that the project includes actual training

### 3. Genetic Optimizer

The system creates a population of timetable drafts, repairs them, evolves them through crossover and mutation, and keeps the strongest candidate over generations.

Simple meaning:

- it tries many candidate timetables
- it keeps the better ones
- it improves them over multiple generations like evolution

### 4. Hybrid Agentic + Genetic

This is the strongest demonstration mode. It combines:

- learned policy guidance
- heuristic draft generation
- local conflict repair
- evolutionary search

Simple meaning:

- this is the best “main demo” mode
- it combines all the major strengths of the system
- if the panel asks which mode represents the final system, this is the best answer

## How training works

The `PolicyModel` is trained through lightweight repeated episodes over feasible scheduling choices. During training, it updates feature weights according to reward signals such as:

- reward for preferred slots
- reward for good room fit
- penalty for late slots
- penalty for repeated same-day placement

This is not large-scale GPU training. It is fast, local, and appropriate for a final year project demo.

## Do you need Google Colab?

Short answer: `No, not for this version.`

You can train and run everything locally on a normal machine.

Use Google Colab only if you want to extend the project into:

- larger experimental datasets
- deeper reinforcement learning
- neural models
- many repeated benchmarking runs

For the current project:

- local training is enough
- local testing is enough
- local demo is enough

So if your question is "Do I need to go to Colab before I can defend this project?" the answer is `No`.

## How to install and run

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the web app

```bash
python -m uvicorn app.main:app --reload
```

### 4. Open the app

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

### Live deployment

The project can also be deployed to Railway for public access and demonstration.

## How to use the app

1. Open the homepage.
2. Choose an optimization strategy.
3. Set `Training Episodes` if you want the policy model to learn before scheduling.
4. Set `Genetic Generations` if using genetic or hybrid mode.
5. Upload your CSV files or leave the form empty to use the sample dataset.
6. Click `Run Agentic Scheduler`.
7. Review:
   - scores
   - hard constraint results
   - soft constraint results
   - learned policy weights
   - optimization summary
8. Download the timetable as CSV.

### Recommended demo settings

For a smooth defense/demo run:

- choose `Hybrid Agentic + Genetic`
- set `Training Episodes` to `10` or `20`
- set `Genetic Generations` to `8` to `12`

For faster comparison runs:

- use `Pure Heuristic` for baseline
- use `Policy-Guided Agent` to show learning
- use `Genetic Optimizer` to show metaheuristic search

## Bundled sample scenarios

The folder [data/sample](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/data/sample) now includes multiple ready-made datasets for testing different cases.

- [data/sample/balanced_baseline](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/data/sample/balanced_baseline) is the normal clean baseline.
- [data/sample/lecturer_unavailability_pressure](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/data/sample/lecturer_unavailability_pressure) stresses lecturer availability.
- [data/sample/room_capacity_pressure](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/data/sample/room_capacity_pressure) stresses room size and hall availability.
- [data/sample/lab_intensive_mix](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/data/sample/lab_intensive_mix) stresses limited lab resources.
- [data/sample/dense_week](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/data/sample/dense_week) tests a denser departmental week.
- [data/sample/unschedulable_edge_case](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/data/sample/unschedulable_edge_case) is intentionally difficult and is useful for showing failure handling.

Scenario notes are also documented in [data/sample/SCENARIOS.md](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/data/sample/SCENARIOS.md).

## How to run tests

```bash
pytest
```

## Verified behavior

The project includes tests for:

- successful schedule generation
- constraint detection
- policy training persistence
- genetic optimization execution

Main test file:

- [tests/test_scheduler.py](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/tests/test_scheduler.py)

## How to explain how you built it

If your panel asks, you can say:

1. I modeled the timetable problem as a University Course Timetabling Problem with hard and soft constraints.
2. I represented courses, rooms, lecturers, and time slots as structured data objects.
3. I used a multi-agent architecture to separate scheduling, checking, repairing, and optimization responsibilities.
4. I implemented a trainable policy model to learn slot-selection preferences from repeated scheduling episodes.
5. I implemented a genetic optimizer to explore multiple candidate timetables and improve the final result.
6. I wrapped everything in a FastAPI web interface so it can be demonstrated interactively.

## How to answer “Where is the AI?”

A strong answer is:

> The AI is in the system's autonomous decision-making and learning layers. The scheduler does not just follow one fixed script. It evaluates candidate placements, learns preference weights through training episodes, repairs conflicts, and can evolve candidate timetables through a genetic optimization process.

You can add:

> So the intelligence is both agentic and model-guided. The model influences the choices, while the agents coordinate planning, validation, and optimization.

## How to answer “Why didn’t you use a deep learning model?”

You can say:

> I intentionally chose a lightweight, explainable model because this project is constraint-heavy and needs transparent decisions. A black-box neural model would be harder to validate, explain, and debug for administrative scheduling. My design still includes learning, but in a way that is practical and academically defensible.

## Evaluation metrics

The project can be evaluated using:

- hard constraint satisfaction percentage
- soft constraint quality percentage
- total schedule score
- number of scheduled sessions
- time taken to generate a timetable
- comparison between strategies
- comparison against manual scheduling

## Suggested experiment table for your report

You can compare:

- Heuristic mode
- Policy mode
- Genetic mode
- Hybrid mode

Across:

- generation time
- hard violations
- soft violations
- final score
- sessions scheduled

That will make your results section much stronger.

## Limitations

Be honest about these:

- the policy model is lightweight rather than deep reinforcement learning
- the system does not yet use a database
- student cohort clash detection is not yet modeled
- there is no manual drag-and-drop editor yet
- training is based on scheduling episodes rather than labeled historical timetable archives

These are good future-work points, not failures.

## Recommended future improvements

- add student-group conflict detection
- add semester-wide and multi-department scheduling
- add database persistence
- add admin edit/override support
- add Excel and PDF export
- add richer visual analytics
- add historical timetable learning from real institutional data
- add a Colab notebook for large-scale experiments
- add a full RL benchmark for comparison

## FAQ for panel questions

### 1. Is there really a model in this project?

Yes. The `PolicyModel` is a trainable model that learns feature weights used during scheduling. It is not just a static rules table.

### 2. Did you train it?

Yes. The app allows training episodes before generation, and the learned weights are displayed in the interface and saved to disk.

### 3. Do I need Google Colab to run or train it?

No. Everything in this implementation can run locally. Colab is optional for larger experiments only.

### 4. Why use a genetic algorithm too?

Because the search space is large. The genetic optimizer helps explore multiple timetable candidates beyond one greedy draft.

### 5. What makes the system agentic?

Responsibility is distributed across cooperating agents: planning, scheduling, validation, repair, and optimization.

### 6. What is the difference between hard and soft constraints?

Hard constraints define validity. Soft constraints define quality.

### 7. Why is this better than manual scheduling?

It is faster, more consistent, more scalable, easier to audit, and easier to improve systematically.

### 8. Can the system adapt when conditions change?

Yes. The project supports retraining, regeneration, and multiple optimization strategies. It is more adaptable than a fixed manual workflow.

### 9. Can this be extended to real university data?

Yes. The project is data-driven and designed around structured institutional inputs.

### 10. If given more time, what would you add?

I would add student-level clash detection, database storage, richer analytics, manual approval tools, and a larger experimental training pipeline.

### 11. Is this the full implementation of the project or just a simplified prototype?

This is the full implementation of the project according to the scope defined in the proposal document. The document asked for an agentic AI-driven timetabling system with autonomous schedule generation, constraint checking, conflict resolution, iterative improvement, and an administrator-facing interface, and all of those components have been implemented here.

What is important is that the proposal itself described a practical academic system, not a massive production platform or a deep-learning-heavy research lab system. So this project should be presented as a complete final-year project implementation, while still acknowledging that future enhancements such as larger-scale deployment, deeper learning models, or database expansion could be added later.

## Important files

- [app/main.py](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/app/main.py)
- [app/agents.py](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/app/agents.py)
- [app/models.py](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/app/models.py)
- [app/io_utils.py](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/app/io_utils.py)
- [app/templates/index.html](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/app/templates/index.html)
- [tests/test_scheduler.py](/C:/FYPs/Big%20Niph/Timetable-Scheduling-System/tests/test_scheduler.py)

## Final summary for defense

> This project implements an agentic AI-driven timetabling system that uses specialized agents, a trainable policy model, and genetic optimization to generate and improve university course timetables. It satisfies hard constraints, optimizes soft constraints, supports interactive experimentation through a web interface, and is practical to run and explain in a final year project setting.
