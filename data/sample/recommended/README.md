# Recommended Scheduler Input

Feed the scheduler five clean datasets:

- `courses.csv`: academic demand only. Department, level, contact hours, room type, and class size matter most.
- `lecturers.csv`: who can teach which department or program, plus workload limits and optional unavailable slots.
- `rooms.csv`: capacity, room type, building/location, and special equipment.
- `timeslots.csv`: the teaching windows the scheduler is allowed to use.
- `student_groups.csv`: the cohorts that must avoid clashes.

Notes:

- `preferred_slots` and `blocked_slots` are supported for backward compatibility, but they are no longer required in the recommended course schema.
- If a course has `duration_hours=3`, the system will split it into one `2` hour session and one `1` hour session automatically.
- If a department has no uploaded lecturers, the app now generates Nigerian lecturer rosters for the supported schools and assigns them department-aware.
