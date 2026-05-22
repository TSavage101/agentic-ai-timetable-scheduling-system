# Sample Scenario Library

This folder contains bundled datasets for demonstrating normal runs and edge cases.

## Default root files

The root CSV files in this folder are the default sample dataset used when the app form is submitted without uploads.

## Scenario folders

### `balanced_baseline`

General-purpose scenario with enough rooms and slots for a smooth valid timetable.

### `lecturer_unavailability_pressure`

Tests how the scheduler behaves when lecturer unavailable slots are tight and preferred slots are harder to satisfy.

### `room_capacity_pressure`

Tests room-size pressure where large classes compete for a small number of suitable halls.

### `lab_intensive_mix`

Tests mixed lecture and lab scheduling where practical courses compete for limited lab rooms.

### `dense_week`

Tests a busier department with many courses and heavier use of the week.

### `unschedulable_edge_case`

An intentionally difficult case that may leave some sessions unscheduled because the constraints are too restrictive. Useful for showing failure handling and hard-constraint reporting.
