# Unified TOP Framework

This package uses the JSON + CSV `interface_case` format directly. The main
solver path no longer calls the legacy Excel-based `model.py` or
`heuristic.py`.

## Files

- `models.py`: shared data classes.
- `io.py`: reads `config.json` and CSV files, validates input, writes outputs.
- `schedule.py`: shared schedule structures, budget helpers, validation.
- `ha.py`: greedy heuristic solver.
- `ea.py`: Gurobi-based exact assignment solver.
- `eao.py`: SCIP/PySCIPOpt-based exact assignment solver.
- `ga.py`: genetic algorithm solver.
- `router.py`: selects the solver from `config.algorithm.name`.
- `inputs/instance*`: normalized sample cases.

## Algorithms

- `ea`: implemented.
- `eao`: implemented.
- `ha`: implemented.
- `ga`: implemented.
- `pso`, `sa`: implemented.
- `ai`: reserved and currently returns `not_implemented`.

## Run

```bash
./.venv/bin/python -m unified_framework unified_framework/inputs/instance1
```

Outputs are written to:

- `<case_dir>/output/result.json`
- `<case_dir>/output/schedule.csv` when a plan is produced
