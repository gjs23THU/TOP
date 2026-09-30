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
- `eah.py`: HiGHS/highspy-based exact assignment solver.
- `eac.py`: COPT/coptpy-based exact assignment solver.
- `ga.py`: genetic algorithm solver.
- `router.py`: selects the solver from `config.algorithm.name`.
- `inputs/instance*`: normalized sample cases.

## Algorithms

- `ea`: implemented.
- `eao`: implemented.
- `eah`: implemented.
- `eac`: implemented; requires `coptpy==8.0.7` and a valid COPT 8 license.
- `ha`: implemented.
- `ga`: implemented.
- `pso`, `sa`: implemented.
- `ai`: reserved and currently returns `not_implemented`.

Example COPT configuration (only `normal` mode is supported):

```json
{
  "algorithm": {
    "name": "eac",
    "mode": "normal",
    "obj": "maxRevenue",
    "timeLimit": null,
    "decimal": 5
  }
}
```

`obj` may also be `minTime` or `minPower`. A null `timeLimit` leaves the
solver without an additional project-injected stopping limit.

## Run

```bash
./.venv/bin/python -m unified_framework unified_framework/inputs/instance1
```

Outputs are written to:

- `<case_dir>/output/result.json`
- `<case_dir>/output/schedule.csv` when a plan is produced
