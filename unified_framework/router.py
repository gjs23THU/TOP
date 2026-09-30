from __future__ import annotations

import argparse
from pathlib import Path
import time

from .io import load_case, write_error, write_outputs
from .models import MaxDistanceError, SolveResult
from .schedule import SchedulePlan, check_remote_requirement


NOT_IMPLEMENTED = {"ai"}


def _dispatch(case) -> SchedulePlan:
    algorithm = case.config.algorithm.name
    mode = case.config.algorithm.mode

    # eac keeps the remote-distance requirement inside the COPT model so an
    # unsatisfiable remote requirement follows the standard infeasible/IIS
    # path.  Preserve the established pre-check behavior for all other
    # algorithms.
    if algorithm != "eac":
        remote_check = check_remote_requirement(case)
        if not remote_check.ok:
            raise MaxDistanceError(remote_check.message or "")

    if algorithm == "ea":
        from . import ea

        return ea.solve(case, mode)
    elif algorithm == "eao":
        from . import eao

        return eao.solve(case, mode)
    elif algorithm == "eah":
        from . import eah

        return eah.solve(case, mode)
    elif algorithm == "eac":
        from . import eac

        return eac.solve(case, mode)
    elif algorithm == "ga":
        from . import ga

        return ga.solve(case, mode)
    elif algorithm == "ha":
        from . import ha

        return ha.solve(case, mode)
    elif algorithm == "pso":
        from . import pso

        return pso.solve(case, mode)
    elif algorithm == "sa":
        from . import sa

        return sa.solve(case, mode)
    elif algorithm in NOT_IMPLEMENTED:
        raise NotImplementedError(f"Algorithm {algorithm} is not implemented yet.")
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def solve_case(case_dir: str | Path) -> SchedulePlan:
    case = load_case(case_dir)
    plan = _dispatch(case)
    write_outputs(case, plan)
    return plan


def run_case(case_dir: str | Path) -> SolveResult:
    case_path = Path(case_dir).resolve()
    case_id = case_path.name
    started = time.perf_counter()
    algorithm_name = None
    try:
        case = load_case(case_path)
        case_id = case.config.case_id
        algorithm_name = case.config.algorithm.name
        plan = _dispatch(case)
        metrics = dict(plan.metrics or {})
        metrics.setdefault("solver_name", algorithm_name)
        metrics["total_seconds"] = time.perf_counter() - started
        plan.metrics = metrics
        return write_outputs(case, plan)
    except Exception as exc:
        return write_error(
            case_path,
            case_id,
            exc,
            metrics={
                "solver_name": algorithm_name,
                "termination_status": "solver_error",
                "total_seconds": time.perf_counter() - started,
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified TOP solver")
    parser.add_argument("case_dir", help="Case directory with config.json and CSV inputs")
    args = parser.parse_args()
    result = run_case(args.case_dir)
    print(result)
    return 0 if result.status == "success" else 1
