from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

# FastAPI only needs this package for multipart endpoints. The registry test
# calls a plain GET endpoint directly, so provide the version probe when the
# developer environment has not installed the optional HTTP upload dependency.
try:
    import python_multipart  # noqa: F401
except ImportError:
    multipart_probe = types.ModuleType("python_multipart")
    multipart_probe.__version__ = "0.0.20"
    sys.modules["python_multipart"] = multipart_probe

from algorithm_service.app import create_app
from algorithm_service.settings import ServiceSettings
from unified_framework import router
from unified_framework.io import _load_config, status_from_exception
from unified_framework.models import MIPError
from unified_framework.schedule import SchedulePlan


@contextmanager
def import_eac_with_fake_copt(fake_coptpy: types.ModuleType):
    """Import eac without requiring COPT or openpyxl in the host test env."""
    fake_openpyxl = types.ModuleType("openpyxl")
    fake_openpyxl_styles = types.ModuleType("openpyxl.styles")
    fake_openpyxl_styles.PatternFill = object
    old_module = sys.modules.pop("unified_framework.eac", None)
    package = importlib.import_module("unified_framework")
    old_attr = getattr(package, "eac", None)
    had_attr = hasattr(package, "eac")
    if had_attr:
        delattr(package, "eac")
    try:
        with mock.patch.dict(
            sys.modules,
            {
                "coptpy": fake_coptpy,
                "openpyxl": fake_openpyxl,
                "openpyxl.styles": fake_openpyxl_styles,
            },
        ):
            yield importlib.import_module("unified_framework.eac")
    finally:
        sys.modules.pop("unified_framework.eac", None)
        if old_module is not None:
            sys.modules["unified_framework.eac"] = old_module
        if had_attr:
            setattr(package, "eac", old_attr)
        elif hasattr(package, "eac"):
            delattr(package, "eac")


class EacIntegrationTests(unittest.TestCase):
    def test_config_accepts_eac(self) -> None:
        payload = {
            "case_id": "eac-config-test",
            "algorithm": {
                "name": "eac",
                "mode": "normal",
                "obj": "maxRevenue",
                "timeLimit": None,
                "decimal": 5,
            },
            "max-distance": 10,
            "total-time/day": [100, 100, 100],
            "total-power/day": [100, 100, 100],
            "min-continuous": 0,
            "12-gap": 0,
            "23-gap": 0,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir)
            (case_dir / "config.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            config = _load_config(case_dir)

        self.assertEqual(config.algorithm.name, "eac")

    def test_router_dispatches_eac_without_eager_copt_import(self) -> None:
        fake_eac = types.ModuleType("unified_framework.eac")
        expected = SchedulePlan(steps=[], objective_value=7.0)
        fake_eac.solve = mock.Mock(return_value=expected)
        case = SimpleNamespace(
            tasks=[],
            config=SimpleNamespace(
                algorithm=SimpleNamespace(name="eac", mode="normal")
            ),
        )

        package = importlib.import_module("unified_framework")
        previous = getattr(package, "eac", None)
        had_previous = hasattr(package, "eac")
        try:
            setattr(package, "eac", fake_eac)
            with mock.patch.dict(
                sys.modules, {"unified_framework.eac": fake_eac}
            ), mock.patch.object(
                router,
                "check_remote_requirement",
                side_effect=AssertionError("eac remote feasibility belongs to COPT"),
            ):
                result = router._dispatch(case)
        finally:
            if had_previous:
                setattr(package, "eac", previous)
            else:
                delattr(package, "eac")

        self.assertIs(result, expected)
        fake_eac.solve.assert_called_once_with(case, "normal")

    def test_algorithms_endpoint_advertises_eac(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = create_app(
                settings=ServiceSettings(run_root=Path(temp_dir))
            )
            endpoint = next(
                route.endpoint
                for route in app.routes
                if getattr(route, "path", None) == "/api/v1/algorithms"
            )
            payload = endpoint()

        self.assertIn("eac", payload["name"])

    def test_eac_rejects_non_normal_mode_before_creating_solver(self) -> None:
        fake_coptpy = types.ModuleType("coptpy")
        with import_eac_with_fake_copt(fake_coptpy) as eac:
            with self.assertRaisesRegex(
                NotImplementedError, "only supports algorithm.mode=normal"
            ):
                eac.solve(None, "back")

    def test_remote_constraint_uses_only_qualified_locations(self) -> None:
        fake_coptpy = types.ModuleType("coptpy")
        fake_coptpy.quicksum = sum
        with import_eac_with_fake_copt(fake_coptpy) as eac:
            optimizer = eac.task_optimize.__new__(eac.task_optimize)
            model = mock.Mock()

            class EdgeVars:
                def __init__(self) -> None:
                    self.calls: list[tuple[object, ...]] = []

                def sum(self, *args: object) -> int:
                    self.calls.append(args)
                    return 1

            edges = EdgeVars()
            optimizer._task_optimize__m = model
            optimizer._task_optimize__x = edges
            optimizer._task_optimize__remtaskindex = []
            optimizer.add_remote_constrs()
            model.addConstr.assert_not_called()

            optimizer._task_optimize__remtaskindex = [7]
            optimizer._task_optimize__point = [(0, 7), (1, 7)]
            optimizer._task_optimize__pointdf = pd.DataFrame(
                {"No": [0, 1]}, index=["near", "far"]
            )
            optimizer._task_optimize__dmatrix = pd.DataFrame(
                {"near": [5.0], "far": [15.0]},
                index=["探测起点1"],
            )
            optimizer._task_optimize__MDistance = 10.0

            optimizer.add_remote_constrs()

            self.assertEqual(edges.calls, [("*", "*", 1, 7)])
            model.addConstr.assert_called_once_with(True, name="remote")

    def test_no_remote_tasks_do_not_create_constraints_or_files(self) -> None:
        fake_coptpy = types.ModuleType("coptpy")
        fake_coptpy.quicksum = sum
        with import_eac_with_fake_copt(fake_coptpy) as eac:
            optimizer = eac.task_optimize.__new__(eac.task_optimize)
            optimizer._task_optimize__task = pd.DataFrame({"remote": [False]})
            optimizer._task_optimize__m = mock.Mock()
            optimizer._task_optimize__remtaskindex = []

            optimizer.check_remote()
            optimizer.add_remote_constrs()

            optimizer._task_optimize__m.addConstr.assert_not_called()

    def test_empty_tupledict_sum_remains_a_linear_expression(self) -> None:
        fake_coptpy = types.ModuleType("coptpy")
        fake_coptpy.LinExpr = lambda value: ("linear", value)

        class EmptyTupleDict:
            def sum(self, *patterns: object) -> int:
                return 0

        with import_eac_with_fake_copt(fake_coptpy) as eac:
            variables = eac._LinearTupleDict(EmptyTupleDict())

            self.assertEqual(
                variables.sum("*", "*", 7, 9),
                ("linear", 0),
            )

    def test_unqualified_remote_locations_add_infeasible_constraint(self) -> None:
        fake_coptpy = types.ModuleType("coptpy")
        fake_coptpy.quicksum = sum
        with import_eac_with_fake_copt(fake_coptpy) as eac:
            optimizer = eac.task_optimize.__new__(eac.task_optimize)
            model = mock.Mock()
            edges = mock.Mock()
            optimizer._task_optimize__m = model
            optimizer._task_optimize__x = edges
            optimizer._task_optimize__remtaskindex = [7]
            optimizer._task_optimize__point = [(0, 7)]
            optimizer._task_optimize__pointdf = pd.DataFrame(
                {"No": [0]}, index=["near"]
            )
            optimizer._task_optimize__dmatrix = pd.DataFrame(
                {"near": [5.0]}, index=["探测起点1"]
            )
            optimizer._task_optimize__MDistance = 10.0

            optimizer.add_remote_constrs()

            edges.sum.assert_not_called()
            model.addConstr.assert_called_once_with(False, name="remote")

    def test_three_objectives_are_translated_to_copt(self) -> None:
        fake_coptpy = types.ModuleType("coptpy")
        fake_coptpy.quicksum = sum
        fake_coptpy.COPT = SimpleNamespace(MAXIMIZE=1)
        with import_eac_with_fake_copt(fake_coptpy) as eac:
            optimizer = eac.task_optimize.__new__(eac.task_optimize)
            model = mock.Mock()

            class EdgeVars:
                def sum(self, *args: object) -> float:
                    return float(args[-1] + 1)

            optimizer._task_optimize__m = model
            optimizer._task_optimize__Obj = [
                eac.CONST.MAX_REVENUE,
                eac.CONST.MIN_TIME,
                eac.CONST.MIN_POWER,
            ]
            optimizer._task_optimize__task = pd.DataFrame(
                {"revenue": [3.0, 5.0]}, index=[0, 1]
            )
            optimizer._task_optimize__x = EdgeVars()
            optimizer._task_optimize__opoint = [(0, 0), (1, 1), (2, 2), (3, 3)]
            optimizer._task_optimize__W = {(3, 3): 12.0}
            optimizer._task_optimize__Q = {(3, 3): 21.0}

            expected = {
                eac.CONST.MAX_REVENUE: 13.0,
                eac.CONST.MIN_TIME: -12.0,
                eac.CONST.MIN_POWER: -21.0,
            }
            for objective, value in expected.items():
                optimizer._task_optimize__objective = objective
                self.assertEqual(optimizer.set_objective(), value)
                model.setObjective.assert_called_with(value, fake_coptpy.COPT.MAXIMIZE)

    def test_resource_variables_use_dynamic_budget_bounds(self) -> None:
        fake_coptpy = types.ModuleType("coptpy")
        fake_coptpy.COPT = SimpleNamespace(BINARY="B", CONTINUOUS="C")
        with import_eac_with_fake_copt(fake_coptpy) as eac:
            optimizer = eac.task_optimize.__new__(eac.task_optimize)
            model = mock.Mock()
            model.addVars.side_effect = [object() for _ in range(9)]
            optimizer._task_optimize__m = model
            optimizer._task_optimize__edges = [(0, 0, 1, 1)]
            optimizer._task_optimize__point = [(0, 0), (1, 1)]
            optimizer._task_optimize__TTime = [10.0, 20.0, 30.0]
            optimizer._task_optimize__TPower = [100.0, 200.0, 300.0]
            optimizer._task_optimize__time_upper_bound = None
            optimizer._task_optimize__power_upper_bound = None

            optimizer.add_variables()

            self.assertEqual(model.addVars.call_args_list[1].kwargs["ub"], 60.0)
            self.assertEqual(model.addVars.call_args_list[2].kwargs["ub"], 600.0)
            self.assertEqual(eac.exact_big_m(60.0, 5.0), 65.0)

    def test_time_limit_is_forwarded_and_environment_is_closed(self) -> None:
        fake_coptpy = types.ModuleType("coptpy")
        fake_coptpy.COPT = SimpleNamespace(
            Param=SimpleNamespace(TimeLimit="TimeLimit")
        )
        model = SimpleNamespace(setParam=mock.Mock())
        environment = SimpleNamespace(
            createModel=mock.Mock(return_value=model), close=mock.Mock()
        )
        fake_coptpy.Envr = mock.Mock(return_value=environment)

        with import_eac_with_fake_copt(fake_coptpy) as eac:
            optimizer = eac.task_optimize(timeLimit=17)
            model.setParam.assert_called_once_with("TimeLimit", 17)
            optimizer.close()
            environment.close.assert_called_once_with()

    def test_infeasible_model_writes_iis_to_requested_output_path(self) -> None:
        fake_coptpy = types.ModuleType("coptpy")
        fake_coptpy.COPT = SimpleNamespace(INFEASIBLE=3, OPTIMAL=1)
        with import_eac_with_fake_copt(fake_coptpy) as eac:
            with tempfile.TemporaryDirectory() as temp_dir:
                iis_path = Path(temp_dir) / "output" / "eac_infeasible.ilp"
                model = SimpleNamespace(
                    Status=fake_coptpy.COPT.INFEASIBLE,
                    HasSol=0,
                    solve=mock.Mock(),
                    computeIIS=mock.Mock(),
                    writeIIS=mock.Mock(),
                )
                optimizer = eac.task_optimize.__new__(eac.task_optimize)
                optimizer._task_optimize__m = model
                optimizer._task_optimize__lppath = None
                optimizer._task_optimize__iispath = str(iis_path)

                with self.assertRaisesRegex(eac.MIPError, "Model is infeasible"):
                    optimizer.run_opt()

                self.assertTrue(iis_path.parent.is_dir())
                model.computeIIS.assert_called_once_with()
                model.writeIIS.assert_called_once_with(str(iis_path))

    def test_time_limited_incumbent_is_accepted(self) -> None:
        fake_coptpy = types.ModuleType("coptpy")
        fake_coptpy.COPT = SimpleNamespace(INFEASIBLE=3, OPTIMAL=1)
        with import_eac_with_fake_copt(fake_coptpy) as eac:
            model = SimpleNamespace(
                Status=9,
                HasSol=1,
                ObjVal=42.5,
                solve=mock.Mock(),
            )
            optimizer = eac.task_optimize.__new__(eac.task_optimize)
            optimizer._task_optimize__m = model
            optimizer._task_optimize__lppath = None
            optimizer._task_optimize__iispath = None

            optimizer.run_opt()
            optimizer.print_status()

            self.assertEqual(optimizer._task_optimize__objvalue, 42.5)

    def test_copt_initialization_failure_has_actionable_error(self) -> None:
        fake_coptpy = types.ModuleType("coptpy")

        def fail_environment() -> object:
            raise RuntimeError("license unavailable")

        fake_coptpy.Envr = fail_environment
        with import_eac_with_fake_copt(fake_coptpy) as eac:
            with self.assertRaisesRegex(
                RuntimeError,
                "verify coptpy==8.0.7.*COPT_LICENSE_DIR",
            ):
                eac.task_optimize()

    def test_eac_error_classes_map_to_public_statuses(self) -> None:
        self.assertEqual(
            status_from_exception(RuntimeError("COPT license unavailable")),
            "solver_error",
        )
        self.assertEqual(
            status_from_exception(NotImplementedError("unsupported mode")),
            "input_error",
        )
        self.assertEqual(
            status_from_exception(MIPError("infeasible")),
            "infeasible",
        )


if __name__ == "__main__":
    unittest.main()
