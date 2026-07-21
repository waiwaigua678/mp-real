from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not prior else src + os.pathsep + prior
    return env


def _run_python(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        cwd=REPO_ROOT,
        env=_python_env(),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


class H6OptionalDependencyAndDocsTests(unittest.TestCase):
    def test_core_imports_do_not_require_av_or_pyarrow(self) -> None:
        result = _run_python(
            """
            import importlib
            import importlib.abc
            import json
            import sys

            class Blocker(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "av" or fullname == "pyarrow" or fullname.startswith("pyarrow."):
                        raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
                    return None

            sys.meta_path.insert(0, Blocker())
            for name in (
                "mp_real",
                "mp_real.runtime.models",
                "mp_real.runtime.inference",
                "mp_real.web.server",
                "mp_real.robots.registry",
                "mp_real.robots.piper.infer",
                "mp_real.robots.rm2.infer",
            ):
                importlib.import_module(name)
            print(json.dumps({"av": "av" in sys.modules, "pyarrow": "pyarrow" in sys.modules}))
            """
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout), {"av": False, "pyarrow": False})

    def test_data_cli_reports_missing_recording_extra(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            (root / "meta").mkdir(parents=True)
            (root / "data" / "chunk-000").mkdir(parents=True)
            (root / "data" / "chunk-000" / "episode_000000.parquet").touch()
            (root / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "codebase_version": "v2.1",
                        "robot_type": "fake",
                        "fps": 10.0,
                        "total_episodes": 1,
                        "total_frames": 1,
                        "features": {
                            "observation.state": {"dtype": "float32", "shape": [1], "names": [["state"]]},
                            "action": {"dtype": "float32", "shape": [1], "names": [["action"]]},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "meta" / "episodes.jsonl").write_text(
                json.dumps({"episode_index": 0, "tasks": ["task"], "length": 1}) + "\n",
                encoding="utf-8",
            )
            (root / "meta" / "tasks.jsonl").write_text(
                json.dumps({"task_index": 0, "task": "task"}) + "\n",
                encoding="utf-8",
            )
            (root / "meta" / "episodes_stats.jsonl").write_text("", encoding="utf-8")
            (root / "meta" / "stats.json").write_text("{}", encoding="utf-8")
            result = _run_python(
                """
                import importlib.abc
                import sys

                class Blocker(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path=None, target=None):
                        if fullname == "pyarrow" or fullname.startswith("pyarrow."):
                            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
                        return None

                sys.meta_path.insert(0, Blocker())
                from mp_real.data.cli import validate_cli
                sys.argv = ["mp-data-validate", sys.argv[1], "--skip-video-check"]
                validate_cli()
                """,
                str(root),
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("uv sync --extra recording", result.stderr)
        self.assertIn("pyarrow", result.stderr)

    def test_cli_help_does_not_require_recording_dependencies(self) -> None:
        commands = {
            "mp-piper-infer": "from mp_real.robots.piper.infer import cli; cli()",
            "mp-rm2-infer": "from mp_real.robots.rm2.infer import cli; cli()",
            "mp-camera-preview": "from mp_real.web.camera_preview import main; main()",
            "mp-data-inspect": "from mp_real.data.cli import inspect_cli; inspect_cli()",
            "mp-data-validate": "from mp_real.data.cli import validate_cli; validate_cli()",
            "mp-data-audit": "from mp_real.data.audit import cli; cli()",
            "mp-robot-replay": "from mp_real.replay.cli import cli; raise SystemExit(cli())",
            "mp-open-loop-eval": "from mp_real.evaluation.open_loop.cli import cli; cli()",
        }
        for command, snippet in commands.items():
            with self.subTest(command=command):
                result = _run_python(
                    """
                    import importlib.abc
                    import sys

                    class Blocker(importlib.abc.MetaPathFinder):
                        def find_spec(self, fullname, path=None, target=None):
                            if fullname == "av" or fullname == "pyarrow" or fullname.startswith("pyarrow."):
                                raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
                            if fullname == "pyAgxArm":
                                raise AssertionError("offline/help command imported the Piper SDK")
                            return None

                    sys.meta_path.insert(0, Blocker())
                    command = sys.argv[1]
                    snippet = sys.argv[2]
                    sys.argv = [command, "--help"]
                    exec(snippet)
                    """,
                    command,
                    snippet,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout.lower())

    def test_readme_local_doc_links_exist(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        linked_paths = {match.group(1) for match in re.finditer(r"\\((docs/[^)#]+)(?:#[^)]+)?\\)", readme)}
        missing = sorted(path for path in linked_paths if not (REPO_ROOT / path).exists())
        self.assertEqual(missing, [])

    def test_ci_workflow_has_core_and_recording_jobs(self) -> None:
        workflow = REPO_ROOT / ".github" / "workflows" / "no-hardware-ci.yml"
        self.assertTrue(workflow.is_file())
        content = workflow.read_text(encoding="utf-8")
        self.assertIn("core-only", content)
        self.assertIn("recording-extra", content)
        self.assertIn("ruff check .", content)
        self.assertNotIn("--execute", content)

    def test_av_and_pyarrow_are_optional_extras_not_core_dependencies(self) -> None:
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = tuple(
            str(item).split(">=", 1)[0].split("[", 1)[0] for item in pyproject["project"]["dependencies"]
        )
        self.assertNotIn("av", dependencies)
        self.assertNotIn("pyarrow", dependencies)
        optional = pyproject["project"]["optional-dependencies"]
        self.assertTrue(any(str(item).startswith("av") for item in optional["recording"]))
        self.assertTrue(any(str(item).startswith("pyarrow") for item in optional["recording"]))


if __name__ == "__main__":
    unittest.main()
