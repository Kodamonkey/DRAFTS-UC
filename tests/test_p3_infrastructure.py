"""Regression tests for the Phase 3 infrastructure fixes from the 2026-09 audit.

These guard the parts of the project that are not Python: the container, the
compose file, CI, the logging destination and the configuration files. Every one
of them drifted precisely because nothing checked it.

P1-21/22  The image installed a hand-written dependency list -- Python 3.10,
          torch 2.1, numpy 1.24, fifteen packages unpinned -- while the project
          declared and tested something else entirely, and CI never built it.
P2-28     config.yaml marked values as [CURRENT] that were not the current ones.
P2-29     advanced-config/logging.yaml was loaded and never read.
P2-30     CPU_THREADS was loaded and never passed to apply_thread_settings.
P2-31     The log level was a literal in the code.
P2-32/33  Logs were unrotated and written inside the source tree, so
          `docker compose run --rm` took the whole run's log with the container.
P2-34/36  The mutation job could not run, and CI declared no token permissions.
P2-03     calculate_optimal_chunk_size raised NameError on its fallback path.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DOCKERFILE = PROJECT_ROOT / "Dockerfile"
COMPOSE = PROJECT_ROOT / "docker-compose.yml"
CI = PROJECT_ROOT / ".github/workflows/ci.yml"
CONFIG_YAML = PROJECT_ROOT / "config.yaml"


def _lock_version(package: str) -> str:
    text = (PROJECT_ROOT / "requirements.lock.txt").read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(package)}==([^\s\\]+)", text, re.MULTILINE)
    assert match, f"{package} not found in requirements.lock.txt"
    return match.group(1)


class TestContainerMatchesTheProject:
    def test_base_image_satisfies_requires_python(self):
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        required = re.search(r'requires-python\s*=\s*"[^0-9]*(\d+)\.(\d+)', pyproject)
        assert required, "requires-python missing from pyproject.toml"
        minimum = (int(required.group(1)), int(required.group(2)))

        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        images = re.findall(r"FROM python:(\d+)\.(\d+)", dockerfile)
        assert images, "no python base image found"
        for major, minor in images:
            assert (int(major), int(minor)) >= minimum, (
                f"base image python {major}.{minor} is older than requires-python "
                f"{minimum[0]}.{minimum[1]}: the image runs an interpreter the "
                "project does not claim to support and CI never tests"
            )

    def test_torch_matches_the_lockfile(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        pinned = set(re.findall(r"torch==([\d.]+)", dockerfile))
        assert pinned, "torch is not pinned in the Dockerfile"
        assert pinned == {_lock_version("torch")}, (
            f"Dockerfile pins torch {pinned}, lockfile has {_lock_version('torch')}"
        )

    def test_no_unpinned_packages_are_installed(self):
        """The old CPU stage installed fifteen packages with no version at all,
        so two builds a week apart produced different images."""
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        in_pip_install = False
        for line in dockerfile.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(r"pip install", stripped):
                in_pip_install = True
            if not in_pip_install:
                continue
            body = stripped.rstrip("\\").strip()
            # A bare package name on a continuation line of a pip install is the
            # pattern that produced the drift (apt packages are not pinned and
            # are not what this guards).
            if re.fullmatch(r"[a-z][a-z0-9_.-]+", body):
                pytest.fail(f"unpinned dependency in a pip install: {body!r}")
            if not stripped.endswith("\\"):
                in_pip_install = False

    def test_the_gpu_stage_installs_the_lockfile_with_hashes(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        assert "--require-hashes -r requirements.lock.txt" in dockerfile, (
            "the lockfile's 1273 hashes protect nothing if the image ignores them"
        )

    def test_advanced_config_reaches_the_image(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        assert dockerfile.count("advanced-config/ /app/advanced-config/") >= 1, (
            "advanced-config/ is absent from the image, so its settings apply "
            "locally and are silently ignored in the container"
        )

    def test_entrypoint_allows_passing_arguments(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        assert 'ENTRYPOINT ["python", "main.py"]' in dockerfile


class TestTheBuildWouldFindItsInputs:
    """What a ``docker build`` would catch, checked without running one.

    Item 23 is committed as code and its build has still never been executed
    anywhere this project can reach. It was attempted here, with the exact
    command CI runs -- ``docker build --target cpu-final -t drafts-uc:ci .``
    -- against a daemon started for the purpose. The daemon came up, the build
    context loaded, and the base image could not be fetched: the network policy
    answers 403 to ``production.cloudfront.docker.com``, which is the CDN the
    Docker registry redirects blobs to. Even ``docker build --check``, which
    only lints, needs that metadata.

    So the build stays unverified, and these tests cover the subset of build
    failures that do not need one: a COPY whose source is not in the
    repository, and a pinned version that has drifted from the lockfile the
    tests actually run against. That second one is the failure mode the audit
    describes -- the image "drifted to a different Python and a different torch
    from the one the tests run against" -- and it is entirely visible from the
    files.
    """

    def _copy_sources(self) -> list[str]:
        """Every path a COPY reads from the build context."""
        sources: list[str] = []
        for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.upper().startswith("COPY "):
                continue
            parts = stripped.split()[1:]
            # Skip flags (--from=, --chown=); the last token is the destination.
            operands = [p for p in parts if not p.startswith("--")]
            if len(operands) < 2:
                continue
            if any(p.startswith("--from=") for p in parts):
                continue  # from another stage, not from the context
            sources.extend(operands[:-1])
        return sources

    def test_every_copy_reads_something_that_exists(self):
        """A COPY of a path that is not in the repository fails the build at
        that layer, after everything before it has been paid for."""
        missing = [
            source for source in self._copy_sources()
            if not (PROJECT_ROOT / source.rstrip("/")).exists()
        ]
        assert not missing, (
            f"the Dockerfile copies {missing}, which are not in the repository"
        )

    def test_the_copies_cover_what_the_entrypoint_needs(self):
        """``ENTRYPOINT ["python", "main.py"]`` cannot run without these."""
        sources = {s.rstrip("/") for s in self._copy_sources()}
        for needed in ("src", "main.py", "config.yaml", "advanced-config"):
            assert needed in sources, f"{needed} never reaches the image"

    def test_nothing_the_image_needs_is_excluded_by_dockerignore(self):
        """A path can be present in the repository and still absent from the
        build context, which fails the same way and is harder to see."""
        ignore = PROJECT_ROOT / ".dockerignore"
        if not ignore.exists():
            pytest.skip("no .dockerignore")
        patterns = {
            line.strip() for line in ignore.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
            and not line.strip().startswith("!")
        }
        for needed in ("src", "main.py", "config.yaml", "advanced-config"):
            assert needed not in patterns, (
                f".dockerignore excludes {needed}, which the Dockerfile copies"
            )

    def test_the_cpu_stage_pins_the_torch_the_lockfile_pins(self):
        """The CPU stage installs torch from the CPU index by an inline pin
        rather than from the lockfile, so nothing but a test keeps the two in
        step -- and the version the tests run against is the lockfile's."""
        text = DOCKERFILE.read_text(encoding="utf-8")
        for package in ("torch", "torchvision"):
            pinned = set(re.findall(rf"{package}==([\d.]+)", text))
            assert pinned == {_lock_version(package)}, (
                f"Dockerfile pins {package} {pinned}, lockfile has "
                f"{_lock_version(package)}"
            )

    def test_the_versions_ci_asserts_are_the_versions_that_get_installed(self):
        """The CI job runs python inside the image and asserts a numpy major
        and a torch minor. Those literals and the lockfile are two statements
        of one fact, and only this keeps them from disagreeing."""
        ci_text = CI.read_text(encoding="utf-8")
        for package, pattern in (
            ("numpy", r'numpy\.__version__\.startswith\("([\d.]+)"\)'),
            ("torch", r'torch\.__version__\.startswith\("([\d.]+)"\)'),
        ):
            asserted = re.search(pattern, ci_text)
            assert asserted, f"CI no longer asserts a {package} version"
            prefix = asserted.group(1)
            assert _lock_version(package).startswith(prefix), (
                f"CI asserts {package} {prefix}*, lockfile has "
                f"{_lock_version(package)}"
            )


class TestComposeRunsAfterAClone:
    def test_no_developer_paths_remain(self):
        compose = COMPOSE.read_text(encoding="utf-8")
        assert "D:/Your/Data/Path" not in compose
        assert not re.search(r"-\s+[A-Za-z]:/(?!app)", compose), (
            "an absolute host path is hardcoded; the project must start after a "
            "clone without editing the compose file"
        )

    def test_the_data_directory_is_required_and_explained(self):
        compose = COMPOSE.read_text(encoding="utf-8")
        assert "${DRAFTS_DATA_DIR:?" in compose, (
            "compose should fail with an actionable message, not create a "
            "directory named after a sample path"
        )

    def test_config_is_mounted_read_only(self):
        compose = COMPOSE.read_text(encoding="utf-8")
        assert "/app/config.yaml:rw" not in compose
        assert compose.count("/app/config.yaml:ro") == 2

    def test_logs_are_persisted_outside_the_container(self):
        compose = COMPOSE.read_text(encoding="utf-8")
        assert compose.count("./logs:/app/logs:rw") == 2, (
            "the documented way to run this is `run --rm`, which takes the log "
            "with the container unless the directory is mounted"
        )
        assert "DRAFTS_LOG_DIR=/app/logs" in compose

    def test_an_env_example_exists(self):
        example = PROJECT_ROOT / ".env.example"
        assert example.exists()
        assert "DRAFTS_DATA_DIR" in example.read_text(encoding="utf-8")


class TestContinuousIntegration:
    def _ci(self) -> dict:
        return yaml.safe_load(CI.read_text(encoding="utf-8"))

    def test_token_permissions_are_declared(self):
        assert self._ci().get("permissions") == {"contents": "read"}, (
            "without this the job inherits the repository default, which is "
            "read-write on contents in older configurations"
        )

    def test_the_image_is_built(self):
        assert "docker" in self._ci()["jobs"], (
            "the container is the documented way to run this and was never built "
            "in CI, which is how it drifted"
        )

    def test_the_lockfile_is_installed_with_hashes(self):
        assert "--require-hashes -r requirements.lock.txt" in CI.read_text(encoding="utf-8")

    def test_a_static_check_gate_exists(self):
        ci = self._ci()
        assert "lint" in ci["jobs"]
        steps = " ".join(str(step) for step in ci["jobs"]["lint"]["steps"])
        assert "ruff check" in steps and "F821" in steps

    def test_the_mutation_job_initialises_its_sessions(self):
        """`cosmic-ray exec` needs a session database; without `init` the job
        could never have run at all."""
        # Ignore comments: one of them mentions `cosmic-ray exec` while
        # explaining why init is needed.
        commands = [
            line for line in CI.read_text(encoding="utf-8").splitlines()
            if "cosmic-ray" in line and not line.strip().startswith("#")
        ]
        text = chr(10).join(commands)
        assert "cosmic-ray init" in text
        assert text.index("cosmic-ray init") < text.index("cosmic-ray exec")

    def test_windows_is_covered(self):
        ci = self._ci()
        matrix = str(ci["jobs"]["test"]["strategy"]["matrix"])
        assert "windows" in matrix, (
            "console encoding and path defects only show up on Windows"
        )


class TestConfigurationIsInternallyConsistent:
    def test_current_markers_point_at_the_current_values(self):
        """The file used to mark 0.15 as [CURRENT] while the value was 0.10, and
        document a threshold table computed for the wrong fraction."""
        text = CONFIG_YAML.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        performance = data["performance"]

        checks = {
            "max_ram_fraction": performance["max_ram_fraction"],
            "max_chunk_samples": performance["max_chunk_samples"],
            "max_dm_cube_size_gb": performance["max_dm_cube_size_gb"],
        }
        for line in text.splitlines():
            if "[CURRENT]" not in line or not line.strip().startswith("#"):
                continue
            numbers = re.findall(r"[\d][\d,.]*", line)
            assert numbers, f"[CURRENT] marker with no value: {line!r}"
            normalised = {n.replace(",", "").rstrip(".") for n in numbers}
            assert any(
                str(value) in normalised or f"{value:.2f}" in normalised
                or str(int(value)) in normalised
                for value in checks.values()
            ), f"[CURRENT] marks a value that is not set anywhere: {line.strip()!r}"

    def test_logging_yaml_actually_drives_the_logger(self):
        """These keys were loaded into a dict and never read again."""
        from src.config import config

        logging_yaml = yaml.safe_load(
            (PROJECT_ROOT / "advanced-config/logging.yaml").read_text(encoding="utf-8")
        )
        general = logging_yaml["general"]
        assert config.LOG_LEVEL == str(general["level"]).upper()
        assert config.LOG_MAX_BYTES == int(general["max_file_size_mb"] * 1024 * 1024)
        assert config.LOG_BACKUP_COUNT == int(general["backup_count"])

    def test_cpu_threads_reaches_the_thread_settings(self):
        from src.config import config

        assert hasattr(config, "CPU_THREADS")
        source = (PROJECT_ROOT / "src/core/pipeline.py").read_text(encoding="utf-8")
        assert "apply_thread_settings(hw, user_threads=" in source, (
            "the option exists in performance.yaml and used to do nothing"
        )

    def test_the_log_level_is_not_a_literal(self):
        source = (PROJECT_ROOT / "src/core/pipeline.py").read_text(encoding="utf-8")
        assert 'setup_logging(level="INFO"' not in source
        assert "LOG_LEVEL" in source


class TestLogRotation:
    def test_the_file_handler_rotates(self):
        source = (PROJECT_ROOT / "src/log_utils/logging_config.py").read_text(encoding="utf-8")
        assert "RotatingFileHandler" in source, (
            "a run lasting days writes an unbounded file otherwise"
        )
        assert "logging.FileHandler(" not in source

    def test_the_destination_can_be_moved_out_of_the_source_tree(self, tmp_path, monkeypatch):
        import importlib
        import logging as stdlib_logging

        monkeypatch.setenv("DRAFTS_LOG_DIR", str(tmp_path))
        module = importlib.import_module("src.log_utils.logging_config")
        stdlib_logging.getLogger("DRAFTS_ROTATION_TEST").handlers.clear()
        module.DRAFTSLogger(name="DRAFTS_ROTATION_TEST", level="INFO", use_colors=False)

        assert list(tmp_path.glob("drafts_pipeline_*.log")), (
            "DRAFTS_LOG_DIR was ignored; in a container the log stays in the "
            "writable layer and disappears with `run --rm`"
        )


class TestNoPackageShadowsTheStandardLibrary:
    """`src/logging/` shadowed the stdlib `logging` module.

    With src/ on sys.path -- which seven scripts in this repository put there --
    any library doing `import logging` got the project's package instead, and
    the project's own logging_config subclasses `logging.Formatter`, so the
    import failed half-initialised. `import pandas` was enough to trigger it.
    The seven scripts survived only because they happened to import logging
    before inserting the path.
    """

    def test_importing_a_third_party_library_works_with_src_on_the_path(self):
        import subprocess

        script = (
            "import sys; sys.path.insert(0, r'%s');"
            "import logging; assert hasattr(logging, 'Formatter'), logging.__file__;"
            "import json, csv;"
            "print(logging.__file__)"
        ) % str(PROJECT_ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert "src" not in result.stdout.replace(str(PROJECT_ROOT), ""), (
            f"stdlib logging resolved to a project package: {result.stdout!r}"
        )

    def test_no_package_is_named_after_a_standard_module(self):
        import sysconfig

        stdlib = {
            path.stem
            for path in Path(sysconfig.get_paths()["stdlib"]).glob("*.py")
        } | {"logging", "json", "csv", "types", "typing", "queue", "select", "signal"}
        for package in (PROJECT_ROOT / "src").iterdir():
            if package.is_dir() and (package / "__init__.py").exists():
                assert package.name not in stdlib, (
                    f"src/{package.name}/ shadows the standard library module "
                    f"'{package.name}' whenever src/ is on sys.path"
                )


class TestChunkSizeFallbackDoesNotCrash:
    def test_calculate_optimal_chunk_size_returns_a_size(self):
        """It referenced a name defined in another function, so the fallback --
        which only runs once the primary budget calculation has failed -- raised
        NameError and took the whole file down with it."""
        import numpy as np

        from src.config import config
        from src.preprocessing.slice_len_calculator import calculate_optimal_chunk_size

        config.FILE_LENG = 1_000_000
        config.FREQ_RESO = 64
        config.TIME_RESO = 0.001
        config.DOWN_TIME_RATE = 1
        config.DOWN_FREQ_RATE = 1
        config.FREQ = np.linspace(1200.0, 1500.0, 64)
        config.DM_min, config.DM_max = 0.0, 500.0

        result = calculate_optimal_chunk_size(slice_len=512)
        assert isinstance(result, int) and result > 0
