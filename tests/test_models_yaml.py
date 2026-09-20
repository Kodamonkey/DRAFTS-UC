"""``advanced-config/models.yaml`` is read, and says only what it can deliver.

Audit item 27 (P2-29, REF-07). The file used to be 177 lines that
``user_config.py`` loaded into a dictionary nothing ever looked at: an operator
who edited a threshold, a batch size or a weights path here changed nothing.
The resolution was per key -- wire up what a run can honour, delete what
describes an intention rather than a behaviour -- and these tests hold both
halves in place.

The second half is the one that rots. A configuration file grows keys; nothing
stops a key being added with no consumer, and then the file is back where it
started. ``test_every_key_in_the_file_is_read`` is what stops it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from src.config.user_config import resolve_model_settings

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = REPO_ROOT / "advanced-config" / "models.yaml"


def _leaf_keys(node, prefix: str = "") -> set[str]:
    """Every dotted path in a nested mapping that ends at a value."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict) and value:
                found |= _leaf_keys(value, path)
            else:
                found.add(path)
    return found


class TestTheFileIsRead:
    def test_the_defaults_are_the_literals_config_used_to_carry(self, tmp_path):
        """An installation with no models.yaml, or with the keys removed, must
        behave exactly as before the file was wired."""
        settings = resolve_model_settings({}, tmp_path)
        assert settings["MODEL_NAME"] == "resnet18"
        assert settings["CLASS_MODEL_NAME"] == "resnet18"
        assert settings["MODEL_DIR"] == (tmp_path / "models").resolve()
        assert settings["MODEL_PATH"].name == "cent_resnet18.pth"
        assert settings["CLASS_MODEL_PATH"].name == "class_resnet18.pth"

    def test_the_shipped_file_resolves_to_what_the_literals_produced(self, tmp_path):
        """The file as committed must not change the installed behaviour. If it
        did, wiring it up would have been a silent migration."""
        shipped = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
        assert resolve_model_settings(shipped, tmp_path) == resolve_model_settings({}, tmp_path)

    def test_a_different_weights_file_actually_reaches_the_path(self, tmp_path):
        """The point of the whole item: editing the file changes what loads."""
        settings = resolve_model_settings(
            {"paths": {"model_dir": "weights",
                       "detection_model": "cent_custom.pth",
                       "classification_model": "class_custom.pth"}},
            tmp_path,
        )
        assert settings["MODEL_DIR"] == (tmp_path / "weights").resolve()
        assert settings["MODEL_PATH"] == (tmp_path / "weights" / "cent_custom.pth").resolve()
        assert settings["CLASS_MODEL_PATH"] == (tmp_path / "weights" / "class_custom.pth").resolve()

    def test_a_different_backbone_reaches_both_the_name_and_the_filename(self, tmp_path):
        """The name is passed to ``centernet(model_name=...)`` AND used to build
        the conventional file name, so changing it must move both."""
        settings = resolve_model_settings(
            {"architecture": {"detection": {"name": "resnet50"},
                              "classification": {"name": "resnet34"}}},
            tmp_path,
        )
        assert settings["MODEL_NAME"] == "resnet50"
        assert settings["CLASS_MODEL_NAME"] == "resnet34"
        assert settings["MODEL_PATH"].name == "cent_resnet50.pth"
        assert settings["CLASS_MODEL_PATH"].name == "class_resnet34.pth"

    def test_an_explicit_file_name_wins_over_the_convention(self, tmp_path):
        settings = resolve_model_settings(
            {"architecture": {"detection": {"name": "resnet50"}},
             "paths": {"detection_model": "whatever.pth"}},
            tmp_path,
        )
        assert settings["MODEL_NAME"] == "resnet50"
        assert settings["MODEL_PATH"].name == "whatever.pth"

    @pytest.mark.parametrize("empty", [None, "", {}])
    def test_an_empty_value_falls_back_rather_than_producing_a_broken_path(
        self, tmp_path, empty
    ):
        """A key left blank in YAML arrives as None. Interpolating that into a
        path gives 'None.pth', which fails at load time with a confusing
        error rather than at configuration time."""
        settings = resolve_model_settings(
            {"paths": {"detection_model": empty, "model_dir": empty}}, tmp_path
        )
        assert settings["MODEL_PATH"] == (tmp_path / "models" / "cent_resnet18.pth").resolve()

    def test_the_whole_mapping_being_absent_is_not_an_error(self, tmp_path):
        assert resolve_model_settings(None, tmp_path)["MODEL_NAME"] == "resnet18"


class TestTheFileOnlyPromisesWhatItDelivers:
    """The half that rots. Every key here must have a consumer; a key with none
    is the defect item 27 was about, and it costs nothing to add one."""

    #: Exactly the keys ``resolve_model_settings`` reads. Adding a key to the
    #: YAML without adding it here -- and to the resolver -- fails this test.
    CONSUMED = {
        "paths.model_dir",
        "paths.detection_model",
        "paths.classification_model",
        "architecture.detection.name",
        "architecture.classification.name",
    }

    def test_every_key_in_the_file_is_read(self):
        shipped = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
        keys = _leaf_keys(shipped)
        unread = keys - self.CONSUMED
        assert not unread, (
            f"models.yaml declares {sorted(unread)}, which nothing reads. "
            "Either wire the key up in resolve_model_settings or delete it: a "
            "key that describes an intention rather than a behaviour is what "
            "audit item 27 was about."
        )

    def test_every_key_the_resolver_reads_is_in_the_file(self):
        """The other direction. A documented key that vanished from the file
        still works (the resolver defaults it), but the file stops describing
        what can be configured."""
        shipped = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
        missing = self.CONSUMED - _leaf_keys(shipped)
        assert not missing, missing

    @pytest.mark.parametrize("gone", [
        "inference", "detection_model", "classification_model",
        "validation", "logging", "experimental",
    ])
    def test_the_sections_with_no_implementation_are_gone(self, gone):
        """Named individually so the deletions cannot be quietly undone.

        ``validation.check_weights_integrity`` was the worst of them: it
        declared an integrity check against a reference checksum that does not
        exist anywhere in the repository (audit P2-37).
        """
        shipped = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
        assert gone not in shipped

    def test_the_thresholds_live_in_config_yaml_and_only_there(self):
        """``detection.det_prob`` and ``classification.class_prob`` in
        config.yaml are read. models.yaml used to declare the same two numbers
        under different names, which is how two files end up disagreeing --
        audit P2-27 is exactly that bug for prewhitening."""
        shipped = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
        flat = _leaf_keys(shipped)
        assert not {k for k in flat if "threshold" in k}, flat
