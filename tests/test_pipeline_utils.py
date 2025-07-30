import sys, os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from DRAFTS.pipeline import _slice_parameters


def test_slice_parameters():
    assert _slice_parameters(0, 512) == (0, 0)
    assert _slice_parameters(100, 512) == (100, 1)
    assert _slice_parameters(1024, 512) == (512, 2)
from pathlib import Path
from DRAFTS import config
from DRAFTS.pipeline import _find_data_files

def test_find_data_files(tmp_path):
    (tmp_path / "A_test.fits").touch()
    (tmp_path / "A_test.fil").touch()
    config.DATA_DIR = tmp_path
    files = _find_data_files("A_test")
    assert len(files) == 2
    assert any(f.suffix == ".fits" for f in files)
    assert any(f.suffix == ".fil" for f in files)

