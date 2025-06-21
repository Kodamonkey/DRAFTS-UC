import sys, os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from DRAFTS.pipeline import _slice_parameters


def test_slice_parameters():
    assert _slice_parameters(0, 512) == (0, 0)
    assert _slice_parameters(100, 512) == (100, 1)
    assert _slice_parameters(1024, 512) == (512, 2)
