import sys, os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from DRAFTS.candidate import Candidate


def test_to_row():
    cand = Candidate(
        file="test.fits",
        slice_id=1,
        band_id=0,
        prob=0.9,
        dm=10.0,
        t_sec=0.5,
        t_sample=100,
        box=(1, 2, 3, 4),
        snr=6.5,
        class_prob=0.8,
        is_burst=True,
        patch_file="patch.png",
    )
    row = cand.to_row()
    assert row[0] == "test.fits"
    assert row[-1] == "patch.png"
