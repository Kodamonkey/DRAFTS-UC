"""The candidate CSV and the checkpoint are written and read as UTF-8.

Found by a Windows CI failure. Two tests of this project's own making read a
source file with ``Path.read_text()`` and no encoding; on Linux that is UTF-8
and the omission is invisible, on a Windows runner it is cp1252, and
``src/core/pipeline.py`` contains ``pc cm⁻³`` in a log line::

    UnicodeDecodeError: 'charmap' codec can't decode byte 0x81

Chasing it turned up the same defect in production, and there it was worse
than an omission: ``candidate_manager`` **read** the candidate CSV with
``encoding="utf-8"`` on one line and **wrote** it with the locale encoding on
four others. On Linux those agree and nothing shows. On Windows the writer is
cp1252 and the reader is UTF-8, so the moment a row carries a byte above 127
-- a source or file name outside ASCII is all it takes -- the file cannot be
read back by the code that wrote it.

These tests pin the round trip with non-ASCII content, which is the only input
that can tell the two encodings apart.
"""
from __future__ import annotations

import csv
import json

from src.output.candidate_manager import (
    CANDIDATE_HEADER,
    CandidateWriter,
    append_candidate,
    ensure_csv_header,
)

#: Characters chosen to break each wrong answer in turn. "ñ" and "é" survive
#: cp1252 and would hide the defect on their own; "π", "⁻³" and "日" do not
#: exist in cp1252 at all, so a cp1252 writer raises on them rather than
#: silently producing bytes a UTF-8 reader will mangle. Both failure modes
#: matter and this string carries both.
NON_ASCII = "FRB-ñé-π-cm⁻³-日"


def _row(name: str) -> list:
    row = [""] * len(CANDIDATE_HEADER)
    row[0] = name
    row[CANDIDATE_HEADER.index("dm_status")] = "measured"
    return row


class TestTheCandidateCSVRoundTrips:
    def test_a_non_ascii_file_name_survives_the_write_and_the_read(self, tmp_path):
        """The writer and the reader have to agree. They did not: one line said
        UTF-8 and four did not say anything."""
        csv_path = tmp_path / "cands.csv"
        ensure_csv_header(csv_path)
        append_candidate(csv_path, _row(f"{NON_ASCII}.fits"))
        CandidateWriter.flush_all()

        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["file"] == f"{NON_ASCII}.fits"

    def test_the_bytes_on_disk_are_utf8(self, tmp_path):
        """Not merely 'readable by the reader we happen to use': the file is
        UTF-8, which is what every other consumer of this CSV will assume."""
        csv_path = tmp_path / "cands.csv"
        ensure_csv_header(csv_path)
        append_candidate(csv_path, _row(f"{NON_ASCII}.fits"))
        CandidateWriter.flush_all()

        raw = csv_path.read_bytes()
        assert NON_ASCII.encode("utf-8") in raw
        raw.decode("utf-8")  # raises if it is anything else

    def test_the_header_repair_path_keeps_the_encoding(self, tmp_path):
        """``ensure_csv_header`` rewrites a file whose header has drifted. That
        read-modify-write is three of the four call sites that had no encoding,
        and it is the one that touches rows somebody else already wrote."""
        csv_path = tmp_path / "cands.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "something_else"])
            writer.writerow([f"{NON_ASCII}.fits", "x"])

        ensure_csv_header(csv_path)

        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == CANDIDATE_HEADER
        assert rows[1][0] == f"{NON_ASCII}.fits"

    def test_an_ascii_only_file_is_byte_identical_either_way(self, tmp_path):
        """The change is a no-op wherever the locale is already UTF-8, which is
        why the golden CSV does not move. Stated as a test so that stays true."""
        csv_path = tmp_path / "cands.csv"
        ensure_csv_header(csv_path)
        append_candidate(csv_path, _row("plain.fits"))
        CandidateWriter.flush_all()

        raw = csv_path.read_bytes()
        assert raw.decode("ascii") == raw.decode("utf-8")


class TestTheCheckpointRoundTrips:
    def test_a_non_ascii_file_stem_survives(self, tmp_path):
        """The checkpoint is keyed by the file stem and stores the run
        fingerprint. Writer and reader used the locale on both sides, so they
        agreed on one machine and disagreed across two."""
        from src.core.checkpoint import load_checkpoint, save_checkpoint

        save_checkpoint(tmp_path, NON_ASCII, chunk_idx=3, total_chunks=9,
                        fingerprint=f"fp-{NON_ASCII}")
        assert load_checkpoint(tmp_path, NON_ASCII, f"fp-{NON_ASCII}") == 3

    def test_the_checkpoint_on_disk_is_utf8_json(self, tmp_path):
        from src.core.checkpoint import save_checkpoint

        save_checkpoint(tmp_path, NON_ASCII, chunk_idx=1, total_chunks=2,
                        fingerprint=f"fp-{NON_ASCII}")
        written = [p for p in tmp_path.rglob("*") if p.is_file()]
        assert written, "save_checkpoint wrote nothing"
        for path in written:
            payload = json.loads(path.read_bytes().decode("utf-8"))
            assert payload

    def test_a_checkpoint_written_as_utf8_elsewhere_is_read_back(self, tmp_path):
        """The cross-machine case: bytes written as UTF-8 on another host must
        load here whatever this machine's locale is.

        The payload is produced by ``save_checkpoint`` and then rewritten as
        explicit UTF-8 bytes, rather than hand-built -- a hand-built one pins
        the schema as well as the encoding, and when the schema moves the test
        starts skipping, which is a test that has stopped testing.
        """
        from src.core.checkpoint import (
            _checkpoint_path,
            load_checkpoint,
            save_checkpoint,
        )

        save_checkpoint(tmp_path, NON_ASCII, chunk_idx=5, total_chunks=10,
                        fingerprint=f"fp-{NON_ASCII}")
        cp_path = _checkpoint_path(tmp_path, NON_ASCII)
        payload = json.loads(cp_path.read_bytes().decode("utf-8"))
        cp_path.write_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )

        assert load_checkpoint(tmp_path, NON_ASCII, f"fp-{NON_ASCII}") == 5
