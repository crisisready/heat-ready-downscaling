import json
import os
import sys
from datetime import date
from unittest.mock import patch

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from heatready_downscaling import ssod_guard as sg  # noqa: E402


def test_affected_is_non_us_ghcnd_on_or_after_the_switch():
    assert sg.affected("SPE00119882", "2025-01-01")
    assert sg.affected("SPE00119882", date(2025, 6, 3))
    assert not sg.affected("SPE00119882", "2024-12-31")
    assert not sg.affected("USW00094728", "2025-03-01")      # US stations are not switched
    assert not sg.affected("EC000309", "2025-03-01")         # ECA&D, 8 characters
    assert not sg.affected("AEMET12345", "2025-03-01")       # 10 characters, not GHCN-D


def test_apply_guard_drops_unlisted_affected_rows_only():
    rows = [{"station_id": s, "date": d} for s, d in [("SPE00000001", "2025-02-01"), ("SPE00000001", "2024-02-01"),
                                                       ("SPE00000002", "2025-02-01"), ("USW00000003", "2025-02-01")]]
    kept, rep = sg.apply_guard(rows, {"SPE00000001"})
    assert [(r["station_id"], r["date"]) for r in kept] == [("SPE00000001", "2025-02-01"), ("SPE00000001", "2024-02-01"),
                                                           ("USW00000003", "2025-02-01")]
    assert rep == {"rows_dropped": 1, "stations_dropped": 1, "rows_affected_kept": 1}


def test_read_allowlist_header_only_means_drop_everything(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("station_id\n")
    assert sg.read_allowlist(str(p)) == set()
    p.write_text("station_id,twin,n\nSPE00000001,1-2,200\n")
    assert sg.read_allowlist(str(p)) == {"SPE00000001"}


def test_committed_allowlist_is_well_formed():
    path = os.path.join(_ROOT, "scorecard", "ssod_guard", "allowlist_2025.csv")
    ids = sg.read_allowlist(path)
    assert ids and all(len(i) == 11 and not i.startswith("US") for i in ids)


def _spanish_rows():
    from test_train_downscaling import _make_rows
    rows = _make_rows(80, ["US", "FR", "CH"], {"US": "Cfa", "FR": "Cfb", "CH": "Dwa"})
    for r in rows:
        k = int(r["station_id"].split("_")[1])
        r["station_id"] = f"{r['region']}_{k % 4}"
        r["date"] = date(2023, 1, 1 + k // 4)
    extra = [{**r, "station_id": "SPE00000001", "region": "SP", "date": date(2025, 1, 2 + i)} for i, r in enumerate(rows[:3])]
    return rows + extra


def test_trainer_refuses_affected_rows_without_allowlist_and_records_guard_with_one(tmp_path):
    import train_downscaling as td
    rows = _spanish_rows()
    base = ["t", "--model-version", "x", "--candidate-only", "--cv-n-jobs", "1"]
    with patch.object(td, "load_training_rows", return_value=rows), patch.object(td, "_bucket_from_credentials", return_value="b"), \
         patch("sys.argv", base):
        with pytest.raises(SystemExit, match="SSOD-v2"):
            td.main()
    allow = tmp_path / "a.csv"
    allow.write_text("station_id\n")
    with patch.object(td, "load_training_rows", return_value=rows), patch.object(td, "_bucket_from_credentials", return_value="b"), \
         patch.object(td, "_MIN_FOLD_TRAIN_ROWS", 10), patch.object(td, "save_model_artifacts") as save, \
         patch("sys.argv", base + ["--ssod-allowlist", str(allow)]):
        td.main()
    meta = save.call_args[0][3]
    assert meta["ssod_guard"]["rows_dropped"] == 3 and meta["ssod_guard"]["rows_affected_kept"] == 0
    assert meta["training_rows"] == len(rows) - 3
