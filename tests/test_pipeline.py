"""Unit tests on a tiny hand-made panel (no download needed):  pytest -q

They pin down the three rules that matter most for leakage and correctness:
  1. YTD de-cumulation: Q1 = YTD, Qk = YTD_k - YTD_(k-1) only for consecutive quarters of the same year.
  2. Label: y=1 iff the next failure is strictly after REPDTE and <= end of the 4th following quarter.
  3. Features at q do not change when rows after q are deleted (point-in-time).
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

TMP = Path(tempfile.mkdtemp())
os.environ["FDIC_DATA_DIR"] = str(TMP)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import config as C  # noqa: E402
import polars as pl  # noqa: E402


def _write_fixture():
    raw = C.RAW_DIR / "financials" / "REPDTE=all"
    raw.mkdir(parents=True, exist_ok=True)
    # bank 1: reports 2007Q1..2008Q4, fails 2009-01-16  (failure 1 quarter after the last report)
    # bank 2: reports 2007Q1, 2007Q2, (gap 2007Q3), 2007Q4, 2008Q1 ; never fails
    rows = []
    for i, d in enumerate(["20070331", "20070630", "20070930", "20071231",
                           "20080331", "20080630", "20080930", "20081231"]):
        rows.append({"CERT": 1, "REPDTE": d, "NETINC": 10.0 * ((i % 4) + 1)})   # YTD 10,20,30,40
    for d, ni in [("20070331", 5.0), ("20070630", 9.0), ("20071231", 30.0), ("20080331", 4.0)]:
        rows.append({"CERT": 2, "REPDTE": d, "NETINC": ni})
    cols = {f: [] for f in C.FIN_FIELDS}
    for r in rows:
        for f in C.FIN_FIELDS:
            if f == "CERT":
                cols[f].append(r["CERT"])
            elif f == "REPDTE":
                cols[f].append(r["REPDTE"])
            elif f == "STNAME":
                cols[f].append("Test")
            elif f in C.YTD_FLOW_FIELDS:
                cols[f].append(r["NETINC"])
            else:
                cols[f].append(100.0)
    schema = pa.schema([pa.field("CERT", pa.int32()), pa.field("REPDTE", pa.string()), pa.field("STNAME", pa.string())]
                       + [pa.field(f, pa.float64()) for f in C.FIN_FIELDS[3:]])
    pq.write_table(pa.table(cols, schema=schema), raw / "part-0.parquet")
    fails = {f: [None] for f in C.FAIL_FIELDS}
    fails.update({"CERT": ["1"], "FAILDATE": ["1/16/2009"], "RESTYPE": ["FAILURE"], "COST": ["50"], "QBFASSET": ["100"]})
    pq.write_table(pa.table({k: pa.array(v, type=pa.string()) for k, v in fails.items()}), C.RAW_DIR / "failures.parquet")


_write_fixture()
from s04_build_panel import panel_lazy  # noqa: E402
from s07_features import assert_point_in_time  # noqa: E402

PANEL = panel_lazy().collect().sort(["CERT", "REPDTE"])


def test_decumulation_consecutive_same_year():
    b1 = PANEL.filter(pl.col("CERT") == 1)["NETINC_Q"].to_list()
    assert b1 == [10, 10, 10, 10, 10, 10, 10, 10]


def test_decumulation_gap_gives_null():
    b2 = PANEL.filter(pl.col("CERT") == 2)["NETINC_Q"].to_list()
    # 2007Q1=5, Q2=9-5=4, Q4 after a gap (Q3 missing) -> null, 2008Q1=4 (new year)
    assert b2 == [5, 4, None, 4]


def test_label_window_is_future_only_and_four_quarters():
    b1 = PANEL.filter(pl.col("CERT") == 1).select("REPDTE", "y")
    y = dict(zip(b1["REPDTE"].to_list(), b1["y"].to_list()))
    # failure 2009-01-16 lies in (q, q+4Q] for q = 2008Q1 .. 2008Q4 only
    assert y[date(2007, 12, 31)] == 0     # window ends 2008-12-31 < failure
    assert y[date(2008, 3, 31)] == 1      # window ends 2009-03-31 >= failure
    assert y[date(2008, 12, 31)] == 1
    assert PANEL.filter(pl.col("CERT") == 2)["y"].sum() == 0


def test_features_point_in_time():
    r = assert_point_in_time(PANEL, date(2007, 12, 31))
    assert r["status"] == "PASS"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
