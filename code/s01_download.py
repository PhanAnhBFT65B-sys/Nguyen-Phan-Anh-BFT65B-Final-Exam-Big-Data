"""Step 1 - Acquisition: download FDIC BankFind data through the public API.

ADAPTED FROM the course starter script `fdic_download_starter.py` (BFT65B course page).
Changes made for this project (documented in the report, Part 1):
  * extended field list (config.FIN_FIELDS: starter's 30 fields + 8 extra fields);
  * explicit, typed Arrow schema written for every quarter (no type drift between files);
  * Hive-style partition layout  raw/financials/REPDTE=YYYYMMDD/part-0.parquet, zstd;
  * download log with timing per quarter and the API index name (data vintage);
  * resumable: a quarter whose file exists AND is marked complete in the manifest is skipped;
  * completeness verification against the API's own `meta.total` and the grand total.

Usage
    python code/s01_download.py --start 2023-03-31 --end 2023-12-31   # quick test (4 quarters)
    python code/s01_download.py                                         # full 1984Q1-2026Q2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import requests

import config as C

BASE = "https://api.fdic.gov/banks"
PAGE_SIZE = 10_000  # maximum rows the API returns per call

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "BFT65B-student-pipeline/1.0"})

STRING_COLS = {"REPDTE", "STNAME"}


def fin_schema() -> pa.Schema:
    """Explicit schema: CERT int32, REPDTE/STNAME string, all amounts/ratios float64.

    float64 (not int64) because the API returns ratios as decimals and amounts can be null;
    a fixed schema prevents one quarter being inferred as int and another as double.
    """
    fields = []
    for f in C.FIN_FIELDS:
        if f == "CERT":
            fields.append(pa.field(f, pa.int32()))
        elif f in STRING_COLS:
            fields.append(pa.field(f, pa.string()))
        else:
            fields.append(pa.field(f, pa.float64()))
    return pa.schema(fields)


def get_json(endpoint: str, params: dict, max_retries: int = 6, pause: float = 0.5) -> dict:
    """GET with retry and exponential back-off (from the starter script)."""
    url = f"{BASE}/{endpoint}"
    for attempt in range(1, max_retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=180)
            if r.status_code == 200:
                time.sleep(pause)
                return r.json()
            print(f"    HTTP {r.status_code} (attempt {attempt}) - retrying", flush=True)
        except requests.RequestException as exc:
            print(f"    {type(exc).__name__} (attempt {attempt}) - retrying", flush=True)
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Giving up on {url} with {params}")


def fetch_all(endpoint: str, fields: list[str], filters: str | None = None,
              sort_by: str = "CERT") -> tuple[list[dict], int, str]:
    """Page through an endpoint with limit/offset. Returns (rows, api_total, index_name)."""
    rows, offset, total, index_name = [], 0, None, ""
    while True:
        params = {"fields": ",".join(fields), "limit": PAGE_SIZE, "offset": offset,
                  "sort_by": sort_by, "sort_order": "ASC", "format": "json"}
        if filters:
            params["filters"] = filters
        js = get_json(endpoint, params)
        total = js["meta"]["total"]
        index_name = js["meta"].get("index", {}).get("name", "")
        page = [rec["data"] for rec in js["data"]]
        rows.extend(page)
        offset += len(page)
        if not page or offset >= total:
            break
    for r in rows:
        r.pop("ID", None)
    return rows, total, index_name


def to_table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    cols = {}
    for f in schema:
        vals = [r.get(f.name) for r in rows]
        if pa.types.is_floating(f.type):
            vals = [None if v is None or v == "" else float(v) for v in vals]
        elif pa.types.is_integer(f.type):
            vals = [None if v is None else int(v) for v in vals]
        cols[f.name] = pa.array(vals, type=f.type)
    return pa.table(cols, schema=schema)


def quarter_ends(start: date, end: date) -> list[str]:
    out = []
    for y in range(start.year, end.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            q = date(y, m, d)
            if start <= q <= end:
                out.append(q.strftime("%Y%m%d"))
    return out


def download_financials(quarters: list[str]) -> None:
    fin_dir = C.RAW_DIR / "financials"
    fin_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = C.RAW_DIR / "manifest_financials.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    schema = fin_schema()

    for i, q in enumerate(quarters, 1):
        target = fin_dir / f"REPDTE={q}" / "part-0.parquet"
        if target.exists() and manifest.get(q, {}).get("complete"):
            print(f"[{i}/{len(quarters)}] {q}: already complete, skipped", flush=True)
            continue
        t0 = time.time()
        rows, total, index_name = fetch_all("financials", C.FIN_FIELDS, filters=f"REPDTE:{q}")
        if not rows:
            print(f"[{i}/{len(quarters)}] {q}: no data (not published yet?)", flush=True)
            continue
        tbl = to_table(rows, schema)
        target.parent.mkdir(parents=True, exist_ok=True)
        # REPDTE is kept inside the file too: files stay self-describing if moved.
        pq.write_table(tbl, target, compression="zstd")
        keys = set(zip(tbl.column("CERT").to_pylist(), tbl.column("REPDTE").to_pylist()))
        complete = tbl.num_rows == total and len(keys) == tbl.num_rows
        manifest[q] = {"rows": tbl.num_rows, "api_total": total, "complete": bool(complete),
                       "seconds": round(time.time() - t0, 2), "api_index": index_name,
                       "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"[{i}/{len(quarters)}] {q}: {tbl.num_rows:>6,} rows (API {total:,}) "
              f"{'OK' if complete else 'INCOMPLETE'} {time.time() - t0:5.1f}s", flush=True)


def download_small_tables() -> None:
    meta = {}
    for name, fields, sort_by in (("failures", C.FAIL_FIELDS, "FAILDATE"),
                                  ("institutions", C.INST_FIELDS, "CERT")):
        t0 = time.time()
        rows, total, index_name = fetch_all(name, fields, sort_by=sort_by)
        # Small tables: keep every field as string in the raw layer; typing happens in s04.
        cols = {f: pa.array([None if r.get(f) is None else str(r.get(f)) for r in rows],
                            type=pa.string()) for f in fields}
        pq.write_table(pa.table(cols), C.RAW_DIR / f"{name}.parquet", compression="zstd")
        meta[name] = {"rows": len(rows), "api_total": total, "api_index": index_name,
                      "complete": len(rows) == total}
        print(f"{name:<12}: {len(rows):>6,} rows (API {total:,}) {time.time() - t0:5.1f}s", flush=True)
    (C.RAW_DIR / "manifest_small_tables.json").write_text(json.dumps(meta, indent=1))


def verify() -> bool:
    """Completeness check: every quarter complete, no duplicate keys, grand total matches API."""
    import polars as pl
    manifest = json.loads((C.RAW_DIR / "manifest_financials.json").read_text())
    panel = pl.scan_parquet(C.RAW_DIR / "financials" / "**" / "*.parquet",
                            hive_partitioning=False).select("CERT", "REPDTE").collect()
    n_dup = panel.height - panel.unique().height
    bad = [q for q, m in manifest.items() if not m["complete"]]
    grand = get_json("financials", {"fields": "CERT", "limit": 1, "format": "json",
                                    "filters": f"REPDTE:[{min(manifest)} TO {max(manifest)}]"})
    api_grand = grand["meta"]["total"]
    res = {"quarters": len(manifest), "rows": panel.height, "api_total_same_range": api_grand,
           "distinct_cert": panel["CERT"].n_unique(), "duplicate_keys": n_dup,
           "incomplete_quarters": bad, "first_q": min(manifest), "last_q": max(manifest)}
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    (C.TAB_DIR / "p1_completeness.json").write_text(json.dumps(res, indent=1))
    print("\n=== Completeness check ===")
    for k, v in res.items():
        print(f"{k:<24}: {v}")
    return n_dup == 0 and not bad and panel.height == api_grand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=C.START_Q)
    ap.add_argument("--end", default=C.END_Q)
    args = ap.parse_args()
    C.ensure_dirs()
    quarters = quarter_ends(date.fromisoformat(args.start), date.fromisoformat(args.end))
    print(f"Downloading {len(quarters)} quarters {quarters[0]} -> {quarters[-1]} into {C.RAW_DIR}\n")
    t0 = time.time()
    download_financials(quarters)
    download_small_tables()
    ok = verify()
    print(f"\nTotal time {time.time() - t0:,.0f}s ; verification {'PASSED' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
