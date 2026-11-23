"""Step 2 - Storage layout benchmark (Part 1).

Question: which on-disk layout should hold 1.68M bank-quarters x 41 columns?
We write the SAME data in six layouts and measure, on this machine:
  size on disk, number of files, write time, and read time for three typical queries
    Q1 full scan (all columns)          - e.g. rebuilding the panel
    Q2 projection (4 columns)            - e.g. a quick size/capital chart
    Q3 projection + filter (year 2008)   - e.g. studying one crisis year
Read times are medians of 3 runs after one warm-up (warm OS page cache, which is the
realistic case on a laptop that reads the same files repeatedly).

Output: outputs/tables/p1_storage_benchmark.csv  and  outputs/figures/p1_storage.png
"""
from __future__ import annotations

import shutil
import time

import pandas as pd
import polars as pl
import pyarrow.parquet as pq

import config as C
from utils import dir_size_mb, n_files, save_json, time_it

COLS4 = ["CERT", "REPDTE", "ASSET", "EQ"]


def load_raw() -> pl.DataFrame:
    return pl.read_parquet(C.RAW_DIR / "financials" / "**" / "*.parquet", hive_partitioning=False)


def write_layouts(df: pl.DataFrame) -> dict:
    B = C.BENCH_DIR
    if B.exists():
        shutil.rmtree(B)
    B.mkdir(parents=True)
    pdf = df.to_pandas()
    sorted_df = df.sort(["REPDTE", "CERT"])
    with_year = sorted_df.with_columns(pl.col("REPDTE").str.slice(0, 4).cast(pl.Int16).alias("YEAR"))
    paths, wt = {}, {}

    def w(name, path, fn):
        t0 = time.perf_counter()
        fn()
        wt[name] = time.perf_counter() - t0
        paths[name] = path

    w("csv", B / "panel.csv", lambda: pdf.to_csv(B / "panel.csv", index=False))
    w("csv_gzip", B / "panel.csv.gz", lambda: pdf.to_csv(B / "panel.csv.gz", index=False, compression="gzip"))
    w("parquet_snappy_1file", B / "panel_snappy.parquet",
      lambda: df.write_parquet(B / "panel_snappy.parquet", compression="snappy"))
    w("parquet_zstd_1file_sorted", B / "panel_zstd.parquet",
      lambda: sorted_df.write_parquet(B / "panel_zstd.parquet", compression="zstd",
                                      row_group_size=100_000, statistics=True))
    w("parquet_zstd_by_year", B / "by_year",
      lambda: with_year.write_parquet(B / "by_year", compression="zstd", partition_by="YEAR"))
    # the raw layer written by s01 (one file per quarter) - copied so timing is comparable
    w("parquet_zstd_by_quarter", B / "by_quarter",
      lambda: shutil.copytree(C.RAW_DIR / "financials", B / "by_quarter"))
    return {"paths": paths, "write_s": wt}


def scan(name: str, path):
    if name.startswith("csv"):
        return pl.scan_csv(path, infer_schema_length=10_000)
    if name in ("parquet_zstd_by_year", "parquet_zstd_by_quarter"):
        return pl.scan_parquet(path / "**" / "*.parquet", hive_partitioning=True)
    return pl.scan_parquet(path)


def queries(name: str, path) -> dict:
    is_csv = name.startswith("csv")
    rep = pl.col("REPDTE").cast(pl.Utf8)
    res = {}
    res["polars_full_s"] = time_it(lambda: scan(name, path).collect())["median_s"]
    res["polars_4cols_s"] = time_it(lambda: scan(name, path).select(COLS4).collect())["median_s"]
    if name == "parquet_zstd_by_year":
        # partition pruning: the YEAR directory key is used directly
        q3 = lambda: scan(name, path).filter(pl.col("YEAR") == 2008).select(COLS4).collect()
    else:
        q3 = lambda: scan(name, path).filter(rep.str.starts_with("2008")).select(COLS4).collect()
    res["polars_2008_4cols_s"] = time_it(q3)["median_s"]
    if is_csv:
        res["pandas_full_s"] = time_it(lambda: pd.read_csv(path), repeat=1)["median_s"]
        res["pandas_4cols_s"] = time_it(lambda: pd.read_csv(path, usecols=COLS4), repeat=1)["median_s"]
    elif name in ("parquet_snappy_1file", "parquet_zstd_1file_sorted"):
        res["pandas_full_s"] = time_it(lambda: pd.read_parquet(path))["median_s"]
        res["pandas_4cols_s"] = time_it(lambda: pd.read_parquet(path, columns=COLS4))["median_s"]
    return res


def schema_check(paths: dict) -> dict:
    """Does the format preserve types? CSV re-infers types; Parquet stores them."""
    csv_schema = pl.read_csv(paths["csv"], n_rows=50_000).schema
    pq_schema = pl.read_parquet_schema(paths["parquet_zstd_1file_sorted"])
    diff = {k: (str(pq_schema[k]), str(csv_schema.get(k))) for k in pq_schema
            if str(pq_schema[k]) != str(csv_schema.get(k))}
    return diff


def main() -> None:
    C.ensure_dirs()
    df = load_raw()
    print(f"rows {df.height:,} cols {df.width}; in-memory polars {df.estimated_size('mb'):.1f} MB; "
          f"pandas {df.to_pandas().memory_usage(deep=True).sum() / 1e6:.1f} MB")
    lay = write_layouts(df)
    rows = []
    for name, path in lay["paths"].items():
        r = {"layout": name, "size_mb": round(dir_size_mb(path), 1), "files": n_files(path),
             "write_s": round(lay["write_s"][name], 2)}
        r.update({k: round(v, 3) for k, v in queries(name, path).items()})
        rows.append(r)
        print(r, flush=True)
    res = pd.DataFrame(rows)
    base = res.loc[res.layout == "csv"].iloc[0]
    res["size_vs_csv"] = (res["size_mb"] / base["size_mb"]).round(3)
    res["full_read_speedup_vs_csv"] = (base["polars_full_s"] / res["polars_full_s"]).round(1)
    res.to_csv(C.TAB_DIR / "p1_storage_benchmark.csv", index=False)

    # row-group statistics of the sorted file: how many row groups can be skipped for 2008?
    md = pq.ParquetFile(lay["paths"]["parquet_zstd_1file_sorted"]).metadata
    rg_total = md.num_row_groups
    idx = md.schema.to_arrow_schema().get_field_index("REPDTE")
    rg_2008 = sum(1 for i in range(rg_total)
                  if md.row_group(i).column(idx).statistics.min <= "20081231"
                  and md.row_group(i).column(idx).statistics.max >= "20080101")
    extra = {"rows": df.height, "cols": df.width,
             "polars_mem_mb": round(df.estimated_size("mb"), 1),
             "pandas_mem_mb": round(df.to_pandas().memory_usage(deep=True).sum() / 1e6, 1),
             "csv_type_changes": schema_check(lay["paths"]),
             "sorted_file_row_groups": rg_total, "row_groups_touching_2008": rg_2008}
    save_json(extra, C.TAB_DIR / "p1_storage_extra.json")
    print(res.to_string())
    print(extra)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    lab = res["layout"].str.replace("parquet_", "pq_")
    ax[0].barh(lab, res["size_mb"], color="#4C72B0")
    ax[0].set_xlabel("Size on disk (MB)")
    ax[0].invert_yaxis()
    x = range(len(res))
    ax[1].barh([i - 0.27 for i in x], res["polars_full_s"], height=0.27, label="Q1 full scan")
    ax[1].barh(list(x), res["polars_4cols_s"], height=0.27, label="Q2 4 columns")
    ax[1].barh([i + 0.27 for i in x], res["polars_2008_4cols_s"], height=0.27, label="Q3 2008, 4 cols")
    ax[1].set_yticks(list(x), lab)
    ax[1].invert_yaxis()
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Polars read time (s, log scale, median of 3)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "p1_storage.png", dpi=150)


if __name__ == "__main__":
    main()
