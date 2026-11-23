"""Step 4 - Build the clean, point-in-time bank-quarter panel with labels (Part 2 + Part 3).

Everything is one Polars LAZY query: Polars builds a logical plan, optimises it
(projection + predicate pushdown into the Parquet scan, common sub-plan elimination)
and only then executes it in parallel. The optimised plan is saved to
outputs/tables/p3_polars_plan_panel.txt for the report.

Fixes applied (ids refer to the audit in s03_data_quality.py):
  F01 duplicates -> unique on (CERT, REPDTE)          F02/F03 null key / ASSET<=0 -> drop
  F06 EQ > ASSET -> EQ null                            F07 negative stock values -> null
  F16-F18 RBC ratios reported as 0 when not applicable -> null
  X07 rows dated on/after a FAILURE event (RTC conservatorships kept reporting) -> drop
  F14 YTD income items -> de-cumulated quarterly flows (only when the previous quarter
      of the SAME year exists, otherwise null; never filled from the future)
  F15 reporting gaps -> lag logic uses the quarter index, not the row position
  X01-X04 failures: parse dates, drop null CERT / pre-1984, multiple events per CERT ->
      label uses the NEXT event strictly after the report date (as-of join forward)

Label (exam 2.3): y_{i,q} = 1{ bank i fails in (q, q + 4 quarters] }. Mergers are not failures.
Only the failure DATE of a FUTURE event enters the label; no future value enters a feature.
Rows with REPDTE > 2025-06-30 get label_complete = False (window not yet observed).
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import date

import polars as pl

import config as C

FIN = C.RAW_DIR / "financials" / "**" / "*.parquet"
STOCKS_NONNEG = ["DEP", "LNLSGR", "CHBAL", "SC", "LNATRES", "NCLNLS", "ORE", "BRO", "DEPDOM",
                 "LNRE", "LNCI", "LNCON", "P3ASSET", "P9ASSET", "NAASSET", "INTAN",
                 "LNRECONS", "LNRENRES", "LNREMULT", "LNRERES", "COREDEP", "FREPP", "OTHBOR"]
END_DATE = date.fromisoformat(C.END_Q)
LAST_LABEL_DATE = date(int(C.LAST_LABEL_Q[:4]), int(C.LAST_LABEL_Q[4:6]), int(C.LAST_LABEL_Q[6:]))


def failures_lazy() -> pl.LazyFrame:
    return (pl.scan_parquet(C.RAW_DIR / "failures.parquet")
            .with_columns(pl.col("FAILDATE").str.strptime(pl.Date, "%m/%d/%Y", strict=False).alias("FAIL_DATE"),
                          pl.col("CERT").cast(pl.Int32),
                          pl.col("COST").cast(pl.Float64, strict=False),
                          pl.col("QBFASSET").cast(pl.Float64, strict=False))
            .filter(pl.col("CERT").is_not_null() & (pl.col("FAIL_DATE") >= pl.date(1984, 1, 1)))
            .select("CERT", "FAIL_DATE", "RESTYPE", "COST", "QBFASSET")
            .unique(["CERT", "FAIL_DATE"])
            .sort("FAIL_DATE"))


def panel_lazy(fin_path=FIN, variant: str = "sorted_shift") -> pl.LazyFrame:
    """Clean panel + labels as ONE lazy query.

    variant="sorted_shift" (default, fastest): after sorting by (CERT, Q) the previous row is the
        previous report of the same bank whenever CERT == CERT.shift(1); a plain shift is a cheap
        memory offset, and the label is computed with a hash join on CERT (failures are tiny).
    variant="window_over" (first version, kept for the benchmark): every lag uses .over("CERT"),
        i.e. a group-by + scatter per expression, and the label uses a global sort + join_asof.
    Both variants return identical results (checked in s05_benchmark.py).
    """
    lf = pl.scan_parquet(fin_path, hive_partitioning=False)
    lf = (lf
          .filter(pl.col("CERT").is_not_null() & pl.col("REPDTE").is_not_null())          # F02
          .filter(pl.col("ASSET").is_not_null() & (pl.col("ASSET") > 0))                  # F03
          .with_columns(pl.col("REPDTE").str.strptime(pl.Date, "%Y%m%d").alias("REPDTE"))
          .with_columns((pl.col("REPDTE").dt.year() * 4 + pl.col("REPDTE").dt.quarter() - 1)
                        .cast(pl.Int32).alias("Q"),
                        pl.col("REPDTE").dt.year().cast(pl.Int16).alias("YEAR"))
          .with_columns(pl.when(pl.col("EQ") > pl.col("ASSET")).then(None)
                        .otherwise(pl.col("EQ")).alias("EQ"))                              # F06
          .with_columns([pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c)
                         for c in STOCKS_NONNEG])                                          # F07
          .with_columns([pl.when(pl.col(c) == 0).then(None).otherwise(pl.col(c)).alias(c)
                         for c in ("RBC1AAJ", "RBCRWAJ")])                                # F16-F18
          .sort(["CERT", "Q"]))
    # F01 duplicates: after the sort a duplicate key is simply equal to the previous row's key,
    # which is much cheaper than a hash-based .unique() (measured: 0.48 s -> ~0.01 s)
    dup = (pl.col("CERT") == pl.col("CERT").shift(1)) & (pl.col("Q") == pl.col("Q").shift(1))
    lf = lf.filter(~dup.fill_null(False))
    # F14 + F15: de-cumulate YTD flows. prevQ is the previous REPORTED quarter of the bank.
    if variant == "window_over":
        prev_q = pl.col("Q").shift(1).over("CERT")
        prev = {f: pl.col(f).shift(1).over("CERT") for f in C.YTD_FLOW_FIELDS}
    else:
        same_bank = pl.col("CERT") == pl.col("CERT").shift(1)
        prev_q = pl.when(same_bank).then(pl.col("Q").shift(1))
        prev = {f: pl.when(same_bank).then(pl.col(f).shift(1)) for f in C.YTD_FLOW_FIELDS}
    same_year_consecutive = (pl.col("Q") - prev_q == 1) & (pl.col("Q") // 4 == prev_q // 4)
    is_q1 = pl.col("REPDTE").dt.quarter() == 1
    lf = lf.with_columns([
        pl.when(is_q1).then(pl.col(f))
          .when(same_year_consecutive).then(pl.col(f) - prev[f])
          .otherwise(None).alias(f"{f}_Q") for f in C.YTD_FLOW_FIELDS
    ]).with_columns(
        # interest income is a sum of positive items: a negative quarterly flow is an artefact
        pl.when(pl.col("INTINC_Q") < 0).then(None).otherwise(pl.col("INTINC_Q")).alias("INTINC_Q"),
        (pl.col("Q") - prev_q).alias("GAP_PREV_Q"),
    )
    # ---- X07: drop rows dated on/after the bank's first FAILURE-type event (conservatorship) ----
    first_closure = (failures_lazy().filter(pl.col("RESTYPE") == "FAILURE")
                     .group_by("CERT").agg(pl.col("FAIL_DATE").min().alias("_FIRST_CLOSURE")))
    lf = (lf.join(first_closure, on="CERT", how="left")
            .filter(pl.col("_FIRST_CLOSURE").is_null() | (pl.col("REPDTE") < pl.col("_FIRST_CLOSURE")))
            .drop("_FIRST_CLOSURE"))
    # ---- label: NEXT failure strictly after the report date ----
    fails = failures_lazy().select("CERT", "FAIL_DATE", "RESTYPE")
    if variant == "window_over":
        lf = (lf.sort("REPDTE")
                .join_asof(fails.sort("FAIL_DATE"), left_on="REPDTE", right_on="FAIL_DATE", by="CERT",
                           strategy="forward", allow_exact_matches=False))
    else:
        nxt = (lf.select("CERT", "REPDTE")
                 .join(fails, on="CERT", how="inner")                  # only ~3k failing banks survive
                 .filter(pl.col("FAIL_DATE") > pl.col("REPDTE"))       # strictly in the future
                 .group_by("CERT", "REPDTE")
                 .agg(pl.col("FAIL_DATE").min(), pl.col("RESTYPE").sort_by("FAIL_DATE").first()))
        lf = lf.join(nxt, on=["CERT", "REPDTE"], how="left", maintain_order="left")
    lf = (lf.with_columns(pl.col("REPDTE").dt.offset_by("12mo").dt.month_end().alias("_H"))
            .with_columns((pl.col("FAIL_DATE").is_not_null() & (pl.col("FAIL_DATE") <= pl.col("_H")))
                          .cast(pl.Int8).alias("y"),
                          (pl.col("REPDTE") <= LAST_LABEL_DATE).alias("label_complete"),
                          ((pl.col("FAIL_DATE").dt.year() * 4 + pl.col("FAIL_DATE").dt.quarter() - 1)
                           - pl.col("Q")).alias("Q_TO_FAIL"))
            .rename({"FAIL_DATE": "NEXT_FAIL_DATE", "RESTYPE": "NEXT_FAIL_TYPE"})
            .drop("_H"))
    return lf


def exits_table(panel: pl.DataFrame) -> pl.DataFrame:
    """One row per bank: first/last report and exit type (active / failure / other exit)."""
    end_q = END_DATE.year * 4 + (END_DATE.month - 1) // 3
    last = panel.group_by("CERT").agg(pl.col("Q").min().alias("FIRST_Q"), pl.col("Q").max().alias("LAST_Q"),
                                      pl.col("REPDTE").max().alias("LAST_REPDTE"))
    f = failures_lazy().collect()
    # a failure "explains" the exit if it happens after the last report (within 8 quarters)
    nxt = (last.sort("LAST_REPDTE")
               .join_asof(f.select("CERT", "FAIL_DATE", "RESTYPE").sort("FAIL_DATE"),
                          left_on="LAST_REPDTE", right_on="FAIL_DATE", by="CERT", strategy="forward"))
    inst = (pl.read_parquet(C.RAW_DIR / "institutions.parquet")
              .select(pl.col("CERT").cast(pl.Int32), "CHANGEC1"))
    out = (nxt.join(inst, on="CERT", how="left")
              .with_columns(((pl.col("FAIL_DATE").dt.year() * 4 + pl.col("FAIL_DATE").dt.quarter() - 1)
                             - pl.col("LAST_Q")).alias("Q_LAST_TO_FAIL"))
              .with_columns(
                  pl.when(pl.col("LAST_Q") == end_q).then(pl.lit("active"))
                    .when(pl.col("FAIL_DATE").is_not_null() & (pl.col("Q_LAST_TO_FAIL") <= 8))
                    .then(pl.lit("failure"))
                    .when(pl.col("CHANGEC1").cast(pl.Int32, strict=False).is_between(200, 299))
                    .then(pl.lit("merger/absorption"))
                    .otherwise(pl.lit("other exit")).alias("EXIT_TYPE")))
    return out


def main() -> None:
    C.ensure_dirs()
    lf = panel_lazy()
    (C.TAB_DIR / "p3_polars_plan_panel.txt").write_text(
        "=== NAIVE (unoptimised) PLAN ===\n" + lf.explain(optimized=False) +
        "\n\n=== OPTIMISED PLAN ===\n" + lf.explain())
    t0 = time.perf_counter()
    panel = lf.collect(engine="in-memory")
    t_build = time.perf_counter() - t0
    exits = exits_table(panel)
    panel = panel.join(exits.select("CERT", "EXIT_TYPE"), on="CERT", how="left").sort(["REPDTE", "CERT"])

    out = C.CLEAN_DIR / "panel"
    if out.exists():
        shutil.rmtree(out)
    panel.write_parquet(out, compression="zstd", partition_by="YEAR")
    exits.write_parquet(C.CLEAN_DIR / "exits.parquet")

    lab = panel.filter("label_complete")
    stats = {
        "build_seconds": round(t_build, 2),
        "rows": panel.height, "banks": panel["CERT"].n_unique(),
        "rows_label_complete": lab.height, "positives": int(lab["y"].sum()),
        "positive_rate_pct": round(100 * lab["y"].mean(), 3),
        "failing_banks_labelled": lab.filter(pl.col("y") == 1)["CERT"].n_unique(),
        "exit_types": dict(zip(*exits.group_by("EXIT_TYPE").len().sort("EXIT_TYPE").to_dict(as_series=False).values())),
        "decumulated_null_share": {f: round(panel[f"{f}_Q"].null_count() / panel.height, 4)
                                   for f in C.YTD_FLOW_FIELDS},
    }
    (C.TAB_DIR / "p2_panel_stats.json").write_text(json.dumps(stats, indent=1))
    by_year = (panel.group_by("YEAR").agg(pl.len().alias("rows"), pl.col("CERT").n_unique().alias("banks"),
                                          pl.col("y").sum().alias("pos_rows"))
               .join(exits.filter(pl.col("EXIT_TYPE") == "failure")
                          .group_by(pl.col("FAIL_DATE").dt.year().cast(pl.Int16).alias("YEAR"))
                          .len().rename({"len": "failures_linked"}), on="YEAR", how="left")
               .join(failures_lazy().collect().group_by(pl.col("FAIL_DATE").dt.year().cast(pl.Int16).alias("YEAR"))
                     .len().rename({"len": "failures_all"}), on="YEAR", how="left")
               .sort("YEAR").fill_null(0))
    by_year.write_csv(C.TAB_DIR / "p2_panel_by_year.csv")
    print(json.dumps(stats, indent=1))
    print(by_year.to_pandas().to_string())


if __name__ == "__main__":
    main()
