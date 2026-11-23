"""Step 6 - PySpark reproduction of the key panel steps (Part 3).

Runs in Spark LOCAL mode (laptop, or Google Colab: `pip install pyspark` and upload the two
Parquet inputs). Reproduces:
  T1  cleaning + YTD de-cumulation (window lag) + 4-quarter failure label (range join)
  T3  per-quarter aggregates
and checks that the results MATCH the Polars panel row by row.
Saves the formatted physical plans (outputs/tables/p3_spark_plan_*.txt) for interpretation.
"""
from __future__ import annotations

import json
import os
import time

import polars as pl

import config as C
from s04_build_panel import STOCKS_NONNEG, panel_lazy

os.environ.setdefault("PYSPARK_PYTHON", os.sys.executable)


def main() -> None:
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    spark = (SparkSession.builder.master("local[*]").appName("fdic-ews")
             .config("spark.driver.memory", "6g")
             .config("spark.sql.shuffle.partitions", "8")      # default 200 is far too many for 1.7M rows
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    src = str(C.BENCH_DIR / "panel_zstd.parquet")

    t0 = time.perf_counter()
    df = (spark.read.parquet(src)
          .where(F.col("CERT").isNotNull() & F.col("REPDTE").isNotNull())
          .where(F.col("ASSET").isNotNull() & (F.col("ASSET") > 0))
          .withColumn("REPDTE", F.to_date("REPDTE", "yyyyMMdd"))
          .withColumn("Q", (F.year("REPDTE") * 4 + F.quarter("REPDTE") - 1).cast("int"))
          .dropDuplicates(["CERT", "REPDTE"])
          .withColumn("EQ", F.when(F.col("EQ") > F.col("ASSET"), None).otherwise(F.col("EQ"))))
    for c in STOCKS_NONNEG:
        df = df.withColumn(c, F.when(F.col(c) < 0, None).otherwise(F.col(c)))
    for c in ("RBC1AAJ", "RBCRWAJ"):
        df = df.withColumn(c, F.when(F.col(c) == 0, None).otherwise(F.col(c)))
    w = Window.partitionBy("CERT").orderBy("Q")
    prev_q = F.lag("Q").over(w)
    same = ((F.col("Q") - prev_q) == 1) & (F.floor(F.col("Q") / 4) == F.floor(prev_q / 4))
    for f in C.YTD_FLOW_FIELDS:
        df = df.withColumn(f + "_Q", F.when(F.quarter("REPDTE") == 1, F.col(f))
                           .when(same, F.col(f) - F.lag(f).over(w)).otherwise(None))
    df = df.withColumn("INTINC_Q", F.when(F.col("INTINC_Q") < 0, None).otherwise(F.col("INTINC_Q")))

    fails = (spark.read.parquet(str(C.RAW_DIR / "failures.parquet"))
             .withColumn("FAIL_DATE", F.to_date("FAILDATE", "M/d/yyyy"))
             .withColumn("CERT", F.col("CERT").cast("int"))
             .where(F.col("CERT").isNotNull() & (F.col("FAIL_DATE") >= F.lit("1984-01-01")))
             .select("CERT", "FAIL_DATE", "RESTYPE").dropDuplicates(["CERT", "FAIL_DATE"]))
    first_closure = (fails.where(F.col("RESTYPE") == "FAILURE").groupBy("CERT")
                     .agg(F.min("FAIL_DATE").alias("_FC")))
    df = (df.join(F.broadcast(first_closure), "CERT", "left")
            .where(F.col("_FC").isNull() | (F.col("REPDTE") < F.col("_FC"))).drop("_FC"))
    # label: does ANY failure fall in (REPDTE, end of month(REPDTE + 12 months)]?
    lab = (df.select("CERT", "REPDTE").alias("p")
           .join(F.broadcast(fails.alias("f")),
                 (F.col("p.CERT") == F.col("f.CERT")) & (F.col("f.FAIL_DATE") > F.col("p.REPDTE")) &
                 (F.col("f.FAIL_DATE") <= F.last_day(F.add_months(F.col("p.REPDTE"), 12))), "inner")
           .select(F.col("p.CERT").alias("CERT"), F.col("p.REPDTE").alias("REPDTE")).distinct()
           .withColumn("y", F.lit(1)))
    panel = df.join(lab, ["CERT", "REPDTE"], "left").fillna({"y": 0})
    panel = panel.cache()
    n = panel.count()
    t_spark_t1 = time.perf_counter() - t0

    agg = (panel.groupBy("Q").agg(F.countDistinct("CERT").alias("banks"), F.sum("ASSET").alias("assets"),
                                  F.percentile_approx(F.col("EQ") / F.col("ASSET"), 0.5, 10000).alias("eq_med"),
                                  F.sum("y").alias("pos")).orderBy("Q"))
    t0 = time.perf_counter()
    agg_pd = agg.toPandas()
    t_spark_t3 = time.perf_counter() - t0

    (C.TAB_DIR / "p3_spark_plan_panel.txt").write_text(panel._jdf.queryExecution().toString())
    (C.TAB_DIR / "p3_spark_plan_label_join.txt").write_text(lab._jdf.queryExecution().executedPlan().toString())
    (C.TAB_DIR / "p3_spark_plan_agg.txt").write_text(agg._jdf.queryExecution().executedPlan().toString())

    # ------------------- compare with Polars, row by row -------------------
    sp = pl.from_pandas(panel.select("CERT", "REPDTE", "y", "NETINC_Q", "INTINC_Q").toPandas()) \
           .with_columns(pl.col("CERT").cast(pl.Int32), pl.col("REPDTE").cast(pl.Date))
    t0 = time.perf_counter()
    po = panel_lazy(C.BENCH_DIR / "panel_zstd.parquet").select("CERT", "REPDTE", "y", "NETINC_Q", "INTINC_Q").collect()
    t_polars_t1 = time.perf_counter() - t0
    m = po.join(sp, on=["CERT", "REPDTE"], how="full", suffix="_spark", coalesce=True)
    cmp = {
        "rows_polars": po.height, "rows_spark": n,
        "rows_only_in_one_engine": int(m.filter(pl.col("y").is_null() | pl.col("y_spark").is_null()).height),
        "y_mismatch": int(m.filter(pl.col("y") != pl.col("y_spark")).height),
        "positives_polars": int(po["y"].sum()), "positives_spark": int(sp["y"].sum()),
        "netinc_q_max_abs_diff": float((m["NETINC_Q"] - m["NETINC_Q_spark"]).abs().max() or 0.0),
        "netinc_q_null_mismatch": int(m.filter(pl.col("NETINC_Q").is_null() != pl.col("NETINC_Q_spark").is_null()).height),
        "spark_T1_s_incl_jvm_warm": round(t_spark_t1, 2), "spark_T3_s": round(t_spark_t3, 2),
        "polars_T1_s": round(t_polars_t1, 2), "spark_version": spark.version,
        "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
    }
    (C.TAB_DIR / "p3_spark_vs_polars.json").write_text(json.dumps(cmp, indent=1))
    agg_pd.to_csv(C.TAB_DIR / "p3_spark_agg.csv", index=False)
    print(json.dumps(cmp, indent=1))
    spark.stop()


if __name__ == "__main__":
    main()
