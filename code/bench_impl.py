"""Engine implementations compared in s05_benchmark.py.

The SAME logical transformations, written idiomatically for each engine:
  T1 panel     : clean + de-cumulate YTD flows (group-wise shift) + as-of label join
  T2 dynamics  : gap-aware year-on-year changes (self-join on CERT, Q-4) + 4-quarter rolling sum
  T3 aggregate : per-quarter cross-sectional statistics (group-by)
Each function returns a small dict of checksums so the engines can be checked for equality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

import config as C
from s04_build_panel import STOCKS_NONNEG, panel_lazy, LAST_LABEL_DATE

FLOWS = C.YTD_FLOW_FIELDS


def _fails_pd() -> pd.DataFrame:
    f = pd.read_parquet(C.RAW_DIR / "failures.parquet", columns=["CERT", "FAILDATE", "RESTYPE"])
    f["FAIL_DATE"] = pd.to_datetime(f["FAILDATE"], format="%m/%d/%Y", errors="coerce")
    f = f[f["CERT"].notna() & (f["FAIL_DATE"] >= "1984-01-01")].copy()
    f["CERT"] = f["CERT"].astype("int32")
    f = f.drop_duplicates(["CERT", "FAIL_DATE"])[["CERT", "FAIL_DATE", "RESTYPE"]]
    return f.sort_values("FAIL_DATE")


# --------------------------------------------------------------------------- T1
def pandas_panel(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["CERT"].notna() & df["REPDTE"].notna()]
    df = df[df["ASSET"].notna() & (df["ASSET"] > 0)].copy()
    df["REPDTE"] = pd.to_datetime(df["REPDTE"], format="%Y%m%d")
    df["Q"] = (df["REPDTE"].dt.year * 4 + df["REPDTE"].dt.quarter - 1).astype("int32")
    df["YEAR"] = df["REPDTE"].dt.year.astype("int16")
    df = df.drop_duplicates(["CERT", "REPDTE"])
    df.loc[df["EQ"] > df["ASSET"], "EQ"] = np.nan
    for c in STOCKS_NONNEG:
        df.loc[df[c] < 0, c] = np.nan
    for c in ("RBC1AAJ", "RBCRWAJ"):
        df.loc[df[c] == 0, c] = np.nan
    df = df.sort_values(["CERT", "Q"])
    g = df.groupby("CERT", sort=False)
    prev_q = g["Q"].shift(1)
    same = ((df["Q"] - prev_q) == 1) & ((df["Q"] // 4) == (prev_q // 4))
    q1 = df["REPDTE"].dt.quarter == 1
    for f in FLOWS:
        prev = g[f].shift(1)
        df[f + "_Q"] = np.where(q1, df[f], np.where(same, df[f] - prev, np.nan))
    df.loc[df["INTINC_Q"] < 0, "INTINC_Q"] = np.nan
    df["GAP_PREV_Q"] = df["Q"] - prev_q
    fails = _fails_pd()
    first_closure = fails[fails["RESTYPE"] == "FAILURE"].groupby("CERT")["FAIL_DATE"].min().rename("_FC")
    df = df.merge(first_closure, left_on="CERT", right_index=True, how="left")
    df = df[df["_FC"].isna() | (df["REPDTE"] < df["_FC"])].drop(columns="_FC")
    df = df.sort_values("REPDTE")
    df = pd.merge_asof(df, fails, left_on="REPDTE", right_on="FAIL_DATE", by="CERT",
                       direction="forward", allow_exact_matches=False)
    horizon = df["REPDTE"] + pd.offsets.MonthEnd(12)
    df["y"] = (df["FAIL_DATE"].notna() & (df["FAIL_DATE"] <= horizon)).astype("int8")
    df["label_complete"] = df["REPDTE"] <= pd.Timestamp(LAST_LABEL_DATE)
    return df


def polars_panel(path, mode: str = "lazy") -> pl.DataFrame:
    if mode == "lazy_window_over":
        return panel_lazy(path, variant="window_over").collect(engine="in-memory")
    lf = panel_lazy(path)
    if mode == "lazy":
        return lf.collect(engine="in-memory")
    if mode == "lazy_noopt":
        return lf.collect(engine="in-memory", optimizations=pl.QueryOptFlags.none())
    if mode == "streaming":
        return lf.collect(engine="streaming")
    raise ValueError(mode)


def checksum_panel(df) -> dict:
    if isinstance(df, pd.DataFrame):
        return {"rows": int(len(df)), "y": int(df["y"].sum()),
                "netinc_q": float(np.nansum(df["NETINC_Q"])), "intinc_q_null": int(df["INTINC_Q"].isna().sum())}
    return {"rows": df.height, "y": int(df["y"].sum()), "netinc_q": float(df["NETINC_Q"].sum()),
            "intinc_q_null": int(df["INTINC_Q"].null_count())}


# --------------------------------------------------------------------------- T2
def pandas_dynamics(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel[["CERT", "Q", "ASSET", "EQ", "NCLNLS", "LNLSGR", "NETINC_Q"]].copy()
    p["eq_ratio"] = p["EQ"] / p["ASSET"]
    p["npl_ratio"] = p["NCLNLS"] / p["LNLSGR"].where(p["LNLSGR"] > 0)
    lag = p[["CERT", "Q", "ASSET", "eq_ratio", "npl_ratio"]].copy()
    lag["Q"] = lag["Q"] + 4
    p = p.merge(lag, on=["CERT", "Q"], how="left", suffixes=("", "_l4"))
    p["asset_growth_yoy"] = np.log(p["ASSET"] / p["ASSET_l4"])
    p["d_eq_ratio_yoy"] = p["eq_ratio"] - p["eq_ratio_l4"]
    p["d_npl_yoy"] = p["npl_ratio"] - p["npl_ratio_l4"]
    p = p.sort_values(["CERT", "Q"])
    # rolling sum of the last 4 REPORTED quarters, valid only if they span exactly 4 quarters
    g = p.groupby("CERT", sort=False)
    # idiomatic, vectorised pandas (no Python lambda) to keep the baseline fair
    p["ni_4q"] = g["NETINC_Q"].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
    span = p["Q"] - g["Q"].shift(3)
    p.loc[span != 3, "ni_4q"] = np.nan
    return p


def polars_dynamics(panel: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    p = (panel.lazy().select("CERT", "Q", "ASSET", "EQ", "NCLNLS", "LNLSGR", "NETINC_Q")
         .with_columns((pl.col("EQ") / pl.col("ASSET")).alias("eq_ratio"),
                       (pl.col("NCLNLS") / pl.when(pl.col("LNLSGR") > 0).then(pl.col("LNLSGR"))).alias("npl_ratio")))
    lag = p.select("CERT", (pl.col("Q") + 4).alias("Q"), pl.col("ASSET").alias("ASSET_l4"),
                   pl.col("eq_ratio").alias("eq_ratio_l4"), pl.col("npl_ratio").alias("npl_ratio_l4"))
    out = (p.join(lag, on=["CERT", "Q"], how="left")
           .with_columns((pl.col("ASSET") / pl.col("ASSET_l4")).log().alias("asset_growth_yoy"),
                         (pl.col("eq_ratio") - pl.col("eq_ratio_l4")).alias("d_eq_ratio_yoy"),
                         (pl.col("npl_ratio") - pl.col("npl_ratio_l4")).alias("d_npl_yoy"))
           .sort(["CERT", "Q"])
           .with_columns(pl.when(pl.col("Q") - pl.col("Q").shift(3).over("CERT") == 3)
                         .then(pl.col("NETINC_Q").rolling_sum(4).over("CERT")).alias("ni_4q")))
    return out.collect()


def checksum_dyn(df) -> dict:
    if isinstance(df, pd.DataFrame):
        g = df["asset_growth_yoy"].replace([np.inf, -np.inf], np.nan)
        return {"rows": int(len(df)), "growth_sum": round(float(np.nansum(g)), 3),
                "ni4q_sum": round(float(np.nansum(df["ni_4q"])), 0), "ni4q_n": int(df["ni_4q"].notna().sum())}
    g = df["asset_growth_yoy"].filter(df["asset_growth_yoy"].is_finite())
    return {"rows": df.height, "growth_sum": round(float(g.sum()), 3),
            "ni4q_sum": round(float(df["ni_4q"].sum()), 0), "ni4q_n": int(df["ni_4q"].drop_nulls().len())}


# --------------------------------------------------------------------------- T3
def pandas_aggregate(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.assign(eq_ratio=panel["EQ"] / panel["ASSET"], roa_q=panel["NETINC_Q"] / panel["ASSET"])
    g = p.groupby("Q")
    out = g.agg(banks=("CERT", "nunique"), assets=("ASSET", "sum"),
                eq_med=("eq_ratio", "median"), pos=("y", "sum"))
    out["roa_p10"] = g["roa_q"].quantile(0.1)   # vectorised group quantile (no lambda)
    return out.reset_index()


def polars_aggregate(panel) -> pl.DataFrame:
    return (panel.lazy().group_by("Q")
            .agg(pl.col("CERT").n_unique().alias("banks"), pl.col("ASSET").sum().alias("assets"),
                 (pl.col("EQ") / pl.col("ASSET")).median().alias("eq_med"),
                 (pl.col("NETINC_Q") / pl.col("ASSET")).quantile(0.1, interpolation="linear").alias("roa_p10"),
                 pl.col("y").sum().alias("pos"))
            .sort("Q").collect())


def checksum_agg(df) -> dict:
    if isinstance(df, pd.DataFrame):
        return {"rows": int(len(df)), "banks": int(df["banks"].sum()), "eq_med": round(float(df["eq_med"].sum()), 6)}
    return {"rows": df.height, "banks": int(df["banks"].sum()), "eq_med": round(float(df["eq_med"].sum()), 6)}
