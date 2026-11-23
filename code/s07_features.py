"""Step 7 - Leak-free feature table (Part 4).

Point-in-time rule: a feature for bank i at quarter q may only use Call Reports of bank i
(or of other banks) dated <= q. Concretely:
  * level ratios use the row at q only;
  * quarterly flows come from s04 (YTD de-cumulation uses q and q-1 of the same year);
  * dynamics use an explicit self-join on (CERT, Q-4) / (CERT, Q-3..Q) - NEVER shift(-k);
  * cross-sectional context (system stress) uses all banks at the SAME quarter q.
The label y (future failure) is joined at the end and is never an input of any feature.

A built-in leakage test (`assert_point_in_time`) recomputes all features from a panel
TRUNCATED at a cut-off date and checks that features at the cut-off are identical: if any
feature used a later row, the two would differ.

NOTE: statistics that need fitting (winsorisation limits, imputation medians, scaling) are
NOT computed here - they live in the sklearn Pipeline (s08) and are fitted on training data only.
"""
from __future__ import annotations

import json
from datetime import date

import polars as pl

import config as C

CLEAN = C.CLEAN_DIR / "panel" / "**" / "*.parquet"


def sdiv(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Safe ratio: null when the denominator is null or <= 0."""
    return pl.when(den > 0).then(num / den).otherwise(None)


def clip01(e: pl.Expr) -> pl.Expr:
    return e.clip(0.0, 1.0)


FEATURES_LEVEL = {
    # size
    "log_asset": pl.col("ASSET").log(),
    # Capital
    "eq_ta": sdiv(pl.col("EQ"), pl.col("ASSET")),
    "tce_ta": sdiv(pl.col("EQ") - pl.col("INTAN").fill_null(0), pl.col("ASSET") - pl.col("INTAN").fill_null(0)),
    "lev_pca": pl.col("RBC1AAJ") / 100,
    "trbc_pca": pl.col("RBCRWAJ") / 100,
    # Asset quality
    "npl_ln": clip01(sdiv(pl.col("NCLNLS"), pl.col("LNLSGR"))),
    "nonaccr_ta": sdiv(pl.col("NAASSET"), pl.col("ASSET")),
    "pd30_ta": sdiv(pl.col("P3ASSET"), pl.col("ASSET")),
    "pd90_ta": sdiv(pl.col("P9ASSET"), pl.col("ASSET")),
    "ore_ta": sdiv(pl.col("ORE"), pl.col("ASSET")),
    "alll_ln": sdiv(pl.col("LNATRES"), pl.col("LNLSGR")),
    "texas": sdiv(pl.col("NCLNLS") + pl.col("ORE").fill_null(0),
                  pl.col("EQ") - pl.col("INTAN").fill_null(0) + pl.col("LNATRES").fill_null(0)),
    # Earnings (quarterly flows annualised x4)
    "roa_q": 4 * sdiv(pl.col("NETINC_Q"), pl.col("ASSET")),
    "nim_q": 4 * sdiv(pl.col("INTINC_Q") - pl.col("EINTEXP_Q"), pl.col("ASSET")),
    "cof_q": 4 * sdiv(pl.col("EINTEXP_Q"), pl.col("ASSET")),
    "prov_ln_q": 4 * sdiv(pl.col("ELNATR_Q"), pl.col("LNLSGR")),
    "nco_ln_q": 4 * sdiv(pl.col("NTLNLS_Q"), pl.col("LNLSGR")),
    "eff_q": sdiv(pl.col("NONIX_Q"), pl.col("INTINC_Q") - pl.col("EINTEXP_Q") + pl.col("NONII_Q")),
    # Liquidity & funding
    "liq_ta": sdiv(pl.col("CHBAL") + pl.col("SC"), pl.col("ASSET")),
    "bro_dep": clip01(sdiv(pl.col("BRO"), pl.col("DEP"))),
    "whsl_ta": sdiv(pl.col("FREPP").fill_null(0) + pl.col("OTHBOR").fill_null(0), pl.col("ASSET")),
    "core_ta": sdiv(pl.col("COREDEP"), pl.col("ASSET")),
    "ln_dep": sdiv(pl.col("LNLSNET"), pl.col("DEP")),
    # Concentration (loan mix)
    "re_ln": clip01(sdiv(pl.col("LNRE"), pl.col("LNLSGR"))),
    "constr_ln": clip01(sdiv(pl.col("LNRECONS"), pl.col("LNLSGR"))),
    "cre_eq": sdiv(pl.col("LNRECONS") + pl.col("LNRENRES") + pl.col("LNREMULT"), pl.col("EQ")),
    "ci_ln": clip01(sdiv(pl.col("LNCI"), pl.col("LNLSGR"))),
    "con_ln": clip01(sdiv(pl.col("LNCON"), pl.col("LNLSGR"))),
}
FEATURES_DYN = ["asset_growth_yoy", "loan_growth_yoy", "d_eq_ta_yoy", "d_npl_ln_yoy", "roa_4q", "roa_4q_min"]
FEATURES_CTX = ["sys_neg_roa_share", "sys_npl_median"]
FEATURES = list(FEATURES_LEVEL) + FEATURES_DYN + FEATURES_CTX
KEEP = ["CERT", "REPDTE", "Q", "YEAR", "STNAME", "ASSET", "y", "label_complete", "NEXT_FAIL_DATE",
        "NEXT_FAIL_TYPE", "Q_TO_FAIL", "EXIT_TYPE"]


def build_features(panel: pl.LazyFrame) -> pl.LazyFrame:
    base = panel.with_columns([e.alias(k) for k, e in FEATURES_LEVEL.items()])
    # ---- dynamics: explicit self-joins on the quarter index (gap-aware, backward only) ----
    lag4 = base.select("CERT", (pl.col("Q") + 4).alias("Q"), pl.col("ASSET").alias("_asset_l4"),
                       pl.col("LNLSGR").alias("_ln_l4"), pl.col("eq_ta").alias("_eq_l4"),
                       pl.col("npl_ln").alias("_npl_l4"))
    # trailing 4 quarters (q-3..q) of quarterly ROA: join the 3 previous quarters explicitly
    lags = [base.select("CERT", (pl.col("Q") + k).alias("Q"), pl.col("roa_q").alias(f"_roa_l{k}"))
            for k in (1, 2, 3)]
    f = base.join(lag4, on=["CERT", "Q"], how="left")
    for lg in lags:
        f = f.join(lg, on=["CERT", "Q"], how="left")
    roa4 = [pl.col("roa_q"), pl.col("_roa_l1"), pl.col("_roa_l2"), pl.col("_roa_l3")]
    f = f.with_columns(
        sdiv(pl.col("ASSET"), pl.col("_asset_l4")).log().alias("asset_growth_yoy"),
        sdiv(pl.col("LNLSGR"), pl.col("_ln_l4")).log().alias("loan_growth_yoy"),
        (pl.col("eq_ta") - pl.col("_eq_l4")).alias("d_eq_ta_yoy"),
        (pl.col("npl_ln") - pl.col("_npl_l4")).alias("d_npl_ln_yoy"),
        pl.mean_horizontal(roa4).alias("roa_4q"),   # mean of available quarters
        pl.min_horizontal(roa4).alias("roa_4q_min"),
    )
    # ---- cross-sectional context at the same quarter (information available at q) ----
    ctx = (base.group_by("Q").agg((pl.col("roa_q") < 0).mean().alias("sys_neg_roa_share"),
                                  pl.col("npl_ln").median().alias("sys_npl_median")))
    f = f.join(ctx, on="Q", how="left")
    names = f.collect_schema().names()
    return f.select([c for c in KEEP if c in names] + FEATURES)


def assert_point_in_time(panel: pl.DataFrame, cut: date) -> dict:
    """Recompute features on data truncated at `cut`; features at `cut` must be identical."""
    full = build_features(panel.lazy()).filter(pl.col("REPDTE") == cut).select(["CERT"] + FEATURES).collect()
    trunc = build_features(panel.lazy().filter(pl.col("REPDTE") <= cut)) \
        .filter(pl.col("REPDTE") == cut).select(["CERT"] + FEATURES).collect()
    a, b = full.sort("CERT"), trunc.sort("CERT")
    assert a.height == b.height and a["CERT"].equals(b["CERT"])
    diffs = {}
    for c in FEATURES:
        x, y = a[c], b[c]
        same_null = (x.is_null() == y.is_null()).all()
        both = x.is_not_null() & y.is_not_null()
        maxdiff = float((x.filter(both) - y.filter(both)).abs().max() or 0.0)
        if not same_null or maxdiff > 1e-12:
            diffs[c] = maxdiff
    if diffs:
        raise AssertionError(f"LEAKAGE: features changed when future rows were removed: {diffs}")
    return {"cut": str(cut), "banks": a.height, "features_checked": len(FEATURES), "status": "PASS"}


def main() -> None:
    C.ensure_dirs()
    panel = pl.read_parquet(CLEAN, hive_partitioning=True)
    lf = build_features(panel.lazy())
    (C.TAB_DIR / "p4_polars_plan_features.txt").write_text(lf.explain())
    feats = lf.sort(["REPDTE", "CERT"]).collect()
    feats.write_parquet(C.FEAT_DIR / "features.parquet", compression="zstd")

    cuts = [d for d in (date(1990, 12, 31), date(2008, 12, 31), date(2023, 3, 31))
            if panel.filter(pl.col("REPDTE") == d).height > 0]
    if not cuts:  # small test downloads: use the second available quarter
        cuts = [sorted(panel["REPDTE"].unique().to_list())[min(1, panel["REPDTE"].n_unique() - 1)]]
    checks = [assert_point_in_time(panel, c) for c in cuts]
    # describe features (on complete-label rows), split by label
    m = feats.filter("label_complete")
    desc = []
    for c in FEATURES:
        s = m[c]
        desc.append({"feature": c, "missing_pct": round(100 * s.null_count() / m.height, 2),
                     "median_y0": m.filter(pl.col("y") == 0)[c].median(),
                     "median_y1": m.filter(pl.col("y") == 1)[c].median(),
                     "p01": s.quantile(0.01), "p99": s.quantile(0.99)})
    pl.DataFrame(desc).write_csv(C.TAB_DIR / "p4_feature_summary.csv")
    out = {"rows": feats.height, "features": len(FEATURES), "leakage_checks": checks}
    (C.TAB_DIR / "p4_feature_checks.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    print(pl.DataFrame(desc))


if __name__ == "__main__":
    main()
