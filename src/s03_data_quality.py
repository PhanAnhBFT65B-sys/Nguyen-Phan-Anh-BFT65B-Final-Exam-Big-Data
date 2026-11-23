"""Step 3 - Automated data-quality audit of the RAW layer (Part 2).

Runs a fixed battery of checks on the three raw tables and writes
    outputs/dq_report.md                 human-readable report (regenerated on every run)
    outputs/tables/p2_dq_checks.csv      one row per check: what, how many, % , handling
    outputs/tables/p2_missing_by_year.csv and outputs/figures/p2_missing_heatmap.png
    outputs/tables/p2_fail_gap.csv       quarters between last Call Report and failure date

The audit only DESCRIBES problems; fixes are applied in s04_build_panel.py and every
fix is referenced back to a check id (F01, F02, ...).
"""
from __future__ import annotations

import json

import polars as pl

import config as C

FIN = C.RAW_DIR / "financials" / "**" / "*.parquet"
NUM = C.STARTER_FIELDS + C.EXTRA_FIELDS


def qidx(col: str) -> pl.Expr:
    """YYYYMMDD string -> integer quarter index (year*4 + quarter-1); consecutive quarters differ by 1."""
    c = pl.col(col)
    return c.str.slice(0, 4).cast(pl.Int32) * 4 + (c.str.slice(4, 2).cast(pl.Int32) - 1) // 3


def load():
    fin = pl.scan_parquet(FIN, hive_partitioning=False)
    fail = pl.read_parquet(C.RAW_DIR / "failures.parquet")
    inst = pl.read_parquet(C.RAW_DIR / "institutions.parquet")
    return fin, fail, inst


def main() -> None:
    C.ensure_dirs()
    fin, fail, inst = load()
    df = fin.with_columns(qidx("REPDTE").alias("Q"),
                          pl.col("REPDTE").str.slice(0, 4).cast(pl.Int32).alias("YEAR")).collect()
    N = df.height
    checks = []

    def add(cid, table, what, n, denom, handling, severity="medium"):
        checks.append({"id": cid, "table": table, "check": what, "n_affected": int(n),
                       "pct": round(100 * n / denom, 3) if denom else None,
                       "severity": severity, "handling_in_s04": handling})

    # ---------------- Financials ----------------
    n_dup = N - df.select("CERT", "REPDTE").unique().height
    add("F01", "financials", "duplicate (CERT, REPDTE) keys", n_dup, N, "assert none; keep first if any", "high")
    add("F02", "financials", "null CERT or REPDTE", df.filter(pl.col("CERT").is_null() | pl.col("REPDTE").is_null()).height,
        N, "drop row", "high")
    add("F03", "financials", "ASSET null or <= 0", df.filter(pl.col("ASSET").is_null() | (pl.col("ASSET") <= 0)).height,
        N, "drop row (no balance sheet = no usable report); ratios undefined", "high")
    add("F04", "financials", "tiny bank: 0 < ASSET < 5,000 (USD < 5m)",
        df.filter((pl.col("ASSET") > 0) & (pl.col("ASSET") < 5000)).height, N,
        "keep; ratios winsorised inside the model pipeline (fit on training data)", "low")
    add("F05", "financials", "negative equity (EQ < 0)", df.filter(pl.col("EQ") < 0).height, N,
        "keep: genuine insolvency signal, not an error", "info")
    add("F06", "financials", "EQ > ASSET (impossible)", df.filter(pl.col("EQ") > pl.col("ASSET")).height, N,
        "set EQ-based ratios to null", "high")
    neg_cols = ["DEP", "LNLSGR", "CHBAL", "SC", "LNATRES", "NCLNLS", "ORE", "INTINC"]
    for c in neg_cols:
        add(f"F07_{c}", "financials", f"negative {c} (stock that cannot be < 0)",
            df.filter(pl.col(c) < 0).height, N, "set value to null", "medium")
    ident = df.filter(pl.col("LNLSGR") > 0).with_columns(
        (pl.col("LNLSNET") - (pl.col("LNLSGR") - pl.col("LNATRES"))).abs().alias("gap"))
    add("F08", "financials", "identity LNLSNET = LNLSGR - LNATRES - unearned income violated by > 2% of LNLSGR",
        ident.filter(pl.col("gap") > 0.02 * pl.col("LNLSGR") + 1).height, ident.height,
        "keep (unearned income not in field set); use LNLSGR as loan denominator", "low")
    add("F09", "financials", "sub-total exceeds total: DEPDOM > DEP*1.001",
        df.filter(pl.col("DEPDOM") > pl.col("DEP") * 1.001 + 1).height, N, "cap ratio at 1", "low")
    add("F10", "financials", "sub-total exceeds total: LNRE > LNLSGR*1.001",
        df.filter(pl.col("LNRE") > pl.col("LNLSGR") * 1.001 + 1).height, N, "cap ratio at 1", "low")
    add("F11", "financials", "brokered deposits > total deposits",
        df.filter(pl.col("BRO") > pl.col("DEP") * 1.001 + 1).height, N, "cap ratio at 1", "low")
    add("F12", "financials", "|ROA| > 50 % (extreme ratio, usually tiny or new banks)",
        df.filter(pl.col("ROA").abs() > 50).height, N, "recompute ROA from NETINC/ASSET; winsorise in pipeline", "medium")
    add("F13", "financials", "|ROE| > 500 % (equity near zero)",
        df.filter(pl.col("ROE").abs() > 500).height, N, "ROE not used as feature (unstable when EQ ~ 0)", "low")

    # F14: income-statement items are year-to-date (YTD). Evidence: ratio Q4/Q1 of INTINC.
    ytd = (df.filter(pl.col("INTINC") > 0)
             .with_columns(pl.col("REPDTE").str.slice(4, 2).alias("MM"))
             .group_by("MM").agg(pl.col("INTINC").median().alias("median_INTINC")).sort("MM"))
    q1 = ytd.filter(pl.col("MM") == "03")["median_INTINC"][0]
    ytd = ytd.with_columns((pl.col("median_INTINC") / q1).round(2).alias("ratio_to_Q1"))
    ytd.write_csv(C.TAB_DIR / "p2_ytd_evidence.csv")
    s = df.sort(["CERT", "Q"]).with_columns(
        pl.col("INTINC").shift(1).over("CERT").alias("prev"), pl.col("Q").shift(1).over("CERT").alias("prevQ"))
    same_year = (pl.col("Q") // 4 == pl.col("prevQ") // 4) & (pl.col("Q") - pl.col("prevQ") == 1)
    dec = s.filter(same_year & (pl.col("INTINC") < pl.col("prev")))
    add("F14", "financials", "YTD flow decreases within a year (INTINC_t < INTINC_t-1): restatement/merger artefact",
        dec.height, s.filter(same_year).height,
        "de-cumulate: flow_q = YTD_q - YTD_(q-1) (same year, consecutive); null if not computable or negative INTINC flow",
        "high")
    # F15: reporting gaps
    gaps = s.filter(pl.col("prevQ").is_not_null() & (pl.col("Q") - pl.col("prevQ") > 1))
    add("F15", "financials", "reporting gap: bank skips >= 1 quarter then reports again",
        gaps.height, N, "lags computed by quarter index, not by row position (gap -> null lag)", "medium")
    # F16-F17 structural missingness
    miss_era = (df.group_by("YEAR").agg([pl.col(c).is_null().mean().alias(c) for c in NUM]).sort("YEAR"))
    miss_era.write_csv(C.TAB_DIR / "p2_missing_by_year.csv")
    early = df.filter(pl.col("YEAR") < 1990)
    add("F16", "financials", "RBCRWAJ reported as 0 (not null) before 1990: risk-based capital not yet defined; "
        "missing value ENCODED AS ZERO", early.filter(pl.col("RBCRWAJ") == 0).height, early.height,
        "0 -> null (a true total capital ratio of exactly 0 is not plausible for a reporting bank) + missing flag",
        "high")
    add("F17", "financials", "RBC1AAJ (leverage ratio) reported as 0 before 1990",
        early.filter(pl.col("RBC1AAJ") == 0).height, early.height, "0 -> null; EQ/ASSET is the always-available capital ratio",
        "high")
    late = df.filter(pl.col("YEAR") >= 2020)
    add("F18", "financials", "RBCRWAJ = 0 while leverage ratio > 0, 2020+ (community bank leverage ratio filers "
        "do not report risk-based ratios)", late.filter((pl.col("RBCRWAJ") == 0) & (pl.col("RBC1AAJ") > 0)).height,
        late.height, "0 -> null; missingness is informative (not at random): indicator kept by the imputer", "high")
    pre01 = df.filter(pl.col("YEAR") <= 2000)
    add("F19", "financials", "P3ASSET (30-89 days past due) entirely missing 1984-2000 (field not collected)",
        pre01["P3ASSET"].null_count(), pre01.height,
        "kept; imputed inside the pipeline + indicator; a model cannot learn it from the 1980s crisis", "medium")
    for c in NUM:
        nn = df[c].null_count()
        if nn:
            add(f"M_{c}", "financials", f"null {c}", nn, N, "imputed in model pipeline (median fit on train) + indicator", "low")

    # ---------------- Failures ----------------
    nf = fail.height
    add("X01", "failures", "CERT null", fail["CERT"].null_count(), nf, "cannot link to panel: excluded", "medium")
    fd = fail.with_columns(pl.col("FAILDATE").str.strptime(pl.Date, "%m/%d/%Y", strict=False).alias("FD"))
    add("X02", "failures", "FAILDATE stored as text M/D/YYYY; unparseable", fd["FD"].null_count(), nf,
        "parse with explicit format", "medium")
    add("X03", "failures", "failure before 1984 (outside panel)", fd.filter(pl.col("FD") < pl.date(1984, 1, 1)).height,
        nf, "excluded (no features available)", "info")
    in_scope = fd.filter(pl.col("FD") >= pl.date(1984, 1, 1)).with_columns(pl.col("CERT").cast(pl.Int32))
    dup = in_scope.filter(pl.col("CERT").is_not_null()).group_by("CERT").len().filter(pl.col("len") > 1)
    add("X04", "failures", "same CERT appears more than once (e.g. assistance then failure)", dup.height,
        in_scope.height, "use FIRST event after each quarter in label window", "medium")
    add("X05", "failures", "RESTYPE = ASSISTANCE (open-bank assistance, not a closure)",
        in_scope.filter(pl.col("RESTYPE") == "ASSISTANCE").height, in_scope.height,
        "counted as failure event (FDIC failure list = failures + assisted resolutions); sensitivity reported", "info")
    last = df.group_by("CERT").agg(pl.col("Q").max().alias("lastQ"), pl.col("Q").min().alias("firstQ"))
    j = in_scope.filter(pl.col("CERT").is_not_null()).with_columns(
        (pl.col("FD").dt.year() * 4 + (pl.col("FD").dt.month() - 1) // 3).alias("FQ")).join(last, on="CERT", how="left")
    add("X06", "failures", "failed CERT never appears in Financials", j["lastQ"].null_count(), j.height,
        "cannot be labelled; mostly savings institutions filing with another regulator; reported as coverage limit", "high")
    jj = j.filter(pl.col("lastQ").is_not_null()).with_columns((pl.col("FQ") - pl.col("lastQ")).alias("gapQ"))
    gap_tab = jj.group_by("gapQ").len().sort("gapQ")
    gap_tab.write_csv(C.TAB_DIR / "p2_fail_gap.csv")
    add("X07", "failures", "bank keeps reporting AFTER a FAILURE event (gapQ < 0), mostly 1989 RTC conservatorships",
        jj.filter((pl.col("gapQ") < 0) & (pl.col("RESTYPE") == "FAILURE")).height, jj.height,
        "rows dated on/after the failure date dropped (bank already failed: known, not to be predicted)", "high")
    add("X07b", "failures", "bank keeps reporting after an ASSISTANCE event (open-bank assistance)",
        jj.filter((pl.col("gapQ") < 0) & (pl.col("RESTYPE") == "ASSISTANCE")).height, jj.height,
        "rows kept: the bank continues as an open institution; label refers to the NEXT event", "medium")
    add("X08", "failures", "last report > 4 quarters before failure (stale: label window never reaches failure)",
        jj.filter(pl.col("gapQ") > 4).height, jj.height, "kept as data limitation (cannot be predicted from panel)", "medium")

    # ---------------- Institutions / exits ----------------
    END_Q = int(C.END_Q[:4]) * 4 + (int(C.END_Q[5:7]) - 1) // 3
    exited = last.filter(pl.col("lastQ") < END_Q)
    failed_certs = set(jj["CERT"].to_list())
    ii = inst.with_columns(pl.col("CERT").cast(pl.Int32))
    ex = exited.join(ii.select("CERT", "CHANGEC1", "ACTIVE").with_columns(pl.lit(True).alias("in_inst")), on="CERT", how="left").with_columns(
        pl.col("CERT").is_in(list(failed_certs)).alias("in_failure_list"))
    tab = ex.group_by("in_failure_list", "CHANGEC1").len().sort("len", descending=True)
    tab.write_csv(C.TAB_DIR / "p2_exit_changec1.csv")
    add("I01", "institutions", "banks that stopped reporting before 2026Q2 (exits)", exited.height, last.height,
        "classified: failure (in failure list) vs other exit (merger, charter change, voluntary)", "info")
    add("I02", "institutions", "exited bank not found in Institutions table", ex["in_inst"].null_count(), ex.height,
        "class 'other exit/unknown'", "low")
    add("I03", "institutions", "Institutions table says ACTIVE=1 but bank stopped reporting",
        ex.filter(pl.col("ACTIVE") == "1").height, ex.height, "trust Financials (point-in-time reports)", "low")

    # ------------- point-in-time / revision note (not testable from API) -------------
    add("P01", "all", "API returns the LATEST revised values, not the vintage first published", 0, None,
        "documented limitation; features at q only use reports dated <= q", "info")

    res = pl.DataFrame(checks)
    res.write_csv(C.TAB_DIR / "p2_dq_checks.csv")
    summary = {"financial_rows": N, "banks": df["CERT"].n_unique(), "failure_rows": nf,
               "failures_in_scope": in_scope.height, "failures_linked": jj.height,
               "fail_gap_dist": dict(zip(gap_tab["gapQ"].to_list(), gap_tab["len"].to_list())),
               "ytd_evidence": ytd.to_dicts()}
    (C.TAB_DIR / "p2_dq_summary.json").write_text(json.dumps(summary, indent=1, default=str))

    # heat-map of missingness by year
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = miss_era.to_pandas().set_index("YEAR")
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(m.T.values, aspect="auto", cmap="viridis_r", vmin=0, vmax=1)
    ax.set_yticks(range(len(m.columns)), m.columns, fontsize=6)
    ax.set_xticks(range(0, len(m), 4), m.index[::4], fontsize=7)
    fig.colorbar(im, label="share missing")
    ax.set_title("Raw Financials: share of missing values by field and year")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "p2_missing_heatmap.png", dpi=150)

    # markdown report
    lines = ["# Automated data-quality report (raw layer)", "",
             f"Generated by `src/s03_data_quality.py`. Financial rows: {N:,}; banks: {summary['banks']:,}; "
             f"failure records: {nf:,} (in scope >=1984: {in_scope.height:,}, linked to panel: {jj.height:,}).", "",
             "| id | table | check | n | % | severity | handling |", "|---|---|---|---:|---:|---|---|"]
    for r in checks:
        if r["id"].startswith("M_"):
            continue
        lines.append(f"| {r['id']} | {r['table']} | {r['check']} | {r['n_affected']:,} | {r['pct']} | "
                     f"{r['severity']} | {r['handling_in_s04']} |")
    lines += ["", "## Missing values per field (all years)", "", "| field | n missing | % |", "|---|---:|---:|"]
    for r in checks:
        if r["id"].startswith("M_"):
            lines.append(f"| {r['id'][2:]} | {r['n_affected']:,} | {r['pct']} |")
    lines += ["", "## YTD evidence (median INTINC by report month)", "", ytd.to_pandas().to_markdown(index=False),
              "", "## Quarters between last Call Report and failure (gapQ)", "",
              gap_tab.to_pandas().to_markdown(index=False)]
    (C.OUT_DIR / "dq_report.md").write_text("\n".join(lines))
    print(res.to_pandas().to_string())


if __name__ == "__main__":
    main()
