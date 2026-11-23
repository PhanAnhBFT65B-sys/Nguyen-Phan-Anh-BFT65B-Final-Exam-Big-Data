"""Step 11 - Extra figures and tables for the report (descriptive only; nothing is refitted).

  p1_download_profile.png   rows and download seconds per quarter (from the download manifest)
  p3_benchmark_bars.png     T1 time and peak memory by engine/variant (from p3_benchmark.csv)
  p4_event_study.png        median of key ratios by quarters before failure, DEVELOPMENT period only
  p5_importance.png         permutation importance on the last development fold
  p5_test_by_year.png       test PR-AUC by year with the number of positive rows
  p4_event_study.csv        the numbers behind the event-study figure
"""
from __future__ import annotations

import json

import matplotlib
import pandas as pd
import polars as pl

import config as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def download_profile() -> None:
    m = json.loads((C.RAW_DIR / "manifest_financials.json").read_text())
    d = pd.DataFrame([{"q": pd.Timestamp(k), **v} for k, v in m.items()]).sort_values("q")
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(d["q"], d["rows"], c="#4C72B0", label="rows per quarter")
    ax.set_ylabel("bank reports per quarter")
    ax2 = ax.twinx()
    ax2.bar(d["q"], d["seconds"], width=60, color="#DD8452", alpha=0.5, label="download seconds")
    ax2.set_ylabel("seconds per quarter")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "p1_download_profile.png", dpi=150)


def benchmark_bars() -> None:
    b = pd.read_csv(C.TAB_DIR / "p3_benchmark.csv")
    b = b[b.task == "T1_panel"].copy()
    b["label"] = b["engine"] + " (" + b["threads"].astype(str).str.replace("all", "4 threads").str.replace("^1$", "1 thread", regex=True) + ")"
    b.loc[b.engine == "pandas", "label"] = "pandas (single-threaded)"
    b = b.sort_values("task_s")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].barh(b["label"], b["task_s"], color=["#DD8452" if e == "pandas" else "#4C72B0" for e in b["engine"]])
    ax[0].set_xlabel("T1 panel build time (s, median of 3)")
    ax[0].invert_yaxis()
    ax[1].barh(b["label"], b["peak_rss_mb"] / 1024, color=["#DD8452" if e == "pandas" else "#55A868" for e in b["engine"]])
    ax[1].set_xlabel("peak memory (GB)")
    ax[1].invert_yaxis()
    ax[1].set_yticklabels([])
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "p3_benchmark_bars.png", dpi=150)


def event_study() -> None:
    """Median ratios k quarters before failure vs banks that do not fail within 12 quarters.
    Development period only (REPDTE <= 2006-12-31) so the test period is not looked at."""
    cols = ["eq_ta", "npl_ln", "roa_4q", "texas", "cre_eq", "liq_ta"]
    f = (pl.scan_parquet(C.FEAT_DIR / "features.parquet")
         .filter(pl.col("REPDTE") <= pl.date(2006, 12, 31))
         .select(["Q_TO_FAIL"] + cols).collect())
    fail = (f.filter(pl.col("Q_TO_FAIL").is_between(1, 12)).group_by("Q_TO_FAIL")
            .agg([pl.col(c).median() for c in cols] + [pl.len().alias("n")]).sort("Q_TO_FAIL"))
    calm = f.filter(pl.col("Q_TO_FAIL").is_null() | (pl.col("Q_TO_FAIL") > 12)).select([pl.col(c).median() for c in cols])
    fail.write_csv(C.TAB_DIR / "p4_event_study.csv")
    calm.write_csv(C.TAB_DIR / "p4_event_study_reference.csv")
    titles = {"eq_ta": "equity / assets", "npl_ln": "non-current loans / loans", "roa_4q": "ROA, mean last 4 q",
              "texas": "Texas ratio", "cre_eq": "CRE loans / equity", "liq_ta": "(cash+securities) / assets"}
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.2))
    for ax, c in zip(axs.ravel(), cols):
        ax.plot(-fail["Q_TO_FAIL"], fail[c], marker="o", ms=3, c="#C44E52", label="banks that fail")
        ax.axhline(calm[c][0], c="grey", ls="--", label="no failure within 12 q")
        ax.set_title(titles[c], fontsize=9)
        ax.set_xlabel("quarters before failure", fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "p4_event_study.png", dpi=150)


def importance() -> None:
    imp = pd.read_csv(C.TAB_DIR / "p5_permutation_importance_dev.csv").head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(imp["feature"], imp["pr_auc_drop"], xerr=imp["std"], color="#4C72B0")
    ax.set_xlabel("drop in PR-AUC when the feature is shuffled (development fold 2001)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "p5_importance.png", dpi=150)


def test_by_year() -> None:
    t = pd.read_csv(C.TAB_DIR / "p5_test_by_year.csv")
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.bar(t["year"], t["pos"], color="#8172B2", alpha=0.6, label="positive bank-quarters")
    ax.set_ylabel("positive rows")
    ax2 = ax.twinx()
    ax2.plot(t["year"], t["pr_auc"], marker="o", c="#C44E52", label="PR-AUC")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("PR-AUC")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "p5_test_by_year.png", dpi=150)


def main() -> None:
    C.ensure_dirs()
    download_profile()
    benchmark_bars()
    event_study()
    importance()
    test_by_year()
    print(pd.read_csv(C.TAB_DIR / "p4_event_study.csv").round(4).to_string())
    print(pd.read_csv(C.TAB_DIR / "p4_event_study_reference.csv").round(4).to_string())


if __name__ == "__main__":
    main()
