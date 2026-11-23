"""Step 10 - Summary figures and test-set uncertainty (no model is refitted, no choice changes).

  * failures per year with DEV / gap / TEST shading           -> outputs/figures/p5_split.png
  * bank-cluster bootstrap 95% CI of test PR-AUC and ROC-AUC   -> outputs/tables/p5_test_bootstrap.json
    (resampling BANKS, not rows, because rows of one bank are correlated over time)
  * cover-page numbers                                          -> outputs/tables/cover_numbers.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import config as C


def main() -> None:
    by_year = pd.read_csv(C.TAB_DIR / "p2_panel_by_year.csv")
    frozen = json.loads((C.TAB_DIR / "p5_frozen_choices.json").read_text())
    tp = pd.read_parquet(C.FEAT_DIR / "test_predictions.parquet").reset_index(drop=True)

    rng = np.random.default_rng(C.SEED)
    banks = tp["CERT"].unique()
    groups = {c: g.index.values for c, g in tp.groupby("CERT")}
    pr, roc = [], []
    for _ in range(300):
        sample = rng.choice(banks, size=len(banks), replace=True)
        idx = np.concatenate([groups[c] for c in sample])
        y, p = tp["y"].values[idx], tp["p"].values[idx]
        pr.append(average_precision_score(y, p))
        roc.append(roc_auc_score(y, p))
    boot = {"reps": 300, "pr_auc_point": average_precision_score(tp.y, tp.p),
            "pr_auc_ci95": [float(np.percentile(pr, 2.5)), float(np.percentile(pr, 97.5))],
            "roc_auc_ci95": [float(np.percentile(roc, 2.5)), float(np.percentile(roc, 97.5))]}
    (C.TAB_DIR / "p5_test_bootstrap.json").write_text(json.dumps(boot, indent=1))
    print(boot)

    # ---- publication-lag check: a Call Report for quarter q becomes public ~30-60 days after q.
    # Failures within 60 days of REPDTE may happen before the supervisor could see the report.
    # We re-score the SAME test predictions without those positives (evaluation slice only).
    feats = pd.read_parquet(C.FEAT_DIR / "features.parquet", columns=["CERT", "REPDTE", "NEXT_FAIL_DATE", "Q_TO_FAIL"])
    feats["REPDTE"] = pd.to_datetime(feats["REPDTE"])
    tp["REPDTE"] = pd.to_datetime(tp["REPDTE"])
    t2 = tp.merge(feats, on=["CERT", "REPDTE"], how="left")
    days = (pd.to_datetime(t2["NEXT_FAIL_DATE"]) - t2["REPDTE"]).dt.days
    early = (t2["y"] == 1) & (days <= 60)
    keep = ~early
    thr = frozen["threshold"]
    by_h = (t2[t2.y == 1].assign(flag=lambda d: d.p >= thr).groupby("Q_TO_FAIL")
            .agg(pos_rows=("y", "size"), recall_at_thr=("flag", "mean")).reset_index())
    by_h.to_csv(C.TAB_DIR / "p5_test_recall_by_horizon.csv", index=False)
    lag = {"positives_failing_within_60d_of_report": int(early.sum()),
           "share_of_positives": float(early.sum() / t2.y.sum()),
           "pr_auc_without_them": average_precision_score(t2.y[keep], t2.p[keep]),
           "roc_auc_without_them": roc_auc_score(t2.y[keep], t2.p[keep])}
    (C.TAB_DIR / "p5_publication_lag_check.json").write_text(json.dumps(lag, indent=1))
    print(lag)
    print(by_h)

    comp = json.loads((C.TAB_DIR / "p1_completeness.json").read_text())
    stats = json.loads((C.TAB_DIR / "p2_panel_stats.json").read_text())
    small = json.loads((C.RAW_DIR / "manifest_small_tables.json").read_text())
    man = json.loads((C.RAW_DIR / "manifest_financials.json").read_text())
    cover = {"download_method": "FDIC BankFind public API (api.fdic.gov/banks), src/s01_download.py",
             "download_date_utc": max(m["downloaded_utc"] for m in man.values())[:10],
             "api_index": sorted({m["api_index"] for m in man.values()})[-1],
             "financials_rows": comp["rows"], "quarters": comp["quarters"], "banks": comp["distinct_cert"],
             "failures_rows": small["failures"]["rows"], "institutions_rows": small["institutions"]["rows"],
             "clean_panel_rows": stats["rows"], "dev_period": frozen["split"]["dev"],
             "test_period": frozen["split"]["test"],
             "dev_failing_banks": frozen["split"]["dev_failing_banks"],
             "test_failing_banks": frozen["split"]["test_failing_banks"]}
    (C.TAB_DIR / "cover_numbers.json").write_text(json.dumps(cover, indent=1))
    print(cover)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(by_year["YEAR"], by_year["failures_all"], color="#8172B2", label="failures + assisted (FDIC list, CERT known)")
    ax.axvspan(1983.5, 2006.5, color="#4C72B0", alpha=0.08, label="development 1984-2006")
    ax.axvspan(2006.5, 2007.5, color="grey", alpha=0.25, label="gap 2007 (4 quarters)")
    ax.axvspan(2007.5, 2025.5, color="#DD8452", alpha=0.10, label="test 2008Q1-2025Q2")
    ax.set_ylabel("events per year")
    ax2 = ax.twinx()
    ax2.plot(by_year["YEAR"], by_year["banks"], c="k", lw=1.2, label="reporting banks")
    ax2.set_ylabel("reporting banks")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "p5_split.png", dpi=150)


if __name__ == "__main__":
    main()
