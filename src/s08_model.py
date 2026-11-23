"""Step 8 - Modelling & validation (Part 5).

Design (all choices frozen BEFORE the test period is touched):
  * Modelling sample: rows with REPDTE <= 2025-06-30 (complete labels).
  * Development period DEV = 1984Q1 .. DEV_END ; gap of 4 quarters ; TEST = TEST_START .. 2025Q2.
    Gap rows are discarded, so no label window of a DEV row overlaps the TEST period.
  * Validation inside DEV: rolling-origin (expanding window) folds with a 4-quarter purge:
    train on quarters <= T, validate on the year starting 5 quarters later. Random K-fold is
    NOT used: it would mix years and leak crisis-wide information across folds.
  * Metrics: PR-AUC (average precision; primary, robust to ~0.5% positives), ROC-AUC, Brier,
    recall/precision when flagging the top 2% of banks (supervisory capacity view).
  * Models: B0 = 3-ratio logistic baseline (capital, NPL, ROA); M1 = regularised logistic on all
    features; M2 = histogram gradient boosting. All preprocessing (winsorising, imputation, scaling)
    is INSIDE an sklearn Pipeline, fitted on the training fold only.
  * Threshold: minimises expected cost  C_FN * FN + C_FP * FP  on pooled out-of-fold DEV
    predictions. C_FN is estimated from the data (median loss to the deposit insurance fund of
    DEV-period failures); C_FP (cost of a false alarm = one targeted examination) is an explicit
    assumption with a sensitivity table.
  * The chosen model is refitted on all DEV rows and evaluated ONCE on TEST. The frozen
    choices are written to disk before the test evaluation runs.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from s07_features import FEATURES

DEV_END = date(2006, 12, 31)
TEST_START = date(2008, 3, 31)     # 2007Q1..2007Q4 = 4-quarter gap
TEST_END = date(2025, 6, 30)
# validation years inside DEV (train <= Dec 31 of year-2, purge year-1)
VAL_YEARS = [1988, 1990, 1992, 2001]
BASELINE_FEATURES = ["eq_ta", "npl_ln", "roa_4q"]
TOP_SHARE = 0.02
C_FP_ASSUMED = 250.0   # USD thousands: assumed cost of one false alarm (targeted exam). See report.


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to [q_low, q_high] quantiles LEARNED ON THE TRAINING DATA ONLY."""

    def __init__(self, low=0.005, high=0.995):
        self.low, self.high = low, high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.low, axis=0)
        self.hi_ = np.nanquantile(X, self.high, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lo_, self.hi_)   # NaN stays NaN (imputed next)


class DropEmpty(BaseEstimator, TransformerMixin):
    """Drop columns that are entirely missing IN THE TRAINING DATA (e.g. P3ASSET before 2001)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.keep_ = ~np.all(np.isnan(X), axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


def make_logit(C_=1.0):
    return Pipeline([("drop_empty", DropEmpty()), ("winsor", Winsorizer()),
                     ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=C_, max_iter=2000, class_weight="balanced"))])


def make_hgb(lr=0.05, leaves=15, min_leaf=200, seed=C.SEED):
    # trees are invariant to monotone transforms; winsorising kept for robustness, NaN handled natively
    return Pipeline([("drop_empty", DropEmpty()), ("winsor", Winsorizer()),
                     ("clf", HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves,
                                                            min_samples_leaf=min_leaf, max_iter=300,
                                                            l2_regularization=1.0, early_stopping=False,
                                                            random_state=seed))])


CANDIDATES = {
    "B0_logit_3ratios": (BASELINE_FEATURES, lambda: make_logit(1.0)),
    "M1_logit_C0.1": (FEATURES, lambda: make_logit(0.1)),
    "M1_logit_C1": (FEATURES, lambda: make_logit(1.0)),
    "M2_hgb_lr0.05_l15": (FEATURES, lambda: make_hgb(0.05, 15, 200)),
    "M2_hgb_lr0.05_l31": (FEATURES, lambda: make_hgb(0.05, 31, 200)),
    "M2_hgb_lr0.1_l15": (FEATURES, lambda: make_hgb(0.1, 15, 500)),
}


def metrics(y, p, top_share=TOP_SHARE) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(round(top_share * len(p))))
    idx = np.argsort(-p)[:k]
    return {"n": int(len(y)), "pos": int(y.sum()), "pr_auc": average_precision_score(y, p),
            "roc_auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p),
            f"recall_top{int(top_share*100)}pct": y[idx].sum() / max(1, y.sum()),
            f"precision_top{int(top_share*100)}pct": y[idx].mean()}


def cost_curve(y, p, c_fn, c_fp, grid=None):
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.9999, 400))) if grid is None else grid
    rows = []
    P = y.sum()
    for t in grid:
        flag = p >= t
        tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum()); fn = int(P - tp)
        rows.append({"threshold": t, "tp": tp, "fp": fp, "fn": fn, "flag_rate": flag.mean(),
                     "cost": c_fn * fn + c_fp * fp})
    return pd.DataFrame(rows)


def main() -> None:
    C.ensure_dirs()
    df = (pl.read_parquet(C.FEAT_DIR / "features.parquet").filter("label_complete").to_pandas())
    df["REPDTE"] = pd.to_datetime(df["REPDTE"])
    dev = df[df["REPDTE"] <= pd.Timestamp(DEV_END)].copy()
    test = df[(df["REPDTE"] >= pd.Timestamp(TEST_START)) & (df["REPDTE"] <= pd.Timestamp(TEST_END))].copy()
    gap = df[(df["REPDTE"] > pd.Timestamp(DEV_END)) & (df["REPDTE"] < pd.Timestamp(TEST_START))]
    split = {"dev": [str(dev.REPDTE.min().date()), str(DEV_END)], "test": [str(TEST_START), str(TEST_END)],
             "dev_rows": len(dev), "dev_pos_rows": int(dev.y.sum()),
             "dev_failing_banks": int(dev.loc[dev.y == 1, "CERT"].nunique()),
             "gap_rows_dropped": len(gap), "test_rows": len(test), "test_pos_rows": int(test.y.sum()),
             "test_failing_banks": int(test.loc[test.y == 1, "CERT"].nunique())}
    # sanity: no DEV label window reaches the test period
    assert (dev["NEXT_FAIL_DATE"].where(dev.y == 1).dropna() < pd.Timestamp(TEST_START)).all()
    print(json.dumps(split, indent=1))

    # ---------------- rolling-origin CV inside DEV ----------------
    folds = []
    for vy in VAL_YEARS:
        tr = dev[dev["REPDTE"] <= pd.Timestamp(f"{vy - 2}-12-31")]
        va = dev[(dev["REPDTE"] >= pd.Timestamp(f"{vy}-01-01")) & (dev["REPDTE"] <= pd.Timestamp(f"{vy}-12-31"))]
        if tr.y.sum() >= 20 and va.y.sum() >= 5:
            folds.append((vy, tr, va))
    cv_rows, oof = [], {}
    for name, (feats, make) in CANDIDATES.items():
        preds = []
        for vy, tr, va in folds:
            t0 = time.perf_counter()
            m = make().fit(tr[feats], tr["y"])
            p = m.predict_proba(va[feats])[:, 1]
            r = metrics(va["y"], p)
            r.update({"model": name, "val_year": vy, "train_rows": len(tr), "fit_s": time.perf_counter() - t0})
            cv_rows.append(r)
            preds.append(pd.DataFrame({"idx": va.index, "y": va["y"].values, "p": p, "val_year": vy}))
        oof[name] = pd.concat(preds)
        print(name, "mean PR-AUC", np.mean([r["pr_auc"] for r in cv_rows if r["model"] == name]), flush=True)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(C.TAB_DIR / "p5_cv_folds.csv", index=False)
    summ = cv.groupby("model").agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_min=("pr_auc", "min"),
                                   roc_auc_mean=("roc_auc", "mean"), brier_mean=("brier", "mean"),
                                   recall_top2_mean=("recall_top2pct", "mean")).sort_values("pr_auc_mean", ascending=False)
    summ.to_csv(C.TAB_DIR / "p5_cv_summary.csv")
    print(summ)
    best = summ.index[0]

    # ---------------- cost-based threshold on pooled OOF predictions (DEV only) ----------------
    fails = pl.read_parquet(C.RAW_DIR / "failures.parquet").with_columns(
        pl.col("FAILDATE").str.strptime(pl.Date, "%m/%d/%Y", strict=False).alias("FD"),
        pl.col("COST").cast(pl.Float64, strict=False), pl.col("QBFASSET").cast(pl.Float64, strict=False))
    dev_f = fails.filter((pl.col("FD") >= pl.date(1984, 1, 1)) & (pl.col("FD") <= DEV_END) & pl.col("COST").is_not_null())
    c_fn = float(dev_f["COST"].median())
    loss_rate = float((dev_f["COST"] / dev_f["QBFASSET"]).median())
    o = oof[best]
    curve = cost_curve(o["y"].values, o["p"].values, c_fn, C_FP_ASSUMED)
    thr = float(curve.loc[curve["cost"].idxmin(), "threshold"])
    curve.to_csv(C.TAB_DIR / "p5_cost_curve_dev.csv", index=False)
    sens = []
    for c_fp in (50, 100, 250, 500, 1000, 2500):
        cc = cost_curve(o["y"].values, o["p"].values, c_fn, c_fp)
        b = cc.loc[cc["cost"].idxmin()]
        sens.append({"C_FP": c_fp, "C_FN/C_FP": round(c_fn / c_fp, 1), "threshold": b["threshold"],
                     "flag_rate": b["flag_rate"], "recall": b["tp"] / max(1, b["tp"] + b["fn"]),
                     "precision": b["tp"] / max(1, b["tp"] + b["fp"])})
    pd.DataFrame(sens).to_csv(C.TAB_DIR / "p5_threshold_sensitivity.csv", index=False)

    frozen = {"model": best, "features": CANDIDATES[best][0], "threshold": thr,
              "C_FN_median_DIF_loss_kUSD": c_fn, "C_FP_assumed_kUSD": C_FP_ASSUMED,
              "dev_median_loss_rate": loss_rate, "split": split, "val_years": [f[0] for f in folds],
              "seed": C.SEED, "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    blob = json.dumps(frozen, indent=1, default=str)
    frozen["sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    (C.TAB_DIR / "p5_frozen_choices.json").write_text(json.dumps(frozen, indent=1, default=str))
    print("FROZEN:", json.dumps(frozen, indent=1, default=str))

    # ---------------- single evaluation on TEST ----------------
    results = []
    final_models = {}
    for name in ("B0_logit_3ratios", best):
        feats, make = CANDIDATES[name]
        m = make().fit(dev[feats], dev["y"])
        final_models[name] = m
        p = m.predict_proba(test[feats])[:, 1]
        r = metrics(test["y"], p)
        r["model"] = name
        if name == best:
            flag = p >= thr
            tp = int((flag & (test.y == 1)).sum()); fp = int((flag & (test.y == 0)).sum())
            fn = int(test.y.sum() - tp)
            r.update({"threshold": thr, "tp": tp, "fp": fp, "fn": fn, "tn": int(len(test) - tp - fp - fn),
                      "recall_at_thr": tp / max(1, tp + fn), "precision_at_thr": tp / max(1, tp + fp),
                      "flag_rate_at_thr": float(flag.mean()),
                      "cost_at_thr_kUSD": c_fn * fn + C_FP_ASSUMED * fp,
                      "cost_flag_nobody_kUSD": c_fn * int(test.y.sum())})
            test_pred = test[["CERT", "REPDTE", "y"]].assign(p=p)
            test_pred.to_parquet(C.FEAT_DIR / "test_predictions.parquet")
            # how early are failures caught? share of failing banks flagged at >=1 quarter before failure
            tp_ = test.assign(p=p, flag=flag)
            fb = tp_[tp_.y == 1].groupby("CERT").agg(any_flag=("flag", "max"))
            r["failing_banks_flagged_share"] = float(fb["any_flag"].mean())
        results.append(r)
    res = pd.DataFrame(results)
    res.to_csv(C.TAB_DIR / "p5_test_results.csv", index=False)
    print(res.T)

    # yearly PR-AUC on test (stability over time)
    tp_ = pd.read_parquet(C.FEAT_DIR / "test_predictions.parquet")
    yr = []
    for yv, g in tp_.groupby(tp_.REPDTE.dt.year):
        if g.y.sum() > 0:
            yr.append({"year": yv, "rows": len(g), "pos": int(g.y.sum()), "pr_auc": average_precision_score(g.y, g.p),
                       "roc_auc": roc_auc_score(g.y, g.p)})
    pd.DataFrame(yr).to_csv(C.TAB_DIR / "p5_test_by_year.csv", index=False)

    # interpretability: standardised logistic coefficients (if M1) or permutation importance (DEV last fold)
    from sklearn.inspection import permutation_importance
    vy, tr, va = folds[-1]
    m = CANDIDATES[best][1]().fit(tr[CANDIDATES[best][0]], tr["y"])
    pi = permutation_importance(m, va[CANDIDATES[best][0]], va["y"], scoring="average_precision",
                                n_repeats=3, random_state=C.SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": CANDIDATES[best][0], "pr_auc_drop": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("pr_auc_drop", ascending=False)
    imp.to_csv(C.TAB_DIR / "p5_permutation_importance_dev.csv", index=False)

    # figures: PR curves on test + cost curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, m in final_models.items():
        p = m.predict_proba(test[CANDIDATES[name][0]])[:, 1]
        pr, rc, _ = precision_recall_curve(test["y"], p)
        ax[0].plot(rc, pr, label=f"{name} (AP={average_precision_score(test['y'], p):.3f})")
    ax[0].axhline(test.y.mean(), ls=":", c="grey", label="no-skill")
    ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].legend(fontsize=7)
    ax[0].set_title("Test period: precision-recall")
    ax[1].plot(curve["flag_rate"] * 100, curve["cost"] / 1e6)
    ax[1].axvline(curve.loc[curve.cost.idxmin(), "flag_rate"] * 100, ls="--", c="red", label="chosen threshold")
    ax[1].set_xscale("log"); ax[1].set_xlabel("% of bank-quarters flagged (DEV out-of-fold)")
    ax[1].set_ylabel("expected cost (USD bn)"); ax[1].legend(fontsize=7)
    ax[1].set_title("Threshold from costs (DEV only)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "p5_pr_cost.png", dpi=150)


if __name__ == "__main__":
    main()
