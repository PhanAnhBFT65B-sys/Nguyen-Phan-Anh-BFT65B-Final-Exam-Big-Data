"""Step 9 - Short Spark MLlib demo (Part 5).

Trains a Spark ML Pipeline (Imputer -> VectorAssembler -> StandardScaler -> LogisticRegression)
on one development fold (train <= 1990Q4, validate 1992) and compares it with the
equivalent scikit-learn logistic regression on exactly the same rows.
The TEST period is deliberately NOT used here: it is evaluated once, in s08 only.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import config as C
from s07_features import FEATURES
from s08_model import make_logit

os.environ.setdefault("PYSPARK_PYTHON", os.sys.executable)
TRAIN_END, VAL_START, VAL_END = "1990-12-31", "1992-01-01", "1992-12-31"


def main() -> None:
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = (SparkSession.builder.master("local[*]").appName("fdic-mllib")
             .config("spark.driver.memory", "6g").config("spark.sql.shuffle.partitions", "8")
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    df = (spark.read.parquet(str(C.FEAT_DIR / "features.parquet")).where("label_complete")
          .select("REPDTE", "y", *FEATURES))
    # Spark's Imputer does not accept +/-inf; turn non-finite values into null like sklearn's NaN
    for c in FEATURES:
        df = df.withColumn(c, F.when(F.col(c).isNull() | F.isnan(c) | (F.abs(F.col(c)) == float("inf")), None)
                           .otherwise(F.col(c)).cast("double"))
    tr = df.where(F.col("REPDTE") <= F.lit(TRAIN_END)).cache()
    va = df.where((F.col("REPDTE") >= F.lit(VAL_START)) & (F.col("REPDTE") <= F.lit(VAL_END))).cache()

    # class weights = 'balanced' as in sklearn: w_c = n / (2 * n_c)
    n, npos = tr.count(), tr.where("y = 1").count()
    tr = tr.withColumn("w", F.when(F.col("y") == 1, n / (2.0 * npos)).otherwise(n / (2.0 * (n - npos))))
    # like DropEmpty in s08: drop features with no value at all in the TRAINING rows (e.g. pd30_ta pre-2001)
    counts = tr.select([F.count(c).alias(c) for c in FEATURES]).first().asDict()
    feats = [c for c in FEATURES if counts[c] > 0]
    imp_cols = [f"{c}_imp" for c in feats]
    pipe = Pipeline(stages=[
        Imputer(inputCols=feats, outputCols=imp_cols, strategy="median"),
        VectorAssembler(inputCols=imp_cols, outputCol="raw"),
        StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True),
        LogisticRegression(featuresCol="features", labelCol="y", weightCol="w", regParam=1.0 / n,  # ~ sklearn C=1 (sum-loss vs mean-loss scaling)

                           elasticNetParam=0.0, maxIter=200),
    ])
    t0 = time.perf_counter()
    model = pipe.fit(tr)
    fit_s = time.perf_counter() - t0
    pred = model.transform(va)
    ev = BinaryClassificationEvaluator(labelCol="y", rawPredictionCol="rawPrediction")
    spark_pr = ev.evaluate(pred, {ev.metricName: "areaUnderPR"})
    spark_roc = ev.evaluate(pred, {ev.metricName: "areaUnderROC"})

    # identical rows in scikit-learn
    trp = tr.drop("w").toPandas()
    vap = va.toPandas()
    t0 = time.perf_counter()
    sk = make_logit(1.0).fit(trp[FEATURES], trp["y"])
    sk_fit = time.perf_counter() - t0
    p = sk.predict_proba(vap[FEATURES])[:, 1]
    out = {"features_used": len(feats), "features_dropped_all_null_in_train": [c for c in FEATURES if c not in feats],
           "train_rows": int(n), "train_pos": int(npos), "val_rows": len(vap), "val_pos": int(vap.y.sum()),
           "spark_pr_auc": spark_pr, "spark_roc_auc": spark_roc, "spark_fit_s": round(fit_s, 2),
           "sklearn_pr_auc": average_precision_score(vap.y, p), "sklearn_roc_auc": roc_auc_score(vap.y, p),
           "sklearn_fit_s": round(sk_fit, 2),
           "note": "Differences vs sklearn: no winsorising and no missing-indicator columns in the Spark pipeline; Spark areaUnderPR uses trapezoidal interpolation; sklearn average_precision uses step-wise "
                   "sums, so small differences are expected. Spark version " + spark.version}
    (C.TAB_DIR / "p5_spark_mllib.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    spark.stop()


if __name__ == "__main__":
    main()
