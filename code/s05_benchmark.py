"""Step 5 - Fair benchmark: Polars vs pandas, plus a scaling test (Part 3).

Fairness rules
  * same input file (the sorted zstd Parquet from s02), same logical output, checked by checksums;
  * every run in a FRESH Python process (no warm caches inside the interpreter, honest peak memory);
  * pandas written idiomatically/vectorised (no Python lambdas in hot paths);
  * Polars measured with all cores AND with POLARS_MAX_THREADS=1, so the gain can be split into
    "better engine" (columnar, vectorised, query optimiser) and "parallelism";
  * timing reported = task time measured inside the process (imports excluded), median of 3 runs.

Scaling test: the panel is replicated x1, x2, x4, x8 (CERT offset per copy) to see how time and
peak memory grow, and to extrapolate when a single laptop stops being enough.

Outputs: outputs/tables/p3_benchmark.csv, p3_scaling.csv, outputs/figures/p3_scaling.png
"""
from __future__ import annotations

import statistics

import pandas as pd
import polars as pl

import config as C
from utils import run_isolated

SRC = C.ROOT / "code"
BASE_FILE = C.BENCH_DIR / "panel_zstd.parquet"


def replicated(k: int):
    if k == 1:
        return BASE_FILE
    out = C.BENCH_DIR / f"panel_x{k}.parquet"
    if not out.exists():
        base = pl.read_parquet(BASE_FILE)
        pl.concat([base.with_columns((pl.col("CERT") + i * 1_000_000).cast(pl.Int32)) for i in range(k)]) \
          .write_parquet(out, compression="zstd")
    return out


T1 = {
    "pandas": "import bench_impl as B\nt=time.perf_counter()\ndf=B.pandas_panel(r'{p}')\nRESULT={{'task_s':time.perf_counter()-t, **B.checksum_panel(df)}}",
    "polars_lazy": "import bench_impl as B\nt=time.perf_counter()\ndf=B.polars_panel(r'{p}','lazy')\nRESULT={{'task_s':time.perf_counter()-t, **B.checksum_panel(df)}}",
    "polars_lazy_window_over": "import bench_impl as B\nt=time.perf_counter()\ndf=B.polars_panel(r'{p}','lazy_window_over')\nRESULT={{'task_s':time.perf_counter()-t, **B.checksum_panel(df)}}",
    "polars_lazy_noopt": "import bench_impl as B\nt=time.perf_counter()\ndf=B.polars_panel(r'{p}','lazy_noopt')\nRESULT={{'task_s':time.perf_counter()-t, **B.checksum_panel(df)}}",
    "polars_streaming": "import bench_impl as B\nt=time.perf_counter()\ndf=B.polars_panel(r'{p}','streaming')\nRESULT={{'task_s':time.perf_counter()-t, **B.checksum_panel(df)}}",
}
T23 = {
    "pandas": ("import bench_impl as B\np=B.pandas_panel(r'{p}')\n"
               "t=time.perf_counter()\nd=B.pandas_dynamics(p)\nt2=time.perf_counter()-t\n"
               "t=time.perf_counter()\na=B.pandas_aggregate(p)\nt3=time.perf_counter()-t\n"
               "RESULT={{'T2_s':t2,'T3_s':t3,**{{'T2_'+k:v for k,v in B.checksum_dyn(d).items()}},"
               "**{{'T3_'+k:v for k,v in B.checksum_agg(a).items()}}}}"),
    "polars": ("import bench_impl as B\np=B.polars_panel(r'{p}','lazy')\n"
               "t=time.perf_counter()\nd=B.polars_dynamics(p)\nt2=time.perf_counter()-t\n"
               "t=time.perf_counter()\na=B.polars_aggregate(p)\nt3=time.perf_counter()-t\n"
               "RESULT={{'T2_s':t2,'T3_s':t3,**{{'T2_'+k:v for k,v in B.checksum_dyn(d).items()}},"
               "**{{'T3_'+k:v for k,v in B.checksum_agg(a).items()}}}}"),
}


def runs(code: str, n: int, env=None) -> dict:
    rs = [run_isolated(code, SRC, env) for _ in range(n)]
    out = dict(rs[0])
    for k in ("task_s", "T2_s", "T3_s", "wall_s"):
        if k in out:
            out[k] = round(statistics.median(r[k] for r in rs), 3)
    out["peak_rss_mb"] = round(max(r["peak_rss_mb"] for r in rs), 0)
    out["runs"] = n
    return out


def main() -> None:
    C.ensure_dirs()
    if not BASE_FILE.exists():
        raise SystemExit("Run s02_storage_benchmark.py first (it writes the benchmark input file).")
    rows = []
    # ---- T1 at x1, all engines, all cores and 1 thread ----
    for eng, code in T1.items():
        for threads in (None, "1"):
            if eng == "pandas" and threads:
                continue
            env = {"POLARS_MAX_THREADS": threads} if threads else None
            r = runs(code.format(p=BASE_FILE), 3, env)
            r.update({"task": "T1_panel", "engine": eng, "threads": threads or "all"})
            rows.append(r)
            print(r, flush=True)
    # ---- T2 / T3 at x1 ----
    for eng, code in T23.items():
        r = runs(code.format(p=BASE_FILE), 3)
        for t in ("T2", "T3"):
            rows.append({"task": f"{t}_{'dynamics' if t == 'T2' else 'aggregate'}", "engine": eng, "threads": "all",
                         "task_s": r[f"{t}_s"], "peak_rss_mb": r["peak_rss_mb"], "runs": 3,
                         **{k: v for k, v in r.items() if k.startswith(t + "_") and k != f"{t}_s"}})
        print(r, flush=True)
    bench = pd.DataFrame(rows)
    bench.to_csv(C.TAB_DIR / "p3_benchmark.csv", index=False)

    # ---- equality checks across engines ----
    t1 = bench[bench.task == "T1_panel"]
    eq = {"T1_rows_equal": t1["rows"].nunique() == 1, "T1_y_equal": t1["y"].nunique() == 1,
          "T1_netinc_q_max_rel_diff": float((t1["netinc_q"].max() - t1["netinc_q"].min()) / abs(t1["netinc_q"].mean()))}
    print(eq)
    pd.Series(eq).to_json(C.TAB_DIR / "p3_benchmark_equality.json")

    # ---- scaling ----
    sc = []
    pandas_ok = True
    for k in (1, 2, 4, 8):
        p = replicated(k)
        for eng in ("pandas", "polars_lazy", "polars_streaming"):
            if eng == "pandas" and not pandas_ok:
                continue
            r = runs(T1[eng].format(p=p), 1)
            r.update({"scale": k, "engine": eng, "rows_in": pl.scan_parquet(p).select(pl.len()).collect().item()})
            sc.append(r)
            print(r, flush=True)
            if eng == "pandas" and r["peak_rss_mb"] > 6000:
                pandas_ok = False   # protect a 16 GB laptop from swapping at the next scale
    scal = pd.DataFrame(sc)
    scal.to_csv(C.TAB_DIR / "p3_scaling.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for eng, g in scal.groupby("engine"):
        ax[0].plot(g["rows_in"] / 1e6, g["task_s"], marker="o", label=eng)
        ax[1].plot(g["rows_in"] / 1e6, g["peak_rss_mb"] / 1024, marker="o", label=eng)
    ax[0].set_xlabel("input rows (millions)")
    ax[0].set_ylabel("T1 time (s)")
    ax[1].set_xlabel("input rows (millions)")
    ax[1].set_ylabel("peak memory (GB)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "p3_scaling.png", dpi=150)


if __name__ == "__main__":
    main()
