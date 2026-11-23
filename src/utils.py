"""Small shared helpers: timing, directory sizes, peak memory of a child process."""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def dir_size_mb(path: Path) -> float:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def n_files(path: Path) -> int:
    path = Path(path)
    return 1 if path.is_file() else sum(1 for p in path.rglob("*") if p.is_file())


def time_it(fn, repeat: int = 3, warmup: bool = True) -> dict:
    """Median wall time over `repeat` runs (after an optional warm-up run)."""
    if warmup:
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(ts), "min_s": min(ts), "max_s": max(ts), "runs": repeat}


def run_isolated(code: str, cwd: Path, env_extra: dict | None = None) -> dict:
    """Run a Python snippet in a FRESH interpreter and report wall time + peak RSS (MB).

    A fresh process is the fair way to compare memory: no caches or allocator
    state are shared between the engines being compared. The snippet must print
    a JSON dict on its last stdout line; it is merged into the result.
    Peak RSS: resource.getrusage (Linux/macOS) or psutil peak working set (Windows).
    """
    wrapper = (
        "import json,time,sys\n"
        "t0=time.perf_counter()\n"
        f"{code}\n"
        "_el=time.perf_counter()-t0\n"
        "try:\n"
        "    import resource\n"
        "    _r=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "    _mb=_r/1024 if sys.platform!='darwin' else _r/1e6\n"
        "except ImportError:  # Windows\n"
        "    import psutil\n"
        "    _mb=psutil.Process().memory_info().peak_wset/1e6\n"
        "out=globals().get('RESULT',{})\n"
        "out.update({'wall_s':_el,'peak_rss_mb':_mb})\n"
        "print(json.dumps(out))\n"
    )
    p = subprocess.run([sys.executable, "-c", wrapper], cwd=cwd, capture_output=True, text=True,
                       env={**os.environ, "PYTHONHASHSEED": "0", **(env_extra or {})})
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str))
