"""Central configuration: paths, field lists, study periods, seeds.

Every script imports from here so a single change propagates through the pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data live OUTSIDE Git. Override with the environment variable FDIC_DATA_DIR.
DATA_DIR = Path(os.environ.get("FDIC_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"            # API output, exactly as downloaded (Parquet per quarter)
BENCH_DIR = DATA_DIR / "bench"        # scratch files for storage benchmarks (CSV etc.)
CLEAN_DIR = DATA_DIR / "clean"        # cleaned, partitioned panel
FEAT_DIR = DATA_DIR / "features"      # feature table joined with labels

OUT_DIR = ROOT / "outputs"            # small result files (tables, figures, logs) -> committed
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

SEED = 42

# ---------------------------------------------------------------------------
# Scope fixed by the exam
# ---------------------------------------------------------------------------
START_Q = "1984-03-31"
END_Q = "2026-06-30"
LAST_LABEL_Q = "20250630"   # modelling uses quarters up to 2025Q2 only (labels complete)
HORIZON_Q = 4               # y = 1{fail in (q, q+4 quarters]}

# ---------------------------------------------------------------------------
# Field selection (codes checked against the FDIC field dictionary
# https://api.fdic.gov/banks/docs/risview_properties.yaml). Amounts: USD thousands.
# The 30 fields of the course starter, plus 8 fields motivated in the report:
#   - real-estate loan breakdown (construction/CRE concentration drove 2008-2012 failures)
#   - wholesale funding (FREPP, OTHBOR) and core deposits (funding fragility)
#   - intangibles (to compute tangible equity)
# ---------------------------------------------------------------------------
KEYS = ["CERT", "REPDTE", "STNAME"]
STARTER_FIELDS = [
    "ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC",
    "LNLSGR", "LNLSNET", "LNRE", "LNCI", "LNCON", "LNATRES",
    "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",
    "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",
    "ROA", "ROE", "NIMY",
    "RBC1AAJ", "RBCRWAJ",
]
EXTRA_FIELDS = ["LNRECONS", "LNRENRES", "LNREMULT", "LNRERES",
                "COREDEP", "FREPP", "OTHBOR", "INTAN"]
FIN_FIELDS = KEYS + STARTER_FIELDS + EXTRA_FIELDS

# Income-statement items are reported YEAR-TO-DATE in the Call Report.
YTD_FLOW_FIELDS = ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "NTLNLS"]

FAIL_FIELDS = ["CERT", "NAME", "CITYST", "FAILDATE", "FAILYR", "COST",
               "QBFASSET", "QBFDEP", "RESTYPE", "RESTYPE1", "SAVR"]
INST_FIELDS = ["CERT", "NAME", "STNAME", "BKCLASS", "ACTIVE", "ESTYMD",
               "ENDEFYMD", "CHANGEC1", "INSDATE", "REGAGNT", "INACTIVE"]


def ensure_dirs() -> None:
    for d in (RAW_DIR, BENCH_DIR, CLEAN_DIR, FEAT_DIR, OUT_DIR, FIG_DIR, TAB_DIR):
        d.mkdir(parents=True, exist_ok=True)
