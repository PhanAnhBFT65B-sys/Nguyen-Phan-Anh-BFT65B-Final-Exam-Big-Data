#!/usr/bin/env bash
# Build report.docx and report.pdf from report.md (needs pandoc; PDF needs LibreOffice).
set -euo pipefail
cd "$(dirname "$0")"
python - <<'PY'
from pathlib import Path
md = Path("report.md").read_text(encoding="utf-8")
import csv
rows = [r for r in csv.DictReader(open("../outputs/tables/p2_dq_checks.csv", encoding="utf-8"))
        if not r["id"].startswith("M_")]
d = lambda n, r=False: "-" * (n - 1) + (":" if r else "-")
dq = ("| id | table | check | n | % | handling in s04 |\n|" + "|".join([d(7), d(10), d(36), d(9, True), d(7, True), d(31)]) + "|\n")
for r in rows:
    n = f"{int(r['n_affected']):,}"
    dq += f"| {r['id']} | {r['table']} | {r['check'].replace('|', '/')} | {n} | {r['pct']} | {r['handling_in_s04'].replace('|', '/')} |\n"
dq += "\nMissing-value profile per field and the full failure-gap histogram: `outputs/dq_report.md` and `outputs/tables/`."
md = md.replace("See `outputs/dq_report.md` in the repository (generated automatically); reproduced in the appendix pages that follow.",
                "Generated automatically by `code/s03_data_quality.py` (`outputs/dq_report.md`).\n\n" + dq)
Path("_build.md").write_text(md, encoding="utf-8")
PY
pandoc _build.md -o report.docx --resource-path=.:.. --reference-doc=reference.docx
if command -v soffice >/dev/null; then soffice --headless --convert-to pdf report.docx >/dev/null 2>&1; fi
rm -f _build.md
