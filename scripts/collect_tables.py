"""Assemble Tables 1 and 4-9 from the per-cell JSON written by run_tables.sh."""
import csv
import json
import signal
import sys
from pathlib import Path

ART = Path(__file__).resolve().parent.parent
TABLES = Path(sys.argv[1]) if len(sys.argv) > 1 else ART / "results" / "tables"

DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa"]
NAMES = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna",
         "scidocs": "SciDocs", "fiqa": "FiQA-2018"}
ROWS = [("union", "Union"), ("procrustes", "A2M"), ("gromov_wasserstein", "GW"), ("cca", "CCA"),
        ("nonlinear", "A2M_MLP"), ("la2m", "LA2M")]
TITLES = {4: "Mistral -> OpenAI-Ada", 5: "NV-Embed-V2 -> OpenAI-Ada", 6: "FastText -> OpenAI-Ada",
          7: "FastText -> NV-Embed-V2", 8: "GTE-Qwen2 -> NV-Embed-V2", 9: "GloVe -> FastText"}
METRICS = [("R", "recall_at_100", 100), ("R1", "recall_d1@100", 100),
           ("RN", "relative_ndcg", 100), ("beta", "integration_error_beta", 1)]


def cell(t, d, m):
    """Metrics for one cell, or a status string if it did not finish."""
    f = TABLES / f"table{t}_{d}_{'procrustes' if m == 'union' else m}.json"
    if not f.exists():
        return "not run"
    j = json.loads(f.read_text())
    if j.get("status") != "ok":
        return j.get("status", "error")
    r = j["results"]
    if m == "union":
        return {"R": 100 * r["direct_concat@100"]}
    return {k: s * r[key] for k, key, s in METRICS}


def fmt(ours, key):
    if isinstance(ours, str):
        return ours
    o = ours.get(key)
    if o is None:
        return "-"
    return f"{o:.2f}"


def main():
    md, rows_csv = ["R, R1, RN in %, beta raw.\n"], []
    for t in range(4, 10):
        md.append(f"## Table {t}: {TITLES[t]}\n")
        md.append("| method | " + " | ".join(f"{NAMES[d]} R / R1 / RN / beta" for d in DATASETS) + " |")
        md.append("|---|" + "---|" * len(DATASETS))
        for m, label in ROWS:
            cells = []
            for d in DATASETS:
                ours = cell(t, d, m)
                cells.append(ours if isinstance(ours, str)
                             else " / ".join(fmt(ours, k) for k, _, _ in METRICS))
                for k, _, _ in METRICS:
                    rows_csv.append({"table": t, "dataset": d, "method": label, "metric": k,
                                     "value": ours.get(k) if isinstance(ours, dict) else ours})
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        md.append("")

    md.append("## Table 1: Recall@100 (subset of Tables 4, 5, 7, 8)\n")
    md.append("| pair | method | " + " | ".join(NAMES[d] for d in DATASETS[:4]) + " |")
    md.append("|---|---|" + "---|" * 4)
    for t in (7, 5, 8, 4):
        for m, label in (("union", "Union"), ("procrustes", "A2M"), ("la2m", "LA2M")):
            vals = [fmt(cell(t, d, m), "R") for d in DATASETS[:4]]
            md.append(f"| {TITLES[t]} | {label} | " + " | ".join(vals) + " |")
    text = "\n".join(md) + "\n"
    (TABLES / "summary.md").write_text(text)
    with open(TABLES / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table", "dataset", "method", "metric", "value"])
        w.writeheader(); w.writerows(rows_csv)
    print(text)


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    main()
