"""Fig. 8 layout: RN and beta against the reference ratio gamma."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = Path(sys.argv[1]), sys.argv[2]
GAMMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
STYLE = {"scifact": ("SciFact", "tab:red"), "arguana": ("ArguAna", "tab:green"),
         "nfcorpus": ("NFCorpus", "tab:blue")}
fig, (ar, ab) = plt.subplots(1, 2, figsize=(9.0, 3.2))
summary = {}
for ds, (name, colour) in STYLE.items():
    for kind, prefix, style in (("Ours", "strat_", "-o"), ("Rand", "", "--o")):
        files = [src / f"{prefix}{ds}_g{g}.json" for g in GAMMAS]
        if not all(f.exists() for f in files):
            continue
        rs = [json.loads(f.read_text())["results"] for f in files]
        rn = [r["relative_ndcg"] for r in rs]; beta = [r["integration_error_beta"] for r in rs]
        summary[f"{kind}@{ds}"] = {"gamma": GAMMAS, "RN": rn, "beta": beta,
                                   "R": [r["recall_at_100"] for r in rs]}
        ar.plot(GAMMAS, rn, style, color=colour, ms=4, label=f"{kind}@{name}")
        ab.plot(GAMMAS, beta, style, color=colour, ms=4, label=f"{kind}@{name}")
ar.set_ylabel("RN"); ab.set_ylabel(r"$\beta$")
for ax in (ar, ab):
    ax.set_xlabel(r"Reference ratio ($\gamma$)"); ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2)
fig.suptitle("Fig. 8 on FastText v.s. NV-Embed V2 (solid: stratified, dashed: random references)", fontsize=9)
fig.tight_layout(); fig.savefig(out, dpi=160)
Path(out).with_suffix(".json").write_text(json.dumps(summary, indent=2))
for ds, v in summary.items():
    print(ds, "RN", [round(x, 3) for x in v["RN"]], "beta", [round(x, 3) for x in v["beta"]], "R", [round(x, 3) for x in v["R"]])
