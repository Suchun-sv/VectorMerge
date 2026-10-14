"""Fig. 9 from two runs of figures.py: alpha from the run without fallback, recall@100 from the Fig 9 version."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOURS = {"FastText v.s. NV-Embed V2": "tab:blue", "GTE-Qwen2 v.s. NV-Embed V2": "tab:red",
           "Mistral v.s. OpenAI Ada": "tab:green"}

alpha_run = json.loads(Path(sys.argv[1]).read_text())
recall_run = json.loads(Path(sys.argv[2]).read_text())
fig, (aa, ar) = plt.subplots(1, 2, figsize=(11, 3.8))
for label in alpha_run:
    c = COLOURS.get(label, "k")
    ks = [k for k in alpha_run[label]["k"] if k <= 1000]
    aa.plot(ks, alpha_run[label]["alpha"][:len(ks)], "-o", color=c, ms=3, label=label)
    ar.plot(recall_run[label]["k"], recall_run[label]["recall"], "-o", color=c, ms=3, label=label)
aa.set_ylabel(r"$\alpha$"); ar.set_ylabel("Recall@100")
for ax in (aa, ar):
    ax.set_xscale("log"); ax.set_xlabel("Number of clusters k (log scale)"); ax.grid(alpha=0.3)
aa.legend(fontsize=7)
fig.suptitle("Fig. 9 on SciFact: global vs local isometries", fontsize=9)
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=160)
print("saved", sys.argv[3])
