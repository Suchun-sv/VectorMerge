"""Fig. 4: a query and its 10 nearest neighbours in FastText and NV-Embed V2 (SciFact).

The paper states the two models "produce exactly the same list of top-10 most similar
SciFact documents" for the query it shows. Fig. 3b of the same paper puts the average
top-10 Jaccard of two strong models near 0.5, so the example is evidently chosen. Here the
overlap is measured for every test query first, then the best-overlapping query is drawn,
so the illustration comes with the base rate it was selected from.
"""
import json
from pathlib import Path

import numpy as np
import faiss
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vectormerge.config import ConfigLoader
from vectormerge.embeddings.generate import _get_single_embedding

OUT = Path("/app/figures"); OUT.mkdir(parents=True, exist_ok=True)
cfg = ConfigLoader().load_config().config
get = lambda m, k: np.asarray(_get_single_embedding(m, "scifact", cfg.embedding_path, type_=k), np.float32)

C, Q, TOP = {}, {}, {}
for m in ("fast-text", "nv-embed"):
    C[m], Q[m] = get(m, "corpus"), get(m, "query")
    idx = faiss.IndexFlatL2(C[m].shape[1]); idx.add(np.ascontiguousarray(C[m]))
    TOP[m] = idx.search(np.ascontiguousarray(Q[m]), 10)[1]

nq = len(TOP["fast-text"])
overlap = np.array([len(set(TOP["fast-text"][i]) & set(TOP["nv-embed"][i])) for i in range(nq)])
dist = {int(v): int((overlap == v).sum()) for v in range(11)}
best = int(np.argmax(overlap))
print(f"queries={nq}  mean top-10 overlap={overlap.mean():.2f}/10  identical lists={dist[10]}  best query={best} ({overlap[best]}/10)")

ranks = {int(d): r for r, d in enumerate(TOP["fast-text"][best])}   # label = FastText rank
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
for ax, m, title in zip(axes, ("fast-text", "nv-embed"), ("FastText", "NV-Embed-V2")):
    nbr = np.unique(np.concatenate([TOP["fast-text"][best], TOP["nv-embed"][best]]))
    # embed the query in the context of its wider neighbourhood, then zoom in
    wide = faiss.IndexFlatL2(C[m].shape[1]); wide.add(np.ascontiguousarray(C[m]))
    ctx = wide.search(Q[m][best:best + 1], 300)[1][0]
    pts = np.vstack([Q[m][best:best + 1], C[m][ctx]])
    z = PCA(n_components=min(50, pts.shape[1], len(pts) - 1)).fit_transform(pts)
    y = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(z)
    pos = {int(d): y[i + 1] for i, d in enumerate(ctx)}
    shown = [d for d in TOP[m][best]]
    for d in shown:
        p = pos[int(d)]
        other = "nv-embed" if m == "fast-text" else "fast-text"
        shared = int(d) in set(int(x) for x in TOP[other][best])
        ax.scatter(*p, s=110, color="gray" if shared else "white", edgecolor="black", zorder=3)
        ax.annotate(str(ranks.get(int(d), "×")), p, xytext=(6, 4), textcoords="offset points", fontsize=11)
    ax.scatter(*y[0], marker="*", s=320, color="crimson", edgecolor="black", zorder=4)
    xy = np.array([pos[int(d)] for d in shown] + [y[0]])
    pad = (xy.max(0) - xy.min(0)) * 0.25 + 1
    ax.set_xlim(xy[:, 0].min() - pad[0], xy[:, 0].max() + pad[0])
    ax.set_ylim(xy[:, 1].min() - pad[1], xy[:, 1].max() + pad[1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
fig.suptitle(f"Fig. 4 repro — SciFact query #{best}: top-10 overlap {overlap[best]}/10  "
             f"(labels = FastText rank; grey = in both lists, white/× = only this model's list)\n"
             f"over all {nq} queries: mean overlap {overlap.mean():.2f}/10, identical lists in {dist[10]} "
             f"({dist[10] / nq:.1%})", fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT / "fig4_repro.png", dpi=170); plt.close(fig)

# overlap histogram as a second panel for the base rate
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.bar(range(11), [dist[v] for v in range(11)], color="slategray")
ax.set_xlabel("# shared documents in the two top-10 lists"); ax.set_ylabel("# SciFact queries")
ax.set_title("Fig. 4 context — FastText vs NV-Embed-V2 top-10 overlap, all SciFact queries", fontsize=8.5)
ax.set_xticks(range(11)); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig4_overlap.png", dpi=170); plt.close(fig)
(OUT / "fig4.json").write_text(json.dumps({"best_query": best, "best_overlap": int(overlap[best]),
                                           "mean_overlap": float(overlap.mean()), "hist": dist}, indent=2))
