"""Fig."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

import figures
from figures import fit_la2m, apply_la2m
from vectormerge.config import ConfigLoader
from vectormerge.dataset import load_dataset
from vectormerge.embeddings.generate import _get_single_embedding
from vectormerge.evaluation import Evaluator


def allocate(sizes, gamma, floor):
    """References per cluster."""
    alloc = np.rint(gamma * sizes).astype(int)
    if floor:
        alloc = np.minimum(sizes, np.maximum(alloc, floor))
        excess = alloc.sum() - int(round(gamma * sizes.sum()))
        while excess > 0:
            spare = np.maximum(alloc - floor, 0)
            if spare.sum() == 0:
                break
            cut = np.minimum(spare, np.floor(excess * spare / spare.sum()).astype(int))
            if cut.sum() == 0:
                cut[np.argmax(spare)] = 1
            alloc -= cut
            excess -= int(cut.sum())
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/app/figures/fig8")
    ap.add_argument("--datasets", default="scifact,arguana,nfcorpus")
    ap.add_argument("--gammas", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--refs-per-cluster", type=int, default=130, help="K: L = round(gamma N / K)")
    ap.add_argument("--min-refs", type=int, default=100,
                    help="clusters with fewer references use the paper's A2M; stratified sampling "
                         "tops clusters up to this many")
    ap.add_argument("--no-floor", action="store_true", help="plain proportional stratified sampling")
    ap.add_argument("--dhat", type=int, default=42)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    figures.SMALL_FALLBACK, figures.MIN_CLUSTER = "paper_a2m", a.min_refs
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = ConfigLoader().load_config().config
    get = lambda m, ds, kind: np.asarray(
        _get_single_embedding(m, ds, cfg.embedding_path, type_=kind), np.float64)
    floor = 0 if a.no_floor else a.min_refs

    for ds in a.datasets.split(","):
        data = load_dataset(dataset_name=ds, data_path=cfg.data_path, force=False,
                            interactive=False, verbose=False)
        c1, c2 = get("fast-text", ds, "corpus"), get("nv-embed", ds, "corpus")
        q1, q2 = get("fast-text", ds, "query"), get("nv-embed", ds, "query")
        n = len(c1)
        for g in (float(x) for x in a.gammas.split(",")):
            L = int(round(g * n / a.refs_per_cluster))
            labels = KMeans(n_clusters=L, n_init=1, random_state=a.seed).fit_predict(c1)
            for kind, prefix in (("random", ""), ("stratified", "strat_")):
                rng = np.random.default_rng(a.seed)
                if kind == "random":
                    d0 = np.sort(rng.choice(n, size=int(round(g * n)), replace=False))
                else:
                    alloc = allocate(np.bincount(labels, minlength=L), g, floor)
                    d0 = np.sort(np.concatenate([
                        rng.choice(np.flatnonzero(labels == c), size=int(alloc[c]), replace=False)
                        for c in range(L)]).astype(int))
                rest = rng.permutation(np.setdiff1d(np.arange(n), d0))
                d1, d2 = np.sort(rest[: len(rest) // 2]), np.sort(rest[len(rest) // 2:])
                groups = [d0[labels[d0] == c] for c in range(L)]
                maps = fit_la2m(c1, c2, d0, groups, a.dhat)
                side = np.concatenate([d0, d1])
                mapped = np.zeros_like(c2)
                mapped[side] = apply_la2m(c1[side], maps, labels[side], c2.shape[1])
                r = Evaluator(corpus_emb_1=c1, corpus_emb_2=c2, query_emb_1=q1, query_emb_2=q2,
                              query_index2answer_index=data.query_index2answer_index,
                              d0=d0, d1=d1, d2=d2, corpus_emb_1_transformed=mapped,
                              k_list=[100], source_side=side).evaluate()
                small = [c for c in range(L) if len(groups[c]) < a.min_refs]
                rec = {"dataset": ds, "kind": kind, "gamma": g, "L": L, "d_hat": a.dhat,
                       "seed": a.seed, "n_refs": int(len(d0)), "min_refs": a.min_refs,
                       "clusters_below_min_refs": len(small),
                       "share_of_translated_docs_in_small_clusters":
                           float(np.isin(labels[side], small).mean()),
                       "results": {k: float(v) for k, v in r.items()}}
                (out / f"{prefix}{ds}_g{g}.json").write_text(json.dumps(rec, indent=2))
                print(f"{ds:8s} gamma={g} {kind:10s} RN {r['relative_ndcg']:.3f} "
                      f"beta {r['integration_error_beta']:.3f} R {r['recall_at_100']:.3f}  L {L}",
                      flush=True)


if __name__ == "__main__":
    main()
