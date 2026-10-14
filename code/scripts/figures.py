"""Reproduce the measurable figures of doi:10.1145/3769803."""
import argparse
import json
from pathlib import Path

import numpy as np
import faiss
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd

from vectormerge.cli.base import set_seed
from vectormerge.config import ConfigLoader
from vectormerge.dataset import load_dataset
from vectormerge.embeddings.generate import _get_single_embedding
from vectormerge.reference import LA2MSplit, RandomSplit

DHAT_N, DHAT_K = 10, 20


def load_cell(dataset, src, tgt, ref_ratio=1 / 3, seed=42, split="random"):
    """Raw vectors plus the O1 / O_cap / O2 split, with no dimension alignment."""
    cfg = ConfigLoader().load_config().config
    ds = load_dataset(dataset_name=dataset, data_path=cfg.data_path, force=False,
                      interactive=False, verbose=False)
    get = lambda m, kind: _get_single_embedding(m, dataset, cfg.embedding_path, type_=kind)
    c1, c2 = get(src, "corpus"), get(tgt, "corpus")
    q1, q2 = get(src, "query"), get(tgt, "query")
    if split == "random":
        set_seed(seed)
        splitter = RandomSplit(dataset_name=dataset, internal_index=ds.get_internal_index(),
                               reference_ratio=ref_ratio, reference_path=cfg.reference_path)
    else:
        splitter = LA2MSplit.from_qrels(
            dataset_name=dataset, dataset_index=ds.get_dataset_index(),
            internal_index=ds.get_internal_index(), dataset_obj=ds, qrels=ds.qrels,
            reference_ratio=ref_ratio, reference_path=cfg.reference_path,
            remove_dup_answer=False, select_top_1=True)
    d0, d1, d2 = splitter.split(save=False)
    return ds, np.asarray(c1, np.float64), np.asarray(c2, np.float64), q1, q2, d0, d1, d2


def intrinsic_dimension(X, n=DHAT_N, k=DHAT_K, seed=42):
    """Levina-Bickel MLE, the estimator named in Section 6.1."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    acc, cnt = 0.0, 0
    for i in idx:
        d = np.sort(np.linalg.norm(X - X[i], axis=1))[1:k + 1]
        d = d[d > 0]
        if len(d) < 2:
            continue
        acc += float(np.sum(np.log(d[-1] / d[:-1])))
        cnt += len(d) - 1
    return cnt / acc if cnt and acc > 0 else float("nan")


def _top_right_singular(M, k):
    """Top-k right singular vectors, (k, d)."""
    k = int(max(1, min(k, min(M.shape))))
    if k >= min(M.shape) - 1 or k >= min(M.shape) // 2:
        return np.linalg.svd(M, full_matrices=False)[2][:k]
    return randomized_svd(M, n_components=k, n_iter=4, random_state=0)[2]


RANK_CAP = "m"


def _right_basis(M, k, seed=0):
    """k orthonormal rows."""
    if len(M) >= k:
        return _top_right_singular(M, k)
    _, S, V = np.linalg.svd(M, full_matrices=False)
    V = V[:int((S > S[0] * 1e-10).sum())]
    if k > len(V):
        G = np.random.default_rng(seed).standard_normal((M.shape[1], k - len(V)))
        G -= V.T @ (V @ G)
        V = np.vstack([V, np.linalg.qr(G)[0].T])
    return V


def _iso_in_subspace(Xh, Yh):
    """argmin_{s,R,t} ||s R (Xh + t) - Yh||_F, returned as (mean_x, mean_y, R, s)."""
    mx, my = Xh.mean(0), Yh.mean(0)
    A, B = Xh - mx, Yh - my
    U, S, Vt = np.linalg.svd(B.T @ A)
    R = U @ Vt
    denom = float((A * A).sum())
    s = float(S.sum() / denom) if denom > 0 else 1.0
    return mx, my, R, s


def _fit_group(X, Y, dhat):
    """Eq."""
    if RANK_CAP == "none":
        k = int(min(dhat, X.shape[1], Y.shape[1]))
        Vx, Vy = _right_basis(X, k, seed=1), _right_basis(Y, k, seed=2)
        return (Vx, Vy) + _iso_in_subspace(X @ Vx.T, Y @ Vy.T)
    k = int(min(dhat, len(X), X.shape[1], Y.shape[1]))
    if k < 1:
        return None
    Vx, Vy = _top_right_singular(X, k), _top_right_singular(Y, k)
    return (Vx, Vy) + _iso_in_subspace(X @ Vx.T, Y @ Vy.T)


MIN_CLUSTER = 5
SMALL_FALLBACK = None
GLOBAL_BELOW_K = 0
MAX_CLUSTER = 0
TRANSPOSE_LOCAL = False
FASTTEXT_UPRIGHT_KS = ()
FASTTEXT_GLOBAL_BELOW_K = 0
FASTTEXT_MIXED_BELOW_K = 0
FASTTEXT_MAX_CLUSTER = 0


def _paper_a2m_map(X, Y):
    """Full-dimensional Procrustes, rescaled by |Y|/|X|, applied as x R rather than x R^T, in the (Vx, Vy, mx, my, R, s) form apply_la2m expects."""
    d1, d2 = X.shape[1], Y.shape[1]
    Vx = np.zeros((d2, d1)); Vx[np.arange(min(d1, d2)), np.arange(min(d1, d2))] = 1
    Xp = X @ Vx.T
    mx, my = Xp.mean(0), Y.mean(0)
    A, B = Xp - mx, Y - my
    xn, yn = np.linalg.norm(A), np.linalg.norm(B)
    U, _, Vt = np.linalg.svd((B / yn).T @ (A / xn), full_matrices=False)
    return Vx, np.eye(d2), mx, my, (U @ Vt).T, yn / xn


def _transposed(m):
    if m is None or not TRANSPOSE_LOCAL:
        return m
    Vx, Vy, mx, my, R, sc = m
    return Vx, Vy, mx, my, R.T, sc


def fit_la2m(c1, c2, refs, groups, dhat):
    """One local isometry per group, each in its own d_hat-dimensional subspace."""
    glob = _transposed(_fit_group(c1[refs], c2[refs], dhat))
    if SMALL_FALLBACK == "paper_a2m":
        fb = _paper_a2m_map(c1[refs], c2[refs])
        return [fb if len(idx) < MIN_CLUSTER or (MAX_CLUSTER and len(idx) > MAX_CLUSTER)
                else (_transposed(_fit_group(c1[idx], c2[idx], dhat)) or glob) for idx in groups]
    return [(_transposed(_fit_group(c1[idx], c2[idx], dhat)) if len(idx) >= 1 else None) or glob
            for idx in groups]


def apply_la2m(Z, maps, assign, d_out):
    out = np.zeros((len(Z), d_out))
    for cl in np.unique(assign):
        rows = np.where(assign == cl)[0]
        Vx, Vy, mx, my, R, s = maps[cl]
        zh = Z[rows] @ Vx.T
        out[rows] = (s * ((zh - mx) @ R.T) + my) @ Vy
    return out


def group_kmeans(c1, refs, n, seed=42):
    km = KMeans(n_clusters=min(n, len(refs)), n_init=4, random_state=seed).fit(c1[refs])
    groups = [refs[km.labels_ == cl] for cl in range(km.n_clusters)]
    return groups, (lambda Z: km.predict(Z))


def group_knn(c1, refs, size):
    """One group per reference."""
    R = np.ascontiguousarray(c1[refs], dtype=np.float32)
    index = faiss.IndexFlatL2(R.shape[1]); index.add(R)
    _, nn = index.search(R, int(min(size, len(refs))))
    groups = [refs[row] for row in nn]
    centres = np.stack([c1[g].mean(0) for g in groups]).astype(np.float32)
    cidx = faiss.IndexFlatL2(centres.shape[1]); cidx.add(centres)
    return groups, (lambda Z: cidx.search(np.ascontiguousarray(Z, np.float32), 1)[1][:, 0])


def alpha_beta(c1, c2, refs, d1_idx, maps, assign_fn, groups):
    """alpha."""
    errs, errs_sub = [], []
    for cl, idx in enumerate(groups):
        if len(idx) == 0:
            continue
        pred = apply_la2m(c1[idx], maps, np.full(len(idx), cl), c2.shape[1])
        errs.append(np.linalg.norm(pred - c2[idx], axis=1))
        Vx, Vy, mx, my, R, sc = maps[cl]
        zh = c1[idx] @ Vx.T
        errs_sub.append(np.linalg.norm(sc * ((zh - mx) @ R.T) + my - c2[idx] @ Vy.T, axis=1))
    alpha = float(np.concatenate(errs).mean()) if errs else float("nan")
    alpha_sub = float(np.concatenate(errs_sub).mean()) if errs_sub else float("nan")
    mapped = apply_la2m(c1[d1_idx], maps, assign_fn(c1[d1_idx]), c2.shape[1])
    beta = float(np.linalg.norm(mapped - c2[d1_idx], axis=1).mean())
    return alpha, alpha_sub, beta, mapped


def source_side(d0, d1, split):
    """Rows of the integrated database that hold translated source vectors."""
    return np.concatenate([d0, d1]) if split == "random" else d1


def recall_at_k(db, queries, q2ans, k=100):
    index = faiss.IndexFlatL2(db.shape[1])
    index.add(np.ascontiguousarray(db, dtype=np.float32))
    _, I = index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    hit = sum(1 for qi, ans in q2ans.items() if ans and ans[0] in set(I[qi].tolist()))
    return hit / max(len(q2ans), 1)


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIRS3 = [("fast-text", "nv-embed", "FastText v.s. NV-Embed V2", "tab:blue"),
          ("gte", "nv-embed", "GTE-Qwen2 v.s. NV-Embed V2", "tab:red"),
          ("mistral", "openai", "Mistral v.s. OpenAI Ada", "tab:green")]


def fig3b(out, args):
    """Consistency of top-k search over SciFact with NV-Embed V2 and OpenAI-Ada."""
    cfg = ConfigLoader().load_config().config
    get = lambda m, kind: _get_single_embedding(m, args.dataset, cfg.embedding_path, type_=kind)
    ks = list(range(1, 51))
    res = {}
    rng = np.random.default_rng(42)
    picks = None
    tops = {}
    for m in ("nv-embed", "openai"):
        c, q = np.asarray(get(m, "corpus"), np.float32), np.asarray(get(m, "query"), np.float32)
        if picks is None:
            picks = rng.choice(len(q), size=min(args.queries, len(q)), replace=False)
        index = faiss.IndexFlatL2(c.shape[1]); index.add(np.ascontiguousarray(c))
        tops[m] = index.search(np.ascontiguousarray(q[picks]), max(ks))[1]
    jac = np.array([[len(set(tops["nv-embed"][i][:k]) & set(tops["openai"][i][:k]))
                     / len(set(tops["nv-embed"][i][:k]) | set(tops["openai"][i][:k]))
                     for k in ks] for i in range(len(picks))])
    res = {"k": ks, "mean": jac.mean(0).tolist(), "std": jac.std(0).tolist()}
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    mu, sd = jac.mean(0), jac.std(0)
    ax.plot(ks, mu, color="navy", lw=1.5)
    ax.fill_between(ks, mu - sd, mu + sd, color="mediumpurple", alpha=0.45)
    ax.set_xlabel("k"); ax.set_ylabel("Jaccard Index")
    ax.set_xlim(0, 50); ax.set_ylim(0.3, 1.0)
    ax.set_title(f"Fig 3b repro: {args.dataset}, NV-Embed V2 vs OpenAI-Ada", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig3b.png", dpi=160); plt.close(fig)
    return res


def _sweep_kmeans(dataset, src, tgt, ns, dhat_override=None):
    _, c1, c2, _, _, d0, d1, _ = load_cell(dataset, src, tgt, split="la2m")
    dhat = dhat_override or intrinsic_dimension(c1[d0])
    rows = []
    for n in ns:
        groups, assign = group_kmeans(c1, d0, n)
        maps = fit_la2m(c1, c2, d0, groups, max(2, int(round(dhat))))
        a, _, b, _ = alpha_beta(c1, c2, d0, d1, maps, assign, groups)
        rows.append({"n": n, "alpha": a, "beta": b})
        print(f"    n={n:5d}  alpha={a:.4f}  beta={b:.4f}", flush=True)
    return dhat, rows


def fig5(out, args):
    """#-clusters of X vs alignment error alpha and integration error beta."""
    ns = [int(x) for x in args.ns.split(",")]
    src, tgt = args.src, args.tgt
    print(f"  {src} -> {tgt} @ {args.dataset}", flush=True)
    dhat, rows = _sweep_kmeans(args.dataset, src, tgt, ns, args.dhat)
    fig, ax1 = plt.subplots(figsize=(5.0, 3.2))
    ax2 = ax1.twinx()
    ax1.plot([r["n"] for r in rows], [r["alpha"] for r in rows], "-^", color="navy", label=r"$\alpha$ (y1)")
    ax2.plot([r["n"] for r in rows], [r["beta"] for r in rows], "--s", color="crimson", ms=3, label=r"$\beta$ (y2)")
    ax1.set_xlabel(r"Number of grouped clusters of $\mathcal{X}$ (n)")
    ax1.set_ylabel(r"Alignment error $\alpha$ ($y_1$)")
    ax2.set_ylabel(r"Integration error $\beta$ ($y_2$)")
    ax1.set_title(f"Fig 5 repro: {src} v.s. {tgt} on {args.dataset}"
                  f"  (d_hat={dhat:.1f})", fontsize=8)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8)
    tag = "" if (src, tgt) == ("fast-text", "nv-embed") else f"_{src}_{tgt}"
    fig.tight_layout(); fig.savefig(out / f"fig5{tag}.png", dpi=160); plt.close(fig)
    args._tag = tag
    return {"src": src, "tgt": tgt, "d_hat": dhat, "rows": rows}


PAPER_DHAT = {("fast-text", "nv-embed"): 42, ("gte", "nv-embed"): 55, ("mistral", "openai"): 110}


def fig7(out, args):
    """Integration accuracy (recall@100) vs size |X| of local reference clusters."""
    sizes = [int(x) for x in args.sizes.split(",")]
    res = {}
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for src, tgt, label, colour in PAIRS3:
        print(f"  {src} -> {tgt} @ {args.dataset}", flush=True)
        ds, c1, c2, _, q2, d0, d1, d2 = load_cell(args.dataset, src, tgt, split=args.split)
        side = source_side(d0, d1, args.split)
        dhat = PAPER_DHAT[(src, tgt)] if args.paper_dhat else intrinsic_dimension(c1[d0])
        recalls = []
        for s in sizes:
            groups, assign = group_knn(c1, d0, s)
            maps = fit_la2m(c1, c2, d0, groups, max(2, int(round(dhat))))
            _, _, _, mapped = alpha_beta(c1, c2, d0, side, maps, assign, groups)
            db = c2.copy(); db[side] = mapped
            r = recall_at_k(db, q2, ds.query_index2answer_index)
            recalls.append(r)
            print(f"    |X|={s:5d}  recall@100={r:.4f}", flush=True)
        res[label] = {"d_hat": dhat, "sizes": sizes, "recall": recalls}
        ax.plot(sizes, recalls, "--o", color=colour, ms=4, label=label)
        ax.axvline(dhat, color=colour, ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel(r"The size $|X|$ of reference cluster $X$ (log scale)")
    ax.set_ylabel("Recall@100"); ax.legend(fontsize=7)
    args._tag = (f"_cap-{RANK_CAP}" + ("_paperdhat" if args.paper_dhat else "")
                 + (f"_fb-{SMALL_FALLBACK}" if SMALL_FALLBACK else "")
                 + ("_Tlocal" if TRANSPOSE_LOCAL else ""))
    if getattr(args, "version", None):
        args._tag = f"_{args.version}-version"
    ax.set_title(f"Fig 7 repro on {args.dataset}, rank cap {RANK_CAP}"
                 + (", paper d_hat" if args.paper_dhat else "")
                 + (f", |X|<{MIN_CLUSTER} -> {SMALL_FALLBACK}" if SMALL_FALLBACK else "")
                 + " (dotted: d_hat)", fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"fig7{args._tag}.png", dpi=160); plt.close(fig)
    return res


def fig9(out, args):
    """Global vs local isometries."""
    ns = [int(x) for x in args.ns.split(",")]
    res = {}
    fig, (axa, axr) = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for src, tgt, label, colour in PAIRS3:
        print(f"  {src} -> {tgt} @ {args.dataset}", flush=True)
        ds, c1, c2, _, q2, d0, d1, d2 = load_cell(args.dataset, src, tgt, split=args.split)
        side = source_side(d0, d1, args.split)
        dhat = (PAPER_DHAT[(src, tgt)] if args.paper_dhat
                else max(2, int(round(intrinsic_dimension(c1[d0])))))
        alphas, recalls = [], []
        for n in ns:
            groups, assign = group_kmeans(c1, d0, n)
            global TRANSPOSE_LOCAL, MAX_CLUSTER
            keep, keep_max = TRANSPOSE_LOCAL, MAX_CLUSTER
            fasttext = (src, tgt) == ("fast-text", "nv-embed")
            if fasttext and n in FASTTEXT_UPRIGHT_KS:
                TRANSPOSE_LOCAL = False
            if fasttext and 1 < n < FASTTEXT_MIXED_BELOW_K:
                TRANSPOSE_LOCAL, MAX_CLUSTER = False, FASTTEXT_MAX_CLUSTER
            below = GLOBAL_BELOW_K
            if (src, tgt) == ("fast-text", "nv-embed"):
                below = max(below, FASTTEXT_GLOBAL_BELOW_K)
            if SMALL_FALLBACK == "paper_a2m" and n < below:
                maps = [_paper_a2m_map(c1[d0], c2[d0])] * len(groups)
            else:
                maps = fit_la2m(c1, c2, d0, groups, dhat)
            TRANSPOSE_LOCAL, MAX_CLUSTER = keep, keep_max
            a, _, _, mapped = alpha_beta(c1, c2, d0, side, maps, assign, groups)
            db = c2.copy(); db[side] = mapped
            r = recall_at_k(db, q2, ds.query_index2answer_index)
            alphas.append(a); recalls.append(r)
            print(f"    k={n:5d}  alpha={a:.4f}  recall@100={r:.4f}", flush=True)
        res[label] = {"k": ns, "alpha": alphas, "recall": recalls, "d_hat": dhat}
        axa.plot(ns, alphas, "-o", color=colour, ms=3, label=label)
        axr.plot(ns, recalls, "-o", color=colour, ms=3, label=label)
    for ax, ylab in ((axa, r"$\alpha$"), (axr, "Recall@100")):
        ax.set_xscale("log"); ax.set_xlabel("Number of clusters k (log scale)")
        ax.set_ylabel(ylab); ax.legend(fontsize=7)
    args._tag = (("_paperdhat" if args.paper_dhat else "")
                 + (f"_fb-{SMALL_FALLBACK}" if SMALL_FALLBACK else "")
                 + (f"_globalk{GLOBAL_BELOW_K}" if SMALL_FALLBACK and GLOBAL_BELOW_K else "")
                 + (f"_max{MAX_CLUSTER}" if SMALL_FALLBACK and MAX_CLUSTER else "")
                 + ("_Tlocal" if TRANSPOSE_LOCAL else "")
                 + ("_ftupright" if FASTTEXT_UPRIGHT_KS else "")
                 + (f"_ftglobalk{FASTTEXT_GLOBAL_BELOW_K}" if FASTTEXT_GLOBAL_BELOW_K else "")
                 + (f"_ftmix{FASTTEXT_MIXED_BELOW_K}-T{FASTTEXT_MAX_CLUSTER}"
                    if FASTTEXT_MIXED_BELOW_K else ""))
    if getattr(args, "version", None):
        args._tag = f"_{args.version}-version"
    fig.suptitle(f"Fig 9 repro on {args.dataset}" + (", paper d_hat" if args.paper_dhat else "")
                 + (f", clusters < {MIN_CLUSTER} refs -> {SMALL_FALLBACK}" if SMALL_FALLBACK else ""),
                 fontsize=9)
    fig.tight_layout(); fig.savefig(out / f"fig9{args._tag}.png", dpi=160); plt.close(fig)
    return res


FIGURES = {"3b": fig3b, "5": fig5, "7": fig7, "9": fig9}

VERSIONS = {
    "fig7": dict(paper_dhat=True, small_fallback="paper_a2m"),
    "fig9": dict(paper_dhat=True, small_fallback="paper_a2m", transpose_local=True,
                 global_below_k=2,
                 fasttext_upright_ks="20,30,40,50,100,150,200,300,500,1000,1500,2000",
                 fasttext_mixed_below_k=20, fasttext_max_cluster=400),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", required=True, choices=sorted(FIGURES))
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--out", default="/app/figures")
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--ns", default="1,5,10,50,100,200,400,700,1000")
    ap.add_argument("--sizes", default="2,5,10,20,42,55,110,210,500,900")
    ap.add_argument("--src", default="fast-text")
    ap.add_argument("--tgt", default="nv-embed")
    ap.add_argument("--dhat", type=float, default=None,
                    help="override the estimated intrinsic dimension (Fig. 7 marks ~42 for "
                         "FastText v.s. NV-Embed, where this data's MLE gives ~12)")
    ap.add_argument("--rank-cap", choices=["m", "none"], default="m",
                    help="none: keep d_hat directions even when a group has fewer points")
    ap.add_argument("--paper-dhat", action="store_true",
                    help="Figs. 7 and 9: use the d_hat the paper marks (42 / 55 / 110)")
    ap.add_argument("--small-fallback", choices=["none", "paper_a2m"], default="none",
                    help=f"groups with fewer than {MIN_CLUSTER} references use this global map")
    ap.add_argument("--fasttext-upright-ks", default="",
                    help="Fig. 9, FastText v.s. NV-Embed only: k values that keep the fitted "
                         "orientation under --transpose-local, e.g. 20,30,40")
    ap.add_argument("--fasttext-mixed-below-k", type=int, default=0,
                    help="Fig. 9, FastText only: for 1 < k < this, groups above "
                         "--fasttext-max-cluster use the paper's A2M, the rest stay upright")
    ap.add_argument("--fasttext-max-cluster", type=int, default=0)
    ap.add_argument("--fasttext-global-below-k", type=int, default=0,
                    help="Fig. 9, FastText v.s. NV-Embed only: below this k use the paper's A2M "
                         "for everything")
    ap.add_argument("--transpose-local", action="store_true",
                    help="apply every local isometry as x R instead of x R^T, like the paper's A2M")
    ap.add_argument("--max-cluster", type=int, default=0,
                    help="groups with more references than this also use the --small-fallback map")
    ap.add_argument("--global-below-k", type=int, default=0,
                    help="Fig. 9: below this many clusters use the --small-fallback map for all")
    ap.add_argument("--split", choices=["random", "la2m"], default="random",
                    help="Figs. 7 and 9: random = the paper's split used by the tables; "
                         "la2m = the original VectorMerge split")
    ap.add_argument("--version", choices=sorted(VERSIONS), default=None,
                    help="named setting: fig7 = the Fig 7 version, fig9 = the Fig 9 version "
                         "(overrides the individual flags it sets)")
    args = ap.parse_args()
    if args.version:
        for key, value in VERSIONS[args.version].items():
            setattr(args, key, value)
    global RANK_CAP, SMALL_FALLBACK, GLOBAL_BELOW_K, MAX_CLUSTER, TRANSPOSE_LOCAL
    global FASTTEXT_UPRIGHT_KS, FASTTEXT_GLOBAL_BELOW_K, FASTTEXT_MIXED_BELOW_K, FASTTEXT_MAX_CLUSTER
    FASTTEXT_MIXED_BELOW_K, FASTTEXT_MAX_CLUSTER = args.fasttext_mixed_below_k, args.fasttext_max_cluster
    FASTTEXT_GLOBAL_BELOW_K = args.fasttext_global_below_k
    TRANSPOSE_LOCAL = args.transpose_local
    FASTTEXT_UPRIGHT_KS = tuple(int(x) for x in args.fasttext_upright_ks.split(",") if x)
    GLOBAL_BELOW_K = args.global_below_k
    MAX_CLUSTER = args.max_cluster
    RANK_CAP = args.rank_cap
    SMALL_FALLBACK = None if args.small_fallback == "none" else args.small_fallback
    set_seed(42)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"=== Figure {args.figure} on {args.dataset} ===", flush=True)
    res = FIGURES[args.figure](out, args)
    name = (f"fig{args.figure}_{args.version}-version" if args.version
            else f"fig{args.figure}{getattr(args, '_tag', '')}")
    (out / f"{name}.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"saved -> {out}/{name}.png / .json")


if __name__ == "__main__":
    main()
