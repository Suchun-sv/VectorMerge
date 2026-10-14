"""Run one cell of a paper table and write the metrics as JSON."""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from vectormerge.cli.base import set_seed
from vectormerge.config import ConfigLoader
from vectormerge.dataset import load_dataset
from vectormerge.embeddings import get_embedding
from vectormerge.embedding_dataset import EmbeddingDataset
from vectormerge.evaluation import Evaluator
from vectormerge.mapping import VectorSpaceMapper


def stratified_split(X, ratio, n_clusters, seed=42):
    """Reference sampling of Section 6.2."""
    rng = np.random.default_rng(seed)
    labels = KMeans(n_clusters=n_clusters, n_init=1, random_state=seed).fit_predict(X)
    d0 = []
    for c in range(n_clusters):
        members = np.flatnonzero(labels == c)
        d0.extend(rng.choice(members, size=int(round(ratio * len(members))), replace=False))
    d0 = np.sort(np.asarray(d0, dtype=int))
    rest = rng.permutation(np.setdiff1d(np.arange(len(X)), d0))
    half = len(rest) // 2
    return d0, np.sort(rest[:half]), np.sort(rest[half:])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--reference-method", choices=["random", "la2m", "stratified"], default="random",
                   help="random: the paper's split, O1 / O_cap / O2 as plain random thirds with the "
                        "answers left wherever they fall, and the integrated database holding "
                        "translated source vectors for O1 and O_cap; la2m: the original package's "
                        "split, answers kept out of O_cap and spread evenly over O1 / O2, only O1 "
                        "translated; stratified: as random, but the references are drawn per "
                        "k-means cluster of the source vectors (Section 6.2, Fig. 8)")
    p.add_argument("--strata", type=int, default=100,
                   help="stratified: number of k-means clusters L")
    p.add_argument("--out", required=True)
    p.add_argument("--reference-ratio", type=float, default=1 / 3,
                   help="share of the corpus drawn as references O_cap (Fig. 8 varies it; "
                        "the rest is split evenly into O1 and O2)")
    p.add_argument("--dhat", type=int, default=None,
                   help="LA2M: fix |X| = pca_dim = DHAT instead of estimating the intrinsic dimension")
    p.add_argument("--cluster-size", type=int, default=None,
                   help="LA2M: |X|, overriding --dhat for the cluster size only")
    p.add_argument("--pca-dim", type=int, default=None,
                   help="LA2M: subspace dimension, overriding --dhat for it only")
    p.add_argument("--pca-mode", choices=["local", "global"], default=None,
                   help="LA2M: global = the original package's one PCA over the database")
    p.add_argument("--transpose-local", action="store_true",
                   help="LA2M: apply every local isometry as x R instead of x R^T")
    p.add_argument("--complement", action="store_true",
                   help="LA2M: fill the component outside each local subspace from the global map")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": args.dataset,
        "source_model": args.source,
        "target_model": args.target,
        "mapping_method": args.method,
        "reference_method": args.reference_method,
        "reference_ratio": args.reference_ratio,
    }

    started = time.time()
    try:
        set_seed(42)
        config = ConfigLoader().load_config().config
        if args.dhat is not None:
            la2m_cfg = config.mapping_config.la2m_config
            la2m_cfg.use_intrinsic_dimension = False
            la2m_cfg.d_prime = la2m_cfg.pca_dim = args.dhat
            config.clustering_config.la2m_config.d_prime = args.dhat
        la2m_cfg = config.mapping_config.la2m_config
        if args.cluster_size is not None or args.pca_dim is not None:
            la2m_cfg.use_intrinsic_dimension = False
        if args.cluster_size is not None:
            la2m_cfg.d_prime = config.clustering_config.la2m_config.d_prime = args.cluster_size
        if args.pca_dim is not None:
            la2m_cfg.pca_dim = args.pca_dim
        if args.pca_mode is not None:
            la2m_cfg.pca_mode = args.pca_mode
        if args.transpose_local:
            config.mapping_config.la2m_config.transpose_local = True
        if args.complement:
            config.mapping_config.la2m_config.complement = True
        record["config"] = config.to_dict()

        dataset = load_dataset(
            dataset_name=args.dataset, data_path=config.data_path,
            force=False, interactive=False, verbose=False,
        )
        corpus_emb_1, corpus_emb_2 = get_embedding(
            dataset_name=args.dataset, model_name=args.source,
            embedding_path=config.embedding_path, target_model_name=args.target,
            type_="corpus", align=True,
        )
        query_emb_1, query_emb_2 = get_embedding(
            dataset_name=args.dataset, model_name=args.source,
            embedding_path=config.embedding_path, target_model_name=args.target,
            type_="query", align=True,
        )

        if args.reference_method == "stratified":
            d0_index, d1_index, d2_index = stratified_split(
                np.asarray(corpus_emb_1), args.reference_ratio, args.strata)
        else:
            embedding_dataset = EmbeddingDataset(
                text_dataset_name=args.dataset,
                source_embedding_model_name=args.source,
                target_embedding_model_name=args.target,
                align_dimension=True,
                dataset_path=config.data_path,
                embedding_path=config.embedding_path,
                reference_method=args.reference_method,
                reference_config={"reference_ratio": args.reference_ratio},
            )
            d0_index = embedding_dataset.d0_index
            d1_index = embedding_dataset.d1_index
            d2_index = embedding_dataset.d2_index

        mapper = VectorSpaceMapper(
            strategy_name=args.method,
            config=config.mapping_config,
            dataset_name=args.dataset,
            source_model=args.source,
            target_model=args.target,
            reference_key=None,
            reference_path=config.reference_path,
            cluster_path=config.cluster_path,
            embedding_path=config.embedding_path,
            mapping_param_path=config.mapping_param_path,
            mapping_embedding_path=config.mapping_embedding_path,
            force=True,
            save_param=False,
            save_embedding=False,
            clustering_config=config.clustering_config,
        )
        transformed = mapper.fit_and_transform(corpus_emb_1, corpus_emb_2, d0_index)

        evaluator = Evaluator(
            corpus_emb_1=corpus_emb_1,
            corpus_emb_2=corpus_emb_2,
            query_emb_1=query_emb_1,
            query_emb_2=query_emb_2,
            query_index2answer_index=dataset.query_index2answer_index,
            d0=d0_index, d1=d1_index, d2=d2_index,
            corpus_emb_1_transformed=transformed,
            k_list=[10, 50, 100, 500, 1000],
            source_side=(np.concatenate([d0_index, d1_index])
                         if args.reference_method != "la2m" else d1_index),
        )
        record["results"] = {k: float(v) for k, v in evaluator.evaluate().items()}
        record["shapes"] = {
            "corpus_source": list(corpus_emb_1.shape),
            "corpus_target": list(corpus_emb_2.shape),
            "query": list(query_emb_1.shape),
            "d0": len(d0_index), "d1": len(d1_index), "d2": len(d2_index),
        }
        record["status"] = "ok"
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()

    record["seconds"] = round(time.time() - started, 1)
    out.write_text(json.dumps(record, indent=2))
    print(f"{record['status']}  {args.dataset}/{args.source}->{args.target}/{args.method}  {record['seconds']}s")
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
