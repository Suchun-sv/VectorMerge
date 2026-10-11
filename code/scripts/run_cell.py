"""Run one cell of a paper table and write the metrics as JSON.

Mirrors `vectormerge evaluate single-run` (src/vectormerge/cli/evaluate.py) but is
importable, non-interactive, and persists structured results so a whole grid can be
assembled and compared against the published tables.

    python scripts/run_cell.py --dataset nfcorpus --source mistral --target openai \
        --method la2m --out results/table4_nfcorpus_la2m.json
"""
import argparse
import json
import time
import traceback
from pathlib import Path

from vectormerge.cli.base import set_seed
from vectormerge.config import ConfigLoader
from vectormerge.dataset import load_dataset
from vectormerge.embeddings import get_embedding
from vectormerge.embedding_dataset import EmbeddingDataset
from vectormerge.evaluation import Evaluator
from vectormerge.mapping import VectorSpaceMapper


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--reference-method", default="la2m")
    p.add_argument("--out", required=True)
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
    }

    started = time.time()
    try:
        # the CLI seeds via its typer callback; the reference split is random, so a
        # missing seed silently changes the numbers between runs
        set_seed(42)
        config = ConfigLoader().load_config().config
        if args.dhat is not None:
            la2m_cfg = config.mapping_config.la2m_config
            la2m_cfg.use_intrinsic_dimension = False
            la2m_cfg.d_prime = la2m_cfg.pca_dim = args.dhat
            # the cluster size is read from the clustering config, not the mapping one
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

        embedding_dataset = EmbeddingDataset(
            text_dataset_name=args.dataset,
            source_embedding_model_name=args.source,
            target_embedding_model_name=args.target,
            align_dimension=True,
            dataset_path=config.data_path,
            embedding_path=config.embedding_path,
            reference_method=args.reference_method,
            reference_config={},
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
            # each cell must compute from scratch: the cached path reloads a cluster
            # result without marking the strategy fitted, and a cached mapping would
            # otherwise be paired with a freshly drawn reference split
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
        )
        record["results"] = {k: float(v) for k, v in evaluator.evaluate().items()}
        record["shapes"] = {
            "corpus_source": list(corpus_emb_1.shape),
            "corpus_target": list(corpus_emb_2.shape),
            "query": list(query_emb_1.shape),
            "d0": len(d0_index), "d1": len(d1_index), "d2": len(d2_index),
        }
        record["status"] = "ok"
    except Exception as exc:  # a failed cell must not take down the grid
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()

    record["seconds"] = round(time.time() - started, 1)
    out.write_text(json.dumps(record, indent=2))
    print(f"{record['status']}  {args.dataset}/{args.source}->{args.target}/{args.method}  {record['seconds']}s")
    return 0 if record["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
