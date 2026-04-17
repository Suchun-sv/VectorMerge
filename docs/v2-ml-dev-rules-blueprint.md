# VectorMerge V2 Blueprint (ml-dev-rules aligned)

## 1. Product Goal

Build a new version that is both:
- a reusable Python library (sklearn-like usage),
- and a benchmark-capable runner for method evaluation.

MVP CLI surface:
1. `vectormerge dataset download -d scifact -m all`
2. `vectormerge tran-emb -m la2m -d scifact -s gte -t openai --set ...`
3. `vectormerge tran-emb -m wasserstein -d scifact -s gte -t openai --set ...`

## 2. Architecture Principles

- CLI is a shell, not the core runtime.
- Core logic lives in reusable Python APIs.
- All runtime inputs use Spec objects.
- All outputs use structured Result objects.
- Method integrations use registry-based plug-in model.
- Reproducibility is mandatory (run_id + config snapshot + artifact manifest).

## 3. Proposed Project Layout (V2)

project-root/
- settings.yaml
- src/vectormerge/
  - __init__.py
  - __main__.py
  - cli/
    - __init__.py
    - dataset.py
    - tran_emb.py
  - core/
    - api.py
    - specs.py
    - results.py
    - runner.py
    - context.py
  - registry/
    - methods.py
    - datasets.py
    - models.py
  - methods/
    - base.py
    - la2m.py
    - wasserstein.py
  - adapters/
    - dataset/
      - base.py
      - beir.py
    - embedding/
      - base.py
      - gte.py
      - openai.py
  - config/
    - models.py
    - loader.py

## 4. Stable Public API

Expose only:
- `prepare_dataset(spec: DatasetPrepareSpec) -> DatasetPrepareResult`
- `run_transform(spec: TransformSpec) -> TransformResult`

These are the integration entrypoints for notebooks/Airflow/services.

## 5. Spec/Result Contract (MVP)

### RunOptions
- force: bool = False
- resume: bool = True
- dry_run: bool = False
- seed: int = 42
- verbose: bool = False
- run_id: str | None = None

### DatasetPrepareSpec
- dataset: str
- model_mode: str = "none"  # none | all | pair
- source_model: str | None = None
- target_model: str | None = None
- overrides: dict[str, object] = {}
- options: RunOptions

### TransformSpec
- method: str  # la2m | wasserstein (MVP)
- dataset: str
- source_model: str
- target_model: str
- overrides: dict[str, object] = {}  # include reference_strategy here
- eval_enabled: bool = True
- options: RunOptions

### TransformResult
- run_id: str
- status: str  # success | failed | partial
- method: str
- dataset: str
- source_model: str
- target_model: str
- metrics: dict[str, object]
- artifacts: dict[str, str]
- stages: list[dict[str, object]]
- config_snapshot: dict[str, object]
- error: dict[str, str] | None

## 6. Override Handling Rules

- All runtime knobs (including `reference_strategy`) come from `overrides`.
- Priority (high -> low):
  1) CLI `--set`
  2) environment variables
  3) settings.yaml
  4) method default
- Dotted keys are supported:
  - `reference_strategy=la2m`
  - `mapping_config.la2m_config.num_clusters=50`
- Unknown keys: fail-fast in strict mode (recommended), warn in dev mode.

## 7. Method Registry (MVP)

- `MethodRegistry.register("la2m", LA2MMethod)`
- `MethodRegistry.register("wasserstein", WassersteinMethod)`
- Common interface:
  - `prepare(context)`
  - `fit(context)`
  - `transform(context)`
  - `evaluate(context)`

## 8. Runner Stage Model

For `run_transform`:
1. resolve_config
2. ensure_dataset
3. ensure_embeddings
4. ensure_reference
5. ensure_cluster (method-dependent)
6. fit_transform_evaluate
7. persist_result_manifest

Each stage writes status for resume support.

## 9. Notebook Integration Example

```python
from vectormerge.core.api import prepare_dataset, run_transform
from vectormerge.core.specs import DatasetPrepareSpec, TransformSpec, RunOptions

prepare_dataset(DatasetPrepareSpec(dataset="scifact", model_mode="all", options=RunOptions()))

res = run_transform(
    TransformSpec(
        method="la2m",
        dataset="scifact",
        source_model="gte",
        target_model="openai",
        overrides={
            "reference_strategy": "la2m",
            "mapping_config.la2m_config.num_clusters": 50,
        },
        options=RunOptions(resume=True),
    )
)
print(res.status, res.metrics, res.artifacts)
```

## 10. Delivery Phases

Phase 1:
- create core specs/results/api, registry, and runner skeleton
- keep old code as adapters

Phase 2:
- implement `dataset download -d ... -m all` through core API
- implement `tran-emb` for `la2m` and `wasserstein`

Phase 3:
- run smoke tests for the 3 target commands
- enforce structured result output and manifest files

Phase 4:
- deprecate legacy command paths
- update docs and AGENTS.md to V2 truth only
