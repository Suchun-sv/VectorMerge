# V2 Migration Checklist

Execution mode: complete this file top-to-bottom. Keep each task independently testable.

## Phase 1 - Core foundation (current priority)

- [x] Create `src/vectormerge/core/specs.py` with:
  - [x] `RunOptions`
  - [x] `DatasetPrepareSpec`
  - [x] `TransformSpec`
  - [x] Minimal validation for method/model mode values

- [x] Create `src/vectormerge/core/results.py` with:
  - [x] `DatasetPrepareResult`
  - [x] `TransformResult`
  - [x] `StageStatus` helper structure

- [x] Create `src/vectormerge/core/context.py` with `RunContext`

- [x] Create `src/vectormerge/core/runner.py` with:
  - [x] `prepare_dataset_runner(spec)`
  - [x] `transform_runner(spec)`
  - [x] stage tracking (`resolve_config`, `ensure_dataset`, `ensure_embeddings`, `fit_transform_evaluate`)

- [x] Create `src/vectormerge/core/api.py` with stable public functions:
  - [x] `prepare_dataset(spec)`
  - [x] `run_transform(spec)`

- [x] Create registry skeleton:
  - [x] `src/vectormerge/registry/methods.py`
  - [x] `src/vectormerge/registry/datasets.py`
  - [x] `src/vectormerge/registry/models.py`

- [x] Create methods skeleton:
  - [x] `src/vectormerge/methods/base.py`
  - [x] `src/vectormerge/methods/la2m.py`
  - [x] `src/vectormerge/methods/wasserstein.py`

- [x] Export core API from package root (`src/vectormerge/__init__.py`)

- [x] Validate Phase 1 with:
  - [x] `python -m compileall src/vectormerge/core src/vectormerge/registry src/vectormerge/methods`

## Phase 2 - CLI wiring for MVP commands

- [x] Keep and adapt `vectormerge dataset download -d <dataset> -m all`
  - [x] Parse `-m/--model-mode` (`none|all|pair`)
  - [x] Build `DatasetPrepareSpec`
  - [x] Call `prepare_dataset()`

- [x] Add new command group/file `src/vectormerge/cli/tran_emb.py`
  - [x] Command: `vectormerge tran-emb`
  - [x] Required args: `-m/--method`, `-d/--dataset`, `-s/--source-model`, `-t/--target-model`
  - [x] `--set` repeated overrides support
  - [x] Build `TransformSpec`
  - [x] Call `run_transform()`

- [x] Register command in `src/vectormerge/cli/__init__.py`

- [x] Validate Phase 2 with:
  - [x] `uv run vectormerge --help`
  - [x] `uv run vectormerge tran-emb --help`

## Phase 3 - Method execution integration

- [x] Connect `la2m` method class to existing mapping pipeline implementation
- [x] Connect `wasserstein` method class to existing mapping pipeline implementation
- [x] Ensure both methods produce the same `TransformResult` schema
- [x] Persist run manifest (`config_snapshot`, `metrics`, `artifacts`) under `output/runs/<run_id>/`

- [ ] Validate Phase 3 with smoke commands:
  - [ ] `vectormerge dataset download -d scifact -m all`
  - [ ] `vectormerge tran-emb -m la2m -d scifact -s gte -t openai --set ...`
  - [ ] `vectormerge tran-emb -m wasserstein -d scifact -s gte -t openai --set ...`

## Phase 4 - Cleanup and deprecation

- [ ] Mark redundant legacy command pathways as deprecated
- [ ] Update README command examples to V2-first usage
- [ ] Update AGENTS.md with final V2 runtime truth
- [ ] Keep backward-compatible aliases temporarily where low cost

## Definition of done

- [ ] The 3 MVP commands run end-to-end
- [ ] Notebook users can call `prepare_dataset()` and `run_transform()` directly
- [ ] New methods can be added via method registry without editing CLI logic
- [ ] Run outputs are structured and reproducible
