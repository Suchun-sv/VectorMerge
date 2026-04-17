# ml-dev-rules Architecture Migration Plan

## 0) Goal and Scope
1. Migrate this repository to the `ml-dev-rules` architecture skeleton (`settings.yaml`, `config/models.py`, `config/loader.py`, `pipeline/`, Typer CLI -> pipeline mapping).
2. Preserve current behavior first; do not rewrite core algorithms during initial migration.
3. Deliver migration in small, reversible phases with verification at each phase.

## 1) Baseline Capture (Regression Guard)
1. Capture current expected behavior for key commands:
   - `vectormerge dataset download -d scifact`
   - `vectormerge embedding generate -m mistral -d scifact --type corpus`
   - `vectormerge create-reference -d scifact --strategy random`
   - `vectormerge create-cluster -d scifact -m mistral ...`
   - `vectormerge map-embedding map ...`
   - `vectormerge evaluate single-run ...`
2. Define pass criteria per command: exit code, output artifacts, and key result fields.
3. Keep this baseline as the acceptance checklist for each migration phase.

## 2) Scaffold the Target Layout
1. Add root `settings.yaml` as the canonical default configuration source.
2. Add `src/vectormerge/config/models.py` with root `Settings` and nested sections.
3. Add `src/vectormerge/config/loader.py` with unified load order and dot-notation overrides.
4. Add `src/vectormerge/pipeline/` package with one pipeline module per command group.
5. Keep old config/CLI paths working during this phase.

## 3) Configuration Migration (Core)
1. Standardize precedence: CLI overrides > env vars > `.env` > `settings.yaml` > defaults.
2. Support `-s/--set key.sub_key=value` plus existing unknown-option dotted args.
3. Bridge to legacy dataclass config so existing execution path keeps working.
4. Resolve current config-source ambiguity by making `settings.yaml` the source of truth.

## 4) CLI to Pipeline Mapping
1. Map each CLI command to exactly one pipeline function:
   - dataset download -> `dataset_pipeline`
   - embedding generate -> `embedding_pipeline`
   - create-reference -> `reference_pipeline`
   - create-cluster -> `cluster_pipeline`
   - map-embedding map -> `mapping_pipeline`
   - evaluate single-run -> `evaluation_pipeline`
2. Keep CLI handlers thin: parse args, load config, call pipeline.
3. Keep business logic in existing modules until later extraction.

## 5) Module Boundary Cleanup
1. Introduce `src/vectormerge/model/` for model-centric components.
2. Keep `dataset/`, `embeddings/`, `reference/`, `clustering/`, `mapping/`, `evaluation/` stable while pipelines mature.
3. Refactor imports incrementally; avoid large one-shot rewrites.

## 6) Compatibility and Deprecation
1. Keep legacy command aliases functional for one migration window.
2. Add deprecation guidance in CLI output and docs.
3. Remove old pathways only after parity is validated.

## 7) Verification Strategy
1. Run a small smoke suite after each phase (single dataset/model).
2. Re-run baseline checklist before and after major merges.
3. Track differences explicitly as acceptable or blocking.

## 8) Documentation Updates
1. Update README to reflect real, current commands only.
2. Update AGENTS.md with final migrated architecture and verification commands.
3. Add config migration notes (old keys -> `settings.yaml` sections).

## 9) Risks and Mitigations
1. Config drift risk: lock verification to baseline artifacts and key metrics.
2. Large-data runtime risk: use scoped smoke tests and staged validation.
3. User workflow breakage risk: preserve compat commands until migration complete.

## 10) Milestones
1. M1: `settings.yaml` + new config models/loader + legacy bridge.
2. M2: Main CLI commands load config through new path.
3. M3: CLI -> pipeline mapping completed across all core commands.
4. M4: Docs + deprecations + final regression verification.
