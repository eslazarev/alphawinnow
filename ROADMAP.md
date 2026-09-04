# AlphaWinnow Roadmap

AlphaWinnow is a local-first Rust engine for formulaic alpha mining,
quantitative factor research, semantic expression deduplication, and causal
factor evaluation on user-supplied data.

This roadmap describes intended product direction, not a promise of delivery
dates. Priorities may change when benchmarks, correctness tests, or community
feedback show that a different order is more useful.

## Available now

- A typed quantitative-expression AST, parser, formatter, and validator.
- A configurable catalog for fields, groups, operators, scalar domains, and
  rolling windows.
- Deterministic generation, mutation, and crossover with bounded complexity.
- Exact fingerprints and conservative semantic deduplication.
- Positive final-root scale equivalence without unsafe rewrites across
  nonlinear, conditional, group, or individual-sleeve boundaries.
- Structural novelty scoring and diversity-aware shortlist selection.
- Parallel local search using a bounded Rayon thread pool.
- Atomic JSONL artifacts, checksummed manifests, and checkpoint/resume.
- Optional causal numeric evaluation with train, validation, and holdout
  diagnostics.
- Optional local Sharadar-compatible Parquet preparation.
- Immutable measured-feedback prioritization with leave-one-out auditing.
- Target-dialect compilation and fail-closed linting.

## Next

### Streaming Arrow and Parquet evaluation

Evaluate datasets larger than memory without first materializing the complete
portable JSON representation.

Planned work:

- column projection and date-range pushdown;
- bounded row-group processing;
- deterministic asset and timestamp ordering;
- point-in-time group classifications;
- input, preprocessing, and schema checksums;
- parity fixtures between JSON and Parquet inputs.

### Shared expression DAG

Compile a candidate batch into a shared typed execution graph so repeated
subexpressions and rolling windows are computed once.

Planned work:

- hash-consed exact subexpression nodes;
- reusable rolling and cross-sectional work buffers;
- bounded caches keyed by dataset and evaluation configuration;
- explicit cache-hit, memory, and throughput diagnostics;
- numeric parity tests against the existing reference evaluator.

### More public catalogs and data adapters

Make AlphaWinnow easier to try without coupling the core engine to one data
vendor or research platform.

Planned work:

- documented catalog-authoring examples;
- small license-safe OHLCV fixtures;
- additional local columnar-data adapters;
- clearer operator-extension documentation;
- validation tools for third-party catalogs and dialects.

## Later

### Measured multi-objective selection

Explore deterministic pool selection across measured quality, stability,
turnover, complexity, and diversity. Any new selector must beat the current
auditable baseline on a preregistered held-out workload before becoming a
default.

### Stronger conservative deduplication

Add a small versioned set of equivalence rules only when symbolic reasoning and
randomized numeric tests establish that they are safe under the active catalog.
Behavioral similarity may identify near-duplicates, but it will not be treated
as proof of exact equivalence.

### Profile-guided CPU optimization

Investigate compact AST storage, shared subexpression hashes, SIMD-friendly
descriptors, and bounded approximate-neighbor indexes. Optimizations must
preserve deterministic candidate identity and pass end-to-end benchmarks.

### Optional GPU evaluation

Consider a GPU backend only if profiling shows that numeric batch evaluation
dominates runtime after CPU vectorization and shared-DAG reuse. The CPU backend
will remain supported and GPU transfer overhead will be included in every
comparison.

### Stable library release

Prepare the public Rust API for a crates.io release with versioned artifact
schemas, migration guidance, examples, and a documented compatibility policy.

## Good first contributions

- Add parser error examples and improve diagnostics without changing accepted
  syntax.
- Add license-safe synthetic datasets covering missing values and rank ties.
- Add an operator by extending the public catalog, evaluator, and parity tests.
- Improve catalog-authoring and custom-dialect documentation.
- Add platform-independent benchmark reporting.
- Add property tests for canonicalization, mutation, and crossover boundaries.
- Improve error messages for malformed JSONL, datasets, and checkpoints.

Before proposing a large feature, open an issue describing its use case,
correctness contract, resource limits, and how it can be tested without private
data.

## Engineering principles

- Local-only by default: no credentials, telemetry, broker, exchange, or remote
  research-platform integration.
- Reproducible artifacts: explicit schemas, checksums, provenance, and atomic
  publication.
- Causal evaluation: no future data may enter a signal or cross a diagnostic
  split boundary.
- Honest metrics: structural scores and measured market evidence remain
  separate.
- Conservative semantics: uncertain equivalences remain distinct.
- Bounded resources: memory, worker pools, search duration, and expression
  complexity have explicit limits.
- Benchmark before optimizing: keep changes only when reproducible evidence
  supports them.

## Non-goals

AlphaWinnow does not plan to provide:

- broker or exchange connectivity;
- live trading or order execution;
- hosted proprietary market data;
- credential storage;
- remote alpha simulation or submission automation;
- guarantees of profitability or future performance.

## How roadmap items graduate

A roadmap item is complete only when:

1. its behavior and failure modes are documented;
2. deterministic or explicitly bounded behavior is tested;
3. relevant correctness and numeric parity tests pass;
4. resource use is measured on a reproducible workload;
5. `cargo fmt`, Clippy with warnings denied, and all-feature tests pass.

