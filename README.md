# AlphaWinnow — Formulaic Alpha Mining in Rust

[![CI](https://github.com/eslazarev/alphawinnow/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/eslazarev/alphawinnow/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/alphawinnow.svg?logo=rust)](https://crates.io/crates/alphawinnow)
[![docs.rs](https://img.shields.io/docsrs/alphawinnow?logo=docsdotrs)](https://docs.rs/alphawinnow)
[![API docs](https://img.shields.io/badge/API%20docs-main-informational?logo=rust)](https://eslazarev.github.io/alphawinnow/)
[![Rust 1.92+](https://img.shields.io/badge/Rust-1.92%2B-black?logo=rust)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Local only](https://img.shields.io/badge/network-local--only-success)](#security-and-scope)

**AlphaWinnow is an open-source Rust CLI and library for formulaic alpha
mining, quantitative factor research, symbolic expression search, and local
factor evaluation.** It uses a typed expression grammar, deterministic genetic
programming, semantic deduplication, structural novelty search, Rayon
parallelism, and optional causal evaluation on your own market data.

Give AlphaWinnow a field/operator catalog and an optional archive of measured
expressions. It explores a large space of valid quantitative formulas, removes
equivalent or trivial signals, and writes a small, diverse, reproducible
shortlist for downstream research.

AlphaWinnow is not a trading bot, broker integration, or hosted alpha database.
It runs locally, performs no network requests, and never claims that a generated
formula is profitable.

## Why AlphaWinnow?

Most alpha factor generators can produce thousands of formulas. The difficult
part is keeping the search typed, bounded, reproducible, and diverse while not
wasting evaluation time on duplicate expressions. AlphaWinnow focuses on that
selection problem.

- **Strongly typed formula generation:** invalid fields, operators, argument
  types, windows, and keyword parameters are rejected before search.
- **Genetic programming for alpha factors:** seeded generation, mutation, and
  crossover explore a configurable quantitative-expression DSL.
- **Semantic alpha deduplication:** exact duplicates, provably trivial signals,
  and positive final-scale equivalents collapse into one family.
- **Diversity-aware factor mining:** structural descriptors discourage a
  shortlist filled with cosmetic variations of the same idea.
- **Fast Rust execution:** a bounded Rayon worker pool evaluates search work in
  parallel with explicit CPU and memory limits.
- **Local factor evaluation:** optional train, validation, and holdout evidence
  includes Pearson signal/return correlation, gross and net returns, turnover,
  costs, Sharpe, drawdown, and auditable daily results.
- **Reproducible quant research:** deterministic seeds, checksums, immutable
  JSONL artifacts, atomic writes, manifests, and checkpoint/resume.
- **Bring your own data:** use a custom field/operator catalog, immutable
  columnar JSON, or the optional local Sharadar Parquet adapter.

## Common use cases

AlphaWinnow is designed for quantitative researchers, systematic traders,
financial machine-learning engineers, and developers building alpha research
pipelines. Typical workflows include:

- generating formulaic alpha-factor candidates from OHLCV or fundamental data;
- building a strongly typed genetic-programming search for trading signals;
- deduplicating large collections of WorldQuant-style expressions;
- evaluating factors locally with causal train/validation/holdout splits;
- ranking candidates by structural novelty before an expensive backtest;
- creating reproducible JSONL datasets for symbolic regression or ML research;
- benchmarking parallel quantitative-expression search in Rust.

## Install

Every prebuilt channel below serves the same platform binaries that
[`release.yml`](.github/workflows/release.yml) built. A source installation
instead compiles the same version from its locked dependencies.

**Homebrew** (macOS and Linux, Apple silicon and x86-64):

```bash
brew tap eslazarev/alphawinnow https://github.com/eslazarev/alphawinnow
brew install alphawinnow
```

**Scoop** (Windows):

```powershell
scoop install https://raw.githubusercontent.com/eslazarev/alphawinnow/main/packaging/scoop/alphawinnow.json
```

**Debian, Ubuntu, Fedora, RHEL** — download the package for your architecture
from the [latest release](https://github.com/eslazarev/alphawinnow/releases/latest)
and install it:

```bash
sudo dpkg -i alphawinnow_<version>-1_amd64.deb     # or _arm64.deb
sudo rpm -i alphawinnow-<version>-1.x86_64.rpm     # or .aarch64.rpm
```

**Prebuilt binary through Cargo**, without compiling:

```bash
cargo binstall alphawinnow
```

**From source**, which needs Rust 1.92 or newer:

```bash
cargo install alphawinnow --locked
```

Or download an archive for your platform straight from the
[latest release](https://github.com/eslazarev/alphawinnow/releases/latest).
Each one carries its own `.sha256`, the README, and the licence.

The `.deb` and `.rpm` are built from the very binary the tarball ships, not from
a separate compilation, and they are not stripped. Each release archive and
native package has its own SHA-256 checksum because the surrounding package
formats have different bytes.

The explicit tap URL is needed because this repository is not named
`homebrew-alphawinnow`; the formula lives in [`Formula/`](Formula) and is
regenerated from the release archives by
[`scripts/update-packaging.sh`](scripts/update-packaging.sh), never by hand.

## Quick start

Rust 1.92 or newer is required to build from source.

Build it from a checkout:

```bash
git clone https://github.com/eslazarev/alphawinnow.git
cd alphawinnow
cargo build --release --workspace

./target/release/alphawinnow doctor --json
./target/release/alphawinnow inspect 'rank(ts_delta(close, 20))'
./target/release/alphawinnow search \
  --output candidates.jsonl \
  --duration-seconds 60 \
  --max-candidates 100000 \
  --threads 4 \
  --seed 20260828 \
  --checkpoint search.checkpoint.json
```

The search writes five durable artifacts:

- `candidates.jsonl`: the diverse expression shortlist;
- `candidates.jsonl.manifest.json`: configuration, counters, checksums, limits,
  and termination evidence;
- `candidates.jsonl.current.json`: the publication pointer naming the last
  complete candidate/manifest pair;
- `.candidates.jsonl.runs/`: that pair, written immutably before the pointer is
  replaced;
- `search.checkpoint.json`: resumable deterministic search state.

Read `candidates.jsonl.current.json` rather than the two adjacent aliases when
you need a guarantee that the candidates and the manifest describe the same
run. [Artifact schemas](#artifact-schemas) documents the publication protocol.

Generated candidates are research hypotheses, not trading recommendations.
`structural_score` measures expression structure and novelty; it is not Sharpe,
Fitness, expected return, or pass probability.

### Use as a library

The same crate is the library. The binary and its argument parser live behind
the default `cli` feature, so turn defaults off to embed the engine without
pulling in a command-line stack:

```toml
[dependencies]
alphawinnow = { version = "0.1", default-features = false }

# With optional causal numeric evaluation and local Parquet preparation:
# alphawinnow = { version = "0.1", default-features = false, features = ["parquet-input"] }
```

```rust
use alphawinnow::{analyze_expression, parse_expression, semantic_fingerprint};

fn main() -> Result<(), alphawinnow::ExpressionError> {
    let expression = parse_expression("rank(ts_delta(close, 20))")?;
    assert!(analyze_expression(&expression).rejection_reason.is_none());
    println!("{}", semantic_fingerprint(&expression));
    Ok(())
}
```

## How it works

```text
typed catalog + optional measured archive
                    |
                    v
        generate / mutate / crossover
                    |
                    v
       parse + type-check + enforce limits
                    |
                    v
    canonicalize + semantic deduplicate
                    |
                    v
       structural novelty + diverse top-k
                    |
                    v
        JSONL shortlist + manifest
                    |
                    v
 optional local factor evaluation on your data
```

The default search is structural and data-independent. Market evidence is an
explicit optional stage, stored separately from structural ranking so a local
metric can never silently masquerade as a remote-platform result.

## CLI commands

| Command | Purpose |
|---|---|
| `doctor` | Verify the parser, hashing, and local runtime |
| `inspect` | Parse and explain canonical and semantic identity |
| `dedup` | Remove exact and positive-root-scale duplicates |
| `search` | Run bounded parallel expression discovery |
| `benchmark` | Compare reproducible one/many-thread workloads |
| `prioritize` | Re-rank candidates using immutable measured feedback |
| `audit-feedback` | Audit feedback guidance with leave-one-out evaluation |
| `compile` | Lower and lint an expression against a target dialect |
| `evaluate` | Evaluate one expression on local data (optional feature) |
| `evaluate-batch` | Evaluate candidate JSONL in bounded batches (optional) |
| `prepare-sharadar` | Build a causal local panel from Sharadar Parquet (optional) |

### Search a custom alpha-expression space

```bash
alphawinnow search \
  --catalog user-catalog.json \
  --archive measured.jsonl \
  --output candidates.jsonl \
  --duration-seconds 3600 \
  --max-candidates 1000000 \
  --threads 8 \
  --seed 20260828 \
  --max-depth 20 \
  --max-nodes 192 \
  --max-operators 96 \
  --checkpoint search.checkpoint.json
```

`--archive` is optional. When present, AlphaWinnow partitions it into accepted,
semantic-duplicate, and provably trivial records. The valid portion supplies
search context without being silently treated as proof of future performance.

Resume an interrupted run with the same catalog, archive, limits, and seed:

```bash
alphawinnow search \
  --archive measured.jsonl \
  --output candidates.jsonl \
  --duration-seconds 3600 \
  --max-candidates 1000000 \
  --threads 8 \
  --seed 20260828 \
  --max-depth 20 \
  --max-nodes 192 \
  --max-operators 96 \
  --resume-from search.checkpoint.json
```

### Deduplicate alpha formulas

```bash
alphawinnow dedup --input candidates.jsonl --output unique.jsonl
```

### Use measured feedback without mixing it into structural score

```bash
alphawinnow prioritize \
  --input candidates.jsonl \
  --feedback examples/public-feedback-v1.json \
  --output guided.jsonl \
  --neighbors 8 \
  --minimum-neighbors 3

alphawinnow audit-feedback \
  --feedback examples/public-feedback-v1.json \
  --neighbors 8 \
  --minimum-neighbors 3
```

### Run local factor evaluation

```bash
cargo run --release --features numeric-evidence -- evaluate-batch \
  --dataset panel-with-warmup.json \
  --catalog optional-dataset-catalog.json \
  --input candidates.jsonl \
  --output measured-evidence.jsonl \
  --horizon-rows 1 \
  --evaluation-start 252 \
  --train-end 1000 \
  --validation-end 1250 \
  --annualization-rows 252 \
  --transaction-cost-bps 5 \
  --signal-decay 8
```

The evaluator loads and validates the dataset once, preserves causal warm-up,
keeps future targets outside the expression inputs, and records failed
expressions without discarding successful rows.

## Custom catalogs and quantitative-expression DSLs

`--catalog` activates a schema-1 user-supplied catalog for the complete run.
Fields, groups, operator availability, generation weights, window domains, and
scalar domains are data rather than hard-coded search assumptions. The catalog
checksum becomes part of the manifest and checkpoint compatibility contract.

This makes AlphaWinnow useful as a generic quantitative factor-mining engine:
you can model an OHLCV formula language, a fundamental-factor DSL, or a
restricted expression dialect supported by another research system. External
catalogs are local files and are never downloaded or embedded into output
beyond the symbols used by an expression.

## Expression subset

The embedded, license-safe catalog is
[`public-v1.json`](crates/alphawinnow/catalog/public-v1.json). It declares
operators, typed arguments, arity, commutativity/associativity, generation
weights, window/non-zero-scalar/keyword domains, keyword relations, fields and
public family labels, field roles, groups, and the scalar domain. Fields are
the catalog identifiers `close`, `open`, `high`, `low`, `volume`, and
`returns`. The parser also accepts finite scalar literals, `true`/`false`, and
catalog group literals such as `group("sector")`; unknown fields and groups are
rejected.

Parsing, validation, generation, compatible-operator replacement, field/group/
scalar/window/keyword mutation, tree-slot typing, novelty families, checkpoint
compatibility, and manifest identity all consume this catalog. Every
generation-enabled operator is reachable in a fixed coverage test, including
`clip`, `divide`, and `negate`. `Catalog::from_json`,
`parse_expression_with_catalog`, `generate_with_catalog`, and
`validate_transformed_with_catalog` form the public import boundary. Adding a
synthetic unary operator or field is covered by tests without adding generator
match arms. External catalogs must be license-safe and are never fetched over
the network.

Supported signal operators are:

- arithmetic: `add(a, b, ...)`, `subtract(a, b)`,
  `multiply(signal, scalar, ...)`, `divide(signal, nonzero_scalar)`, `negate(a)`;
- time series: `ts_rank(a, window)`, `ts_mean(a, window)`,
  `ts_std_dev(a, window)`, `ts_zscore(a, window)`, and
  `ts_delta(a, window)`, with integer windows in `2..=512`;
- cross sectional: `rank(a)`, `zscore(a)`,
  `group_rank(a, group("sector"))`;
- conditions: `greater(signal, scalar)` and
  `if_else(boolean, then_signal, else_signal)`;
- nonlinear bounds: `winsorize(a, std=2)` where `std` is in `0.1..=10`, and
  `clip(a, lower=-2, upper=2)` where both bounds are in `-100..=100` and
  `lower < upper`.

Unknown operators, incorrect arity or kinds, invalid windows, non-finite
scalars, unsupported keywords, non-literal keyword values, and out-of-range
operator parameters are rejected before search or deduplication.

Formatting normalizes numeric spelling and keyword order. `add` and
`multiply` operands are sorted because those registry entries are explicitly
commutative. Exact fingerprints are SHA-256 hashes of exact canonical text.
Semantic analysis flattens associative multiplication only along the final
root, combines finite scalar factors, removes positive magnitude, and retains
a normalized negative direction. Thus `close`, `multiply(close, 6)`,
`multiply(close, -2, -3)`, and `add(close, close)` share a family, while
`multiply(close, -2)` remains directionally distinct. Scaling inside a
nonlinear operator, condition, clipping/winsorization step, or one composite
sleeve is never moved across that boundary.

The same conservative pass rejects roots proven to be zero or constant, such
as `subtract(close, close)`, multiplication by zero, or `rank` of a proven-zero
subtree. Uncertain cases remain eligible. `inspect` exposes `rejection_reason`
as `null`, `provably_zero`, or `provably_constant`. `dedup` writes all accepted
records while separately reporting deterministic duplicate and rejection
counts, so one old trivial record cannot abort the remaining valid stream.

## Search and structural score

The seeded generator, mutation, and typed crossover enforce configured depth,
node, operator, window, scalar, and operator-parameter limits. Immutable paths
identify the root, every argument, and every keyword slot together with its
required kind. Mutation covers fields, windows, ordinary scalars, groups,
bounded keywords, boolean literals, compatible operators, and generated typed
subtrees. Crossover replaces a receiver subtree only with a compatible donor
kind. Every result round-trips through the public parser and passes complete
limits validation.

Retries are capped at 16. Successful lineage records the affected path,
old/new exact subtree fingerprints, retries, and semantic parent fingerprints.
If retries cannot produce a valid child, the actual operation is
`fallback_generation`; its parent list is empty and the requested transform is
recorded separately. A bounded Rayon pool executes indexed offspring
generation, validation, semantic analysis, fingerprinting, and structural
feature computation. Results merge in ordinal order; scored parent selection
and family-balanced elitism bound the population at 512 expressions. The
resident elite, semantic-hash, descriptor, population, and parent structures
have explicit capacities in search-spec schema 2. Elite retention is a
streaming deterministic top-k operation rather than an unbounded list of every
accepted AST. Final descriptor-aware shortlist selection compares bounded
32-record candidate and selected-reference samples and checks the run deadline
between selections. A wide shortlist therefore remains bounded instead of
degrading into a quadratic all-pairs finalization tail.

`structural_score` schema 2 is explicitly non-financial:

```text
0.20 * simplicity
+ 0.15 * archive_descriptor_novelty
+ 0.10 * root_family_diversity
+ 0.10 * lineage_diversity
+ 0.35 * structural_descriptor_novelty
+ 0.10 * transform_diversity
```

The versioned weights, expression limits, and resource capacities are embedded
in the manifest. Search reserves a finalization interval, checks the wall clock
between bounded generation/scoring/merge stages, stops at the hard duration
deadline, and serializes completed work. A versioned checkpoint written after
each completed generation contains the exact population, bounded archives,
counters, next generation, deterministic seed state, configuration checksum,
and operator-catalog checksum. Resume rejects a changed configuration, archive,
or operator catalog. With a candidate ceiling reached
before the deadline, candidate bytes are deterministic for identical input,
seed, configuration, and thread count. Deadline-bound run counts can reflect
host scheduling; the manifest records the actual termination reason and
elapsed time. The current indexed computation and stable tie breaking also
produce matching candidate checksums across thread counts in the integration
benchmark, but callers should rely only on the fixed-thread contract.

## Artifact schemas

Each newly written candidate JSONL line has `schema: 4`; readers also accept
legacy schema 1–3 records. Schema 4 adds the immutable `run_id`; schema 3 added
the score schema, every component, normalized weights, and complete structural
descriptor:

```json
{
  "schema": 4,
  "run_id": "<run SHA-256>",
  "expression": "rank(close)",
  "fingerprint": "<exact SHA-256>",
  "semantic_fingerprint": "<semantic SHA-256>",
  "structural_score": 0.0,
  "score_schema": 2,
  "score_components": {
    "simplicity": 0.0, "archive_novelty": 0.0,
    "family_diversity": 0.0, "lineage_diversity": 0.0,
    "structural_novelty": 0.0, "transform_diversity": 0.0
  },
  "normalized_weights": {
    "simplicity": 0.20, "archive_novelty": 0.15,
    "family_diversity": 0.10, "lineage_diversity": 0.10,
    "structural_novelty": 0.35, "transform_diversity": 0.10
  },
  "structural_descriptor": {
    "operator_shingles": [], "operator_histogram": {}, "ordered_paths": [],
    "fields": [], "field_families": [], "window_buckets": {}, "groups": [],
    "nonlinear": false, "conditional": false, "lineage_parents": []
  },
  "root_family": "rank",
  "nodes": 2,
  "depth": 2,
  "provenance": {
    "kind": "mutation",
    "operation": "mutate_field",
    "parent_fingerprints": ["<semantic SHA-256>"],
    "requested_operation": "mutate_field",
    "affected_path": {
      "segments": [{"slot": "binary_left"}]
    },
    "old_subtree_fingerprint": "<exact subtree SHA-256>",
    "new_subtree_fingerprint": "<exact subtree SHA-256>",
    "retry_count": 0
  }
}
```

The adjacent compatibility file `OUTPUT.manifest.json` has `schema: 6`,
`candidate_schema: 4`, tool
version, complete search specification and scoring weights, configuration
checksum, archive and candidate counters, `rejected_trivial_candidates`, a
machine-readable `rejection_reasons` map, archive accepted/duplicate/rejected
counters and reasons, invalid-transform count, accepted/retained transform-kind
counts, completed generations, all resident capacities and peaks, elapsed
milliseconds, termination reason, and SHA-256 of the exact candidate JSONL
content.

Candidate and manifest bytes are first written and synced as an immutable pair
under `.OUTPUT.runs/`. Only then is `OUTPUT.current.json` atomically replaced.
Readers following that pointer cannot pair a new candidate file with a stale
manifest. The adjacent candidate and manifest paths remain compatibility
aliases and are refreshed after publication; damaging either alias does not
damage the last complete pointed-to run. Checkpoints use schema 1 and the same
temporary-file, `sync_all`, atomic-rename protocol.

## Optional numeric evidence

The `numeric-evidence` feature adds a pure-Rust evaluator and the `evaluate`
command; it is absent from the default binary. Input is immutable schema-1
JSON with increasing `timestamps`, an `assets` vector, one row-major
`NumericColumn` per field, a separate row-major `realized_returns` column,
optional static group labels, a source label, and preprocessing notes. Each
column contains exactly `timestamps.len() * assets.len()` numbers or nulls.

The evaluator supports every operator in the embedded catalog. Time-series
windows read the current and previous rows only. Signal row `t` is paired with
the additive realized return from rows `t+1..=t+horizon`; a target is discarded
if it would cross its train, validation, or holdout boundary. The output records
the dataset ID and checksum, source, canonical expression/fingerprint, split
ranges, horizon, costs, alignment rule, user preprocessing, evaluator
assumptions, observation counts, measured Pearson correlation, measured
per-row gross/net return, and turnover.

`evaluate-batch` accepts the candidate JSONL produced by `search`, loads and
validates the immutable dataset once, evaluates bounded chunks with
`--threads N`, preserves deterministic input order, and atomically publishes
one evidence JSONL. At most one chunk of expression artifacts is materialized
at a time, so increasing workers has an explicit memory tradeoff.
Individual expression failures are recorded without discarding successful
rows. `--include-daily` is optional because daily evidence can materially
increase artifact size.

`evaluation_start` changes only the first scored row: earlier rows remain
available to rolling operators and provide causal warm-up without entering
reported PnL. `overall` is evaluated continuously without resetting positions
at diagnostic split boundaries. `daily` is emitted only with
`--include-daily`; every row records signal and target timestamps, gross and
cost-adjusted returns, and turnover. `--signal-decay N` applies one causal
linear decay at the final signal root, with the current row receiving weight
`N` and the oldest row weight `1`; zero disables the setting.
`--truncation X` caps absolute normalized position weights and deterministically
redistributes remaining gross exposure; zero disables it.

`brain-proxy-v1` is a transparent local convention, not a platform result. It
uses population-volatility annualized gross Sharpe, twice the annualized
unit-gross return, mean unit-gross turnover, twice the maximum drawdown of the
additive gross-PnL curve, mean gross return divided by turnover for margin, and
`Sharpe * sqrt(abs(Returns) / max(Turnover, 0.125))` for Fitness. Gross proxy
metrics never deduct the user-selected transaction cost; measured net metrics
remain separate. Universe and source-data differences can dominate the error,
so proxy values still require out-of-sample calibration.

### Optional Sharadar Parquet preparation

The `parquet-input` feature adds a local, read-only adapter for partitioned
Sharadar `daily` and `stocks` tables. It selects a fresh positive-finite
market-cap universe independently on every date, emits adjusted open/high/low/
close, raw volume, and `closeadj` returns, keeps realized returns outside
current membership for future targets, and atomically writes the same schema-1
JSON consumed by `evaluate`:

```bash
cargo run --release --features parquet-input -- prepare-sharadar \
  --parquet-root /path/to/sharadar/parquet \
  --start-date 2018-01-01 \
  --end-date 2023-12-31 \
  --evaluation-start-date 2019-01-01 \
  --train-end-date 2021-01-01 \
  --validation-end-date 2023-01-01 \
  --universe-size 3000 \
  --output sharadar-top3000.json
```

The command prints the exact row indices to pass to `evaluate`. Loaded dates
before `evaluation_start` are warm-up only. The adapter never contacts a data
vendor or platform, never writes into the source lake, and refuses to
overwrite an existing output file.

## Target-dialect compile and lint

`compile` lowers an expression against an immutable, user-supplied dialect
artifact before an external consumer sees it. A dialect contains explicit
field/operator allowlists and conservative unary-to-scalar lowerings. For
example, a target that lacks `negate` can declare a structural lowering to
`multiply(-1, x)`:

```bash
alphawinnow compile 'rank(negate(returns))' --dialect target-v1.json
```

The command emits source and compiled canonical expressions, symbols used, and
every lowering applied. It fails closed on an unavailable field/operator and
contains no credentials, network, simulation, or submission path.

Numeric evidence is a separate schema-1 artifact. It never enters
`structural_score` (`structural_score_used` is explicitly `false`), is never
represented as measured WorldQuant Fitness/Sharpe, and makes no pass-probability
or future-return claim. Optional `brain_proxy` values remain explicitly local
estimates with `remote_platform_result: false`. The evaluator uses deterministic ordinal tie-breaking for ranks,
listwise missing-value exclusion, additive horizon returns, cross-sectional
demeaning plus unit-gross normalization, and turnover costs; all are repeated
in each artifact for auditability.

## Measured-feedback prioritization

`prioritize` consumes candidate-schema JSONL and one immutable schema-1
feedback dataset. A dataset declares its ID, one context checksum, a descriptive
outcome label, positive scales for an exact structural-feature set, and bounded
outcomes in `[-1, 1]` with confidence in `(0, 1]`. The included
[`public-feedback-v1.json`](examples/public-feedback-v1.json) is synthetic and
contains no market or platform data.

Supported features are field-name-independent: nodes, depth, operator count,
field/family/group counts, short/medium/long window counts, nonlinear and
conditional flags, and lineage-parent count. Deterministic nearest-neighbor
guidance reports dataset/context/checksum, neighbor count, nearest distance,
weighted outcome, dispersion, and a conservative outcome. Dataset checksums
are invariant to record order.

The output nests each original candidate byte-for-byte at the semantic-field
level and adds `measured_guidance`; `candidate.structural_score` is not changed
and every estimate records `structural_score_used: false`. Guided candidates
sort by conservative measured outcome, then measured mean, original structural
score, and semantic fingerprint. Records without enough comparable neighbors
remain explicit and sort after guided records. The outcome is user-defined and
must not be called expected return, Sharpe, Fitness, correlation, or a pass
probability unless the supplied immutable dataset itself establishes that
meaning.

`audit-feedback` applies the identical neighbor rule while excluding each
target record from its own evidence. It reports coverage, MAE, RMSE, sign
accuracy, balanced sign accuracy, and rank AUC (only when both non-zero outcome
signs exist). This is a deterministic leakage check for comparing feature sets
or neighbor settings, not evidence that the supplied outcomes generalize to a
new dataset or context.

## Reproducible benchmark

Command:

```bash
cargo run --release -q -p alphawinnow -- \
  benchmark --candidates 20000 --threads 1,4 --seed 20260828
```

The phase rows were measured 2026-08-28 and the final row on 2026-09-04, both
on the same Apple M1 Pro, Darwin arm64, Rust/Cargo 1.92.0. The review baseline
preceded the Phase 1 semantic/triviality pass:

| Version | Threads | Generated | Semantic families | Trivial rejects | Duplicate rejects | Retained | Wall time | Candidate SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Review baseline | 1 | 20,000 | 15,787 | — | 4,213 | 100 | ~359 ms | `b0134b7ba5c3e29b96b660a616e8e305041cd55c4884c5cd3de9db2fa39c3b95` |
| Review baseline | 4 | 20,000 | 15,787 | — | 4,213 | 100 | ~271 ms | `b0134b7ba5c3e29b96b660a616e8e305041cd55c4884c5cd3de9db2fa39c3b95` |
| Phase 1 | 1 | 20,000 | 15,633 | 52 | 4,315 | 100 | 519 ms | `2b9fe09fda351a8495dd17fcbb38886fbf3867f9d331927390455c595976abc4` |
| Phase 1 | 4 | 20,000 | 15,633 | 52 | 4,315 | 100 | 442 ms | `2b9fe09fda351a8495dd17fcbb38886fbf3867f9d331927390455c595976abc4` |
| Phase 2 | 1 | 20,000 | 18,056 | 49 | 1,895 | 100 | 813 ms | `8bd2a3e1764c240dc1c64db537942de1b0ce79045364e95910c8a0446fed87df` |
| Phase 2 | 4 | 20,000 | 18,056 | 49 | 1,895 | 100 | 712 ms | `8bd2a3e1764c240dc1c64db537942de1b0ce79045364e95910c8a0446fed87df` |
| Phase 2 corrected | 1 | 20,000 | 17,997 | 64 | 1,939 | 100 | 906 ms | `77cd80a7104b8ec849422464695759fd5f8176bc81f149ff658a540344092666` |
| Phase 2 corrected | 4 | 20,000 | 17,997 | 64 | 1,939 | 100 | 912 ms | `77cd80a7104b8ec849422464695759fd5f8176bc81f149ff658a540344092666` |
| Phase 3 | 1 | 20,000 | 7,008 | 132 | 12,860 | 100 | 697 ms | `6a22cccc07c2e6583c67a250986303559162776389abe00b437b2c711b49e2fe` |
| Phase 3 | 4 | 20,000 | 7,008 | 132 | 12,860 | 100 | 361 ms | `6a22cccc07c2e6583c67a250986303559162776389abe00b437b2c711b49e2fe` |
| Phase 4 | 1 | 20,000 | 14,815 | 92 | 5,093 | 100 | 3,289 ms | `180d296a48f9668d96b91db0340e746dd4973f3d1799665e99e5bc8e53a53cab` |
| Phase 4 | 4 | 20,000 | 14,815 | 92 | 5,093 | 100 | 1,268 ms | `180d296a48f9668d96b91db0340e746dd4973f3d1799665e99e5bc8e53a53cab` |
| Phase 5 | 1 | 20,000 | 14,815 | 92 | 5,093 | 100 | 3,479 ms | `5a322ca6d4a5ecfc42ae28ae510ccf46b3ea1f63665e23cb842dc1733b17d97e` |
| Phase 5 | 4 | 20,000 | 14,815 | 92 | 5,093 | 100 | 1,713 ms | `5a322ca6d4a5ecfc42ae28ae510ccf46b3ea1f63665e23cb842dc1733b17d97e` |
| Phase 6 | 1 | 20,000 | 15,135 | 72 | 4,793 | 100 | 3,373 ms | `8b4a83c552ab96f1382b9a9c4b7f5455bcd0d60225750c6f36f7a34395476dfb` |
| Phase 6 | 4 | 20,000 | 15,135 | 72 | 4,793 | 100 | 1,576 ms | `8b4a83c552ab96f1382b9a9c4b7f5455bcd0d60225750c6f36f7a34395476dfb` |
| Phase 9 | 1 | 20,000 | 16,587 | 49 | 3,364 | 100 | 5,042 ms | `af46848c82daf83d57a1a111301e63e4ca99cd7b3f0f0513a1a5aab972e5d8f8` |
| Phase 9 | 4 | 20,000 | 16,587 | 49 | 3,364 | 100 | 2,433 ms | `af46848c82daf83d57a1a111301e63e4ca99cd7b3f0f0513a1a5aab972e5d8f8` |
| Re-measured 2026-09-04 | 1 | 20,000 | 15,850 | 38 | 4,112 | 100 | 4,725 ms | `66df757199e55116f283747f74bdb917c59c1a50ea4f813a66182feb92a40e12` |
| Re-measured 2026-09-04 | 4 | 20,000 | 15,850 | 38 | 4,112 | 100 | 2,046 ms | `66df757199e55116f283747f74bdb917c59c1a50ea4f813a66182feb92a40e12` |

Phase 1 rejected 23 proven-zero and 29 proven-constant expressions. The
semantic-family count fell by 154: 52 explicit trivial rejections plus 102
additional forms merged by the strengthened final-root equivalence rules.
The composite-add and tolerant-archive corrections did not change these fixed
workload counts or checksum.

Phase 2 accepted 64 initial, 4,150 mutation, 12,666 crossover, and 1,176
truthfully labelled fallback-generation records. The retained shortlist still
contains 96 crossover and four initial records; changing selection pressure
belongs to the generation-based engine and diversity phases, not to the
lineage implementation.

After correcting context-aware `multiply` slot compatibility, the same
workload accepted 64 initial, 4,171 mutation, 12,585 crossover, and 1,177
fallback-generation records. It retained the same 96 crossover and four
initial records. One- and four-thread candidate checksums remain identical;
the changed checksum and rejection counts are the expected result of the
expanded typed crossover search space.

Phase 3 completed 157 deterministic batches with a peak population of 512 and
zero invalid emitted expressions. Accepted candidates comprised 45 initial,
915 mutation, 3,927 crossover, and 2,121 truthful fallback-generation records.
The higher duplicate count reflects actual evolutionary convergence under the
still-coarse novelty measure; Phase 4 addresses it with layered structural
descriptors rather than hiding it with random generation.

Phase 4 recovered 14,815 semantic families using deterministic staged
descriptor comparisons. The retained 100 are balanced 25 each across initial,
mutation, crossover, and fallback-generation provenance. Descriptor scoring is
more expensive than root-only balancing, but the four-thread workload remained
byte-identical and reduced measured wall time from 3,289 ms to 1,268 ms.

Phase 5 retains the same accepted/rejected counts while bounding the persistent
population at 512, elite archive at 4,096, semantic index at 100,000, and
imported descriptor archive at 4,096 entries. Its changed checksum is expected:
the bounded archive uses deterministic family-balanced top-k retention and
schema 4 includes the run identifier. A pause/resume regression reaches the
same fixed-budget candidate checksum as an uninterrupted run.

Phase 6 replaces the 13-way generator match with weighted catalog traversal
and makes all 15 parser-supported operators generation-reachable. The workload
found 15,135 semantic families with zero invalid emitted expressions. The
candidate checksum changed because catalog weights/domains now define the
search distribution; one- and four-thread bytes still match.

Phase 9 preserves geometric node-count buckets independently within transform
and root families in the population, elite archive, and shortlist. The fixed
workload retains 16,587 semantic families and a mix of 31 mutation, 28
crossover, 24 fallback-generation, and 17 initial records. A separate bounded
100,000-candidate run with limits 20/192/96 retained structures from 1 through
159 nodes, demonstrating that the wider CLI envelope is operational. The
additional diversity bookkeeping increases wall time; one- and four-thread
candidate checksums remain identical.

The final row is a re-run of the identical fixed workload against the current
tree, not a new optimization. It retains 15,850 semantic families and a mix of
32 mutation, 29 crossover, 22 fallback-generation, and 17 initial records over
157 generations, rejecting 38 trivial expressions (28 provably zero, 10 provably
constant) and 4,112 duplicates. Its counts and candidate checksum differ from
the Phase 9 row, so that row no longer describes this tree; the score-ordered
shortlist fallback for deadline-terminated runs is not the cause, because that
path stays inactive when a run stops on `max_candidates` and disabling it
reproduces the same checksum. Wall times are the median of three consecutive
runs; one- and four-thread bytes still match.

These are observations from one workload and environment, not a hard-coded
speedup claim. The command reports full configuration, counts, checksums,
thread counts, and wall time so results can be compared honestly elsewhere.

## FAQ

### What is formulaic alpha mining?

Formulaic alpha mining searches a domain-specific language of mathematical
expressions for candidate signals or quantitative factors. AlphaWinnow builds
only well-typed expressions, removes known duplicates, and preserves a diverse
shortlist. A separate measured evaluation is still required to determine
whether any candidate contains useful market information.

### Is AlphaWinnow a symbolic regression or genetic programming library?

AlphaWinnow uses strongly typed genetic-programming techniques—generation,
mutation, crossover, selection, and novelty—but specializes them for
quantitative expressions. The default structural search does not fit a numeric
regression target. You can provide measured feedback or evaluate candidates on
your own panel as explicit later stages.

### Is AlphaWinnow a backtesting engine?

It includes an optional causal factor evaluator with transaction costs,
turnover, split diagnostics, and daily long-short evidence. It is not an order
execution simulator: there is no broker, exchange, fill, latency, or live
trading model.

### Does it support WorldQuant BRAIN or FASTEXPR formulas?

The bundled DSL contains familiar WorldQuant-style time-series,
cross-sectional, arithmetic, group, and conditional operators. AlphaWinnow has
no WorldQuant BRAIN API, authentication, simulation, or submission integration.
Use a local catalog and target-dialect file to describe the exact syntax an
external system accepts; always verify compatibility independently.

AlphaWinnow is an independent open-source project and is not affiliated with
or endorsed by WorldQuant.

### Can I use Sharadar market data?

Yes. The optional `parquet-input` feature reads local Sharadar-compatible
Parquet files and prepares a causal dynamic-universe panel. No market data is
bundled, downloaded, or redistributed; you must supply data you are licensed
to use.

### Does AlphaWinnow require Python?

No. The CLI, parser, search engine, semantic deduplication, numeric evaluator,
and Parquet adapter are implemented in Rust.

### Will it discover profitable trading strategies?

There is no such guarantee. Large factor searches are vulnerable to
overfitting, selection bias, regime change, and transaction costs. Treat every
output as a hypothesis requiring independent point-in-time data, out-of-sample
testing, and risk review.

## Security and scope

AlphaWinnow is local-only by design:

- no network client or telemetry;
- no credentials, cookies, or API keys;
- no broker, exchange, or remote research-platform integration;
- no database server or background service;
- no live simulation or alpha submission path.

Commands read explicit immutable inputs and write versioned local artifacts.
The optional Sharadar adapter is also read-only with respect to the source
dataset.

## Development

Two documentation surfaces are published automatically and need no manual
step. [docs.rs/alphawinnow](https://docs.rs/alphawinnow) is built by crates.io
for every released version. [eslazarev.github.io/alphawinnow](https://eslazarev.github.io/alphawinnow/)
is the same `cargo doc` reference built from `main`, so it covers unreleased
work; both are generated from the `///` comments in the source, never
hand-edited.

Run the complete quality gates before opening a pull request:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo build --release --workspace --all-features
```

AlphaWinnow is a single crate with a library target and a binary target.
Reusable search and evaluation logic belongs in the library; `src/main.rs`
should remain a thin argument-parsing and serialization layer that only calls
the public API. See the public [`ROADMAP.md`](ROADMAP.md) for planned work and
[`docs/radiate-decision.md`](docs/radiate-decision.md) for the reproducible
multi-objective-search dependency evaluation.

### Continuous integration

Every push and pull request against `main` runs the same gates on GitHub
Actions, plus a few that are impractical to run locally on every change:

| Workflow | What it enforces |
| --- | --- |
| `ci.yml` | formatting, Clippy with warnings denied, tests on Linux/macOS/Windows, every feature combination, the declared MSRV, `cargo doc` with warnings denied, a `cargo publish` dry run, and the standalone benchmark workspace |
| `ci.yml` &rarr; `library-graph` | that no CLI-only dependency is reachable without the `cli` feature, so `default-features = false` stays genuinely lean |
| `audit.yml` | `cargo audit` and `cargo deny` over advisories, licenses, sources, and the banned-dependency policy in [`deny.toml`](deny.toml) |
| `codeql.yml`, `secrets.yml`, `scorecard.yml` | static analysis, verified-secret scanning, and OpenSSF Scorecard |
| `release.yml` | raises the version from the Conventional Commits since the last tag, then tags, verifies, and publishes it; see [`docs/releasing.md`](docs/releasing.md) |
| `docs.yml` | rustdoc for `main`, published to [GitHub Pages](https://eslazarev.github.io/alphawinnow/) |

`deny.toml` also encodes the local-only product boundary: no HTTP or TLS crate
may enter the dependency graph, so the "performs no network requests" claim is
checked mechanically rather than by review alone.

## License

AlphaWinnow is available under the [MIT License](LICENSE).

## Limitations and next milestone

- The public catalog is deliberately small and synthetic; there is no bundled
  private field catalog or automatic catalog download.
- Admission and deterministic archive merge remain serial by design; indexed
  generation, validation, analysis, and scoring run in the bounded Rayon pool.
- Structural novelty is an auditable AST descriptor distance, not numeric
  correlation or measured market evidence.
- The generic numeric backend reads the documented JSON columnar schema. The
  optional Sharadar adapter reads Parquet but intentionally materializes that
  portable JSON boundary; it is not yet a general streaming Parquet evaluator.
- Static group membership is supported; point-in-time changing classifications
  require a future schema version.
- Persistent search memory is bounded by explicit resource capacities; each
  parallel generation additionally holds one fixed 128-candidate work batch.
- The manifest contains measured elapsed time and therefore is auditable rather
  than byte-identical between repeated runs; candidate JSONL is the
  reproducible content artifact.

The review-driven Phases 1–9 and the Radiate decision gate are complete. The
generic measured-feedback prioritizer remains separate from structural score.
The next smallest milestone is a streaming Arrow/Parquet adapter for the
numeric feature; it must not add remote platform integration.
