# Working on AlphaWinnow

Welcome! AlphaWinnow is a local-first Rust toolkit for exploring, validating,
deduplicating, and evaluating quantitative expressions. The project values
correctness, reproducibility, and useful research artifacts over impressive but
unsupported performance claims.

This guide captures the decisions that help contributions fit naturally into
the existing codebase.

## Product direction

AlphaWinnow works with explicit local inputs and produces portable, versioned
artifacts. The core product does not need an account, hosted service, or network
connection.

New features are especially welcome when they improve one of these workflows:

- typed quantitative-expression parsing and transformation;
- deterministic, diversity-aware candidate search;
- conservative semantic deduplication;
- causal evaluation on user-supplied data;
- reproducible manifests, checkpoints, and benchmarks;
- clear extension points for catalogs, dialects, and local data formats.

Platform integrations, credentials, browser automation, live simulation,
submission, and account-specific storage belong outside this repository. Keep
private expressions, identifiers, PnL, catalogs, credentials, and licensed data
in their original private environments; use small synthetic fixtures here.

## Expression semantics

Semantic deduplication intentionally favors conservative, explainable rules:

- A positive scalar applied only at the final expression root is
  scale-equivalent.
- A negative final scalar changes direction and remains a distinct family.
- Scaling inside a nonlinear operator, condition, clipping or winsorization
  step, or one composite sleeve remains significant.
- Dividing a complete equal-weight composite by its number of legs changes
  readability, not sleeve influence. Standardize individual sleeves when equal
  influence is required.

Structural search produces research candidates, not validated trading signals.
Sharpe, Fitness, PnL, correlation, and similar labels should appear only when
they come from an explicit measured dataset and documented methodology.

## Code organization

AlphaWinnow ships as one crate, `alphawinnow`, with a library target and a
binary target. Reusable parsing, search, evaluation, and artifact logic belongs
in the library (`src/lib.rs` and its modules). Keep `src/main.rs` focused on
arguments, orchestration, diagnostics, and serialization; it may only call the
public library API, exactly as an external consumer would.

The binary and its argument parser sit behind the default `cli` feature, so a
library consumer using `default-features = false` compiles neither. Never move
a CLI-only dependency into the unconditional dependency list; the
`library-graph` CI job fails if one becomes reachable without `cli`.

Prefer changes that are:

- deterministic for identical inputs, seed, thread count, and configuration;
- bounded in memory, worker count, expression complexity, and wall time;
- explicit about schemas, assumptions, and failure modes;
- compatible with atomic output and graceful checkpointing;
- testable with license-safe local fixtures.

JSONL is the preferred format for large streams. Public artifacts should carry
an explicit schema version and enough provenance to reproduce their creation.

## Testing changes

Canonicalization rules should include golden cases. AST generation, mutation,
and crossover benefit from property tests that exercise typing and resource
limits. Performance changes should include a reproducible end-to-end benchmark
and preserve the existing identity and determinism contracts.

Before considering a change ready, run:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

When a proposed optimization conflicts with correctness or reproducibility,
keep the simpler behavior until the faster version can demonstrate parity.
