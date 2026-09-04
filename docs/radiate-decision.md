# Why AlphaWinnow does not currently depend on Radiate

Decision recorded on 2026-08-28: keep Radiate outside the AlphaWinnow runtime
dependency graph. Retain the standalone benchmark so the decision can be
revisited when AlphaWinnow has equivalent measured multi-objective workloads.

## Reviewed release

- project: <https://github.com/pkalivas/radiate>
- crate/tag: `radiate` 1.3.0 / `v1.3.0`
- tag commit: `09c26b5c150d498ae0dfa5809f8ac0d11e70a135`
- repository HEAD observed during the review:
  `f5172b1e7b3b82da5a3aa0c45f629df7889cf488`
- license: MIT, copyright Peter Kalivas, 2019–2026

The adapter is deliberately a standalone Cargo workspace. Radiate is a research
comparison dependency, not part of the AlphaWinnow library, CLI, lockfile, or
default build graph.

Run it with:

```console
cargo run --release --manifest-path tools/radiate-selection-benchmark/Cargo.toml
```

The fixed workload compares the current schedule-derived parent indexing with
Radiate tournament selection, and a deterministic scalar top-k reference with
Radiate NSGA-II and NSGA-III. It reports wall time, deterministic checksums,
selected-index coverage, and mean selected index. The latter two are behavioral
descriptors, not claims that unlike algorithms have equal selection semantics.

## Measured result

Two release-mode runs on an Apple M1 Pro (`arm64`, Darwin, Rust/Cargo 1.92.0)
produced identical checksums for all five result vectors. Timings varied as
expected and should not be generalized to other hardware:

| Operation | Run 1 | Run 2 | Stable checksum |
| --- | ---: | ---: | --- |
| AlphaWinnow schedule-derived parent index | 721 us | 773 us | `36c34c73cc6eb9a5` |
| Radiate tournament, k=3 | 16,247 us | 11,983 us | `e5522562e45039cf` |
| Deterministic scalar top-k reference | 253 us | 226 us | `c16ebf2154ef3aca` |
| Radiate NSGA-II | 62,345 us | 58,945 us | `fa343edc39734b9b` |
| Radiate NSGA-III, 12 partitions | 64,748 us | 64,720 us | `0380121a490e5b8b` |

The tournament comparison performs one million parent draws from 4,096 sorted
members. Radiate was about 15–23 times slower and deliberately selected better
ranked indices more often; AlphaWinnow's schedule covered all 4,096 indices.
That behavioral difference means timing alone is not a quality comparison.

The survivor comparison selects 1,024 of 4,096 members with three deterministic
synthetic objectives. NSGA-II/III were about 233–286 times slower than scalar
top-k, but supply Pareto behavior that scalar top-k does not. AlphaWinnow does
not yet have measured evidence showing that replacing its auditable scalar
structural score plus family/descriptor constraints with Pareto selection
improves candidate quality.

## Decision by reusable concern

- **Tournament selection:** reject for now. Radiate uses a thread-local random
  provider and returns stochastic selections from an already ordered
  population. AlphaWinnow derives each parent choice and offspring identity
  from stable run/generation/slot material and proves equality across thread
  counts. Wrapping every draw in a derived scoped seed would discard most of
  the reusable implementation and was slower in the fixed adapter.
- **NSGA-II/NSGA-III:** defer. They are plausible only after measured numeric
  objectives have an explicitly versioned role in search. Adopting them now
  would change semantics rather than optimize an equivalent workload.
- **Survivor/offspring allocation:** reject for now. The current bounded
  population intentionally preserves competitive transform families and
  below-root descriptor diversity; the generic selectors do not preserve
  those invariants without a custom adapter.
- **Adaptive mutation-rate scheduling:** defer. A schedule would have to be a
  versioned part of candidate identity and checkpoint state. No equivalent
  measured workload currently demonstrates a benefit.
- **Recombination clone efficiency:** no valid drop-in benchmark exists.
  Radiate chromosomes/phenotypes and AlphaWinnow's typed AST have different
  ownership and validity requirements. Comparing unrelated clones would be a
  misleading microbenchmark, while replacing the typed AST is explicitly out
  of scope.

The gate can be reopened when a fixed, causally aligned numeric workload makes
Pareto quality measurable. Any future adapter must preserve typed validation,
semantic fingerprints, truthful lineage, bounded checkpoints, transactional
artifacts, and cross-thread candidate checksums.
