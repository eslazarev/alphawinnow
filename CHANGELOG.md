# Changelog

## 0.3.0 — prepared, not yet published

### Fixed

- Update optional Arrow/Parquet dependencies to 60.0.0 and lock zstd 0.14.3,
  zstd-safe 8.2.0 and zstd-sys 2.1.2. These include the fixes for
  RUSTSEC-2026-0391, -0398, -0403 and -0410 and the related zstd-safe warnings.
  A synthetic compressed/uncompressed Parquet test checks panel-value parity.
- Run `cargo audit` directly in the dependency-audit workflow, retaining only
  `contents: read`. Scheduled audits no longer attempt to create issues, and
  findings fail the job through the CLI exit code rather than a GitHub write API.

- Bootstrap seeds are parsed/fingerprinted once at initialization; restore
  validates seeds directly without reconstructing a second search session.
- Feedback file read/parse errors consistently include their input path.
- The JSONL bridge reports initialization counters from the actual session.

- JSON outcomes preserve exact finite `f64` bits, including signed zero and
  subnormals, through checkpoint restoration (`serde_json/float_roundtrip`).
- Snapshot checkpoints reject missing retained parents; valid under-capacity
  populations and short final batches remain supported.
- Catalog-aware mutation and crossover now carry the supplied catalog through
  tree-slot typing and full operator-context validation. Custom operators no
  longer cause valid edits to be discarded and replaced by fresh generation.
  Added explicit-catalog tree enumeration, kind lookup, and replacement APIs;
  existing built-in tree helpers remain wrappers with the same signatures.
- Window mutation recognizes positional window arguments beyond binary-right
  slots, including the third argument of `ts_corr`. These correctness fixes
  can change generated candidates and RNG choices for affected expressions;
  pin the previous binary when replaying a previously frozen experiment.

### Added

- Opt-in schema-3 measured checkpoints reference a separately persisted,
  SHA-256-bound exclusion archive. `ExcludedHistory` validates once and shares
  immutable storage; standalone schema-1/2 serialization remains available.
- Screening report schema 2 binds parsed candidates and exact CLI input bytes
  to feedback/configuration. `screen-feedback --format jsonl` writes a metadata
  header and one decision per line; JSON remains the default.

- A detailed WorldQuant BRAIN workflow guide with synthetic catalog, dialect,
  context, measured-session and feedback files, plus an executable offline
  JSONL/checkpoint demonstration checked in CI. No platform connector is added.
- Measured-search warm starts from caller-audited parents and historical
  semantic exclusions. Screened-pool feedback updates parents using only real
  measurements; unselected proposals receive no invented negative label.
  Import, measurement and screening counters remain separate. Opt-in schema-2
  checkpoints retain exact resume state; ordinary schema-1 sessions are unchanged.
  The JSONL example exposes both operations. This is integration functionality,
  not a claim of improved financial performance.

- Separate held-out feedback evaluation via `audit_feedback_holdout` and
  `audit-feedback --evaluation-feedback`: immutable training evidence,
  per-record predictions, both dataset checksums, and fail-closed checks for
  context/label/scale mismatch and known identity overlap. Callers remain
  responsible for temporal and economic-family splits.
- Explicit abstention via `screen_candidates` and `screen-feedback`. A required
  conservative-outcome threshold produces one atomic audit artifact retaining
  every eligible and rejected candidate, including missing guidance. Existing
  search, prioritization, defaults, and checkpoints are unchanged. These are
  evaluation and screening tools, not evidence of improved market prediction.

- Opt-in `field_diversity::select_field_diverse` limits identical input-field
  sets in an ordered shortlist without changing ranking or claiming return
  independence; deferred records remain visible.
- Opt-in `identity_reduction::reduce_unit_scaling` removes exact unit
  multiplication/division without changing the default fingerprint contract.

### Changed / migration required

- `max_evaluations` retains its historical name for compatibility, but docs now
  explicitly define it as a proposal budget: 32 proposals selecting 2 of each
  batch of 8 give 8 measurements and 24 screened proposals, not 32 measurements.
- Screening consumers must accept report schema 2 and its two new checksum
  fields. Held-out feedback report schema remains 1.

- `FeedbackError` adds `HoldoutContextMismatch`, `HoldoutOverlap`, and
  `InvalidAdmissionThreshold`. Downstream exhaustive matches must be updated;
  this is why the combined release is 0.3.0 rather than the previously prepared,
  unpublished 0.2.1 checkpoint-only patch.
- Warm-started or screened measured sessions write checkpoint schema 2; old
  readers cannot restore it. Ordinary fully measured sessions retain schema 1.
- Custom-catalog and window mutation fixes can change output/RNG trajectories.
  Keep the old binary for exact historical replay. The earlier 0.2.1 structural
  parity report predates these changes and does not validate this release.

See [0.3.0 release notes](docs/releases/0.3.0.md) for migration and verification.

## 0.2.0

### Added

- Public `measured_search` ask/tell API: typed proposals, semantic deduplication,
  caller-owned evaluation identity, complete finite-outcome snapshots, bounded
  search state, and checkpoint restoration. Includes synthetic Rust and local
  JSONL stdio examples, without requiring CLI dependencies.
- Explicit grammar exploration, applicable mutation, and score/uniform parent
  controls, including latest-batch retention policies.
- Opt-in immutable feedback guidance during search, archive-seeded/frozen
  parents, reserved feedback elite slots, and bounded operator/action weighting
  with provenance in the manifest.
- Configurable initial catalog coverage and public typed-grammar sampling APIs.
- Causal numeric implementations for the expanded embedded operator catalog,
  with fail-closed validation of unsupported generation-enabled operators.
- `inspect --catalog`, configurable population/parent capacities, and repeated
  multi-seed CLI throughput benchmarks.

### Changed / migration required

- Public structs, including `SearchRunOptions`, `RunManifest`, and
  `RunCheckpoint`, have new fields. Update downstream struct literals.
- Structural-search manifest schema is now 8 (was 6); checkpoint schema is
  now 2 (was 1). Candidate JSONL remains schema 4. The separate measured-search
  checkpoint starts at schema 1; it is not interchangeable with CLI checkpoints.
- Nested cross-sectional ranks share semantic identity; binary root products
  participate in root-scale normalization. Recompute affected semantic indexes.
- Expanded generation-enabled operators change the built-in catalog checksum
  and seeded search output. Cross-version candidate/checkpoint parity is not
  promised. Preserve the old executable and inputs for old-run reproduction.

### Release engineering

- Automatic releases preserve an explicitly prepared version instead of
  incrementing it twice; regression tests cover both automatic and explicit bumps.
- Version-specific release/migration notes are attached to the GitHub release.
- Local AlphaGen research workspaces and experiment directories are ignored,
  not included in the public crate. No market results or superiority claims ship.

See [0.2.0 release notes](docs/releases/0.2.0.md) for migration and verification.

## 0.1.2

Previous tagged baseline. See the repository's `v0.1.2` tag for its exact source
and package manifests; this changelog does not reconstruct earlier release notes.
