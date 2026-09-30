# Changelog

## 0.2.0 — prepared, not yet published

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
