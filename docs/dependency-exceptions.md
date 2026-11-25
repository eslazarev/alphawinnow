# Dependency audit exceptions

## RUSTSEC-2026-0543: Chrono maintenance status

- Approved: November 25, 2026.
- Review deadline: December 31, 2026. Any extension requires a new owner decision.
- Scope: only `RUSTSEC-2026-0543` in `cargo deny`.
- Current dependency: `chrono 0.4.45`.
- Reason: Chrono is unmaintained, and the advisory lists no patched version.

The optional `parquet-input` feature uses Chrono for calendar-date conversions.
Arrow/Parquet 60.2 also depends on Chrono. Replacing only our direct usage would
not remove Chrono from the dependency graph.

This exception accepts a maintenance risk. It does not establish that Chrono is
free of security defects. Other advisories, yanked releases, license checks,
source checks, and dependency bans remain subject to the existing policy.
The separate `cargo audit` workflow is unchanged and can still report this notice.

The review deadline is manual. The installed `cargo-deny 0.20.2` does not support
per-exception expiry, so this configuration does not expire automatically.
The Parquet advisory `RUSTSEC-2026-0595` is not ignored; the minimum dependency
version was raised to 60.2 to include its fix.

### Removal plan

1. Check Arrow/Parquet releases for a supported replacement of Chrono.
2. Migrate our date conversions with tests for epoch boundaries, leap days,
   supported date ranges, and unchanged Parquet panel values.
3. Upgrade the upstream dependencies when their migration is available.
4. Use `cargo tree --all-features -i chrono` to check the remaining dependency paths.
5. Remove this exception when Chrono leaves the graph or the advisory is withdrawn.
6. Run the audit, feature tests, and minimum-Rust-version check before merging.

Sources: [RustSec advisory](https://rustsec.org/advisories/RUSTSEC-2026-0543)
and [upstream maintenance announcement](https://github.com/chronotope/chrono/issues/1768).
