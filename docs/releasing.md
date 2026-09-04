# Releasing AlphaWinnow

A release publishes the `alphawinnow` crate to crates.io and attaches
prebuilt `alphawinnow` binaries to a GitHub Release. Merging to `main` triggers
it; the version is raised, tagged, verified, and published by
[`.github/workflows/release.yml`](../.github/workflows/release.yml) in one run.
No version is edited and nothing is uploaded from a workstation.

## One-time repository setup

1. Create a crates.io API token scoped to `publish-new` and `publish-update`.
2. In **Settings → Environments**, create an environment named `crates-io`.
3. Add the token to that environment as the secret `CRATES_IO_TOKEN`.
4. Optionally add required reviewers to the `crates-io` environment. The
   workflow then waits for an approval, which keeps the irreversible action
   behind a human decision. Two jobs enter that environment — the preflight
   token check and the upload itself — so a reviewed release is approved twice,
   once before anything is written and once before the upload.

The environment is the only place the token is readable. Pull-request builds,
scheduled audits, and the binary matrix never see it.

## Cutting a release

Merge to `main`. That is the whole procedure: the workflow raises the version,
tags it, and releases it in one run. Nothing is edited by hand and no tag is
pushed from a workstation.

### Which part of the version moves

`scripts/bump-version.sh` classifies the Conventional Commit subjects since the
last `v*` tag: a `!` marker or a `BREAKING CHANGE:` trailer is breaking, a
`feat:` subject is a feature, anything else is a fix.

Below `1.0.0` the minor position carries breakage, because Cargo treats `0.1.x`
as one compatibility range and `0.2.0` as a different one:

| Change since the last tag | `0.x` | `1.0.0` and later |
| --- | --- | --- |
| `fix:`, `chore:`, `docs:` … | patch | patch |
| `feat:` | patch | minor |
| `feat!:` or `BREAKING CHANGE:` | minor | major |

Promoting a breaking change to `major` while below `1.0.0` would claim a
stability this crate has not declared yet, so the script refuses to.

### When a merge does not release

crates.io versions are immutable, so a merge that cannot affect the published
artifact must not consume one. The run stops before raising anything when:

- no file under `crates/`, `Cargo.toml`, or `Cargo.lock` changed — a
  documentation, workflow, or roadmap merge releases nothing;
- the head commit subject begins with `chore(release):`, which is the bump
  commit itself;
- the head commit message contains `[skip release]`.

### Overriding the level, or releasing by hand

Run the workflow from the Actions tab with **Run workflow**, uncheck `dry_run`,
and pick `level` explicitly to force `patch`, `minor`, or `major`.

A tag pushed by hand still releases exactly that tag and raises nothing:

```bash
git tag -a v0.2.0 -m "v0.2.0"
git push origin v0.2.0
```

That path requires `version = "0.2.0"` to already be in the manifest; the
`verify` job fails the release if the tag and the manifest disagree, before
anything is built or uploaded.

## What the workflow does

1. **preflight** — reads `CRATES_IO_TOKEN` from the `crates-io` environment and
   fails if it is missing. This runs before anything is written, so a
   repository without the token stops here rather than after it has been
   tagged. A rehearsal skips it, because it publishes nothing.
2. **decide** — classifies the run: release the pushed tag, raise and release,
   rehearse, or stop. A merge that changed no crate source or manifest stops
   here.
3. **verify** — formatting, Clippy with warnings denied, the full all-features
   test suite, and a `cargo publish --workspace --dry-run`. Only after all four
   pass does the same job raise the version, refresh `Cargo.lock`, commit
   `chore(release): vX.Y.Z`, and push the commit and the tag. The order is the
   point: every check is read-only, so a failure can never leave a tag behind
   on a commit that does not build. The raise step is skipped for a pushed tag,
   which already names its version. Every later job checks out that tag rather
   than the commit that started the run, so the release is built from exactly
   the tree the tag points at.
4. **binaries** — builds `alphawinnow` with `--all-features` for
   `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`,
   `aarch64-apple-darwin`, `x86_64-apple-darwin`, and
   `x86_64-pc-windows-msvc`. Each archive carries the README, the licence, and
   a SHA-256 checksum, and every natively runnable binary answers
   `--version` and `doctor --json` before it is packaged.
5. **crates-io** — publishes with `cargo publish --workspace --locked`. One
   crate carries both targets, so there is no multi-crate ordering to get
   wrong and no window in which a half-published release is visible.
6. **github-release** — creates the release for the tag and attaches every
   archive and checksum.

The jobs are chained so that each irreversible step happens only after the
previous one succeeded. The publish job waits for the whole binary matrix, so a
target that fails to build stops the upload instead of shipping a half-finished
release. The GitHub Release in turn waits for crates.io, so a failed upload
never leaves a published release whose notes point at a version nobody can
install.

## Rehearsing without publishing

Run the workflow manually from the Actions tab with **Run workflow** and leave
`dry_run` checked. That path raises no version and writes no tag: it runs
`verify` and the full binary matrix against the branch as it stands, uploads
the archives as workflow artifacts, and skips both crates.io and the GitHub
Release. It is the cheapest way to confirm that a new target or a dependency
bump still builds everywhere.

## If a release goes wrong

crates.io versions are immutable. A bad version cannot be replaced, only
yanked:

```bash
cargo yank --version 0.2.0 alphawinnow
```

Yanking keeps existing lockfiles working and stops new dependants from
resolving to that version. Then merge the fix: the next release is raised and
published automatically, so no version has to be chosen by hand. Delete the
GitHub Release and its tag only if no one has fetched them yet.
