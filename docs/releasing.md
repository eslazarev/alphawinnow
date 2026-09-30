# Releasing AlphaWinnow

A release publishes the `alphawinnow` crate to crates.io and attaches
prebuilt `alphawinnow` binaries to a GitHub Release. Merging to `main` triggers
it; the tree is verified, the version resolved, then tagged and published by
[`.github/workflows/release.yml`](../.github/workflows/release.yml) in one run.
A reviewed PR may prepare the version explicitly. Nothing needs to be uploaded
from a workstation.

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

After review and CI, merge to `main`. The workflow verifies the tree, resolves
the version, tags it, and releases it in one run. A merge that changes crate
source or manifests is a publishing action; do not merge just to rehearse.

### Preparing an explicit version

For a planned breaking release such as 0.2.0, update `[workspace.package].version`
and the `alphawinnow` entry in `Cargo.lock` together in the PR. Add a changelog
entry and `docs/releases/<version>.md`; the workflow appends these version-specific
notes to the GitHub release. Keep Homebrew/Scoop manifests unchanged until actual
release archives and checksums exist.

In `auto` mode, `bump-version.sh` preserves a manifest version greater than the
last reachable `v*` tag and verifies its lockfile with `cargo metadata --locked`.
It does not increment a prepared 0.2.0 to 0.2.1, nor create an empty bump commit.
A version behind the last release fails. Explicit `patch`/`minor`/`major` still
increment the manifest, so choose `auto` to publish a prepared version.

Before committing, inspect `git diff` and the exact file list: local research
directories, private datasets, machine-specific handoff notes and result artifacts
must not be included. Do not use `git add .` on a long-lived research checkout.

```bash
python3 -m unittest discover -s scripts -p 'test_*.py'
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
cargo test --workspace --no-default-features --locked
cargo +1.92.0 check --workspace --all-targets --all-features --locked
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --all-features --no-deps --locked
cargo run --example measured_search --no-default-features --locked
cargo package -p alphawinnow --list --locked
cargo publish --workspace --dry-run --locked
```

On a dirty preparation checkout, package/dry-run checks need `--allow-dirty`;
this does not stage or publish anything. Review the packaged sources separately.

### Which part of the version moves

When no newer version is already prepared, `scripts/bump-version.sh` classifies
the Conventional Commit subjects since the
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
stability this crate has not declared yet, so automatic classification uses
`minor`. A manually requested `major` still takes effect.

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
and pick `level` explicitly to force `patch`, `minor`, or `major`. Manual runs
always release the repository's default branch; the ref selector shown by the
Actions UI is deliberately ignored so it cannot inject an unreviewed branch
into the default branch's cache scope.

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
   test suite, version-selection regression tests, and a
   `cargo publish --workspace --dry-run`. Only after checks pass does the same
   job resolve the prepared version or raise it and refresh `Cargo.lock`, commit
   `chore(release): vX.Y.Z` if files changed, and push the commit and tag. The order is the
   point: every check is read-only, so a failure can never leave a tag behind
   on a commit that does not build. The raise step is skipped for a pushed tag,
   which already names its version. Every later job checks out the immutable
   commit SHA verified by this job, so the release cannot move to another tree
   between verification and packaging.
4. **binaries** — builds `alphawinnow` with `--all-features` for
   `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`,
   `aarch64-apple-darwin`, `x86_64-apple-darwin`, and
   `x86_64-pc-windows-msvc`. Each archive carries the README, the licence, and
   a SHA-256 checksum, and every natively runnable binary answers
   `--version` and `doctor --json` before it is packaged. The two Linux jobs
   additionally produce a `.deb` and an `.rpm` from that same unstripped
   binary, each on its own architecture so that `dpkg-shlibdeps` and `ldd`
   resolve real dependencies.
5. **crates-io** — publishes with `cargo publish --workspace --locked`. One
   crate carries both targets, so there is no multi-crate ordering to get
   wrong and no window in which a half-published release is visible.
6. **github-release** — creates the release for the tag and attaches every
   archive, package, and checksum.
7. **packaging** — regenerates `Formula/alphawinnow.rb` and the Scoop manifest
   from the published checksums and commits them to the default branch, which
   is what `brew tap` and Scoop read. It runs last because those manifests
   point at release download URLs that do not exist until the release does.

The jobs are chained so that each irreversible step happens only after the
previous one succeeded. The publish job waits for the whole binary matrix, so a
target that fails to build stops the upload instead of shipping a half-finished
release. The GitHub Release in turn waits for crates.io, so a failed upload
never leaves a published release whose notes point at a version nobody can
install.

## Rehearsing without publishing

Run the workflow manually from the Actions tab with **Run workflow** and leave
`dry_run` checked. That path raises no version and writes no tag: it runs
`verify` and the full binary matrix against the default branch as it stands,
uploads the archives as workflow artifacts, and skips both crates.io and the
GitHub Release. It is the cheapest way to confirm that a new target or a
dependency bump still builds everywhere.

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
