# WorldQuant BRAIN alpha research with AlphaWinnow

A detailed, offline-first guide to catalogs, candidate search, measured feedback,
checkpoints, and the boundary between a Rust research engine and a platform
integration. Applies to the prepared **0.3.0** source tree.

**Nothing in this guide logs in to BRAIN, runs a platform simulation, or submits
an alpha.** All supplied files and demonstration outcomes are synthetic. They
are examples of software contracts, not profitable formulas or an account's
available data. AlphaWinnow is independent of and not endorsed by WorldQuant.

[Back to README](../README.md#using-alphawinnow-with-worldquant-brain)

## Contents

1. [Responsibilities and terminology](#1-responsibilities-and-terminology)
2. [Run the complete offline demo](#2-run-the-complete-offline-demo)
3. [Example files](#3-example-files)
4. [Build a small typed catalog](#4-build-a-small-typed-catalog)
5. [Freeze settings and evaluation context](#5-freeze-settings-and-evaluation-context)
6. [Generate and inspect candidates](#6-generate-and-inspect-candidates)
7. [Deduplicate without losing meaning](#7-deduplicate-without-losing-meaning)
8. [Translate a dialect, not a whole platform](#8-translate-a-dialect-not-a-whole-platform)
9. [Audit and screen immutable feedback](#9-audit-and-screen-immutable-feedback)
10. [Adaptive search with real measurements](#10-adaptive-search-with-real-measurements)
11. [Record outcomes and recover safely](#11-record-outcomes-and-recover-safely)
12. [Local evaluation and submission checks](#12-local-evaluation-and-submission-checks)
13. [Choose informative experiments](#13-choose-informative-experiments)
14. [Troubleshooting and production checklist](#14-troubleshooting-and-production-checklist)

## 1. Responsibilities and terminology

AlphaWinnow supplies the **local search engine**. An adapter you maintain supplies
the platform-specific behavior. They should remain separate so credentials,
licensed catalogs, account history, and private formulas never need to enter
the public crate.

```text
your typed catalog + audited historical measurements
                         |
                         v
       AlphaWinnow: ask -> typed proposals + lineage
                         |
                         v
       your selector -> frozen selected candidate set
                         |
                         v
       your adapter -> translate / validate / evaluate externally
                         |
                         v
       durable results -> versioned objective -> tell_screened
                         |
                         v
       updated parents + checkpoint -> next ask

       submission is a separate decision, outside this loop
```

| Term | Meaning here | Does not mean |
| --- | --- | --- |
| Candidate | A generated expression with identity and lineage | A passing or tradable alpha |
| Structural score | A local measure of expression structure and novelty | Sharpe, Fitness, or PASS probability |
| Measured outcome | A finite score supplied by the caller under a declared objective | A result the library verified with BRAIN |
| Feedback guidance | An estimate from a supplied immutable feedback dataset | Guaranteed out-of-sample performance |
| Semantic fingerprint | Identity under AlphaWinnow's documented equivalence rules | A platform ID or a proof of equal PnL on every backend |
| Context | Caller-defined identity for comparable measurements | Authentication or proof that the measurements are genuine |

No BRAIN API client, credentials manager, quota manager, scheduler, portfolio
service or submission automation is included. Use any external integration only
as permitted by the applicable platform and data terms. Obtain current platform
requirements from your own account; this guide intentionally does not hard-code
account-dependent thresholds or promise an accepted API payload.

## 2. Run the complete offline demo

Use a source checkout containing this guide. Repository-level examples are not
installed beside the CLI by `cargo install`. Rust 1.92+ is required to build;
Python 3.8+ is used only for this optional demonstration wrapper, with no extra
Python packages. The library and CLI themselves remain Rust-only.

From the repository root:

```bash
cargo build -p alphawinnow --bin alphawinnow \
  --example measured_search_stdio --locked

python3 examples/worldquant-brain/demo.py \
  --cli target/debug/alphawinnow \
  --bridge target/debug/examples/measured_search_stdio \
  --output dist/brain-guide-demo
```

Choose a **new output directory** for every run. The script refuses an existing
directory rather than replacing prior evidence. Windows users should supply
the corresponding `.exe` executable paths. If using `CARGO_TARGET_DIR`, pass
paths from that directory instead of `target/debug`.

The demo:

1. Searches 256 local candidates using the supplied restricted catalog.
2. Compiles one expression against the toy dialect and confirms a forbidden
   field is rejected.
3. Audits separate training/evaluation feedback and confirms overlapping
   evidence is rejected.
4. Writes a screening report retaining both eligible and rejected records.
5. Imports two synthetic parents, asks for four pools of eight proposals,
   measures the first two per pool, and updates retained parents.
6. Repeats pending `ask` calls, restores checkpoints in fresh processes, and
   asserts identical subsequent proposals.

Expected final counters are **2 imported, 8 measured, 24 screened out, 4 batches,
0 external requests**. These eight measurements are a toy function of expression
text, not live simulations. The CLI shortlist and the adaptive pools are two
separate demonstrations; one is not silently used as the other's measured data.

The output directory contains:

```text
brain-guide-demo/
  candidates.jsonl                   # structural shortlist, not all attempts
  candidates.jsonl.manifest.json     # structural run metadata
  candidates.jsonl.current.json      # pointer to a complete immutable run pair
  .candidates.jsonl.runs/            # immutable candidate/manifest generation
  search.json                       # CLI search summary
  compiled.json                     # toy lowering report
  holdout.json                      # separate-feedback audit
  screening.json                    # every candidate's screening decision
  requests.jsonl                    # primary measured-session JSONL requests
  responses.jsonl                   # matching primary-session responses
  checkpoint.json                   # last completed adaptive-batch checkpoint
  summary.json                      # counters, fixture hashes, no-market flag
```

The transcripts are written after successful completion. This small script is
**not** a crash-safe production ledger. Restore-verification processes are
checked by assertions but are not included in the primary-session transcript.

## 3. Example files

All files live in [`examples/worldquant-brain/`](../examples/worldquant-brain/).
Only the catalog, dialect, measured-session and feedback schemas are library
contracts. The two adapter-oriented templates are documentation conventions.

| File | Consumer | Purpose |
| --- | --- | --- |
| [catalog.json](../examples/worldquant-brain/catalog.json) | `search`, `inspect --catalog`, measured session | Three public toy fields and six typed operators |
| [dialect.json](../examples/worldquant-brain/dialect.json) | `compile` | Allowlist and a `negate` lowering, not a BRAIN compatibility certificate |
| [session-config.json](../examples/worldquant-brain/session-config.json) | JSONL bridge / measured-search API | Seed, complexity limits, proposal budget and retention policy |
| [seeds.json](../examples/worldquant-brain/seeds.json) | `init_measured` | Two caller-measured toy parents |
| [evaluation-context.json](../examples/worldquant-brain/evaluation-context.json) | Demo / your adapter | Illustrative settings, objective identity and no-network scope |
| [feedback-training.json](../examples/worldquant-brain/feedback-training.json) | `audit-feedback`, `screen-feedback` | Three normalized synthetic observations |
| [feedback-evaluation.json](../examples/worldquant-brain/feedback-evaluation.json) | Held-out audit | Two separate synthetic observations |
| [result-record-template.json](../examples/worldquant-brain/result-record-template.json) | Your adapter only | Suggested provenance fields; never send it as an API request |
| [demo.py](../examples/worldquant-brain/demo.py) | Python, optional | Executes and checks the offline examples |

Copy templates into your private research workspace before supplying private
fields or outcomes. Do not replace these public fixtures with account exports.

## 4. Build a small typed catalog

The supplied catalog describes an AlphaWinnow grammar. It is not an official
BRAIN field export and does not assert availability in your account.

A field entry looks like this:

```json
{"name": "close", "kind": "signal", "family": "toy-price", "allowed_roles": ["signal_input"]}
```

`family` is a research grouping label, not a statistical independence guarantee.
The catalog schema requires a nonempty group list, so the example includes a
`sector` group even though none of its six operators consumes a group argument.
That declaration does not provide sector classifications or enable neutralization.
Do not flatten a vector-valued platform field into a scalar signal merely to
make parsing succeed. Choose a supported representation and a separately
verified reduction in your adapter, or leave the field out.

An operator signature explicitly constrains its arguments:

```json
{
  "name": "ts_mean",
  "arity": {"kind": "binary"},
  "inputs": [
    {"kinds": ["signal"]},
    {"kinds": ["scalar"], "domain": {"kind": "window_set", "values": [5, 10, 20, 60]}}
  ],
  "output": "signal",
  "generation_weight": 3
}
```

The windows are illustrative research choices, not optimized values. Catalog
domains and session complexity limits both apply. A catalog's `generation_weight`
changes proposal sampling; it is not a portfolio allocation weight.
The structural CLI also validates implemented operator grammars: this example
keeps `multiply`'s built-in variadic 2–8 signature. Do not assume that a different
target's binary-only signature can replace it in every public entry point;
validate catalogs through the actual workflow you intend to run.

Start with a coherent economic mechanism and a small verified allowlist. More
fields create many more combinations; they do not automatically produce more
independent evidence. Preserve field units, missingness and reporting frequency
in your private catalog audit even when those properties are not encoded in
AlphaWinnow's structural type system.

## 5. Freeze settings and evaluation context

The same formula under different universes, delays or neutralization settings is
a different evaluation experiment. Store the exact settings and period beside
every measurement, not only in a global configuration file that can change.

`evaluation-context.json` illustrates what to track. Its `USA`, `TOP3000`,
`INDUSTRY` and other values are examples, **not recommended settings or a validated
platform request**. The offline demo does not evaluate stocks and does not use
these settings to compute a score.

For a real adapter, derive context identity from a stable serialization of:

- dataset/catalog version and relevant field semantics;
- region, universe, delay and all signal-processing settings;
- evaluation dates, source revision and point-in-time assumptions;
- objective definition, normalization and validation split;
- reference portfolio identity when the objective depends on that portfolio.

Hashing JSON is useful only after fixing key order, numeric representation and
which settings are included. Keep the original metadata as well as its digest.
Do not change the meaning of an existing context string. If the objective changes,
start a new version and re-audit imported labels.

`evaluator_context` identifies evaluation semantics; `pool_context` distinguishes
the objective/reference pool used for parent scores. A cached standalone score
may be reusable in an unchanged context. An old marginal portfolio contribution
is not automatically comparable after the portfolio changes.

## 6. Generate and inspect candidates

Run the CLI manually if you do not need the demo wrapper:

```bash
mkdir -p dist/brain-manual
./target/debug/alphawinnow search \
  --catalog examples/worldquant-brain/catalog.json \
  --output dist/brain-manual/candidates.jsonl \
  --max-candidates 256 --duration-seconds 60 --threads 2 --seed 7

./target/debug/alphawinnow inspect 'rank(ts_delta(close, 20))' \
  --catalog examples/worldquant-brain/catalog.json
```

Inspect lineage, fields, depth, operator counts and rejection reasons before
spending an external evaluation budget. The output is a bounded shortlist, not
a log containing all 256 generation attempts. Rejected duplicates and trivial
expressions still contribute to the run's counters.

Use the `.current.json` pointer when consuming the candidate/manifest pair
transactionally; adjacent aliases alone are not the atomic publication boundary.
Save input catalog/config checksums and the tool version with the experiment.

Two different budgets must remain separate:

| Budget | Counts | Owner |
| --- | --- | --- |
| CLI `--max-candidates` | Structural search attempts | CLI search |
| Measured `max_evaluations` | Emitted unique proposals, including screened-out proposals | Measured session |
| Measured `max_attempts` | Attempted transforms, including rejected attempts | Measured session |
| External simulation budget | Actual authorized external attempts | Your durable adapter ledger |

Rescoring retained parents is also your responsibility. A proposal limit does
not count or authorize those external calls. The engine cannot enforce a live
quota that it does not control.

## 7. Deduplicate without losing meaning

AlphaWinnow separates exact canonical identity from its documented semantic
identity. Under the root-scale convention, a final positive multiplier is
scale-equivalent; a negative multiplier reverses direction. Scaling *inside*
a nonlinear call or only one sleeve remains significant.

```bash
./target/debug/alphawinnow inspect 'multiply(close, 2)'
./target/debug/alphawinnow inspect 'multiply(close, -2)'
./target/debug/alphawinnow inspect 'rank(multiply(close, 2))'
```

These commands explain identities, not platform PnL. Apply only equivalences
that are valid for your target's preprocessing semantics. Keep settings context
alongside semantic identity when deciding whether a historical simulation is
reusable. Do not merge contexts merely because the expression fingerprint matches.

Warm starts accept a set of this library's semantic fingerprints. Exclude both
successful and unsuccessful previously attempted expressions where applicable;
an exclusion is not a training label. Do not provide another engine's hashes.

`identity_reduction::reduce_unit_scaling` is an additional opt-in helper, not an
automatic change to the default fingerprint contract. Version such a policy and
apply it to old and new expressions consistently. Field-set diversity likewise
does not establish low return correlation.

## 8. Translate a dialect, not a whole platform

```bash
./target/debug/alphawinnow compile 'rank(negate(returns))' \
  --dialect examples/worldquant-brain/dialect.json
```

The toy dialect declares `negate` as a unary-to-scalar lowering to
`multiply(-1, x)`. The report includes canonical source/compiled expressions,
used fields/operators and `negate->multiply` in the lowering log. Its output is
still AlphaWinnow-style functional expression text, not a platform response.

This intentional failure demonstrates allowlist enforcement:

```bash
./target/debug/alphawinnow compile 'rank(high)' \
  --dialect examples/worldquant-brain/dialect.json
```

`high` is known to the built-in parser but absent from this target allowlist.
The command exits nonzero rather than silently inventing a field mapping.

Important limitation: in 0.3.0 the `compile` CLI uses the **built-in parser**;
it does not accept a custom `--catalog`. An allowlist does not add parser
operators or fields. A custom search catalog can therefore describe expressions
this command cannot compile. Your adapter must support that mapping separately.

Before external evaluation, verify arity, scalar spelling, keyword behavior,
group representation, vector reductions, units, missing-value semantics and
platform resource limits. Do not use naive text replacement across nested
expressions. Never overwrite the source formula with its translated form;
retain both and the translator version.

## 9. Audit and screen immutable feedback

This is separate from the measured-session API. The nearest-neighbor feedback
format requires outcomes in `[-1, 1]` and confidence in `(0, 1]`; the measured
session accepts arbitrary **finite** outcome scores with larger values better.
Do not pass raw platform metrics into the normalized feedback schema without a
documented transformation. Confidence is caller-supplied evidence weighting,
not an automatically estimated probability.

The supplied feedback files contain invented toy labels on structural features;
they are deliberately **not** the call-count objective used by the adaptive demo.
They have a different context string and should not be mixed into that session.

```bash
./target/debug/alphawinnow audit-feedback \
  --feedback examples/worldquant-brain/feedback-training.json \
  --evaluation-feedback examples/worldquant-brain/feedback-evaluation.json

./target/debug/alphawinnow screen-feedback \
  --input dist/brain-manual/candidates.jsonl \
  --feedback examples/worldquant-brain/feedback-training.json \
  --output dist/brain-manual/screening.json \
  --minimum-conservative-outcome 0
```

Audit output includes both evidence identities, per-record predictions, coverage
and aggregate errors. Known record-ID or supplied semantic-fingerprint overlap
fails closed. If you omit fingerprints or rename near-duplicates, the library
cannot infer all leakage from structural features. Split by time and economic
family yourself. A repeatedly inspected split is not an untouched holdout.

The screening report retains `eligible`, `below_threshold`, and `missing_guidance`
records. Zero is an illustrative threshold in the chosen label's units. Choose
your threshold before viewing validation results. An empty eligible set is valid;
the tool does not fill it with unrelated candidates to meet a desired quota.
The conservative score is a dispersion heuristic, not a statistical confidence
bound, BRAIN PASS probability or guarantee of future returns.

Screening uses report schema 2. `input_file_checksum` identifies the exact
candidate JSONL bytes read by the CLI, and `candidate_records_checksum`
identifies the ordered parsed records independently of whitespace. Preserve
these with `feedback_checksum`, configuration and the CLI's output
`content_checksum`; they bind the decision artifact to its inputs.

For a larger report, add `--format jsonl` and use `--output screening.jsonl`.
The first line has `type: "metadata"` and all report metadata/checksums; each
following line has `type: "candidate"` and a `record` containing the guided
candidate and its decision. Both line types use schema 2. This streams output
serialization, not the ranking computation: candidates still fit in memory.

## 10. Adaptive search with real measurements

Use [`MeasuredSearchSession`](../crates/alphawinnow/src/measured_search.rs) directly
from Rust, or the [`measured_search_stdio`](../crates/alphawinnow/examples/measured_search_stdio.rs)
example from another language. The executable reads **one JSON object per line**
and writes one response per line. Keep one process alive for a session; a new
process must initialize or restore. A reported `error` is not a successful update.

The demo produces complete runnable transcripts rather than requiring you to
copy placeholder fingerprints. The protocol's sequence is:

1. `init_measured`: pass `config`, `catalog`, `evaluator_context`, `pool_context`,
   `seeds`, and `excluded`. Seeds must fit parent capacity and have compatible
   real measurements. `excluded: []` is intentional for the small demonstration.
2. `ask`: receive `proposals` and `score_requests`. Repeating `ask` while pending
   returns the same proposals; it does not authorize repeating external requests.
3. Freeze your selected subset before evaluation. The demo takes the first two
   only to make the example deterministic, **not** because that is a useful
   financial selector.
4. Build outcomes for every retained incumbent and the selected pending proposals.
   Match by `semantic_fingerprint`, not by ordinal position in a response list.
5. `tell_screened`: supply both contexts and those finite outcomes. Omitted
   proposals stay deduplicated without invented failure labels. At least one
   pending proposal must be measured; an entirely unevaluated batch is not a
   normal successful `tell_screened` update.
6. `checkpoint`: persist the returned checkpoint string. To restore, use a fresh
   process with `op: "restore"`, the identical config/catalog and that string.

Two literal protocol requests are:

```jsonl
{"op":"ask"}
{"op":"checkpoint"}
```

They require an initialized process. Inspect the generated `requests.jsonl` for
the complete `init_measured` and `tell_screened` objects. Requests and responses
have matching line order, but no built-in concurrent request IDs: serialize calls
on a bridge process instead of multiplexing unordered callers.

The returned checkpoint is a JSON **string** inside the outer response, not a
nested object. The demo writes its decoded content as `checkpoint.json`; reading
that file as text gives the string needed for `restore`. Warm-started/screened
sessions use measured checkpoint schema 2, which differs from CLI structural
checkpoint schema 2. Matching numbers do not make the formats interchangeable.

For large historical exclusion sets, Rust callers can use
`checkpoint_with_history_json` and `from_checkpoint_with_history_json` instead.
The schema-3 envelope references an immutable `ExcludedHistory` archive by SHA-256.
Persist that archive first, retain it while referenced, and load it once per
process. Seeds and evolving state stay inside the checkpoint. The JSONL bridge
and this demo intentionally keep standalone checkpoints; they do not implement
the separate-archive protocol. See the [Rust example and lifecycle notes](../README.md#drive-search-with-your-own-evaluator)
and the [synthetic checkpoint benchmark](../crates/alphawinnow/examples/checkpoint_history.rs).

The toy objective is `-min(1 + count('('), 20) / 20`. Thus `rank(close)` and
`ts_mean(volume, 20)` both have outcome `-0.1`. It provides a reproducible software
test, not economic evidence. In production, define your own versioned objective
from actual results and retain the raw metrics so that the transformation is
auditable. A predicted score must not be recorded as an observed outcome.

## 11. Record outcomes and recover safely

The [result template](../examples/worldquant-brain/result-record-template.json)
shows provenance to preserve. Nulls mean unknown/not yet recorded, not zero
performance. It is **not** an accepted BRAIN API body or AlphaWinnow input file.
Your adapter owns its storage schema and state machine.

At minimum, retain source expression, source identity, exact wire expression,
full settings and context, lineage, selection timestamp, request/response
timestamps, external identifier, raw metrics, checks, terminal status, failure
reason and objective version. Keep rejected observations too.

A suggested recovery distinction is:

| State | Meaning | Safe next action |
| --- | --- | --- |
| Not sent | No external attempt was made | Submit only within an explicitly authorized budget |
| Known pending | Remote identity exists; result not terminal | Poll the known request instead of creating another |
| Unknown effect | Response lost; cannot prove whether request took effect | Quarantine and reconcile before considering a retry |
| Completed | Durable result exists | Reuse it only in a compatible context |
| Known validation failure | Failure response durably identified | Retain diagnostic evidence; do not silently relabel or retry |

An AlphaWinnow checkpoint contains search state, **not** remote request state.
Restoring it cannot establish whether an interrupted request spent quota. Freeze
selected identities before external work, persist observations before feedback,
and make each adapter transition idempotent. Do not claim exactly-once remote
execution unless your external protocol and recovery evidence establish it.

Daily quota resets must not expand a user's total authorized budget. Network
errors and lack of progress need bounded backoff and a resumable stop, not an
infinite cycle that spends more evaluations. Those mechanisms are architectural
guidance for your adapter; this repository does not implement them for BRAIN.

## 12. Local evaluation and submission checks

The optional numeric evaluator uses data you provide. `brain-proxy-v1` is a
transparent local metric convention, not a reproduction certificate. Platform
data coverage, universe membership, date alignment, adjustments, costs,
missing-value handling and signal processing can all differ.

Treat these as distinct questions:

- Does the expression parse and satisfy local structural limits?
- Does the selected target accept its syntax and settings?
- Does measured performance pass the checks for the chosen evaluation?
- Is it sufficiently different from the current portfolio?
- Is the economic hypothesis stable across reasonable nearby parameters and
  genuinely independent validation periods?
- Is it eligible for submission **now**, under the current account state?

One answer does not imply the next. A low-correlation observation from an old
book is not a current-book check. Even a platform PASS is not evidence of
independent profitability or immunity to search-induced overfitting.
Submission remains a separate authorized action outside AlphaWinnow.

## 13. Choose informative experiments

Use the tool to refine a hypothesis, not merely to manufacture many similar
formulas. Before external evaluation, write down the economic reasoning,
fields, periods/settings, selection rule, rejection criteria and bounded budget.

- Prefer a coherent dataset/mechanism; explain any cross-dataset combination.
- Exclude semantic/root-scale duplicates and cap repeated parent families.
- Keep a deliberate exploration allocation instead of trusting only the top
  predicted scores. There is no universally validated default mix in this guide.
- Track both predictive calibration and independent family/context coverage.
- Save every attempted result; unknown outcomes remain unknown.
- Validate tuned constants in a preregistered neighborhood. A sharp isolated
  maximum is a warning; choosing the best neighbor is another optimization,
  not independent validation of the original choice.
- Do not count neighboring parameter variants as independent alpha discoveries.

When a lane saturates, audit a genuinely different source or mechanism instead
of just increasing the candidate ceiling. More simulation labels help only if
they add useful information under a sound validation design.

## 14. Troubleshooting and production checklist

| Symptom | Check first |
| --- | --- |
| Unknown symbol | Which parser/catalog is active? `compile` does not consume a custom search catalog in 0.3.0 |
| Invalid arity or window | Match the typed signature and both catalog/session limits; do not rewrite a measured expression and keep its old score |
| Target rejects compiled text | The dialect is a local contract, not platform validation; check scalar spelling, kwargs, types and translation |
| No candidates / generation exhausted | Inspect duplicates, grammar breadth and limits; do not silently change a frozen experiment |
| No screening-eligible records | Check evidence coverage and threshold; abstention is an expected result |
| Missing parent outcome | `score_requests` includes incumbents; do not score only the newly chosen proposals |
| Context mismatch | Verify settings, objective, pool, catalog and versions; do not just rename a context to bypass the check |
| Checkpoint cannot restore | Use the original executable/config/catalog and correct checkpoint format; do not edit counters or schema numbers |
| Repeat remote simulations after restart | Check your external ledger; the Rust checkpoint is not a simulation ledger |
| Demo output directory exists | Use a new output directory; preserve the earlier transcript |

Before adapting the demonstration to private work:

- [ ] Verify account-visible fields, operators and permitted integration methods.
- [ ] Preserve data licenses and keep private artifacts outside the public repo.
- [ ] Freeze the grammar, semantics, settings, objective, version and budget.
- [ ] Validate source-to-wire translation on a small explicitly authorized set.
- [ ] Separate local structural guidance, proxy metrics and actual observations.
- [ ] Record requests durably and reconcile ambiguous outcomes without blind retries.
- [ ] Test interruption, restore and pending-result recovery independently.
- [ ] Evaluate on separate time/family evidence; report negative findings too.
- [ ] Review correlation, robustness and current platform checks before submission.

For library details, see the [README](../README.md),
[0.3.0 migration notes](releases/0.3.0.md), and the runnable
[adaptive Rust example](../crates/alphawinnow/examples/adaptive_measured_search.rs).

### Example verification

Verified locally on 2026-10-27: two complete offline demo runs produced
byte-identical candidates, primary JSONL transcripts, checkpoints, summaries,
compiled reports, holdout reports and screening reports. Timing-bearing run
manifests were excluded from that byte comparison. Both runs imported 2 parents,
measured 8 toy proposals, screened out 24 and restored identical next batches.
Forbidden-field and overlapping-feedback checks failed as intended. CI includes
the runnable example smoke test so future contract changes can be detected.
No live platform compatibility or financial performance was tested.
