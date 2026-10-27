use std::collections::BTreeSet;

use alphawinnow::measured_search::{
    ExcludedHistory, MeasuredExploration, MeasuredMutation, MeasuredOutcome,
    MeasuredParentSelection, MeasuredSearchConfig, MeasuredSearchSession, MeasuredSeed,
};
use alphawinnow::{Limits, builtin_catalog, parse_expression, semantic_fingerprint};

fn config() -> MeasuredSearchConfig {
    MeasuredSearchConfig {
        seed: 71,
        limits: Limits::default(),
        max_evaluations: 11,
        max_attempts: 2000,
        batch_size: 4,
        parent_capacity: 2,
        exploration_every: 4,
        exploration_policy: MeasuredExploration::Grammar,
        mutation_policy: MeasuredMutation::Applicable,
        parent_selection: MeasuredParentSelection::Score,
        evaluator_context: "synthetic-fixed-objective-v1".into(),
    }
}

fn seeds() -> Vec<MeasuredSeed> {
    ["rank(close)", "ts_mean(volume, 20)"]
        .into_iter()
        .enumerate()
        .map(|(i, e)| MeasuredSeed {
            record_id: format!("synthetic-{i}"),
            expression: e.into(),
            outcome: -1.,
        })
        .collect()
}

fn start(excluded: &BTreeSet<String>) -> MeasuredSearchSession {
    MeasuredSearchSession::from_measured_parents(
        config(),
        builtin_catalog().clone(),
        &config().evaluator_context,
        "fixed-pool",
        &seeds(),
        excluded,
    )
    .unwrap()
}

fn restore(s: &MeasuredSearchSession) -> MeasuredSearchSession {
    MeasuredSearchSession::from_checkpoint_json(
        &s.checkpoint_json().unwrap(),
        config(),
        builtin_catalog().clone(),
    )
    .unwrap()
}

#[test]
fn referenced_checkpoint_preserves_pending_screened_state_and_next_batch() {
    let excluded = (0..1024).map(|i| format!("{i:064x}")).collect();
    let mut s = start(&excluded);
    let history =
        ExcludedHistory::from_json(&s.excluded_history().unwrap().to_json().unwrap()).unwrap();
    let resume = |s: &MeasuredSearchSession| {
        let compact = s.checkpoint_with_history_json().unwrap();
        let embedded = s.checkpoint_json().unwrap();
        assert!(compact.len() * 4 < embedded.len());
        let restored = MeasuredSearchSession::from_checkpoint_with_history_json(
            &compact,
            config(),
            builtin_catalog().clone(),
            &history,
        )
        .unwrap();
        assert_eq!(restored.checkpoint_json().unwrap(), embedded);
        restored
    };
    assert_eq!(resume(&s).ask().unwrap(), s.ask().unwrap());
    assert_eq!(resume(&s).ask().unwrap(), s.ask().unwrap());
    let selected = s.ask().unwrap()[0].clone();
    let scores: Vec<_> = s
        .parents()
        .into_iter()
        .chain([selected])
        .map(|p| MeasuredOutcome {
            semantic_fingerprint: p.semantic_fingerprint,
            outcome: 0.25,
        })
        .collect();
    s.tell_screened(&config().evaluator_context, "fixed-pool", &scores)
        .unwrap();
    let mut restored = resume(&s);
    assert_eq!(restored.measured(), 1);
    assert_eq!(restored.screened_out(), 3);
    assert_eq!(restored.ask().unwrap(), s.ask().unwrap());
}

#[test]
fn referenced_checkpoint_rejects_wrong_archive_and_corrupt_state() {
    let mut s = start(&BTreeSet::from(["a".repeat(64)]));
    s.ask().unwrap();
    let compact = s.checkpoint_with_history_json().unwrap();
    let history = s.excluded_history().unwrap();
    let read = |content: &str, history: &ExcludedHistory| {
        MeasuredSearchSession::from_checkpoint_with_history_json(
            content,
            config(),
            builtin_catalog().clone(),
            history,
        )
    };
    assert!(read(&compact, &ExcludedHistory::new(BTreeSet::new()).unwrap()).is_err());
    assert!(
        MeasuredSearchSession::from_checkpoint_json(&compact, config(), builtin_catalog().clone(),)
            .is_err()
    );
    let original: serde_json::Value = serde_json::from_str(&compact).unwrap();
    for (pointer, value) in [
        ("/schema", serde_json::json!(2)),
        ("/excluded_sha256", serde_json::json!("b".repeat(64))),
        (
            "/state/bootstrap/excluded",
            serde_json::json!(["a".repeat(64)]),
        ),
        ("/state/bootstrap", serde_json::Value::Null),
        ("/state/seen", serde_json::json!([])),
        ("/state/pending/0/ordinal", serde_json::json!(9999)),
    ] {
        let mut bad = original.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(
            read(&bad.to_string(), history).is_err(),
            "accepted {pointer}"
        );
    }
    let plain = MeasuredSearchSession::new(config(), builtin_catalog().clone()).unwrap();
    assert!(plain.checkpoint_with_history_json().is_err());
    assert!(ExcludedHistory::from_json("[\"bad\"]").is_err());
    assert!(ExcludedHistory::from_json(&" ".repeat(8 * 1024 * 1024 + 1)).is_err());
    assert!(ExcludedHistory::new((0..100_001).map(|i| format!("{i:064x}")).collect()).is_err());
}

#[test]
fn measured_winners_replace_seeds_and_drive_next_batch_with_exact_resume() {
    let mut s = start(&BTreeSet::new());
    assert_eq!(s.imported(), 2);
    assert_eq!(s.measured(), 0);
    assert_eq!(s.attempted(), 0);
    assert_eq!(
        restore(&s).checkpoint_json().unwrap(),
        s.checkpoint_json().unwrap()
    );
    let initial = s.parents();
    let first = s.ask().unwrap();
    assert_eq!(first, s.ask().unwrap());
    assert!(
        first
            .iter()
            .any(|p| !p.provenance.parent_fingerprints.is_empty())
    );
    let scores = s
        .score_requests()
        .iter()
        .map(|p| MeasuredOutcome {
            semantic_fingerprint: p.semantic_fingerprint.clone(),
            outcome: if initial.contains(p) { -1. } else { 1. },
        })
        .collect::<Vec<_>>();
    let before = s.checkpoint_json().unwrap();
    assert!(
        s.tell(&config().evaluator_context, "fixed-pool", &scores[1..])
            .is_err()
    );
    assert_eq!(s.checkpoint_json().unwrap(), before);
    s.tell(&config().evaluator_context, "fixed-pool", &scores)
        .unwrap();
    assert_eq!(s.measured(), 4);
    assert!(s.parents().iter().all(|p| first.contains(p)));
    let parents = s.parents();
    let mut resumed = restore(&s);
    let next = s.ask().unwrap();
    assert_eq!(resumed.ask().unwrap(), next);
    assert!(
        next.iter()
            .any(|p| !p.provenance.parent_fingerprints.is_empty())
    );
    assert!(
        next.iter()
            .all(|p| p.provenance.parent_fingerprints.iter().all(|id| parents
                .iter()
                .any(|parent| &parent.semantic_fingerprint == id)))
    );
    assert_eq!(restore(&s).ask().unwrap(), next);
}

#[test]
fn imports_do_not_spend_budget_and_short_final_batch_restores() {
    let mut s = start(&BTreeSet::new());
    let mut sizes = Vec::new();
    loop {
        let batch = s.ask().unwrap();
        if batch.is_empty() {
            break;
        }
        sizes.push(batch.len());
        let scores = s
            .score_requests()
            .iter()
            .map(|p| MeasuredOutcome {
                semantic_fingerprint: p.semantic_fingerprint.clone(),
                outcome: 0.25,
            })
            .collect::<Vec<_>>();
        s.tell(&config().evaluator_context, "fixed-pool", &scores)
            .unwrap();
        s = restore(&s);
    }
    assert_eq!(sizes, vec![4, 4, 3]);
    assert_eq!(s.measured(), 11);
    assert_eq!(s.imported(), 2);
}

#[test]
fn history_excludes_previously_emitted_and_scale_equivalent_seeds() {
    let baseline = start(&BTreeSet::new()).ask().unwrap();
    let mut excluded: BTreeSet<_> = baseline
        .iter()
        .map(|p| p.semantic_fingerprint.clone())
        .collect();
    excluded.insert(semantic_fingerprint(
        &parse_expression("multiply(rank(close), 3)").unwrap(),
    ));
    let mut s = start(&excluded);
    let batch = s.ask().unwrap();
    assert_eq!(batch.len(), 4);
    assert!(
        batch
            .iter()
            .all(|p| !excluded.contains(&p.semantic_fingerprint))
    );
    assert_eq!(restore(&s).ask().unwrap(), batch);
}

#[test]
fn bootstrap_is_order_independent_and_rejects_bad_evidence() {
    let original = start(&BTreeSet::new());
    let mut reversed = seeds();
    reversed.reverse();
    let make = |seeds: &[MeasuredSeed], context: &str, excluded: &BTreeSet<String>| {
        MeasuredSearchSession::from_measured_parents(
            config(),
            builtin_catalog().clone(),
            context,
            "fixed-pool",
            seeds,
            excluded,
        )
    };
    assert_eq!(
        original.checkpoint_json().unwrap(),
        make(&reversed, &config().evaluator_context, &BTreeSet::new())
            .unwrap()
            .checkpoint_json()
            .unwrap()
    );
    assert!(make(&seeds(), "wrong-context", &BTreeSet::new()).is_err());
    assert!(make(&[], &config().evaluator_context, &BTreeSet::new()).is_err());
    assert!(
        make(
            &seeds(),
            &config().evaluator_context,
            &BTreeSet::from(["foreign-hash".into()])
        )
        .is_err()
    );
    for expression in ["multiply(rank(close), 3)", "unknown_field", "1"] {
        let mut bad = seeds();
        bad[1].expression = expression.into();
        assert!(make(&bad, &config().evaluator_context, &BTreeSet::new()).is_err());
    }
    let mut bad = seeds();
    bad[1].outcome = f64::NAN;
    assert!(make(&bad, &config().evaluator_context, &BTreeSet::new()).is_err());
    let mut bad = seeds();
    bad[1].record_id = bad[0].record_id.clone();
    assert!(make(&bad, &config().evaluator_context, &BTreeSet::new()).is_err());
}

#[test]
fn checkpoint_rejects_missing_imports_excluded_proposals_and_schema_downgrade() {
    let mut s = start(&BTreeSet::new());
    let pending = s.ask().unwrap();
    let valid = s.checkpoint_value().unwrap();
    for case in 0..5 {
        let mut bad = valid.clone();
        match case {
            0 => bad["schema"] = 1.into(),
            1 => bad["measured"] = 0.into(),
            2 => {
                bad["parents"].as_array_mut().unwrap().pop();
            }
            3 => {
                bad["bootstrap"]["excluded"] = serde_json::json!([pending[0].semantic_fingerprint]);
            }
            _ => bad["bootstrap"]["seeds"][0]["outcome"] = 9.into(),
        }
        assert!(
            MeasuredSearchSession::from_checkpoint_json(
                &bad.to_string(),
                config(),
                builtin_catalog().clone()
            )
            .is_err(),
            "case {case}"
        );
    }
}

#[test]
fn unsupported_retention_policies_are_explicitly_rejected() {
    for policy in [
        MeasuredParentSelection::Uniform,
        MeasuredParentSelection::LatestBatchScoreV1,
        MeasuredParentSelection::LatestBatchUniformV1,
    ] {
        let mut cfg = config();
        cfg.parent_selection = policy;
        assert!(
            MeasuredSearchSession::from_measured_parents(
                cfg,
                builtin_catalog().clone(),
                &config().evaluator_context,
                "fixed-pool",
                &seeds(),
                &BTreeSet::new()
            )
            .is_err()
        );
    }
}

#[test]
fn screened_pools_promote_only_measured_winners_and_count_budget_honestly() {
    for bootstrapped in [false, true] {
        let mut s = if bootstrapped {
            start(&BTreeSet::new())
        } else {
            MeasuredSearchSession::new(config(), builtin_catalog().clone()).unwrap()
        };
        let mut total = 0;
        let mut identities = BTreeSet::new();
        while !s.ask().unwrap().is_empty() {
            let pending = s.ask().unwrap();
            total += pending.len();
            for p in &pending {
                assert!(identities.insert(p.semantic_fingerprint.clone()));
            }
            let winner = pending[0].clone();
            let mut scores: Vec<_> = s
                .parents()
                .iter()
                .map(|p| MeasuredOutcome {
                    semantic_fingerprint: p.semantic_fingerprint.clone(),
                    outcome: -1.,
                })
                .collect();
            scores.push(MeasuredOutcome {
                semantic_fingerprint: winner.semantic_fingerprint.clone(),
                outcome: 1.,
            });
            s.tell_screened(&config().evaluator_context, "fixed-pool", &scores)
                .unwrap();
            assert_eq!(s.parents()[0], winner);
            assert!(pending[1..].iter().all(|p| !s.parents().contains(p)));
            assert_eq!(s.measured() + s.screened_out(), total);
            s = restore(&s);
        }
        assert_eq!(s.measured(), 3);
        assert_eq!(s.screened_out(), 8);
        assert_eq!(s.imported(), if bootstrapped { 2 } else { 0 });
    }
}

#[test]
fn screened_snapshot_rejects_partial_parents_unknown_ids_and_invalid_checkpoints() {
    let mut s = start(&BTreeSet::new());
    let pending = s.ask().unwrap();
    let parents = s.parents();
    let scores: Vec<_> = parents
        .iter()
        .map(|p| MeasuredOutcome {
            semantic_fingerprint: p.semantic_fingerprint.clone(),
            outcome: 1.,
        })
        .collect();
    let before = s.checkpoint_json().unwrap();
    // No measured proposal, missing incumbents, unknown identity, duplicate.
    let selected = MeasuredOutcome {
        semantic_fingerprint: pending[0].semantic_fingerprint.clone(),
        outcome: -2.,
    };
    for bad in [
        scores.clone(),
        vec![selected.clone()],
        [
            scores.clone(),
            vec![MeasuredOutcome {
                semantic_fingerprint: "unknown".into(),
                outcome: 1.,
            }],
        ]
        .concat(),
        [scores.clone(), vec![selected.clone(), selected.clone()]].concat(),
    ] {
        assert!(
            s.tell_screened(&config().evaluator_context, "fixed-pool", &bad)
                .is_err()
        );
        assert_eq!(s.checkpoint_json().unwrap(), before);
    }
    let valid = [scores, vec![selected]].concat();
    s.tell_screened(&config().evaluator_context, "fixed-pool", &valid)
        .unwrap();
    assert_eq!(s.parents(), parents); // A measured loser does not displace incumbents.
    let checkpoint = s.checkpoint_value().unwrap();
    for ordinal in [0, 1, 1000] {
        let mut bad = checkpoint.clone();
        bad["screened_out"]
            .as_array_mut()
            .unwrap()
            .push(ordinal.into());
        assert!(
            MeasuredSearchSession::from_checkpoint_json(
                &bad.to_string(),
                config(),
                builtin_catalog().clone()
            )
            .is_err()
        );
    }
}
