use alphawinnow::measured_search::{
    MeasuredExploration, MeasuredMutation, MeasuredOutcome, MeasuredParentSelection,
    MeasuredSearchConfig, MeasuredSearchSession,
};
use alphawinnow::{Limits, builtin_catalog};

fn config() -> MeasuredSearchConfig {
    MeasuredSearchConfig {
        seed: 71,
        limits: Limits::default(),
        max_evaluations: 40,
        max_attempts: 2_000,
        batch_size: 8,
        parent_capacity: 1,
        exploration_every: 5,
        exploration_policy: MeasuredExploration::FixedCover,
        mutation_policy: MeasuredMutation::LegacyUniform,
        parent_selection: MeasuredParentSelection::Score,
        evaluator_context: "synthetic-v1".to_owned(),
    }
}

fn session() -> MeasuredSearchSession {
    MeasuredSearchSession::new(config(), builtin_catalog().clone()).unwrap()
}

#[test]
fn embedded_checkpoint_preserves_observed_float_bits_without_reparse() {
    let mut cfg = config();
    cfg.parent_selection = MeasuredParentSelection::LatestBatchScoreV1;
    let mut session = MeasuredSearchSession::new(cfg, builtin_catalog().clone()).unwrap();
    let proposals = session.ask().unwrap();
    // Exact diagnostic cases that previously shifted by one ULP when the
    // string checkpoint was parsed into Value for embedding in another file.
    for value in [
        0.029_358_551_875_080_758_f64,
        0.027_007_301_935_505_726,
        0.020_855_982_990_714_006,
        0.000_985_977_915_838_431_9,
    ] {
        let mut trial = session.clone();
        let scores: Vec<_> = proposals
            .iter()
            .map(|p| MeasuredOutcome {
                semantic_fingerprint: p.semantic_fingerprint.clone(),
                outcome: value,
            })
            .collect();
        trial.tell("synthetic-v1", "batch-events", &scores).unwrap();
        let embedded = trial.checkpoint_value().unwrap();
        let stored = embedded["parents"][0]["outcome"].as_f64().unwrap();
        assert_eq!(stored.to_bits(), value.to_bits());
        assert_eq!(
            serde_json::to_string(&embedded["parents"][0]["outcome"]).unwrap(),
            serde_json::to_string(&value).unwrap()
        );
        assert!(trial.checkpoint_json().unwrap().contains(&format!(
            "\"outcome\":{}",
            serde_json::to_string(&value).unwrap()
        )));
    }
}

#[test]
fn latest_batch_feedback_drops_incumbents_and_resumes_with_exact_identity() {
    let mut cfg = config();
    cfg.max_evaluations = 26; // Also exercise a short final cohort.
    cfg.parent_capacity = 4;
    cfg.parent_selection = MeasuredParentSelection::LatestBatchScoreV1;
    let catalog = builtin_catalog().clone();
    let mut session = MeasuredSearchSession::new(cfg.clone(), catalog.clone()).unwrap();
    while !session.ask().unwrap().is_empty() {
        let pending = session.ask().unwrap();
        let old = session.parents();
        assert_eq!(session.score_requests(), pending);
        let checkpoint = session.checkpoint_json().unwrap();
        let mut restored =
            MeasuredSearchSession::from_checkpoint_json(&checkpoint, cfg.clone(), catalog.clone())
                .unwrap();
        let values: Vec<_> = pending
            .iter()
            .map(|p| MeasuredOutcome {
                semantic_fingerprint: p.semantic_fingerprint.clone(),
                outcome: -f64::from(u32::try_from(p.ordinal).unwrap()), // Incumbents would win.
            })
            .collect();
        let mut invalid = values.clone();
        invalid[0].outcome = f64::NAN;
        for bad in [invalid, values[..values.len() - 1].to_vec()] {
            assert!(session.tell("synthetic-v1", "events", &bad).is_err());
            assert_eq!(session.checkpoint_json().unwrap(), checkpoint);
        }
        if let Some(parent) = old.first() {
            let mut bad = values.clone();
            bad.push(MeasuredOutcome {
                semantic_fingerprint: parent.semantic_fingerprint.clone(),
                outcome: 1.,
            });
            assert!(session.tell("synthetic-v1", "events", &bad).is_err());
            assert_eq!(session.checkpoint_json().unwrap(), checkpoint);
        }
        let mut alternative = session.clone();
        let flipped: Vec<_> = values
            .iter()
            .map(|v| MeasuredOutcome {
                semantic_fingerprint: v.semantic_fingerprint.clone(),
                outcome: -v.outcome,
            })
            .collect();
        alternative
            .tell("synthetic-v1", "events", &flipped)
            .unwrap();
        session.tell("synthetic-v1", "events", &values).unwrap();
        assert_ne!(alternative.parents(), session.parents());
        let mut reversed = values;
        reversed.reverse();
        restored.tell("synthetic-v1", "events", &reversed).unwrap();
        assert_eq!(session.parents(), pending[..pending.len().min(4)]);
        assert!(session.parents().iter().all(|p| !old.contains(p)));
        assert_eq!(
            session.checkpoint_json().unwrap(),
            restored.checkpoint_json().unwrap()
        );
        let checkpoint = session.checkpoint_json().unwrap();
        session =
            MeasuredSearchSession::from_checkpoint_json(&checkpoint, cfg.clone(), catalog.clone())
                .unwrap();
        let mut wrong = cfg.clone();
        wrong.parent_selection = MeasuredParentSelection::Score;
        assert!(
            MeasuredSearchSession::from_checkpoint_json(&checkpoint, wrong, catalog.clone())
                .is_err()
        );
    }
    assert_eq!(session.measured(), 26);
}

#[test]
fn latest_uniform_is_score_blind_latest_only_and_resumes_partial_batches() {
    let mut cfg = config();
    cfg.max_evaluations = 26;
    cfg.parent_capacity = 4;
    cfg.parent_selection = MeasuredParentSelection::LatestBatchUniformV1;
    cfg.exploration_policy = MeasuredExploration::Grammar;
    cfg.mutation_policy = MeasuredMutation::Applicable;
    let catalog = builtin_catalog().clone();
    let mut left = MeasuredSearchSession::new(cfg.clone(), catalog.clone()).unwrap();
    let mut blind = left.clone();
    while !left.ask().unwrap().is_empty() {
        let pending = left.ask().unwrap();
        assert_eq!(pending, blind.ask().unwrap());
        assert_eq!(left.score_requests(), pending);
        let old = left.parents();
        let before = left.checkpoint_json().unwrap();
        let mut restored =
            MeasuredSearchSession::from_checkpoint_json(&before, cfg.clone(), catalog.clone())
                .unwrap();
        let scores = outcomes(&left, 1.);
        let mut sentinels = scores.clone();
        for value in &mut sentinels {
            value.outcome = -1.;
        }
        sentinels.reverse();
        let mut bad = scores.clone();
        bad[0].outcome = f64::NAN;
        assert!(left.tell("synthetic-v1", "events", &bad).is_err());
        assert_eq!(left.checkpoint_json().unwrap(), before);
        left.tell("synthetic-v1", "events", &scores).unwrap();
        restored.tell("synthetic-v1", "events", &scores).unwrap();
        blind
            .tell("synthetic-v1", "other-events", &sentinels)
            .unwrap();
        assert_eq!(left.parents(), blind.parents());
        assert_eq!(left.parents().len(), pending.len().min(4));
        assert!(
            left.parents()
                .iter()
                .all(|p| pending.contains(p) && !old.contains(p))
        );
        assert!(
            left.parents()
                .windows(2)
                .all(|p| p[0].ordinal < p[1].ordinal)
        );
        assert_eq!(
            left.checkpoint_json().unwrap(),
            restored.checkpoint_json().unwrap()
        );
        left = MeasuredSearchSession::from_checkpoint_json(
            &left.checkpoint_json().unwrap(),
            cfg.clone(),
            catalog.clone(),
        )
        .unwrap();
        blind = MeasuredSearchSession::from_checkpoint_json(
            &blind.checkpoint_json().unwrap(),
            cfg.clone(),
            catalog.clone(),
        )
        .unwrap();
    }
    assert_eq!(left.measured(), 26);
    assert_eq!(blind.measured(), 26);
}

#[test]
fn latest_uniform_checkpoint_rejects_wrong_policy_stale_parents_and_wrong_order() {
    let mut cfg = config();
    cfg.parent_capacity = 4;
    cfg.parent_selection = MeasuredParentSelection::LatestBatchUniformV1;
    let catalog = builtin_catalog().clone();
    let mut session = MeasuredSearchSession::new(cfg.clone(), catalog.clone()).unwrap();
    session.ask().unwrap();
    session
        .tell("synthetic-v1", "first", &outcomes(&session, 1.))
        .unwrap();
    let old = session.checkpoint_value().unwrap()["parents"][0].clone();
    session.ask().unwrap();
    session
        .tell("synthetic-v1", "second", &outcomes(&session, 1.))
        .unwrap();
    let checkpoint = session.checkpoint_json().unwrap();
    assert!(checkpoint.contains("\"parent_selection\":\"latest_batch_uniform_v1\""));
    for policy in [
        MeasuredParentSelection::Uniform,
        MeasuredParentSelection::LatestBatchScoreV1,
    ] {
        let mut wrong = cfg.clone();
        wrong.parent_selection = policy;
        assert!(
            MeasuredSearchSession::from_checkpoint_json(&checkpoint, wrong, catalog.clone())
                .is_err()
        );
    }
    let mut stale = session.checkpoint_value().unwrap();
    stale["parents"][0] = old;
    let mut reversed = session.checkpoint_value().unwrap();
    reversed["parents"].as_array_mut().unwrap().swap(0, 1);
    let mut missing = session.checkpoint_value().unwrap();
    missing["parents"].as_array_mut().unwrap().pop();
    for malformed in [stale, reversed, missing] {
        assert!(
            MeasuredSearchSession::from_checkpoint_json(
                &serde_json::to_string(&malformed).unwrap(),
                cfg.clone(),
                catalog.clone(),
            )
            .is_err()
        );
    }
}

#[test]
fn latest_uniform_is_not_first_four_and_initial_proposals_match_score_policy() {
    let mut selections = std::collections::BTreeSet::new();
    for seed in 0..16 {
        let mut cfg = config();
        cfg.seed = seed;
        cfg.parent_capacity = 4;
        cfg.parent_selection = MeasuredParentSelection::LatestBatchUniformV1;
        let mut control =
            MeasuredSearchSession::new(cfg.clone(), builtin_catalog().clone()).unwrap();
        cfg.parent_selection = MeasuredParentSelection::LatestBatchScoreV1;
        let mut score = MeasuredSearchSession::new(cfg, builtin_catalog().clone()).unwrap();
        assert_eq!(control.ask().unwrap(), score.ask().unwrap());
        control
            .tell("synthetic-v1", "events", &outcomes(&control, 0.))
            .unwrap();
        selections.insert(
            control
                .parents()
                .iter()
                .map(|p| p.ordinal)
                .collect::<Vec<_>>(),
        );
    }
    assert!(selections.len() > 1);
    assert!(selections.iter().any(|s| s != &[1, 2, 3, 4]));
}

#[test]
fn diverse_retention_preserves_score_elites_and_resumes_byte_identically() {
    let mut cfg = config();
    cfg.parent_capacity = 4;
    cfg.parent_selection = MeasuredParentSelection::ScoreDiverseV1;
    let catalog = builtin_catalog().clone();
    let mut left = MeasuredSearchSession::new(cfg.clone(), catalog.clone()).unwrap();
    while !left.ask().unwrap().is_empty() {
        let checkpoint = left.checkpoint_json().unwrap();
        let mut wrong = cfg.clone();
        wrong.parent_selection = MeasuredParentSelection::Score;
        assert!(
            MeasuredSearchSession::from_checkpoint_json(&checkpoint, wrong, catalog.clone())
                .is_err()
        );
        let mut right =
            MeasuredSearchSession::from_checkpoint_json(&checkpoint, cfg.clone(), catalog.clone())
                .unwrap();
        let scores = outcomes(&left, 1.);
        let mut reversed = scores.clone();
        reversed.reverse();
        let mut ranked = left.score_requests();
        ranked.sort_by_key(|p| std::cmp::Reverse(p.ordinal));
        left.tell("synthetic-v1", "pool", &scores).unwrap();
        right.tell("synthetic-v1", "pool", &reversed).unwrap();
        assert_eq!(
            left.checkpoint_json().unwrap(),
            right.checkpoint_json().unwrap()
        );
        assert!(ranked[..2].iter().all(|p| left.parents().contains(p)));
        assert_eq!(left.parents().len(), 4);
        let restored = MeasuredSearchSession::from_checkpoint_json(
            &left.checkpoint_json().unwrap(),
            cfg.clone(),
            catalog.clone(),
        )
        .unwrap();
        assert_eq!(left.parents(), restored.parents());
    }
    assert_eq!(left.measured(), 40);
}

#[test]
fn uniform_retention_is_score_blind_order_independent_and_resumable() {
    let mut cfg = config();
    cfg.parent_capacity = 4;
    cfg.parent_selection = MeasuredParentSelection::Uniform;
    let catalog = builtin_catalog().clone();
    let mut left = MeasuredSearchSession::new(cfg.clone(), catalog.clone()).unwrap();
    let mut right = left.clone();
    let mut selected_sentinel = false;
    while !left.ask().unwrap().is_empty() {
        assert_eq!(left.ask().unwrap(), right.ask().unwrap());
        let checkpoint = left.checkpoint_json().unwrap();
        let mut restored =
            MeasuredSearchSession::from_checkpoint_json(&checkpoint, cfg.clone(), catalog.clone())
                .unwrap();
        let positive = outcomes(&left, 1.);
        let mut blind = outcomes(&right, 0.);
        for s in &mut blind {
            s.outcome = -1.;
        }
        blind.reverse();
        let candidates = left.score_requests();
        left.tell("synthetic-v1", "pool", &positive).unwrap();
        restored.tell("synthetic-v1", "pool", &positive).unwrap();
        right
            .tell("synthetic-v1", "different-pool-context", &blind)
            .unwrap();
        assert_eq!(
            left.checkpoint_json().unwrap(),
            restored.checkpoint_json().unwrap()
        );
        assert_eq!(left.parents(), right.parents());
        assert_eq!(left.parents().len(), 4);
        assert!(left.parents().iter().all(|p| candidates.contains(p)));
        assert!(
            left.parents()
                .windows(2)
                .all(|p| p[0].ordinal < p[1].ordinal)
        );
        selected_sentinel |= !right.parents().is_empty();
        right = MeasuredSearchSession::from_checkpoint_json(
            &right.checkpoint_json().unwrap(),
            cfg.clone(),
            catalog.clone(),
        )
        .unwrap();
    }
    assert_eq!(left.measured(), 40);
    assert!(
        selected_sentinel,
        "numerical rejection scores must not filter the control"
    );
}

#[test]
fn retention_identity_is_opt_in_and_uniform_checkpoint_order_is_validated() {
    let legacy = config();
    let encoded = serde_json::to_string(&legacy).unwrap();
    assert!(!encoded.contains("parent_selection"));
    assert_eq!(
        serde_json::from_str::<MeasuredSearchConfig>(&encoded).unwrap(),
        legacy
    );
    let mut cfg = legacy.clone();
    cfg.parent_capacity = 4;
    cfg.parent_selection = MeasuredParentSelection::Uniform;
    let catalog = builtin_catalog().clone();
    let mut session = MeasuredSearchSession::new(cfg.clone(), catalog.clone()).unwrap();
    session.ask().unwrap();
    session
        .tell("synthetic-v1", "pool", &outcomes(&session, 1.))
        .unwrap();
    let checkpoint = session.checkpoint_json().unwrap();
    assert!(checkpoint.contains("\"parent_selection\":\"uniform\""));
    let mut wrong = cfg.clone();
    wrong.parent_selection = MeasuredParentSelection::Score;
    assert!(
        MeasuredSearchSession::from_checkpoint_json(&checkpoint, wrong, catalog.clone()).is_err()
    );
    let mut malformed: serde_json::Value = serde_json::from_str(&checkpoint).unwrap();
    malformed["parents"].as_array_mut().unwrap().swap(0, 1);
    assert!(
        MeasuredSearchSession::from_checkpoint_json(&malformed.to_string(), cfg, catalog).is_err()
    );
}

#[test]
fn applicable_mutation_avoids_missing_classes_and_preserves_bounds() {
    use alphawinnow::{
        TransformKind, mutate_applicable_with_catalog, parse_expression,
        validate_transformed_with_catalog,
    };
    let catalog = builtin_catalog();
    let limits = Limits::default();
    for source in [
        "close",
        "rank(close)",
        "ts_mean(close, 20)",
        "add(close, volume)",
    ] {
        let parent = parse_expression(source).unwrap();
        for index in 0..128 {
            let result = mutate_applicable_with_catalog(&parent, 9, index, &limits, catalog);
            assert_eq!(
                result,
                mutate_applicable_with_catalog(&parent, 9, index, &limits, catalog)
            );
            assert_eq!(result.1.kind, TransformKind::Mutation);
            assert!(
                ![
                    "mutate_group",
                    "mutate_boolean_literal",
                    "mutate_bounded_keyword"
                ]
                .contains(&result.1.operation.as_str())
            );
            assert_ne!(result.0, parent);
            validate_transformed_with_catalog(&result.0, &limits, catalog).unwrap();
            assert!(result.1.retry_count < 16);
        }
    }
    // A valid one-field catalog provides no alternative for a root field.
    let mut narrow = catalog.clone();
    narrow.fields.retain(|f| f.name == "close");
    narrow.validate().unwrap();
    let parent = parse_expression("close").unwrap();
    let (_, provenance) = mutate_applicable_with_catalog(&parent, 9, 0, &limits, &narrow);
    assert_eq!(provenance.kind, TransformKind::FallbackGeneration);
    assert_eq!(
        provenance.requested_operation.as_deref(),
        Some("applicable_mutation")
    );
}

#[test]
fn mutation_policy_is_opt_in_and_checkpoint_identity_is_enforced() {
    let legacy = config();
    let encoded = serde_json::to_string(&legacy).unwrap();
    assert!(!encoded.contains("mutation_policy"));
    assert_eq!(
        serde_json::from_str::<MeasuredSearchConfig>(&encoded).unwrap(),
        legacy
    );
    let mut cfg = legacy.clone();
    cfg.mutation_policy = MeasuredMutation::Applicable;
    let mut original = MeasuredSearchSession::new(cfg.clone(), builtin_catalog().clone()).unwrap();
    original.ask().unwrap();
    original
        .tell("synthetic-v1", "pool-0", &outcomes(&original, 1.))
        .unwrap();
    original.ask().unwrap();
    let checkpoint = original.checkpoint_json().unwrap();
    assert!(checkpoint.contains("\"mutation_policy\":\"applicable\""));
    assert!(
        MeasuredSearchSession::from_checkpoint_json(&checkpoint, legacy, builtin_catalog().clone())
            .is_err()
    );
    let mut restored =
        MeasuredSearchSession::from_checkpoint_json(&checkpoint, cfg, builtin_catalog().clone())
            .unwrap();
    loop {
        let pending = original.ask().unwrap();
        assert_eq!(pending, restored.ask().unwrap());
        if pending.is_empty() {
            break;
        }
        let scores = outcomes(&original, 1.);
        original.tell("synthetic-v1", "pool", &scores).unwrap();
        restored.tell("synthetic-v1", "pool", &scores).unwrap();
        assert_eq!(
            original.checkpoint_json().unwrap(),
            restored.checkpoint_json().unwrap()
        );
    }
    assert_eq!(original.measured(), 40);
}

fn outcomes(session: &MeasuredSearchSession, direction: f64) -> Vec<MeasuredOutcome> {
    session
        .score_requests()
        .iter()
        .map(|p| MeasuredOutcome {
            semantic_fingerprint: p.semantic_fingerprint.clone(),
            outcome: direction * f64::from(u32::try_from(p.ordinal).unwrap()),
        })
        .collect()
}

#[test]
fn measured_scores_choose_parents_and_change_descendants() {
    let mut left = session();
    let mut right = session();
    let initial = left.ask().unwrap();
    assert_eq!(right.ask().unwrap(), initial);
    assert!(left.parents().is_empty());
    left.tell("synthetic-v1", "pool-0", &outcomes(&left, 1.0))
        .unwrap();
    let other_parent = initial
        .iter()
        .position(|p| p.expression.contains('('))
        .unwrap();
    assert!(other_parent < 7);
    let other_scores = right
        .score_requests()
        .iter()
        .map(|p| MeasuredOutcome {
            semantic_fingerprint: p.semantic_fingerprint.clone(),
            outcome: if p == &initial[other_parent] {
                1.0
            } else {
                0.0
            },
        })
        .collect::<Vec<_>>();
    right.tell("synthetic-v1", "pool-0", &other_scores).unwrap();
    assert_eq!(left.parents()[0], initial[7]);
    assert_eq!(right.parents()[0], initial[other_parent]);
    let left_next = left.ask().unwrap();
    let right_next = right.ask().unwrap();
    assert_ne!(left_next, right_next);
    for (batch, parent) in [
        (&left_next, &initial[7]),
        (&right_next, &initial[other_parent]),
    ] {
        assert!(
            batch.iter().any(|p| {
                p.provenance
                    .parent_fingerprints
                    .contains(&parent.semantic_fingerprint)
            }),
            "parent={parent:?}, batch={batch:?}"
        );
        assert!(batch.iter().all(|p| {
            p.provenance
                .parent_fingerprints
                .iter()
                .all(|id| id == &parent.semantic_fingerprint)
        }));
    }
}

#[test]
fn pending_retry_and_checkpoint_resume_are_byte_identical() {
    let mut original = session();
    original.ask().unwrap();
    original
        .tell("synthetic-v1", "pool-0", &outcomes(&original, 1.0))
        .unwrap();
    let pending = original.ask().unwrap();
    let checkpoint = original.checkpoint_json().unwrap();
    assert_eq!(original.ask().unwrap(), pending);
    assert_eq!(original.checkpoint_json().unwrap(), checkpoint);
    let mut restored = MeasuredSearchSession::from_checkpoint_json(
        &checkpoint,
        config(),
        builtin_catalog().clone(),
    )
    .unwrap();
    let mut seen = std::collections::BTreeSet::new();
    loop {
        let batch = original.ask().unwrap();
        assert_eq!(restored.ask().unwrap(), batch);
        if batch.is_empty() {
            break;
        }
        assert!(
            batch
                .iter()
                .all(|p| seen.insert(p.semantic_fingerprint.clone()))
        );
        let scores = outcomes(&original, 1.0);
        let reversed = scores.iter().rev().cloned().collect::<Vec<_>>();
        original
            .tell("synthetic-v1", "pool-refit", &scores)
            .unwrap();
        restored
            .tell("synthetic-v1", "pool-refit", &reversed)
            .unwrap();
        assert_eq!(
            restored.checkpoint_json().unwrap(),
            original.checkpoint_json().unwrap()
        );
    }
    assert_eq!(original.measured(), 40);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("session.json");
    original.write_checkpoint(&path).unwrap();
    assert_eq!(
        std::fs::read_to_string(path).unwrap(),
        original.checkpoint_json().unwrap()
    );
}

#[test]
fn incomplete_stale_and_nonfinite_measurements_leave_state_unchanged() {
    let mut session = session();
    session.ask().unwrap();
    session
        .tell("synthetic-v1", "pool-0", &outcomes(&session, 1.0))
        .unwrap();
    let pending = session.ask().unwrap();
    let before = session.checkpoint_json().unwrap();
    let scores = outcomes(&session, 1.0);
    // New pool scores must also refresh the previous parent.
    let pending_only = scores
        .iter()
        .filter(|s| {
            pending
                .iter()
                .any(|p| p.semantic_fingerprint == s.semantic_fingerprint)
        })
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        session
            .tell("synthetic-v1", "pool-1", &pending_only)
            .is_err()
    );
    assert!(session.tell("wrong-data", "pool-1", &scores).is_err());
    let mut invalid = scores.clone();
    invalid[0].outcome = f64::NAN;
    assert!(session.tell("synthetic-v1", "pool-1", &invalid).is_err());
    invalid = scores.clone();
    invalid[0] = invalid[1].clone();
    assert!(session.tell("synthetic-v1", "pool-1", &invalid).is_err());
    assert_eq!(session.checkpoint_json().unwrap(), before);
    session.tell("synthetic-v1", "pool-1", &scores).unwrap();
}

#[test]
fn random_control_is_independent_of_measured_outcomes() {
    let mut cfg = config();
    cfg.exploration_every = 1;
    cfg.exploration_policy = MeasuredExploration::Grammar;
    let mut left = MeasuredSearchSession::new(cfg.clone(), builtin_catalog().clone()).unwrap();
    let mut right = MeasuredSearchSession::new(cfg, builtin_catalog().clone()).unwrap();
    loop {
        let batch = left.ask().unwrap();
        assert_eq!(right.ask().unwrap(), batch);
        if batch.is_empty() {
            break;
        }
        left.tell("synthetic-v1", "pool", &outcomes(&left, 1.0))
            .unwrap();
        right
            .tell("synthetic-v1", "pool", &outcomes(&right, -1.0))
            .unwrap();
    }
}

#[test]
fn grammar_control_changes_with_seed_and_resumes_exactly() {
    let mut cfg = config();
    cfg.exploration_policy = MeasuredExploration::Grammar;
    cfg.exploration_every = 1;
    cfg.max_evaluations = 64;
    let mut left = MeasuredSearchSession::new(cfg.clone(), builtin_catalog().clone()).unwrap();
    let mut other = cfg.clone();
    other.seed += 1;
    let mut right = MeasuredSearchSession::new(other, builtin_catalog().clone()).unwrap();
    let mut left_stream = Vec::new();
    let mut right_stream = Vec::new();
    loop {
        let batch = left.ask().unwrap();
        let checkpoint = left.checkpoint_json().unwrap();
        let mut restored = MeasuredSearchSession::from_checkpoint_json(
            &checkpoint,
            cfg.clone(),
            builtin_catalog().clone(),
        )
        .unwrap();
        assert_eq!(restored.ask().unwrap(), batch);
        let other_batch = right.ask().unwrap();
        assert_eq!(batch.len(), other_batch.len());
        if batch.is_empty() {
            break;
        }
        assert!(
            batch
                .iter()
                .chain(&other_batch)
                .all(|p| p.provenance.operation == "grammar_sample")
        );
        left_stream.extend(batch.into_iter().map(|p| p.expression));
        right_stream.extend(other_batch.into_iter().map(|p| p.expression));
        left.tell("synthetic-v1", "pool", &outcomes(&left, 1.0))
            .unwrap();
        restored
            .tell("synthetic-v1", "pool", &outcomes(&restored, 1.0))
            .unwrap();
        assert_eq!(
            left.checkpoint_json().unwrap(),
            restored.checkpoint_json().unwrap()
        );
        right
            .tell("synthetic-v1", "pool", &outcomes(&right, 1.0))
            .unwrap();
    }
    assert_eq!(left_stream.len(), 64);
    assert_ne!(left_stream, right_stream);
}

#[test]
fn missing_policy_preserves_legacy_config_but_changed_policy_rejects_resume() {
    let mut value = serde_json::to_value(config()).unwrap();
    value.as_object_mut().unwrap().remove("exploration_policy");
    let legacy: MeasuredSearchConfig = serde_json::from_value(value).unwrap();
    assert_eq!(legacy, config());
    let checkpoint = session().checkpoint_json().unwrap();
    let mut changed = config();
    changed.exploration_policy = MeasuredExploration::Grammar;
    assert!(
        MeasuredSearchSession::from_checkpoint_json(
            &checkpoint,
            changed,
            builtin_catalog().clone(),
        )
        .is_err()
    );
}

#[test]
fn checkpoint_rejects_different_config_and_corrupt_counters() {
    let mut session = session();
    session.ask().unwrap();
    let checkpoint = session.checkpoint_json().unwrap();
    let mut other = config();
    other.seed += 1;
    assert!(
        MeasuredSearchSession::from_checkpoint_json(&checkpoint, other, builtin_catalog().clone())
            .is_err()
    );
    let mut state: serde_json::Value = serde_json::from_str(&checkpoint).unwrap();
    state["emitted"] = 999_999.into();
    assert!(
        MeasuredSearchSession::from_checkpoint_json(
            &state.to_string(),
            config(),
            builtin_catalog().clone()
        )
        .is_err()
    );
}
