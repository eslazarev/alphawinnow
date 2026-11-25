use std::collections::BTreeSet;

use alphawinnow::measured_search::{
    MeasuredAskCounters, MeasuredExploration, MeasuredMutation, MeasuredOutcome,
    MeasuredParentSelection, MeasuredSearchConfig, MeasuredSearchSession, MeasuredSearchStatus,
    MeasuredSeed,
};
use alphawinnow::{Limits, builtin_catalog};

fn config() -> MeasuredSearchConfig {
    MeasuredSearchConfig {
        seed: 71,
        limits: Limits::default(),
        max_evaluations: 24,
        max_attempts: 1000,
        batch_size: 8,
        parent_capacity: 2,
        exploration_every: 5,
        exploration_policy: MeasuredExploration::FixedCover,
        mutation_policy: MeasuredMutation::LegacyUniform,
        parent_selection: MeasuredParentSelection::Score,
        evaluator_context: "synthetic-diagnostics-v1".into(),
    }
}

fn new(cfg: &MeasuredSearchConfig) -> MeasuredSearchSession {
    MeasuredSearchSession::new(cfg.clone(), builtin_catalog().clone()).unwrap()
}

fn ids(session: &MeasuredSearchSession) -> Vec<String> {
    session.checkpoint_value().unwrap()["pending"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["semantic_fingerprint"].as_str().unwrap().to_owned())
        .collect()
}

fn score(session: &mut MeasuredSearchSession) {
    let outcomes: Vec<_> = session
        .score_requests()
        .into_iter()
        .map(|p| MeasuredOutcome {
            semantic_fingerprint: p.semantic_fingerprint,
            outcome: 1.0,
        })
        .collect();
    session
        .tell("synthetic-diagnostics-v1", "pool-v1", &outcomes)
        .unwrap();
}

#[test]
fn reports_preserve_proposals_checkpoints_and_pending_retries() {
    let cfg = config();
    let mut plain = new(&cfg);
    let mut reported = new(&cfg);
    assert_eq!(reported.diagnostics().status, MeasuredSearchStatus::Ready);
    while !plain.ask().unwrap().is_empty() {
        let before = reported.attempted();
        let report = reported.ask_with_report().unwrap();
        assert_eq!(report.proposals, plain.ask().unwrap());
        let c = &report.counters;
        assert_eq!(
            c.attempted,
            c.non_signal
                + c.invalid_transform
                + c.analysis_rejected
                + c.excluded_history
                + c.duplicate_seen
                + c.emitted
        );
        assert_eq!(c.attempted, reported.attempted() - before);
        assert_eq!(usize::try_from(c.emitted).unwrap(), report.proposals.len());
        assert!(!report.reused_pending);
        assert_eq!(
            report.diagnostics.status,
            MeasuredSearchStatus::AwaitingFeedback
        );
        let checkpoint = reported.checkpoint_json().unwrap();
        assert_eq!(plain.checkpoint_json().unwrap(), checkpoint);
        let retry = reported.ask_with_report().unwrap();
        assert!(retry.reused_pending);
        assert_eq!(retry.counters, MeasuredAskCounters::default());
        assert_eq!(retry.proposals, report.proposals);
        assert_eq!(reported.checkpoint_json().unwrap(), checkpoint);
        let restored = MeasuredSearchSession::from_checkpoint_json(
            &checkpoint,
            cfg.clone(),
            builtin_catalog().clone(),
        )
        .unwrap();
        assert_eq!(restored.diagnostics(), report.diagnostics);
        score(&mut plain);
        score(&mut reported);
    }
    let report = reported.ask_with_report().unwrap();
    assert!(report.proposals.is_empty());
    assert_eq!(
        report.diagnostics.status,
        MeasuredSearchStatus::ProposalBudgetExhausted
    );
    assert_eq!(
        plain.checkpoint_json().unwrap(),
        reported.checkpoint_json().unwrap()
    );
}

#[test]
fn all_screened_batches_resume_without_measurements_or_refunds() {
    let cfg = config();
    let mut session = new(&cfg);
    let mut all_ids = BTreeSet::new();
    for _ in 0..3 {
        let batch = session.ask().unwrap();
        assert_eq!(batch.len(), 8);
        for p in batch {
            assert!(all_ids.insert(p.semantic_fingerprint));
        }
        let pending = ids(&session);
        session
            .screen_pending(&cfg.evaluator_context, "pool-v1", &pending)
            .unwrap();
        assert_eq!(session.measured(), 0);
        assert!(session.parents().is_empty());
        assert!(
            session
                .screen_pending(&cfg.evaluator_context, "pool-v1", &pending)
                .is_err()
        );
        let checkpoint = session.checkpoint_json().unwrap();
        let restored = MeasuredSearchSession::from_checkpoint_json(
            &checkpoint,
            cfg.clone(),
            builtin_catalog().clone(),
        )
        .unwrap();
        assert_eq!(restored.checkpoint_json().unwrap(), checkpoint);
        session = restored;
    }
    assert!(session.ask().unwrap().is_empty());
    assert_eq!(session.screened_out(), 24);
    assert_eq!(session.diagnostics().remaining_proposals, 0);
}

#[test]
fn screening_preserves_parent_scores_and_rejects_stale_requests_atomically() {
    for policy in [
        MeasuredParentSelection::Score,
        MeasuredParentSelection::ScoreDiverseV1,
    ] {
        let mut cfg = config();
        cfg.parent_selection = policy;
        let mut session = new(&cfg);
        session.ask().unwrap();
        score(&mut session);
        session.ask().unwrap();
        let pending = ids(&session);
        let checkpoint = session.checkpoint_json().unwrap();
        let mut duplicate = pending.clone();
        duplicate[1] = duplicate[0].clone();
        let mut wrong = pending.clone();
        wrong[0] = "not-a-pending-identity".into();
        for supplied in [&pending[..1], &duplicate, &wrong, &[]] {
            assert!(
                session
                    .screen_pending(&cfg.evaluator_context, "pool-v1", supplied)
                    .is_err()
            );
            assert_eq!(session.checkpoint_json().unwrap(), checkpoint);
        }
        for (eval, pool) in [
            ("wrong", "pool-v1"),
            (&*cfg.evaluator_context, "wrong"),
            (&*cfg.evaluator_context, ""),
        ] {
            assert!(session.screen_pending(eval, pool, &pending).is_err());
            assert_eq!(session.checkpoint_json().unwrap(), checkpoint);
        }
        let parents = session.checkpoint_value().unwrap()["parents"].clone();
        let attempts = session.attempted();
        let mut reversed = pending.clone();
        reversed.reverse();
        session
            .screen_pending(&cfg.evaluator_context, "pool-v1", &reversed)
            .unwrap();
        assert_eq!(session.checkpoint_value().unwrap()["parents"], parents);
        assert_eq!(session.attempted(), attempts);
        assert_eq!(session.measured(), 8);
        assert_eq!(session.screened_out(), 8);
        let mut restored = MeasuredSearchSession::from_checkpoint_json(
            &session.checkpoint_json().unwrap(),
            cfg.clone(),
            builtin_catalog().clone(),
        )
        .unwrap();
        assert_eq!(session.ask().unwrap(), restored.ask().unwrap());
        let before = restored.checkpoint_json().unwrap();
        assert!(
            restored
                .screen_pending(&cfg.evaluator_context, "pool-v1", &pending)
                .is_err()
        );
        assert_eq!(restored.checkpoint_json().unwrap(), before);
        score(&mut restored);
        assert_eq!(restored.measured(), 16);
    }
}

#[test]
fn screening_rejects_non_snapshot_policies() {
    for policy in [
        MeasuredParentSelection::Uniform,
        MeasuredParentSelection::LatestBatchScoreV1,
        MeasuredParentSelection::LatestBatchUniformV1,
    ] {
        let mut cfg = config();
        cfg.parent_selection = policy;
        let mut session = new(&cfg);
        session.ask().unwrap();
        let before = session.checkpoint_json().unwrap();
        assert!(
            session
                .screen_pending(&cfg.evaluator_context, "pool-v1", &ids(&session))
                .is_err()
        );
        assert_eq!(session.checkpoint_json().unwrap(), before);
    }
}

#[test]
fn exclusions_explain_empty_batches_and_restore_referenced_history() {
    let mut cfg = config();
    cfg.max_evaluations = 8;
    cfg.max_attempts = 8;
    cfg.exploration_every = 1;
    let mut baseline = new(&cfg);
    let batch = baseline.ask().unwrap();
    assert!(!batch.is_empty());
    let excluded = batch
        .iter()
        .map(|p| p.semantic_fingerprint.clone())
        .collect();
    let mut session = MeasuredSearchSession::from_measured_parents(
        cfg.clone(),
        builtin_catalog().clone(),
        &cfg.evaluator_context,
        "pool-v1",
        &[MeasuredSeed {
            record_id: "seed".into(),
            expression: "rank(close)".into(),
            outcome: 1.0,
        }],
        &excluded,
    )
    .unwrap();
    let report = session.ask_with_report().unwrap();
    assert!(report.proposals.is_empty());
    assert!(report.counters.excluded_history > 0);
    assert_eq!(
        report.diagnostics.status,
        MeasuredSearchStatus::AttemptBudgetExhausted
    );
    assert_eq!(report.diagnostics.emitted, 0);
    assert_eq!(report.diagnostics.imported, 1);
    assert_eq!(report.diagnostics.measured, 0);
    let restored = MeasuredSearchSession::from_checkpoint_with_history_json(
        &session.checkpoint_with_history_json().unwrap(),
        cfg,
        builtin_catalog().clone(),
        session.excluded_history().unwrap(),
    )
    .unwrap();
    assert_eq!(restored.diagnostics(), report.diagnostics);
}

#[test]
fn both_budgets_exhausted_still_requires_pending_feedback() {
    let mut cfg = config();
    cfg.max_evaluations = 1;
    cfg.max_attempts = 1;
    let mut session = new(&cfg);
    assert_eq!(session.ask().unwrap().len(), 1);
    assert_eq!(
        session.diagnostics().status,
        MeasuredSearchStatus::AwaitingFeedback
    );
    assert_eq!(session.diagnostics().remaining_attempts, 0);
    assert_eq!(session.diagnostics().remaining_proposals, 0);
    session
        .screen_pending(&cfg.evaluator_context, "pool-v1", &ids(&session))
        .unwrap();
    assert_eq!(
        session.diagnostics().status,
        MeasuredSearchStatus::BothBudgetsExhausted
    );
}

#[test]
fn bootstrapped_all_screened_batch_preserves_history_and_resumes() {
    let cfg = config();
    let mut session = MeasuredSearchSession::from_measured_parents(
        cfg.clone(),
        builtin_catalog().clone(),
        &cfg.evaluator_context,
        "pool-v1",
        &[MeasuredSeed {
            record_id: "seed".into(),
            expression: "rank(close)".into(),
            outcome: -1.0,
        }],
        &BTreeSet::new(),
    )
    .unwrap();
    session.ask().unwrap();
    let parents = session.checkpoint_value().unwrap()["parents"].clone();
    session
        .screen_pending(&cfg.evaluator_context, "pool-v1", &ids(&session))
        .unwrap();
    assert_eq!(session.imported(), 1);
    assert_eq!(session.measured(), 0);
    assert_eq!(session.screened_out(), 8);
    assert_eq!(session.checkpoint_value().unwrap()["parents"], parents);
    let mut restored = MeasuredSearchSession::from_checkpoint_with_history_json(
        &session.checkpoint_with_history_json().unwrap(),
        cfg,
        builtin_catalog().clone(),
        session.excluded_history().unwrap(),
    )
    .unwrap();
    assert_eq!(restored.diagnostics(), session.diagnostics());
    assert_eq!(
        restored.ask_with_report().unwrap(),
        session.ask_with_report().unwrap()
    );
    score(&mut restored);
    assert_eq!(restored.measured(), 8);
    assert_eq!(restored.screened_out(), 8);
}
