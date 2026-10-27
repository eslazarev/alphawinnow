//! Warm-start, screen, measure, update parents, checkpoint: synthetic only.
use std::collections::BTreeSet;

use alphawinnow::{
    Limits, builtin_catalog,
    measured_search::{
        MeasuredExploration, MeasuredMutation, MeasuredOutcome, MeasuredParentSelection,
        MeasuredSearchConfig, MeasuredSearchSession, MeasuredSeed,
    },
    parse_expression, semantic_fingerprint,
};

fn score(expression: &str) -> Result<f64, Box<dyn std::error::Error>> {
    // A toy external measurement, not a financial metric or surrogate prediction.
    Ok(-f64::from(u32::try_from(
        parse_expression(expression)?.node_count(),
    )?))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let catalog = builtin_catalog().clone();
    let config = MeasuredSearchConfig {
        seed: 7,
        limits: Limits::default(),
        max_evaluations: 32,
        max_attempts: 2000,
        batch_size: 8,
        parent_capacity: 2,
        exploration_every: 4,
        exploration_policy: MeasuredExploration::Grammar,
        mutation_policy: MeasuredMutation::Applicable,
        parent_selection: MeasuredParentSelection::Score,
        evaluator_context: "toy-negative-node-count-v1".into(),
    };
    let seeds = ["rank(close)", "ts_mean(volume, 20)"]
        .into_iter()
        .enumerate()
        .map(|(i, expression)| {
            Ok(MeasuredSeed {
                record_id: format!("toy-{i}"),
                expression: expression.into(),
                outcome: score(expression)?,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let exclusions = BTreeSet::from([semantic_fingerprint(&parse_expression("rank(open)")?)]);
    let mut session = MeasuredSearchSession::from_measured_parents(
        config.clone(),
        catalog.clone(),
        &config.evaluator_context,
        "fixed-toy-objective",
        &seeds,
        &exclusions,
    )?;
    let mut batches = 0;
    while !session.ask()?.is_empty() {
        let pool = session.ask()?;
        // This deterministic selection is illustrative, NOT a predictive model.
        // A real caller persists the selection before invoking its evaluator.
        let selected = pool.into_iter().take(2);
        let outcomes = session
            .parents()
            .into_iter()
            .chain(selected)
            .map(|p| {
                Ok(MeasuredOutcome {
                    outcome: score(&p.expression)?,
                    semantic_fingerprint: p.semantic_fingerprint,
                })
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        session.tell_screened(&config.evaluator_context, "fixed-toy-objective", &outcomes)?;
        session = MeasuredSearchSession::from_checkpoint_json(
            &session.checkpoint_json()?,
            config.clone(),
            catalog.clone(),
        )?;
        batches += 1;
    }
    println!(
        "{}",
        serde_json::json!({"schema": 1, "market_evidence": false,
        "batches": batches, "imported": session.imported(), "measured": session.measured(),
        "screened_out": session.screened_out(), "attempted": session.attempted(),
        "parents": session.parents()})
    );
    Ok(())
}
