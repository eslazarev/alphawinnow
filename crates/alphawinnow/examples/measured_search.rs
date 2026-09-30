//! Minimal ask/tell loop with a synthetic, non-financial objective.
//! Run: `cargo run --example measured_search --no-default-features --locked`

use alphawinnow::{
    Limits, builtin_catalog,
    measured_search::{
        MeasuredExploration, MeasuredMutation, MeasuredOutcome, MeasuredParentSelection,
        MeasuredSearchConfig, MeasuredSearchSession,
    },
    parse_expression,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let catalog = builtin_catalog().clone();
    let config = MeasuredSearchConfig {
        seed: 7,
        limits: Limits::default(),
        max_evaluations: 32,
        max_attempts: 2_000,
        batch_size: 8,
        parent_capacity: 4,
        exploration_every: 4,
        exploration_policy: MeasuredExploration::Grammar,
        mutation_policy: MeasuredMutation::Applicable,
        parent_selection: MeasuredParentSelection::Score,
        evaluator_context: "synthetic-negative-node-count-v1".to_owned(),
    };
    let mut session = MeasuredSearchSession::new(config.clone(), catalog.clone())?;
    while !session.ask()?.is_empty() {
        // Score requests include retained parents for this snapshot policy.
        // A real evaluator must use one consistent dataset/objective/pool context.
        let outcomes = session
            .score_requests()
            .into_iter()
            .map(|proposal| {
                let expression = parse_expression(&proposal.expression)?;
                // Larger is preferred. This toy objective favors small ASTs;
                // it is neither market evidence nor a trading-performance metric.
                let nodes = u32::try_from(expression.node_count())?;
                Ok(MeasuredOutcome {
                    semantic_fingerprint: proposal.semantic_fingerprint,
                    outcome: -f64::from(nodes),
                })
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        session.tell(
            &config.evaluator_context,
            "fixed-toy-objective-v1",
            &outcomes,
        )?;
        // Verify checkpoint restoration with the identical config and catalog.
        session = MeasuredSearchSession::from_checkpoint_json(
            &session.checkpoint_json()?,
            config.clone(),
            catalog.clone(),
        )?;
    }
    println!(
        "{}",
        serde_json::json!({
            "schema": 1,
            "objective": config.evaluator_context,
            "market_evidence": false,
            "attempted": session.attempted(),
            "measured": session.measured(),
            "parents": session.parents(),
        })
    );
    Ok(())
}
