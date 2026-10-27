//! Synthetic benchmark: standalone versus digest-referenced checkpoints.
//! Run with `cargo run --release --no-default-features --example checkpoint_history`.
use std::{collections::BTreeSet, time::Instant};

use alphawinnow::{
    Limits, builtin_catalog,
    measured_search::{
        ExcludedHistory, MeasuredExploration, MeasuredMutation, MeasuredParentSelection,
        MeasuredSearchConfig, MeasuredSearchSession, MeasuredSeed,
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let exclusions: BTreeSet<_> = (0..100_000).map(|i| format!("{i:064x}")).collect();
    let config = MeasuredSearchConfig {
        seed: 71,
        limits: Limits::default(),
        max_evaluations: 32,
        max_attempts: 2000,
        batch_size: 8,
        parent_capacity: 2,
        exploration_every: 4,
        exploration_policy: MeasuredExploration::Grammar,
        mutation_policy: MeasuredMutation::Applicable,
        parent_selection: MeasuredParentSelection::Score,
        evaluator_context: "synthetic-checkpoint-size-v1".into(),
    };
    let seeds: Vec<_> = ["rank(close)", "ts_mean(volume, 20)"]
        .into_iter()
        .enumerate()
        .map(|(i, expression)| MeasuredSeed {
            record_id: format!("synthetic-{i}"),
            expression: expression.into(),
            outcome: -1.,
        })
        .collect();
    let start = Instant::now();
    let mut session = MeasuredSearchSession::from_measured_parents(
        config.clone(),
        builtin_catalog().clone(),
        &config.evaluator_context,
        "fixed-synthetic-pool",
        &seeds,
        &exclusions,
    )?;
    let initialization_us = start.elapsed().as_micros();
    session.ask()?;
    let archive = session.excluded_history().unwrap().to_json()?;
    let start = Instant::now();
    let history = ExcludedHistory::from_json(&archive)?;
    let history_load_us = start.elapsed().as_micros();
    let mut trials = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let standalone = session.checkpoint_json()?;
        let standalone_write_us = start.elapsed().as_micros();
        let start = Instant::now();
        let mut full = MeasuredSearchSession::from_checkpoint_json(
            &standalone,
            config.clone(),
            builtin_catalog().clone(),
        )?;
        let standalone_restore_us = start.elapsed().as_micros();
        let start = Instant::now();
        let compact = session.checkpoint_with_history_json()?;
        let referenced_write_us = start.elapsed().as_micros();
        let start = Instant::now();
        let mut referenced = MeasuredSearchSession::from_checkpoint_with_history_json(
            &compact,
            config.clone(),
            builtin_catalog().clone(),
            &history,
        )?;
        let referenced_restore_us = start.elapsed().as_micros();
        assert_eq!(full.checkpoint_json()?, referenced.checkpoint_json()?);
        assert_eq!(full.ask()?, referenced.ask()?);
        trials.push(serde_json::json!({
            "standalone_bytes": standalone.len(), "referenced_bytes": compact.len(),
            "standalone_write_us": standalone_write_us,
            "standalone_restore_us": standalone_restore_us,
            "referenced_write_us": referenced_write_us,
            "referenced_restore_us": referenced_restore_us,
            "state_and_pending_parity": true,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": 1, "package_version": env!("CARGO_PKG_VERSION"),
            "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
            "profile": "release", "exclusions": exclusions.len(),
            "history_bytes": archive.len(), "history_checksum": history.checksum(),
            "initialization_us": initialization_us, "history_load_us": history_load_us,
            "timing_scope": "in-memory serialization and validation; excludes disk I/O and compilation",
            "comparison": "same current implementation; not a comparison against the previous release",
            "trials": trials,
        }))?
    );
    Ok(())
}
