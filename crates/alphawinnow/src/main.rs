use std::{collections::BTreeMap, path::PathBuf, time::Instant};

use alphawinnow::{
    Catalog, CatalogScaffoldPolicy, DialectSpec, FeedbackConfig, FeedbackDataset, Limits,
    OperatorPolicyConfig, ResourceLimits, SEARCH_SPEC_SCHEMA, SearchGuidance, SearchRunOptions,
    SearchSpec, StructuralScoreConfig, analyze_expression, audit_feedback, builtin_catalog,
    canonical, compile_dialect, deduplicate, fingerprint, parse_expression,
    parse_expression_with_catalog, prioritize_candidates, read_candidates, run_search,
    run_search_with_options, run_search_with_options_and_catalog, write_guided_jsonl_atomic,
    write_jsonl_atomic, write_run_transactional,
};
use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::json;

#[cfg(feature = "numeric-evidence")]
use alphawinnow::{
    ColumnarDataset, EvaluationConfig, NumericEvaluator, evaluate_numeric,
    evaluate_numeric_with_catalog,
};
#[cfg(feature = "parquet-input")]
use alphawinnow::{SharadarPanelConfig, prepare_sharadar_panel};

#[derive(Debug, Parser)]
#[command(
    name = "alphawinnow",
    version,
    about = "Local-only structural quantitative-expression search"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Verify that parsing, hashing, and local computation are usable.
    Doctor(DoctorArgs),
    /// Parse, validate, and describe one expression.
    Inspect(InspectArgs),
    /// Remove exact and positive-root-scale duplicates from candidate JSONL.
    Dedup(DedupArgs),
    /// Search a bounded local expression space.
    Search(SearchArgs),
    /// Run a reproducible structural-search workload.
    Benchmark(BenchmarkArgs),
    /// Prioritize candidate JSONL using separate immutable measured feedback.
    Prioritize(PrioritizeArgs),
    /// Audit measured feedback with leave-one-record-out guidance.
    AuditFeedback(AuditFeedbackArgs),
    /// Lower and lint one expression against an immutable target dialect.
    Compile(CompileArgs),
    /// Measure one expression on immutable local columnar JSON.
    #[cfg(feature = "numeric-evidence")]
    Evaluate(EvaluateArgs),
    /// Measure a candidate JSONL stream while loading the dataset only once.
    #[cfg(feature = "numeric-evidence")]
    EvaluateBatch(EvaluateBatchArgs),
    /// Prepare a causal dynamic-universe panel from local Sharadar Parquet.
    #[cfg(feature = "parquet-input")]
    PrepareSharadar(PrepareSharadarArgs),
}

#[derive(Debug, Args)]
struct DoctorArgs {
    /// Emit a machine-readable response.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct InspectArgs {
    /// Optional schema-1 public catalog controlling fields and signatures.
    #[arg(long)]
    catalog: Option<PathBuf>,
    /// Expression to parse and validate.
    expression: String,
}

#[derive(Debug, Args)]
struct DedupArgs {
    /// Versioned candidate JSONL input.
    #[arg(long)]
    input: PathBuf,
    /// Atomic versioned candidate JSONL output.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Clone, Args)]
struct SearchArgs {
    /// Optional schema-1 public catalog JSON controlling the active grammar.
    #[arg(long)]
    catalog: Option<PathBuf>,
    /// Optional measured-candidate archive in versioned JSONL format.
    #[arg(long)]
    archive: Option<PathBuf>,
    /// Seed the first parent population from the accepted archive records.
    #[arg(long, requires = "archive")]
    seed_from_archive: bool,
    /// Keep the initial guided archive parent pool fixed across generations.
    #[arg(long, requires_all = ["archive", "feedback", "seed_from_archive"])]
    freeze_seed_parents: bool,
    /// Use a balanced seed-keyed initial cover from the wider typed catalog scaffold.
    #[arg(long, conflicts_with = "catalog_scaffold_policy")]
    seeded_catalog_scaffold: bool,
    /// Select the initial catalog-cover policy; the legacy seeded flag remains supported.
    #[arg(long, value_enum, default_value_t = CliCatalogScaffoldPolicy::Fixed)]
    catalog_scaffold_policy: CliCatalogScaffoldPolicy,
    /// Immutable schema-1 measured feedback used for parent selection.
    #[arg(long)]
    feedback: Option<PathBuf>,
    /// Maximum nearest neighbors used for search guidance.
    #[arg(long, default_value_t = 8)]
    feedback_neighbors: usize,
    /// Minimum comparable neighbors required to guide one candidate.
    #[arg(long, default_value_t = 3)]
    feedback_minimum_neighbors: usize,
    /// Maximum normalized feedback distance in [0, 1].
    #[arg(long, default_value_t = 0.75)]
    feedback_maximum_distance: f64,
    /// Destination for shortlisted candidates in versioned JSONL format.
    #[arg(long)]
    output: PathBuf,
    /// Hard wall-clock limit for the run.
    #[arg(long, default_value_t = 3_600)]
    duration_seconds: u64,
    /// Hard candidate-generation ceiling.
    #[arg(long, default_value_t = 1_000_000)]
    max_candidates: u64,
    /// Bounded Rayon worker count.
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Seed controlling deterministic generation.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Maximum number of diverse candidates retained.
    #[arg(long, default_value_t = 100)]
    shortlist_size: usize,
    /// Maximum expression-tree depth.
    #[arg(long, default_value_t = 6)]
    max_depth: usize,
    /// Maximum expression-tree node count.
    #[arg(long, default_value_t = 40)]
    max_nodes: usize,
    /// Maximum operator count per expression.
    #[arg(long, default_value_t = 16)]
    max_operators: usize,
    /// Maximum resident candidates retained between generations.
    #[arg(long, default_value_t = 512)]
    population_capacity: usize,
    /// Highest-ranked resident candidates eligible to become parents.
    #[arg(long, default_value_t = 128)]
    parent_pool_capacity: usize,
    /// Reserve this many parent slots for the global measured-feedback ranking.
    #[arg(long, default_value_t = 0, requires = "feedback")]
    feedback_elite_parent_count: usize,
    /// Opt in to feedback-derived operator generation weights with this strength in [0, 1].
    #[arg(long, requires = "feedback")]
    feedback_operator_policy_strength: Option<f64>,
    /// Minimum multiplier preserving exploration for every enabled operator.
    #[arg(long, default_value_t = 0.5)]
    feedback_operator_exploration_floor: f64,
    /// Maximum feedback-derived operator-weight multiplier.
    #[arg(long, default_value_t = 1.5)]
    feedback_operator_maximum_multiplier: f64,
    /// Minimum measured records containing an operator before its weight may change.
    #[arg(long, default_value_t = 3)]
    feedback_operator_minimum_support: usize,
    /// Integer scale retaining deterministic fractional weight resolution.
    #[arg(long, default_value_t = 100)]
    feedback_operator_weight_scale: u16,
    /// Optional immutable candidate archive used only to estimate transform-action rewards.
    #[arg(long, requires = "feedback_operator_policy_strength")]
    feedback_action_archive: Option<PathBuf>,
    /// Durable checkpoint updated after each completed generation.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Resume from a checkpoint created with the identical configuration.
    #[arg(long)]
    resume_from: Option<PathBuf>,
    /// Gracefully pause after this many completed generations.
    #[arg(long, hide = true)]
    pause_after_generations: Option<u64>,
}

#[derive(Debug, Default, Clone, Copy, ValueEnum)]
enum CliCatalogScaffoldPolicy {
    #[default]
    Fixed,
    SeededDiverse,
    MotifDiverse,
}

impl From<CliCatalogScaffoldPolicy> for CatalogScaffoldPolicy {
    fn from(value: CliCatalogScaffoldPolicy) -> Self {
        match value {
            CliCatalogScaffoldPolicy::Fixed => Self::Fixed,
            CliCatalogScaffoldPolicy::SeededDiverse => Self::SeededDiverse,
            CliCatalogScaffoldPolicy::MotifDiverse => Self::MotifDiverse,
        }
    }
}

#[derive(Debug, Args)]
struct BenchmarkArgs {
    /// Generated-candidate count per trial.
    #[arg(long, default_value_t = 10_000)]
    candidates: u64,
    /// Comma-separated Rayon thread counts.
    #[arg(long, value_delimiter = ',', default_value = "1,2")]
    threads: Vec<usize>,
    /// Workload seed.
    #[arg(long, default_value_t = 20260828)]
    seed: u64,
    /// Optional comma-separated seeds; when supplied, replaces --seed.
    #[arg(long, value_delimiter = ',')]
    seeds: Option<Vec<u64>>,
    /// Repetitions for every seed/thread pair.
    #[arg(long, default_value_t = 1)]
    repetitions: usize,
}

#[derive(Debug, Args)]
struct PrioritizeArgs {
    /// Candidate-schema JSONL to prioritize.
    #[arg(long)]
    input: PathBuf,
    /// Immutable schema-1 measured-feedback JSON.
    #[arg(long)]
    feedback: PathBuf,
    /// Atomic guided-candidate JSONL output.
    #[arg(long)]
    output: PathBuf,
    /// Maximum nearest neighbors used for one estimate.
    #[arg(long, default_value_t = 8)]
    neighbors: usize,
    /// Minimum comparable neighbors required to emit guidance.
    #[arg(long, default_value_t = 3)]
    minimum_neighbors: usize,
    /// Maximum normalized feature distance in [0, 1].
    #[arg(long, default_value_t = 0.75)]
    maximum_distance: f64,
}

#[derive(Debug, Args)]
struct AuditFeedbackArgs {
    /// Immutable schema-1 measured-feedback JSON.
    #[arg(long)]
    feedback: PathBuf,
    /// Maximum nearest neighbors used for one estimate.
    #[arg(long, default_value_t = 8)]
    neighbors: usize,
    /// Minimum comparable neighbors required to emit guidance.
    #[arg(long, default_value_t = 3)]
    minimum_neighbors: usize,
    /// Maximum normalized feature distance in [0, 1].
    #[arg(long, default_value_t = 0.75)]
    maximum_distance: f64,
}

#[derive(Debug, Args)]
struct CompileArgs {
    /// Catalog-valid expression to lower.
    expression: String,
    /// Immutable schema-1 target dialect JSON.
    #[arg(long)]
    dialect: PathBuf,
}

#[cfg(feature = "numeric-evidence")]
#[derive(Debug, Args)]
struct EvaluateArgs {
    /// Immutable schema-1 columnar dataset JSON.
    #[arg(long)]
    dataset: PathBuf,
    /// Optional schema-1 catalog declaring dataset-specific fields.
    #[arg(long)]
    catalog: Option<PathBuf>,
    /// Catalog-valid expression to evaluate.
    #[arg(long)]
    expression: String,
    /// Strictly future target horizon in rows.
    #[arg(long)]
    horizon_rows: usize,
    /// First scored row; earlier dataset rows remain available as warm-up.
    #[arg(long, default_value_t = 0)]
    evaluation_start: usize,
    /// Exclusive final row of the train split.
    #[arg(long)]
    train_end: usize,
    /// Exclusive final row of the validation split; holdout follows.
    #[arg(long)]
    validation_end: usize,
    /// One-way turnover cost in basis points.
    #[arg(long, default_value_t = 0.0)]
    transaction_cost_bps: f64,
    /// Optional rows-per-year metadata; no annualized claim is fabricated.
    #[arg(long)]
    annualization_rows: Option<u32>,
    /// Causal platform-style linear decay applied once at the final signal root.
    #[arg(long, default_value_t = 0)]
    signal_decay: usize,
    /// Maximum absolute normalized position weight; zero disables truncation.
    #[arg(long, default_value_t = 0.0)]
    truncation: f64,
    /// Include continuous per-row gross/net `PnL` and turnover evidence.
    #[arg(long)]
    include_daily: bool,
    /// Emit explicitly labelled local estimates using BRAIN-like metric formulas.
    #[arg(long)]
    brain_proxy: bool,
}

#[cfg(feature = "numeric-evidence")]
#[derive(Debug, Args)]
struct EvaluateBatchArgs {
    /// Immutable schema-1 columnar dataset JSON.
    #[arg(long)]
    dataset: PathBuf,
    /// Optional schema-1 catalog declaring dataset-specific fields.
    #[arg(long)]
    catalog: Option<PathBuf>,
    /// Candidate-schema JSONL to evaluate in input order.
    #[arg(long)]
    input: PathBuf,
    /// New atomic JSONL containing candidate identity and measured evidence.
    #[arg(long)]
    output: PathBuf,
    /// Optional input-record ceiling; zero evaluates the complete stream.
    #[arg(long, default_value_t = 0)]
    max_records: usize,
    /// Bounded expression workers; output order remains identical to input.
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Strictly future target horizon in rows.
    #[arg(long)]
    horizon_rows: usize,
    /// First scored row; earlier dataset rows remain available as warm-up.
    #[arg(long, default_value_t = 0)]
    evaluation_start: usize,
    /// Exclusive final row of the train split.
    #[arg(long)]
    train_end: usize,
    /// Exclusive final row of the validation split; holdout follows.
    #[arg(long)]
    validation_end: usize,
    /// One-way turnover cost in basis points.
    #[arg(long, default_value_t = 0.0)]
    transaction_cost_bps: f64,
    /// Optional rows-per-year metadata.
    #[arg(long)]
    annualization_rows: Option<u32>,
    /// Causal platform-style linear decay applied once at the final signal root.
    #[arg(long, default_value_t = 0)]
    signal_decay: usize,
    /// Maximum absolute normalized position weight; zero disables truncation.
    #[arg(long, default_value_t = 0.0)]
    truncation: f64,
    /// Include continuous per-row gross/net `PnL` and turnover evidence.
    #[arg(long)]
    include_daily: bool,
    /// Emit explicitly labelled local estimates using BRAIN-like formulas.
    #[arg(long)]
    brain_proxy: bool,
}

#[cfg(feature = "parquet-input")]
#[derive(Debug, Args)]
struct PrepareSharadarArgs {
    /// Directory containing the partitioned `daily` and `stocks` tables.
    #[arg(long)]
    parquet_root: PathBuf,
    /// First loaded date; use it for causal operator warm-up.
    #[arg(long)]
    start_date: chrono::NaiveDate,
    /// Last loaded date, inclusive.
    #[arg(long)]
    end_date: chrono::NaiveDate,
    /// First scored date reported as an evaluator row index.
    #[arg(long)]
    evaluation_start_date: chrono::NaiveDate,
    /// Exclusive train boundary reported as an evaluator row index.
    #[arg(long)]
    train_end_date: chrono::NaiveDate,
    /// Exclusive validation boundary reported as an evaluator row index.
    #[arg(long)]
    validation_end_date: chrono::NaiveDate,
    /// Daily positive-finite market-cap universe size.
    #[arg(long, default_value_t = 3000)]
    universe_size: usize,
    /// Immutable dataset identifier; a deterministic default is generated.
    #[arg(long)]
    dataset_id: Option<String>,
    /// New schema-1 row-major JSON dataset.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Doctor(args) => {
            doctor(&args);
            Ok(())
        }
        Command::Inspect(args) => inspect(&args),
        Command::Dedup(args) => dedup(&args),
        Command::Search(args) => search(&args),
        Command::Benchmark(args) => benchmark(&args),
        Command::Prioritize(args) => prioritize(&args),
        Command::AuditFeedback(args) => audit_feedback_command(&args),
        Command::Compile(args) => compile_command(&args),
        #[cfg(feature = "numeric-evidence")]
        Command::Evaluate(args) => evaluate(&args),
        #[cfg(feature = "numeric-evidence")]
        Command::EvaluateBatch(args) => evaluate_batch(&args),
        #[cfg(feature = "parquet-input")]
        Command::PrepareSharadar(args) => prepare_sharadar(&args),
    }
}

fn compile_command(args: &CompileArgs) -> Result<()> {
    let bytes = std::fs::read(&args.dialect)
        .with_context(|| format!("cannot read dialect {}", args.dialect.display()))?;
    let dialect: DialectSpec =
        serde_json::from_slice(&bytes).context("invalid target dialect JSON")?;
    let artifact = compile_dialect(&args.expression, &dialect)
        .context("target dialect compatibility check failed")?;
    println!("{}", serde_json::to_string_pretty(&artifact)?);
    Ok(())
}

fn audit_feedback_command(args: &AuditFeedbackArgs) -> Result<()> {
    let bytes = std::fs::read(&args.feedback)
        .with_context(|| format!("cannot read feedback {}", args.feedback.display()))?;
    let feedback: FeedbackDataset =
        serde_json::from_slice(&bytes).context("invalid measured-feedback JSON")?;
    let config = FeedbackConfig {
        schema: 1,
        neighbors: args.neighbors,
        minimum_neighbors: args.minimum_neighbors,
        maximum_distance: args.maximum_distance,
    };
    let report = audit_feedback(&feedback, &config).context("feedback audit failed")?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

fn prioritize(args: &PrioritizeArgs) -> Result<()> {
    refuse_same_path(&args.input, &args.output)?;
    refuse_same_path(&args.feedback, &args.output)?;
    let candidates = read_candidates(&args.input).context("cannot read candidate input")?;
    let bytes = std::fs::read(&args.feedback)
        .with_context(|| format!("cannot read feedback {}", args.feedback.display()))?;
    let feedback: FeedbackDataset =
        serde_json::from_slice(&bytes).context("invalid measured-feedback JSON")?;
    let config = FeedbackConfig {
        schema: 1,
        neighbors: args.neighbors,
        minimum_neighbors: args.minimum_neighbors,
        maximum_distance: args.maximum_distance,
    };
    let guided = prioritize_candidates(&candidates, &feedback, &config)
        .context("cannot prioritize candidates")?;
    let guided_count = guided
        .iter()
        .filter(|record| record.measured_guidance.is_some())
        .count();
    let checksum = write_guided_jsonl_atomic(&args.output, &guided)
        .context("cannot write guided-candidate output")?;
    println!(
        "{}",
        json!({
            "schema": 1,
            "input_records": candidates.len(),
            "guided_records": guided_count,
            "unguided_records": guided.len() - guided_count,
            "feedback_dataset_id": feedback.dataset_id,
            "feedback_context_checksum": feedback.context_checksum,
            "feedback_checksum": feedback.checksum(),
            "structural_score_modified": false,
            "content_checksum": checksum,
            "output": args.output,
        })
    );
    Ok(())
}

#[cfg(feature = "numeric-evidence")]
fn evaluate(args: &EvaluateArgs) -> Result<()> {
    let bytes = std::fs::read(&args.dataset)
        .with_context(|| format!("cannot read dataset {}", args.dataset.display()))?;
    let dataset: ColumnarDataset =
        serde_json::from_slice(&bytes).context("invalid columnar dataset JSON")?;
    let config = EvaluationConfig {
        schema: 1,
        horizon_rows: args.horizon_rows,
        evaluation_start: args.evaluation_start,
        train_end: args.train_end,
        validation_end: args.validation_end,
        transaction_cost_bps: args.transaction_cost_bps,
        annualization_rows: args.annualization_rows,
        signal_decay: args.signal_decay,
        truncation: args.truncation,
        include_daily: args.include_daily,
        brain_proxy: args.brain_proxy,
    };
    let catalog = load_optional_catalog(args.catalog.as_ref())?;
    let artifact = if let Some(catalog) = &catalog {
        evaluate_numeric_with_catalog(&args.expression, &dataset, &config, catalog)
    } else {
        evaluate_numeric(&args.expression, &dataset, &config)
    }
    .context("numeric evaluation failed")?;
    println!("{}", serde_json::to_string_pretty(&artifact)?);
    Ok(())
}

#[cfg(feature = "numeric-evidence")]
fn evaluate_batch(args: &EvaluateBatchArgs) -> Result<()> {
    use std::io::{BufWriter, Write};

    use rayon::prelude::*;

    refuse_same_path(&args.input, &args.output)?;
    if args.output.exists() {
        bail!("refusing to overwrite {}", args.output.display());
    }
    let bytes = std::fs::read(&args.dataset)
        .with_context(|| format!("cannot read dataset {}", args.dataset.display()))?;
    let dataset: ColumnarDataset =
        serde_json::from_slice(&bytes).context("invalid columnar dataset JSON")?;
    let config = EvaluationConfig {
        schema: 1,
        horizon_rows: args.horizon_rows,
        evaluation_start: args.evaluation_start,
        train_end: args.train_end,
        validation_end: args.validation_end,
        transaction_cost_bps: args.transaction_cost_bps,
        annualization_rows: args.annualization_rows,
        signal_decay: args.signal_decay,
        truncation: args.truncation,
        include_daily: args.include_daily,
        brain_proxy: args.brain_proxy,
    };
    let catalog = load_optional_catalog(args.catalog.as_ref())?;
    let evaluator = if let Some(catalog) = &catalog {
        NumericEvaluator::new_with_catalog(&dataset, config, catalog)
    } else {
        NumericEvaluator::new(&dataset, config)
    }
    .context("numeric evaluator initialization failed")?;
    if args.threads == 0 || args.threads > alphawinnow::MAX_THREADS {
        bail!("threads must be in 1..={}", alphawinnow::MAX_THREADS);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()
        .context("cannot build numeric evaluation worker pool")?;
    let mut candidates = read_candidates(&args.input).context("cannot read candidate input")?;
    if args.max_records > 0 {
        candidates.truncate(args.max_records);
    }

    let parent = args
        .output
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("cannot create output directory {}", parent.display()))?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("cannot create temporary file in {}", parent.display()))?;
    let mut succeeded = 0_usize;
    let mut failed = 0_usize;
    {
        let mut writer = BufWriter::new(temporary.as_file_mut());
        // One chunk bounds simultaneously materialized signal/PnL artifacts
        // while Rayon evaluates expressions independently. Indexed parallel
        // collection preserves deterministic input order.
        for chunk in candidates.chunks(args.threads) {
            let records: Vec<_> = pool.install(|| {
                chunk
                    .par_iter()
                    .map(|candidate| numeric_batch_record(&evaluator, candidate))
                    .collect()
            });
            for (success, record) in records {
                if success {
                    succeeded += 1;
                } else {
                    failed += 1;
                }
                serde_json::to_writer(&mut writer, &record)
                    .context("cannot serialize batch evidence")?;
                writer
                    .write_all(b"\n")
                    .context("cannot write batch evidence")?;
            }
        }
        writer.flush().context("cannot flush batch evidence")?;
    }
    temporary
        .as_file_mut()
        .sync_all()
        .context("cannot sync batch evidence")?;
    temporary
        .persist(&args.output)
        .map_err(|error| error.error)
        .with_context(|| format!("cannot publish {}", args.output.display()))?;
    println!(
        "{}",
        json!({
            "schema": 1,
            "input_records": candidates.len(),
            "succeeded": succeeded,
            "failed": failed,
            "signal_decay": args.signal_decay,
            "truncation": args.truncation,
            "threads": args.threads,
            "output": args.output,
        })
    );
    Ok(())
}

#[cfg(feature = "numeric-evidence")]
fn numeric_batch_record(
    evaluator: &NumericEvaluator<'_>,
    candidate: &alphawinnow::CandidateRecord,
) -> (bool, serde_json::Value) {
    match evaluator.evaluate(&candidate.expression) {
        Ok(evaluation) => (
            true,
            json!({
                "schema": 1,
                "status": "succeeded",
                "candidate": {
                    "expression": candidate.expression,
                    "fingerprint": candidate.fingerprint,
                    "semantic_fingerprint": candidate.semantic_fingerprint,
                    "structural_score": candidate.structural_score,
                },
                "evaluation": evaluation,
            }),
        ),
        Err(error) => (
            false,
            json!({
                "schema": 1,
                "status": "failed",
                "candidate": {
                    "expression": candidate.expression,
                    "fingerprint": candidate.fingerprint,
                    "semantic_fingerprint": candidate.semantic_fingerprint,
                    "structural_score": candidate.structural_score,
                },
                "error": error.to_string(),
            }),
        ),
    }
}

#[cfg(feature = "parquet-input")]
fn prepare_sharadar(args: &PrepareSharadarArgs) -> Result<()> {
    use std::io::Write;

    if args.output.exists() {
        bail!("refusing to overwrite {}", args.output.display());
    }
    if !(args.start_date < args.evaluation_start_date
        && args.evaluation_start_date < args.train_end_date
        && args.train_end_date < args.validation_end_date
        && args.validation_end_date <= args.end_date)
    {
        bail!("require start < evaluation-start < train-end < validation-end <= end");
    }
    let dataset_id = args.dataset_id.clone().unwrap_or_else(|| {
        format!(
            "sharadar-dynamic-top{}-{}-{}-v1",
            args.universe_size,
            args.start_date.format("%Y%m%d"),
            args.end_date.format("%Y%m%d")
        )
    });
    let prepared = prepare_sharadar_panel(&SharadarPanelConfig {
        parquet_root: args.parquet_root.clone(),
        start_date: args.start_date,
        end_date: args.end_date,
        universe_size: args.universe_size,
        dataset_id,
    })
    .context("cannot prepare Sharadar panel")?;
    let boundary = |date: chrono::NaiveDate| -> Result<usize> {
        let key = date.format("%Y%m%d").to_string();
        prepared
            .date_rows
            .range(key..)
            .next()
            .map(|(_, row)| *row)
            .with_context(|| format!("no panel row on or after {date}"))
    };
    let evaluation_start = boundary(args.evaluation_start_date)?;
    let train_end = boundary(args.train_end_date)?;
    let validation_end = boundary(args.validation_end_date)?;

    let parent = args
        .output
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("cannot create output directory {}", parent.display()))?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("cannot create temporary file in {}", parent.display()))?;
    serde_json::to_writer(temporary.as_file_mut(), &prepared.dataset)
        .context("cannot serialize prepared panel")?;
    temporary
        .as_file_mut()
        .write_all(b"\n")
        .context("cannot finalize prepared panel")?;
    temporary
        .as_file_mut()
        .sync_all()
        .context("cannot sync prepared panel")?;
    temporary
        .persist(&args.output)
        .map_err(|error| error.error)
        .with_context(|| format!("cannot publish {}", args.output.display()))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": 1,
            "dataset_id": prepared.dataset.dataset_id,
            "output": args.output,
            "rows": prepared.dataset.timestamps.len(),
            "assets": prepared.dataset.assets.len(),
            "universe_size": args.universe_size,
            "evaluation_start": evaluation_start,
            "train_end": train_end,
            "validation_end": validation_end,
        }))?
    );
    Ok(())
}

fn doctor(args: &DoctorArgs) {
    let expression = parse_expression("rank(ts_delta(close, 5))")
        .expect("built-in doctor expression must remain valid");
    let payload = json!({
        "name": "alphawinnow",
        "ready": true,
        "schema": 1,
        "version": env!("CARGO_PKG_VERSION"),
        "network_required": false,
        "parser_check": canonical(&expression),
    });
    if args.json {
        println!("{payload}");
    } else {
        println!(
            "AlphaWinnow {} is ready (offline)",
            env!("CARGO_PKG_VERSION")
        );
    }
}

fn inspect(args: &InspectArgs) -> Result<()> {
    let catalog = load_optional_catalog(args.catalog.as_ref())?;
    let expression = if let Some(catalog) = &catalog {
        parse_expression_with_catalog(&args.expression, catalog)
    } else {
        parse_expression(&args.expression)
    }
    .context("invalid expression")?;
    let operator_catalog_checksum = catalog
        .as_ref()
        .map_or_else(|| builtin_catalog().checksum(), Catalog::checksum);
    let analysis = analyze_expression(&expression);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": 1,
            "operator_catalog_checksum": operator_catalog_checksum,
            "kind": expression.kind(),
            "canonical": canonical(&expression),
            "semantic_canonical": analysis.semantic_canonical,
            "fingerprint": fingerprint(&expression),
            "semantic_fingerprint": analysis.semantic_fingerprint,
            "rejection_reason": analysis.rejection_reason,
            "nodes": expression.node_count(),
            "depth": expression.depth(),
            "operators": expression.operator_count(),
        }))?
    );
    Ok(())
}

fn dedup(args: &DedupArgs) -> Result<()> {
    refuse_same_path(&args.input, &args.output)?;
    let records = read_candidates(&args.input).context("cannot read dedup input")?;
    let input_count = records.len();
    let result = deduplicate(records).context("cannot deduplicate input")?;
    let rejection_reasons = result.rejection_counts();
    let checksum =
        write_jsonl_atomic(&args.output, &result.accepted).context("cannot write dedup output")?;
    println!(
        "{}",
        json!({
            "schema": 1,
            "input_records": input_count,
            "output_records": result.accepted.len(),
            "accepted_records": result.accepted.len(),
            "duplicate_records": result.duplicates.len(),
            "rejected_records": result.rejected.len(),
            "rejection_reasons": rejection_reasons,
            "removed_records": result.duplicates.len() + result.rejected.len(),
            "content_checksum": checksum,
            "output": args.output,
        })
    );
    Ok(())
}

fn search(args: &SearchArgs) -> Result<()> {
    if let Some(archive) = &args.archive
        && !archive.is_file()
    {
        bail!(
            "archive does not exist or is not a file: {}",
            archive.display()
        );
    }
    let spec = search_spec(args);
    spec.validate().context("invalid search configuration")?;
    let guidance = args
        .feedback
        .as_ref()
        .map(|path| {
            let bytes = std::fs::read(path)
                .with_context(|| format!("cannot read feedback {}", path.display()))?;
            let dataset: FeedbackDataset =
                serde_json::from_slice(&bytes).context("invalid measured-feedback JSON")?;
            Ok::<_, anyhow::Error>(SearchGuidance {
                dataset,
                config: FeedbackConfig {
                    schema: 1,
                    neighbors: args.feedback_neighbors,
                    minimum_neighbors: args.feedback_minimum_neighbors,
                    maximum_distance: args.feedback_maximum_distance,
                },
            })
        })
        .transpose()?;
    let options = SearchRunOptions {
        checkpoint_path: args.checkpoint.clone().or_else(|| args.resume_from.clone()),
        resume_from: args.resume_from.clone(),
        pause_after_generations: args.pause_after_generations,
        guidance,
        seed_from_archive: args.seed_from_archive,
        freeze_seed_parents: args.freeze_seed_parents,
        feedback_elite_parent_count: args.feedback_elite_parent_count,
        operator_policy: operator_policy_config(args),
        operator_policy_action_archive: args.feedback_action_archive.clone(),
        catalog_scaffold_policy: if args.seeded_catalog_scaffold {
            CatalogScaffoldPolicy::SeededDiverse
        } else {
            args.catalog_scaffold_policy.into()
        },
    };
    let catalog = args
        .catalog
        .as_ref()
        .map(|path| {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("cannot read catalog {}", path.display()))?;
            Catalog::from_json(&content).map_err(anyhow::Error::msg)
        })
        .transpose()
        .context("invalid search catalog")?;
    let result = if let Some(catalog) = &catalog {
        run_search_with_options_and_catalog(&spec, args.archive.as_deref(), &options, catalog)
    } else {
        run_search_with_options(&spec, args.archive.as_deref(), &options)
    }
    .context("search failed")?;
    let manifest_path = alphawinnow::artifact::manifest_path(&args.output);
    let publication = write_run_transactional(&args.output, &result.candidates, &result.manifest)
        .context("cannot transactionally publish run")?;
    let checksum = publication.candidate_checksum.clone();
    println!(
        "{}",
        json!({
            "schema": 1,
            "output": args.output,
            "manifest": manifest_path,
            "publication_pointer": alphawinnow::publication_pointer_path(&args.output),
            "run_id": result.manifest.run_id,
            "retained_candidates": result.candidates.len(),
            "attempted_candidates": result.manifest.attempted_candidates,
            "rejected_invalid_candidates": result.manifest.rejected_invalid_candidates,
            "termination_reason": result.manifest.termination_reason,
            "rejected_trivial_candidates": result.manifest.rejected_trivial_candidates,
            "rejection_reasons": result.manifest.rejection_reasons,
            "archive_records": result.manifest.archive_records,
            "archive_accepted_records": result.manifest.archive_accepted_records,
            "archive_duplicate_records": result.manifest.archive_duplicate_records,
            "archive_rejected_records": result.manifest.archive_rejected_records,
            "archive_rejection_reasons": result.manifest.archive_rejection_reasons,
            "accepted_transform_counts": result.manifest.accepted_transform_counts,
            "retained_transform_counts": result.manifest.retained_transform_counts,
            "measured_feedback": result.manifest.measured_feedback,
            "operator_policy": result.manifest.operator_policy,
            "guided_population_candidates": result.manifest.guided_population_candidates,
            "seeded_from_archive": result.manifest.seeded_from_archive,
            "frozen_seed_parents": result.manifest.frozen_seed_parents,
            "feedback_elite_parent_count": result.manifest.feedback_elite_parent_count,
            "content_checksum": checksum,
        })
    );
    Ok(())
}

fn operator_policy_config(args: &SearchArgs) -> Option<OperatorPolicyConfig> {
    args.feedback_operator_policy_strength
        .map(|strength| OperatorPolicyConfig {
            schema: 1,
            strength,
            exploration_floor: args.feedback_operator_exploration_floor,
            maximum_multiplier: args.feedback_operator_maximum_multiplier,
            minimum_support: args.feedback_operator_minimum_support,
            weight_scale: args.feedback_operator_weight_scale,
        })
}

fn load_optional_catalog(path: Option<&PathBuf>) -> Result<Option<Catalog>> {
    path.map(|path| {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read catalog {}", path.display()))?;
        Catalog::from_json(&content).map_err(anyhow::Error::msg)
    })
    .transpose()
    .context("invalid catalog")
}

fn benchmark(args: &BenchmarkArgs) -> Result<()> {
    if args.threads.is_empty()
        || args.threads.contains(&0)
        || args.candidates == 0
        || args.repetitions == 0
    {
        bail!("benchmark candidates, thread counts, and repetitions must be greater than zero");
    }
    let seeds = args.seeds.clone().unwrap_or_else(|| vec![args.seed]);
    if seeds.is_empty() {
        bail!("benchmark seeds must not be empty");
    }

    let mut trials = Vec::new();
    let mut expected_checksums = BTreeMap::new();
    let mut zero_invalid_emissions = true;
    let mut bounded_populations = true;
    let mut duplicate_rejection_observed = true;
    for &seed in &seeds {
        for repetition in 0..args.repetitions {
            for &threads in &args.threads {
                let trial = benchmark_trial(seed, repetition + 1, threads, args.candidates)?;
                zero_invalid_emissions &= trial.zero_invalid_emissions;
                bounded_populations &= trial.bounded_population;
                duplicate_rejection_observed &= trial.duplicate_rejection_observed;
                if let Some(expected) = expected_checksums.insert(seed, trial.checksum.clone())
                    && expected != trial.checksum
                {
                    bail!(
                        "determinism failure for seed {seed}: expected {expected}, got {}",
                        trial.checksum
                    );
                }
                trials.push(trial.payload);
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": 1,
            "workload": "deterministic_structural_search",
            "tool_version": env!("CARGO_PKG_VERSION"),
            "environment": {
                "architecture": std::env::consts::ARCH,
                "os": std::env::consts::OS,
            },
            "seed": (seeds.len() == 1).then_some(seeds[0]),
            "seeds": seeds,
            "repetitions": args.repetitions,
            "candidate_ceiling": args.candidates,
            "deterministic_across_repetitions_and_thread_counts": true,
            "engineering_checks": {
                "zero_invalid_emissions": zero_invalid_emissions,
                "bounded_populations": bounded_populations,
                "duplicate_rejection_observed": duplicate_rejection_observed,
            },
            "trials": trials,
        }))?
    );
    Ok(())
}

struct BenchmarkTrial {
    payload: serde_json::Value,
    checksum: String,
    zero_invalid_emissions: bool,
    bounded_population: bool,
    duplicate_rejection_observed: bool,
}

fn benchmark_trial(
    seed: u64,
    repetition: usize,
    threads: usize,
    candidates: u64,
) -> Result<BenchmarkTrial> {
    let spec = SearchSpec {
        schema: SEARCH_SPEC_SCHEMA,
        seed,
        threads,
        max_candidates: candidates,
        duration_seconds: 3_600,
        limits: Limits::default(),
        scoring: StructuralScoreConfig::default(),
        shortlist_size: 100,
        resources: ResourceLimits::default(),
    };
    let started = Instant::now();
    let result = run_search(&spec, None).context("benchmark workload failed")?;
    let elapsed = started.elapsed();
    let manifest = result.manifest;
    let attempted_float = benchmark_count_as_f64(manifest.attempted_candidates)?;
    let invalid_float = benchmark_count_as_f64(manifest.rejected_invalid_candidates)?;
    let valid_float = benchmark_count_as_f64(manifest.valid_candidates)?;
    let trivial_float = benchmark_count_as_f64(manifest.rejected_trivial_candidates)?;
    let duplicate_float = benchmark_count_as_f64(manifest.duplicate_candidates)?;
    let checksum = manifest.candidate_content_checksum.clone();
    let trial = BenchmarkTrial {
        checksum: checksum.clone(),
        zero_invalid_emissions: manifest.rejected_invalid_candidates == 0,
        bounded_population: manifest.peak_population <= manifest.population_capacity,
        duplicate_rejection_observed: manifest.duplicate_candidates > 0,
        payload: json!({
            "seed": seed,
            "repetition": repetition,
            "threads": threads,
            "generated_candidates": manifest.attempted_candidates,
            "attempted_candidates": manifest.attempted_candidates,
            "rejected_invalid_candidates": manifest.rejected_invalid_candidates,
            "invalid_rate": invalid_float / attempted_float,
            "semantic_families": manifest.valid_candidates,
            "valid_candidates": manifest.valid_candidates,
            "semantic_family_yield": valid_float / attempted_float,
            "rejected_trivial_candidates": manifest.rejected_trivial_candidates,
            "trivial_rate": trivial_float / attempted_float,
            "rejection_reasons": manifest.rejection_reasons,
            "rejected_duplicate_candidates": manifest.duplicate_candidates,
            "duplicate_rate": duplicate_float / attempted_float,
            "evaluated_candidates": manifest.valid_candidates,
            "archived_candidates": manifest.valid_candidates,
            "generations_completed": manifest.generations_completed,
            "population_capacity": manifest.population_capacity,
            "peak_population": manifest.peak_population,
            "accepted_transform_counts": manifest.accepted_transform_counts,
            "retained_transform_counts": manifest.retained_transform_counts,
            "retained_candidates": manifest.retained_candidates,
            "candidate_checksum": checksum,
            "wall_milliseconds": elapsed.as_millis(),
            "attempted_candidates_per_second": attempted_float / elapsed.as_secs_f64(),
            "semantic_families_per_second": valid_float / elapsed.as_secs_f64(),
            "configuration": spec,
        }),
    };
    Ok(trial)
}

fn benchmark_count_as_f64(count: u64) -> Result<f64> {
    u32::try_from(count)
        .map(f64::from)
        .context("benchmark count exceeds the supported candidate ceiling")
}

fn search_spec(args: &SearchArgs) -> SearchSpec {
    let limits = Limits {
        max_depth: args.max_depth,
        max_nodes: args.max_nodes,
        max_operators: args.max_operators,
        ..Limits::default()
    };
    let resources = ResourceLimits {
        population_capacity: args.population_capacity,
        parent_pool_capacity: args.parent_pool_capacity,
        ..ResourceLimits::default()
    };
    SearchSpec {
        schema: SEARCH_SPEC_SCHEMA,
        seed: args.seed,
        threads: args.threads,
        max_candidates: args.max_candidates,
        duration_seconds: args.duration_seconds,
        limits,
        scoring: StructuralScoreConfig::default(),
        shortlist_size: args.shortlist_size,
        resources,
    }
}

fn refuse_same_path(input: &std::path::Path, output: &std::path::Path) -> Result<()> {
    if input == output {
        bail!("input and output paths must differ");
    }
    Ok(())
}
