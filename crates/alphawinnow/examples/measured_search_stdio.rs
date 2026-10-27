//! JSONL bridge showing external use of the measured-search library API.
//! Compile with `cargo build --example measured_search_stdio --no-default-features`.
//! First send init, then alternate ask and tell; each response is one JSON line.

use std::{
    collections::BTreeSet,
    io::{self, BufRead, Write},
};

use alphawinnow::{
    Catalog,
    measured_search::{MeasuredOutcome, MeasuredSearchConfig, MeasuredSearchSession, MeasuredSeed},
};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request {
    Init {
        config: MeasuredSearchConfig,
        catalog: Catalog,
    },
    InitMeasured {
        config: MeasuredSearchConfig,
        catalog: Catalog,
        evaluator_context: String,
        pool_context: String,
        seeds: Vec<MeasuredSeed>,
        excluded: BTreeSet<String>,
    },
    Restore {
        config: MeasuredSearchConfig,
        catalog: Catalog,
        checkpoint: String,
    },
    Ask,
    Tell {
        evaluator_context: String,
        pool_context: String,
        outcomes: Vec<MeasuredOutcome>,
    },
    TellScreened {
        evaluator_context: String,
        pool_context: String,
        outcomes: Vec<MeasuredOutcome>,
    },
    Checkpoint,
}

fn respond(
    session: &mut Option<MeasuredSearchSession>,
    request: Request,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    match request {
        Request::InitMeasured {
            config,
            catalog,
            evaluator_context,
            pool_context,
            seeds,
            excluded,
        } => {
            if session.is_some() {
                return Err("session already initialized".into());
            }
            let initialized = MeasuredSearchSession::from_measured_parents(
                config,
                catalog,
                &evaluator_context,
                &pool_context,
                &seeds,
                &excluded,
            )?;
            let imported = initialized.imported();
            let measured = initialized.measured();
            *session = Some(initialized);
            Ok(json!({"initialized": true, "imported": imported, "measured": measured}))
        }
        Request::Init { config, catalog } => {
            if session.is_some() {
                return Err("session already initialized".into());
            }
            *session = Some(MeasuredSearchSession::new(config, catalog)?);
            Ok(json!({"initialized": true}))
        }
        Request::Restore {
            config,
            catalog,
            checkpoint,
        } => {
            if session.is_some() {
                return Err("session already initialized".into());
            }
            *session = Some(MeasuredSearchSession::from_checkpoint_json(
                &checkpoint,
                config,
                catalog,
            )?);
            Ok(json!({"restored": true}))
        }
        request => {
            let session = session.as_mut().ok_or("initialize the session first")?;
            match request {
                Request::Ask => Ok(
                    json!({"proposals": session.ask()?, "score_requests": session.score_requests(), "attempted": session.attempted(), "measured": session.measured()}),
                ),
                Request::Tell {
                    evaluator_context,
                    pool_context,
                    outcomes,
                } => {
                    session.tell(&evaluator_context, &pool_context, &outcomes)?;
                    Ok(json!({"parents": session.parents(), "measured": session.measured()}))
                }
                Request::TellScreened {
                    evaluator_context,
                    pool_context,
                    outcomes,
                } => {
                    session.tell_screened(&evaluator_context, &pool_context, &outcomes)?;
                    Ok(
                        json!({"parents": session.parents(), "measured": session.measured(),
                        "screened_out": session.screened_out(), "imported": session.imported()}),
                    )
                }
                Request::Checkpoint => Ok(json!({"checkpoint": session.checkpoint_json()?})),
                Request::Init { .. } | Request::InitMeasured { .. } | Request::Restore { .. } => {
                    unreachable!()
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut session = None;
    let mut stdout = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let response = match serde_json::from_str::<Request>(&line?) {
            Ok(request) => {
                respond(&mut session, request).unwrap_or_else(|e| json!({"error": e.to_string()}))
            }
            Err(error) => json!({"error": error.to_string()}),
        };
        writeln!(stdout, "{response}")?;
        stdout.flush()?;
    }
    Ok(())
}
