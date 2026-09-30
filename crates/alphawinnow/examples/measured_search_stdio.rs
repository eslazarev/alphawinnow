//! JSONL bridge showing external use of the measured-search library API.
//! Compile with `cargo build --example measured_search_stdio --no-default-features`.
//! First send init, then alternate ask and tell; each response is one JSON line.

use std::io::{self, BufRead, Write};

use alphawinnow::{
    Catalog,
    measured_search::{MeasuredOutcome, MeasuredSearchConfig, MeasuredSearchSession},
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
    Checkpoint,
}

fn respond(
    session: &mut Option<MeasuredSearchSession>,
    request: Request,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    match request {
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
                Request::Checkpoint => Ok(json!({"checkpoint": session.checkpoint_json()?})),
                Request::Init { .. } | Request::Restore { .. } => unreachable!(),
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
