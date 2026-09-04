//! End-to-end tests for the `alphawinnow` binary.
//!
//! The binary is gated behind the `cli` feature, so `CARGO_BIN_EXE_alphawinnow`
//! only exists when that feature is active.
#![cfg(feature = "cli")]

use std::{fs, process::Command};

use tempfile::tempdir;

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_alphawinnow"))
}

fn candidate_line(expression: &str) -> String {
    serde_json::json!({
        "schema": 1,
        "expression": expression,
        "fingerprint": "",
        "semantic_fingerprint": "",
        "structural_score": 0.0,
        "root_family": "fixture",
        "nodes": 0,
        "depth": 0,
        "provenance": {
            "kind": "initial",
            "operation": "fixture",
            "parent_fingerprints": []
        }
    })
    .to_string()
}

#[test]
fn doctor_reports_ready_json() {
    let output = command().args(["doctor", "--json"]).output().unwrap();
    assert!(output.status.success());
    let payload: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["ready"], true);
    assert_eq!(payload["network_required"], false);
}

#[test]
fn inspect_reports_canonical_semantic_identity() {
    let output = command()
        .args(["inspect", "multiply(close, 2.0)"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let payload: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["canonical"], "multiply(2, close)");
    assert_eq!(payload["semantic_canonical"], "close");
    assert_eq!(payload["semantic_fingerprint"].as_str().unwrap().len(), 64);
}

#[test]
fn inspect_rejects_invalid_expression_with_nonzero_exit() {
    let output = command()
        .args(["inspect", "winsorize(close, std=999)"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid expression"));
}

#[test]
fn inspect_flags_provably_zero_with_a_machine_readable_reason() {
    let output = command()
        .args(["inspect", "subtract(close, close)"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let payload: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["rejection_reason"], "provably_zero");
}

#[test]
fn compile_lowers_and_lints_against_an_immutable_dialect() {
    let directory = tempdir().unwrap();
    let dialect = directory.path().join("dialect.json");
    fs::write(
        &dialect,
        serde_json::json!({
            "schema": 1,
            "name": "fixture-v1",
            "allowed_operators": ["multiply", "rank"],
            "allowed_fields": ["returns"],
            "unary_scalar_lowerings": {
                "negate": {"replacement": "multiply", "scalar": -1}
            }
        })
        .to_string(),
    )
    .unwrap();
    let output = command()
        .args([
            "compile",
            "rank(negate(returns))",
            "--dialect",
            dialect.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        payload["compiled_expression"],
        "rank(multiply(-1, returns))"
    );
    assert_eq!(payload["lowerings_applied"][0], "negate->multiply");
}

#[test]
#[allow(clippy::too_many_lines)]
fn search_writes_candidate_jsonl_and_adjacent_manifest_atomically() {
    let directory = tempdir().unwrap();
    let output_path = directory.path().join("candidates.jsonl");
    let output = command()
        .args([
            "search",
            "--output",
            output_path.to_str().unwrap(),
            "--duration-seconds",
            "10",
            "--max-candidates",
            "300",
            "--threads",
            "2",
            "--seed",
            "7",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let lines: Vec<_> = fs::read_to_string(&output_path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect();
    assert!(!lines.is_empty());
    assert!(lines.iter().all(|record| record["schema"] == 4));
    assert!(lines.iter().all(|record| record["run_id"].is_string()));
    assert!(lines.iter().all(|record| {
        record["score_schema"] == 2
            && record["score_components"]["structural_novelty"].is_number()
            && record["normalized_weights"]["structural_novelty"].is_number()
            && record["structural_descriptor"]["ordered_paths"].is_array()
    }));
    assert!(lines.iter().all(|record| {
        record["provenance"]["retry_count"].is_number()
            && record["provenance"]["new_subtree_fingerprint"].is_string()
    }));
    for record in lines
        .iter()
        .filter(|record| record["provenance"]["kind"] == "crossover")
    {
        assert!(record["provenance"]["affected_path"].is_object());
        assert!(record["provenance"]["old_subtree_fingerprint"].is_string());
        assert_eq!(
            record["provenance"]["parent_fingerprints"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }
    let manifest_path = directory.path().join("candidates.jsonl.manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["schema"], 6);
    assert_eq!(manifest["candidate_schema"], 4);
    assert_eq!(manifest["attempted_candidates"], 300);
    assert_eq!(manifest["rejected_invalid_candidates"], 0);
    assert!(manifest["generations_completed"].as_u64().unwrap() > 1);
    assert!(
        manifest["peak_population"].as_u64().unwrap()
            <= manifest["population_capacity"].as_u64().unwrap()
    );
    let rejected: u64 = manifest["rejection_reasons"]
        .as_object()
        .unwrap()
        .values()
        .map(|value| value.as_u64().unwrap())
        .sum();
    assert_eq!(rejected, manifest["rejected_trivial_candidates"]);
    let accepted_transforms: u64 = manifest["accepted_transform_counts"]
        .as_object()
        .unwrap()
        .values()
        .map(|value| value.as_u64().unwrap())
        .sum();
    assert_eq!(accepted_transforms, manifest["valid_candidates"]);
    let retained_transforms: u64 = manifest["retained_transform_counts"]
        .as_object()
        .unwrap()
        .values()
        .map(|value| value.as_u64().unwrap())
        .sum();
    assert_eq!(retained_transforms, manifest["retained_candidates"]);
    for record in &lines {
        let expression =
            alphawinnow::parse_expression(record["expression"].as_str().unwrap()).unwrap();
        assert_eq!(
            alphawinnow::analyze_expression(&expression).rejection_reason,
            None
        );
    }
    let (published, published_manifest) = alphawinnow::read_published_run(&output_path).unwrap();
    assert_eq!(published.len(), lines.len());
    assert_eq!(published_manifest.run_id, manifest["run_id"]);
    assert!(fs::read_dir(directory.path()).unwrap().count() >= 4);
}

#[test]
fn repeated_search_has_matching_candidate_content() {
    let directory = tempdir().unwrap();
    let left = directory.path().join("left.jsonl");
    let right = directory.path().join("right.jsonl");
    for path in [&left, &right] {
        let output = command()
            .args([
                "search",
                "--output",
                path.to_str().unwrap(),
                "--duration-seconds",
                "10",
                "--max-candidates",
                "500",
                "--threads",
                "2",
                "--seed",
                "99",
            ])
            .output()
            .unwrap();
        assert!(output.status.success());
    }
    assert_eq!(fs::read(left).unwrap(), fs::read(right).unwrap());
}

#[test]
fn dedup_removes_scale_equivalent_records() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source.jsonl");
    let duplicate_source = directory.path().join("duplicates.jsonl");
    let unique = directory.path().join("unique.jsonl");
    let search = command()
        .args([
            "search",
            "--output",
            source.to_str().unwrap(),
            "--duration-seconds",
            "10",
            "--max-candidates",
            "20",
            "--seed",
            "5",
        ])
        .output()
        .unwrap();
    assert!(search.status.success());
    let original = fs::read_to_string(&source).unwrap();
    let first = original.lines().next().unwrap();
    fs::write(&duplicate_source, format!("{first}\n{first}\n")).unwrap();
    let output = command()
        .args([
            "dedup",
            "--input",
            duplicate_source.to_str().unwrap(),
            "--output",
            unique.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(fs::read_to_string(unique).unwrap().lines().count(), 1);
    let payload: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["accepted_records"], 1);
    assert_eq!(payload["duplicate_records"], 1);
    assert_eq!(payload["rejected_records"], 0);
}

#[test]
fn dedup_writes_valid_records_and_reports_rejected_and_duplicate_counts() {
    let directory = tempdir().unwrap();
    let source = directory.path().join("source.jsonl");
    let output_path = directory.path().join("unique.jsonl");
    let input = [
        candidate_line("close"),
        candidate_line("subtract(close, close)"),
        candidate_line("multiply(close, 2)"),
        candidate_line("open"),
    ]
    .join("\n");
    fs::write(&source, format!("{input}\n")).unwrap();
    let output = command()
        .args([
            "dedup",
            "--input",
            source.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let payload: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["accepted_records"], 2);
    assert_eq!(payload["duplicate_records"], 1);
    assert_eq!(payload["rejected_records"], 1);
    assert_eq!(payload["rejection_reasons"]["provably_zero"], 1);
    let content = fs::read_to_string(output_path).unwrap();
    assert_eq!(content.lines().count(), 2);
    assert!(
        content.lines().all(|line| {
            serde_json::from_str::<serde_json::Value>(line).unwrap()["schema"] == 4
        })
    );
}

#[test]
fn search_tolerates_a_mixed_archive_and_reports_its_partition() {
    let directory = tempdir().unwrap();
    let archive = directory.path().join("archive.jsonl");
    let candidates = directory.path().join("candidates.jsonl");
    let input = [
        candidate_line("close"),
        candidate_line("subtract(close, close)"),
        candidate_line("multiply(close, 2)"),
        candidate_line("open"),
    ]
    .join("\n");
    fs::write(&archive, format!("{input}\n")).unwrap();
    let output = command()
        .args([
            "search",
            "--archive",
            archive.to_str().unwrap(),
            "--output",
            candidates.to_str().unwrap(),
            "--duration-seconds",
            "10",
            "--max-candidates",
            "300",
            "--seed",
            "17",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["archive_records"], 4);
    assert_eq!(payload["archive_accepted_records"], 2);
    assert_eq!(payload["archive_duplicate_records"], 1);
    assert_eq!(payload["archive_rejected_records"], 1);
    assert_eq!(payload["archive_rejection_reasons"]["provably_zero"], 1);
    let manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(directory.path().join("candidates.jsonl.manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["archive_accepted_records"], 2);
    assert_eq!(manifest["archive_duplicate_records"], 1);
    assert_eq!(manifest["archive_rejected_records"], 1);
}

#[test]
fn benchmark_reports_reproducible_checksums_and_timings() {
    let output = command()
        .args([
            "benchmark",
            "--candidates",
            "300",
            "--threads",
            "1,2",
            "--seed",
            "8",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["trials"].as_array().unwrap().len(), 2);
    assert_eq!(
        payload["trials"][0]["candidate_checksum"],
        payload["trials"][1]["candidate_checksum"]
    );
    assert!(payload["trials"][0]["rejected_trivial_candidates"].is_number());
    assert!(payload["trials"][0]["rejected_invalid_candidates"].is_number());
    assert!(payload["trials"][0]["rejected_duplicate_candidates"].is_number());
    assert!(payload["trials"][0]["wall_milliseconds"].is_number());
}

#[test]
fn search_help_preserves_public_resource_flags() {
    let output = command().args(["search", "--help"]).output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    for flag in [
        "--duration-seconds",
        "--catalog",
        "--max-candidates",
        "--threads",
        "--seed",
        "--max-depth",
        "--max-nodes",
        "--max-operators",
    ] {
        assert!(stdout.contains(flag));
    }
}

#[test]
fn search_records_configurable_expression_limits() {
    let directory = tempdir().unwrap();
    let output = directory.path().join("bounded.jsonl");
    let result = command()
        .args([
            "search",
            "--output",
            output.to_str().unwrap(),
            "--duration-seconds",
            "10",
            "--max-candidates",
            "100",
            "--threads",
            "1",
            "--max-depth",
            "12",
            "--max-nodes",
            "96",
            "--max-operators",
            "48",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(output.with_extension("jsonl.manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["search_spec"]["limits"]["max_depth"], 12);
    assert_eq!(manifest["search_spec"]["limits"]["max_nodes"], 96);
    assert_eq!(manifest["search_spec"]["limits"]["max_operators"], 48);
}

#[test]
fn search_uses_and_records_an_explicit_catalog() {
    let directory = tempdir().unwrap();
    let catalog_path = directory.path().join("catalog.json");
    let output = directory.path().join("custom.jsonl");
    let builtin = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/public-v1.json");
    let mut catalog: alphawinnow::Catalog =
        serde_json::from_slice(&fs::read(builtin).unwrap()).unwrap();
    catalog.name = "custom-cli-fixture".to_owned();
    catalog.fields = vec![alphawinnow::FieldSpec {
        name: "custom_signal".to_owned(),
        kind: alphawinnow::ExprKind::Signal,
        family: "fixture".to_owned(),
        allowed_roles: vec!["signal_input".to_owned()],
    }];
    catalog.groups = vec![alphawinnow::GroupSpec {
        name: "industry".to_owned(),
        kind: alphawinnow::ExprKind::Group,
    }];
    for input in catalog
        .operators
        .iter_mut()
        .flat_map(|operator| operator.inputs.iter_mut())
    {
        if matches!(input.domain, alphawinnow::ValueDomain::Window { .. }) {
            input.domain = alphawinnow::ValueDomain::WindowSet {
                values: vec![5, 20],
            };
        }
    }
    fs::write(&catalog_path, serde_json::to_vec_pretty(&catalog).unwrap()).unwrap();

    let result = command()
        .args([
            "search",
            "--catalog",
            catalog_path.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--duration-seconds",
            "10",
            "--max-candidates",
            "500",
            "--threads",
            "2",
            "--shortlist-size",
            "20",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let candidates = alphawinnow::read_candidates(&output).unwrap();
    assert!(!candidates.is_empty());
    assert!(candidates.iter().all(|candidate| {
        alphawinnow::parse_expression_with_catalog(&candidate.expression, &catalog).is_ok()
            && candidate.expression.contains("custom_signal")
    }));
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(output.with_extension("jsonl.manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["operator_catalog_checksum"], catalog.checksum());
}

#[test]
fn prioritize_keeps_structural_score_separate_from_measured_guidance() {
    let directory = tempdir().unwrap();
    let input = directory.path().join("candidates.jsonl");
    let feedback = directory.path().join("feedback.json");
    let output = directory.path().join("guided.jsonl");
    let mut simple: serde_json::Value = serde_json::from_str(&candidate_line("close")).unwrap();
    simple["semantic_fingerprint"] = "simple".into();
    simple["structural_score"] = 0.9.into();
    simple["nodes"] = 1.into();
    simple["depth"] = 1.into();
    let mut richer: serde_json::Value =
        serde_json::from_str(&candidate_line("rank(ts_mean(close, 20))")).unwrap();
    richer["semantic_fingerprint"] = "richer".into();
    richer["structural_score"] = 0.1.into();
    richer["nodes"] = 8.into();
    richer["depth"] = 4.into();
    fs::write(
        &input,
        format!(
            "{}\n{}\n",
            serde_json::to_string(&simple).unwrap(),
            serde_json::to_string(&richer).unwrap()
        ),
    )
    .unwrap();
    let records = [
        ("simple-a", 1.0, 1.0, -0.9),
        ("simple-b", 2.0, 1.0, -0.8),
        ("simple-c", 2.0, 2.0, -0.7),
        ("rich-a", 7.0, 4.0, 0.7),
        ("rich-b", 8.0, 4.0, 0.8),
        ("rich-c", 9.0, 5.0, 0.9),
    ]
    .into_iter()
    .map(|(record_id, nodes, depth, outcome)| {
        serde_json::json!({
            "record_id": record_id,
            "features": {"depth": depth, "nodes": nodes},
            "outcome": outcome,
            "confidence": 1.0
        })
    })
    .collect::<Vec<_>>();
    fs::write(
        &feedback,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema": 1,
            "dataset_id": "synthetic-cli-feedback-v1",
            "context_checksum": "fixture-context",
            "outcome_label": "bounded synthetic utility",
            "feature_scales": {"depth": 10.0, "nodes": 40.0},
            "records": records
        }))
        .unwrap(),
    )
    .unwrap();
    let result = command()
        .args([
            "prioritize",
            "--input",
            input.to_str().unwrap(),
            "--feedback",
            feedback.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--maximum-distance",
            "0.2",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let summary: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(summary["guided_records"], 2);
    assert_eq!(summary["structural_score_modified"], false);
    let lines = fs::read_to_string(output).unwrap();
    let guided = lines
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(guided[0]["candidate"]["semantic_fingerprint"], "richer");
    assert_eq!(guided[0]["candidate"]["structural_score"], 0.1);
    assert_eq!(
        guided[0]["measured_guidance"]["structural_score_used"],
        false
    );
}

#[test]
fn audit_feedback_reports_leave_one_out_coverage() {
    let feedback = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("examples/public-feedback-v1.json");
    let output = command()
        .args([
            "audit-feedback",
            "--feedback",
            feedback.to_str().unwrap(),
            "--neighbors",
            "3",
            "--minimum-neighbors",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(payload["records"], 6);
    assert!(payload["coverage"].as_f64().unwrap() > 0.0);
    assert_eq!(payload["feedback_checksum"].as_str().unwrap().len(), 64);
}

#[cfg(feature = "numeric-evidence")]
#[test]
fn evaluate_reports_separate_causally_aligned_evidence() {
    let directory = tempfile::tempdir().unwrap();
    let dataset = directory.path().join("dataset.json");
    let mut close = Vec::new();
    let mut realized = vec![Some(0.0), Some(0.0)];
    for value in [
        1.0, -3.0, 2.0, 7.0, -5.0, 4.0, 9.0, -2.0, 6.0, -8.0, 3.0, 5.0,
    ] {
        close.extend([Some(value), Some(-value)]);
        realized.extend([Some(value), Some(-value)]);
    }
    realized.truncate(24);
    let payload = serde_json::json!({
        "schema": 1,
        "dataset_id": "cli-synthetic-v1",
        "timestamps": (0..12).collect::<Vec<_>>(),
        "assets": ["a", "b"],
        "fields": {"close": {"values": close}},
        "realized_returns": {"values": realized},
        "groups": {"sector": ["one", "one"]},
        "source_label": "license-safe synthetic CLI fixture",
        "preprocessing": ["none"]
    });
    fs::write(&dataset, serde_json::to_vec_pretty(&payload).unwrap()).unwrap();
    let output = command()
        .args([
            "evaluate",
            "--dataset",
            dataset.to_str().unwrap(),
            "--expression",
            "close",
            "--horizon-rows",
            "1",
            "--train-end",
            "6",
            "--validation-end",
            "9",
            "--transaction-cost-bps",
            "5",
            "--annualization-rows",
            "252",
            "--brain-proxy",
            "--include-daily",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let evidence: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(evidence["schema"], 1);
    assert_eq!(evidence["structural_score_used"], false);
    assert_eq!(evidence["splits"][0]["observations"], 10);
    assert_eq!(
        evidence["overall"]["brain_proxy"]["methodology"],
        "brain-proxy-v1"
    );
    assert_eq!(
        evidence["overall"]["brain_proxy"]["remote_platform_result"],
        false
    );
    assert!(!evidence["daily"].as_array().unwrap().is_empty());
    assert!(evidence["alignment"].as_str().unwrap().contains("t+1"));
}

#[cfg(feature = "numeric-evidence")]
#[test]
fn evaluate_batch_loads_one_dataset_and_publishes_every_candidate() {
    let directory = tempfile::tempdir().unwrap();
    let dataset = directory.path().join("dataset.json");
    let input = directory.path().join("candidates.jsonl");
    let evidence = directory.path().join("evidence.jsonl");
    let values = (0..24)
        .map(|index| Some(f64::from(index) - 12.0))
        .collect::<Vec<_>>();
    let payload = serde_json::json!({
        "schema": 1,
        "dataset_id": "cli-batch-synthetic-v1",
        "timestamps": (0..12).collect::<Vec<_>>(),
        "assets": ["a", "b"],
        "fields": {"close": {"values": values}},
        "realized_returns": {"values": (0..24).map(|index| Some(if index % 2 == 0 { 0.01 } else { -0.01 })).collect::<Vec<_>>()},
        "groups": {"sector": ["one", "one"]},
        "source_label": "license-safe synthetic CLI batch fixture",
        "preprocessing": ["none"]
    });
    fs::write(&dataset, serde_json::to_vec_pretty(&payload).unwrap()).unwrap();
    fs::write(
        &input,
        format!(
            "{}\n{}\n",
            candidate_line("rank(close)"),
            candidate_line("ts_zscore(close, 2)")
        ),
    )
    .unwrap();
    let output = command()
        .args([
            "evaluate-batch",
            "--dataset",
            dataset.to_str().unwrap(),
            "--input",
            input.to_str().unwrap(),
            "--output",
            evidence.to_str().unwrap(),
            "--threads",
            "2",
            "--horizon-rows",
            "1",
            "--train-end",
            "6",
            "--validation-end",
            "9",
            "--annualization-rows",
            "252",
            "--brain-proxy",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["succeeded"], 2);
    assert_eq!(summary["failed"], 0);
    assert_eq!(summary["threads"], 2);
    let rows = fs::read_to_string(evidence)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|row| row["status"] == "succeeded"));
    assert!(
        rows.iter()
            .all(|row| row["evaluation"].get("daily").is_none())
    );
}
