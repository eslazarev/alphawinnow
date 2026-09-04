use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    RejectionReason, SearchSpec, StructuralDescriptor, StructuralScoreWeights,
    canonical::digest,
    transform::{Provenance, TransformKind},
};

pub const CANDIDATE_SCHEMA: u32 = 4;
pub const MANIFEST_SCHEMA: u32 = 6;
static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Portable candidate record. The score contains no market-performance evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub schema: u32,
    #[serde(default)]
    pub run_id: String,
    pub expression: String,
    pub fingerprint: String,
    pub semantic_fingerprint: String,
    pub structural_score: f64,
    #[serde(default)]
    pub score_schema: u32,
    #[serde(default)]
    pub score_components: StructuralScoreComponents,
    #[serde(default)]
    pub normalized_weights: StructuralScoreWeights,
    #[serde(default)]
    pub structural_descriptor: StructuralDescriptor,
    pub root_family: String,
    pub nodes: usize,
    pub depth: usize,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StructuralScoreComponents {
    pub simplicity: f64,
    pub archive_novelty: f64,
    pub family_diversity: f64,
    pub lineage_diversity: f64,
    pub structural_novelty: f64,
    pub transform_diversity: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    pub schema: u32,
    pub run_id: String,
    pub candidate_schema: u32,
    pub operator_catalog_checksum: String,
    pub tool_version: String,
    pub search_spec: SearchSpec,
    pub config_checksum: String,
    pub archive_records: u64,
    pub archive_accepted_records: u64,
    pub archive_duplicate_records: u64,
    pub archive_rejected_records: u64,
    pub archive_rejection_reasons: std::collections::BTreeMap<RejectionReason, u64>,
    pub attempted_candidates: u64,
    pub rejected_invalid_candidates: u64,
    pub valid_candidates: u64,
    pub rejected_trivial_candidates: u64,
    pub rejection_reasons: std::collections::BTreeMap<RejectionReason, u64>,
    pub duplicate_candidates: u64,
    pub retained_candidates: u64,
    pub accepted_transform_counts: std::collections::BTreeMap<TransformKind, u64>,
    pub retained_transform_counts: std::collections::BTreeMap<TransformKind, u64>,
    pub generations_completed: u64,
    pub population_capacity: usize,
    pub peak_population: usize,
    pub elite_archive_capacity: usize,
    pub peak_elite_archive: usize,
    pub semantic_archive_capacity: usize,
    pub peak_semantic_archive: usize,
    pub descriptor_archive_capacity: usize,
    pub peak_descriptor_archive: usize,
    pub elapsed_milliseconds: u128,
    pub termination_reason: String,
    pub candidate_content_checksum: String,
}

pub const CHECKPOINT_SCHEMA: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointDraft {
    pub expression: String,
    pub provenance: Provenance,
    pub semantic_fingerprint: String,
    pub descriptor: StructuralDescriptor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunCheckpoint {
    pub schema: u32,
    pub run_id: String,
    pub config_checksum: String,
    pub operator_catalog_checksum: String,
    pub next_generation: u64,
    pub generations_completed: u64,
    pub attempted_candidates: u64,
    pub rejected_invalid_candidates: u64,
    pub duplicate_candidates: u64,
    pub valid_candidates: u64,
    pub rejection_reasons: std::collections::BTreeMap<RejectionReason, u64>,
    pub accepted_transform_counts: std::collections::BTreeMap<TransformKind, u64>,
    pub semantic_archive: Vec<String>,
    pub elite_archive: Vec<CheckpointDraft>,
    pub population: Vec<CheckpointDraft>,
    pub parent_pool: Vec<CheckpointDraft>,
    pub peak_population: usize,
    pub peak_elite_archive: usize,
    pub peak_semantic_archive: usize,
    pub termination_state: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicationPointer {
    pub schema: u32,
    pub run_id: String,
    pub candidate_path: PathBuf,
    pub manifest_path: PathBuf,
    pub candidate_checksum: String,
    pub manifest_checksum: String,
}

#[derive(Debug, Error)]
pub enum ArtifactError {
    #[error("cannot read artifact {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid JSON on line {line} of {path}: {source}")]
    JsonLine {
        path: PathBuf,
        line: usize,
        source: serde_json::Error,
    },
    #[error("unsupported candidate schema {schema} on line {line} of {path}")]
    Schema {
        path: PathBuf,
        line: usize,
        schema: u32,
    },
    #[error("cannot serialize artifact: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("cannot atomically write artifact {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
}

/// Read a versioned candidate stream.
///
/// # Errors
/// Returns an error for I/O failures, malformed JSON, or unsupported schemas.
pub fn read_candidates(path: &Path) -> Result<Vec<CandidateRecord>, ArtifactError> {
    let file = File::open(path).map_err(|source| ArtifactError::Read {
        path: path.to_owned(),
        source,
    })?;
    let mut records = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = line.map_err(|source| ArtifactError::Read {
            path: path.to_owned(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let record: CandidateRecord =
            serde_json::from_str(&line).map_err(|source| ArtifactError::JsonLine {
                path: path.to_owned(),
                line: line_number,
                source,
            })?;
        if record.schema == 0 || record.schema > CANDIDATE_SCHEMA {
            return Err(ArtifactError::Schema {
                path: path.to_owned(),
                line: line_number,
                schema: record.schema,
            });
        }
        records.push(record);
    }
    Ok(records)
}

/// Serialize records exactly as they will appear in JSONL.
///
/// # Errors
/// Returns an error if a record cannot be serialized.
pub fn candidate_content(records: &[CandidateRecord]) -> Result<String, ArtifactError> {
    let mut content = String::new();
    for record in records {
        content.push_str(&serde_json::to_string(record)?);
        content.push('\n');
    }
    Ok(content)
}

/// Serialize and atomically replace a candidate artifact, returning its checksum.
///
/// # Errors
/// Returns an error on serialization, temporary-file, sync, or rename failures.
pub fn write_jsonl_atomic(
    path: &Path,
    records: &[CandidateRecord],
) -> Result<String, ArtifactError> {
    let content = candidate_content(records)?;
    atomic_write(path, content.as_bytes())?;
    Ok(digest(&content))
}

/// Atomically serialize a run manifest.
///
/// # Errors
/// Returns an error on serialization, temporary-file, sync, or rename failures.
pub fn write_manifest_atomic(path: &Path, manifest: &RunManifest) -> Result<(), ArtifactError> {
    let mut content = serde_json::to_vec_pretty(manifest)?;
    content.push(b'\n');
    atomic_write(path, &content)
}

/// Atomically write a deterministic resume checkpoint.
///
/// # Errors
/// Returns an error on serialization or durable replacement failure.
pub fn write_checkpoint_atomic(
    path: &Path,
    checkpoint: &RunCheckpoint,
) -> Result<(), ArtifactError> {
    let mut content = serde_json::to_vec(checkpoint)?;
    content.push(b'\n');
    atomic_write(path, &content)
}

/// Read a versioned resume checkpoint.
///
/// # Errors
/// Returns an error for I/O, JSON, or unsupported checkpoint schema failures.
pub fn read_checkpoint(path: &Path) -> Result<RunCheckpoint, ArtifactError> {
    let content = fs::read(path).map_err(|source| ArtifactError::Read {
        path: path.to_owned(),
        source,
    })?;
    let checkpoint: RunCheckpoint =
        serde_json::from_slice(&content).map_err(|source| ArtifactError::JsonLine {
            path: path.to_owned(),
            line: 1,
            source,
        })?;
    if checkpoint.schema != CHECKPOINT_SCHEMA {
        return Err(ArtifactError::Schema {
            path: path.to_owned(),
            line: 1,
            schema: checkpoint.schema,
        });
    }
    Ok(checkpoint)
}

#[must_use]
pub fn manifest_path(output: &Path) -> PathBuf {
    let name = output
        .file_name()
        .map_or_else(|| "candidates".into(), std::ffi::OsStr::to_os_string);
    let mut adjacent = name;
    adjacent.push(".manifest.json");
    output.with_file_name(adjacent)
}

#[must_use]
pub fn publication_pointer_path(output: &Path) -> PathBuf {
    let name = output
        .file_name()
        .map_or_else(|| "candidates".into(), std::ffi::OsStr::to_os_string);
    let mut adjacent = name;
    adjacent.push(".current.json");
    output.with_file_name(adjacent)
}

/// Durably publish an immutable candidate/manifest pair, then atomically move
/// one pointer to the complete version. Compatibility aliases are refreshed
/// only after the authoritative pair is durable.
///
/// # Errors
/// Returns an error for inconsistent run IDs, serialization, or filesystem
/// durability failures.
pub fn write_run_transactional(
    output: &Path,
    records: &[CandidateRecord],
    manifest: &RunManifest,
) -> Result<PublicationPointer, ArtifactError> {
    if records
        .iter()
        .any(|record| record.run_id != manifest.run_id)
    {
        return Err(ArtifactError::Write {
            path: output.to_owned(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "candidate and manifest run IDs differ",
            ),
        });
    }
    let candidate_bytes = candidate_content(records)?.into_bytes();
    let candidate_checksum = digest(std::str::from_utf8(&candidate_bytes).unwrap_or_default());
    if candidate_checksum != manifest.candidate_content_checksum {
        return Err(ArtifactError::Write {
            path: output.to_owned(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "candidate checksum does not match manifest",
            ),
        });
    }
    let mut manifest_bytes = serde_json::to_vec_pretty(manifest)?;
    manifest_bytes.push(b'\n');
    let manifest_checksum = digest(std::str::from_utf8(&manifest_bytes).unwrap_or_default());
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let file_name = output.file_name().unwrap_or_default().to_string_lossy();
    let runs = parent.join(format!(".{file_name}.runs"));
    fs::create_dir_all(&runs).map_err(|source| ArtifactError::Write {
        path: runs.clone(),
        source,
    })?;
    let version = format!(
        "{}-{}-{}",
        &manifest.run_id[..manifest.run_id.len().min(16)],
        &candidate_checksum[..candidate_checksum.len().min(16)],
        &manifest_checksum[..manifest_checksum.len().min(16)]
    );
    let final_directory = runs.join(version);
    if !final_directory.exists() {
        let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let staging = runs.join(format!(".stage-{}-{sequence}", std::process::id()));
        fs::create_dir(&staging).map_err(|source| ArtifactError::Write {
            path: staging.clone(),
            source,
        })?;
        let result = (|| -> Result<(), std::io::Error> {
            write_new_file(&staging.join("candidates.jsonl"), &candidate_bytes)?;
            write_new_file(&staging.join("manifest.json"), &manifest_bytes)?;
            File::open(&staging)?.sync_all()?;
            fs::rename(&staging, &final_directory)?;
            File::open(&runs)?.sync_all()?;
            Ok(())
        })();
        if let Err(source) = result {
            let _ = fs::remove_dir_all(&staging);
            return Err(ArtifactError::Write {
                path: final_directory,
                source,
            });
        }
    }
    let pointer = PublicationPointer {
        schema: 1,
        run_id: manifest.run_id.clone(),
        candidate_path: final_directory.join("candidates.jsonl"),
        manifest_path: final_directory.join("manifest.json"),
        candidate_checksum,
        manifest_checksum,
    };
    let mut pointer_bytes = serde_json::to_vec_pretty(&pointer)?;
    pointer_bytes.push(b'\n');
    atomic_write(&publication_pointer_path(output), &pointer_bytes)?;
    write_jsonl_atomic(output, records)?;
    write_manifest_atomic(&manifest_path(output), manifest)?;
    Ok(pointer)
}

/// Read the immutable pair selected by the authoritative publication pointer.
///
/// # Errors
/// Returns an error for a missing/inconsistent pointer, candidate stream, or
/// manifest.
pub fn read_published_run(
    output: &Path,
) -> Result<(Vec<CandidateRecord>, RunManifest), ArtifactError> {
    let pointer_path = publication_pointer_path(output);
    let pointer_bytes = fs::read(&pointer_path).map_err(|source| ArtifactError::Read {
        path: pointer_path.clone(),
        source,
    })?;
    let pointer: PublicationPointer =
        serde_json::from_slice(&pointer_bytes).map_err(|source| ArtifactError::JsonLine {
            path: pointer_path,
            line: 1,
            source,
        })?;
    if pointer.schema != 1 {
        return Err(ArtifactError::Schema {
            path: publication_pointer_path(output),
            line: 1,
            schema: pointer.schema,
        });
    }
    let candidate_bytes =
        fs::read(&pointer.candidate_path).map_err(|source| ArtifactError::Read {
            path: pointer.candidate_path.clone(),
            source,
        })?;
    let records = read_candidates(&pointer.candidate_path)?;
    let manifest_bytes =
        fs::read(&pointer.manifest_path).map_err(|source| ArtifactError::Read {
            path: pointer.manifest_path.clone(),
            source,
        })?;
    let manifest: RunManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|source| ArtifactError::JsonLine {
            path: pointer.manifest_path,
            line: 1,
            source,
        })?;
    if manifest.run_id != pointer.run_id
        || records.iter().any(|record| record.run_id != pointer.run_id)
    {
        return Err(ArtifactError::Write {
            path: output.to_owned(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "published pair has inconsistent run IDs",
            ),
        });
    }
    let candidate_checksum = digest(std::str::from_utf8(&candidate_bytes).unwrap_or_default());
    let manifest_checksum = digest(std::str::from_utf8(&manifest_bytes).unwrap_or_default());
    if candidate_checksum != pointer.candidate_checksum
        || candidate_checksum != manifest.candidate_content_checksum
    {
        return Err(ArtifactError::Write {
            path: output.to_owned(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "published candidate checksum verification failed",
            ),
        });
    }
    if manifest_checksum != pointer.manifest_checksum {
        return Err(ArtifactError::Write {
            path: output.to_owned(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "published manifest checksum verification failed",
            ),
        });
    }
    Ok((records, manifest))
}

fn write_new_file(path: &Path, content: &[u8]) -> Result<(), std::io::Error> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(content)?;
    file.sync_all()
}

pub(crate) fn atomic_write(path: &Path, content: &[u8]) -> Result<(), ArtifactError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path.file_name().unwrap_or_default().to_string_lossy();
    let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{file_name}.{}.{sequence}.tmp",
        std::process::id()
    ));
    let result = (|| -> Result<(), std::io::Error> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(content)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    })();
    if let Err(source) = result {
        let _ = fs::remove_file(&temporary);
        return Err(ArtifactError::Write {
            path: path.to_owned(),
            source,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::transform::{Provenance, TransformKind};

    fn record(expression: &str) -> CandidateRecord {
        CandidateRecord {
            schema: CANDIDATE_SCHEMA,
            run_id: "test-run".to_owned(),
            expression: expression.to_owned(),
            fingerprint: "a".repeat(64),
            semantic_fingerprint: "b".repeat(64),
            structural_score: 0.5,
            score_schema: 2,
            score_components: StructuralScoreComponents::default(),
            normalized_weights: StructuralScoreWeights::default(),
            structural_descriptor: StructuralDescriptor::default(),
            root_family: "field".to_owned(),
            nodes: 1,
            depth: 1,
            provenance: Provenance {
                kind: TransformKind::Initial,
                operation: "test".to_owned(),
                parent_fingerprints: Vec::new(),
                requested_operation: None,
                affected_path: None,
                old_subtree_fingerprint: None,
                new_subtree_fingerprint: None,
                retry_count: 0,
            },
        }
    }

    #[test]
    fn atomic_jsonl_replaces_complete_artifact_and_leaves_no_temp() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("out.jsonl");
        fs::write(&path, "old\n").unwrap();
        write_jsonl_atomic(&path, &[record("close")]).unwrap();
        assert_eq!(read_candidates(&path).unwrap()[0].expression, "close");
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }
}
