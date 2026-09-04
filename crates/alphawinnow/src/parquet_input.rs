//! Optional immutable Parquet adapters for local numeric evidence.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Path, PathBuf},
};

use arrow_array::{Array, Date32Array, Float64Array, StringArray};
use chrono::{Datelike, NaiveDate};
use parquet::{arrow::arrow_reader::ParquetRecordBatchReaderBuilder, errors::ParquetError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ColumnarDataset, NumericColumn};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharadarPanelConfig {
    pub parquet_root: PathBuf,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub universe_size: usize,
    pub dataset_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedPanel {
    pub dataset: ColumnarDataset,
    pub date_rows: BTreeMap<String, usize>,
}

#[derive(Debug, Error)]
pub enum ParquetInputError {
    #[error("invalid Parquet panel configuration: {0}")]
    Config(String),
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot decode {path}: {source}")]
    Parquet { path: PathBuf, source: ParquetError },
    #[error("{path} does not contain the required `{column}` column/type")]
    Column { path: PathBuf, column: String },
}

#[derive(Debug)]
struct PricePoint {
    date: i32,
    open_adjusted: f64,
    high_adjusted: f64,
    low_adjusted: f64,
    close_adjusted: f64,
    volume: f64,
}

/// Build a deterministic row-major panel from immutable Sharadar Parquet files.
///
/// Each date independently selects the largest positive finite market caps.
/// Signal fields are null outside membership, while the separate realized
/// return column remains available for the next-row target. Adjusted OHLC,
/// raw volume, and `returns` are emitted for local research; callers can
/// retain earlier rows as causal warm-up via `EvaluationConfig::evaluation_start`.
///
/// # Errors
/// Returns configuration, filesystem, Parquet, or schema diagnostics.
#[allow(clippy::too_many_lines)]
pub fn prepare_sharadar_panel(
    config: &SharadarPanelConfig,
) -> Result<PreparedPanel, ParquetInputError> {
    validate_config(config)?;
    let minimum_date = date32(config.start_date);
    let maximum_date = date32(config.end_date);
    let years = config.start_date.year()..=config.end_date.year();

    let mut daily: BTreeMap<i32, Vec<(String, f64)>> = BTreeMap::new();
    for year in years.clone() {
        for path in partition_files(&config.parquet_root, "daily", year)? {
            read_daily(&path, minimum_date, maximum_date, &mut daily)?;
        }
    }
    for values in daily.values_mut() {
        let mut unique = BTreeMap::new();
        for (ticker, marketcap) in values.drain(..) {
            unique
                .entry(ticker)
                .and_modify(|current: &mut f64| *current = current.max(marketcap))
                .or_insert(marketcap);
        }
        values.extend(unique);
        values.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        values.truncate(config.universe_size);
    }
    daily.retain(|_, values| !values.is_empty());
    let assets: Vec<_> = daily
        .values()
        .flat_map(|values| values.iter().map(|(ticker, _)| ticker.clone()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if daily.len() < 3 || assets.len() < 2 {
        return Err(ParquetInputError::Config(
            "selected range must contain at least three dates and two assets".to_owned(),
        ));
    }
    let selected: BTreeSet<_> = assets.iter().cloned().collect();
    let mut prices: BTreeMap<String, Vec<PricePoint>> = BTreeMap::new();
    for year in years {
        for path in partition_files(&config.parquet_root, "stocks", year)? {
            read_stocks(&path, minimum_date, maximum_date, &selected, &mut prices)?;
        }
    }

    let dates: Vec<_> = daily.keys().copied().collect();
    let date_index: BTreeMap<_, _> = dates
        .iter()
        .enumerate()
        .map(|(index, date)| (*date, index))
        .collect();
    let asset_index: BTreeMap<_, _> = assets
        .iter()
        .enumerate()
        .map(|(index, asset)| (asset.as_str(), index))
        .collect();
    let length = dates.len().saturating_mul(assets.len());
    let mut membership = vec![false; length];
    for (date, members) in &daily {
        let row = date_index[date];
        for (ticker, _) in members {
            let asset = asset_index[ticker.as_str()];
            membership[row * assets.len() + asset] = true;
        }
    }
    let mut realized = vec![None; length];
    let mut open = vec![None; length];
    let mut high = vec![None; length];
    let mut low = vec![None; length];
    let mut close = vec![None; length];
    let mut volume = vec![None; length];
    for (ticker, values) in &mut prices {
        let Some(&asset) = asset_index.get(ticker.as_str()) else {
            continue;
        };
        values.sort_by_key(|point| point.date);
        values.dedup_by_key(|point| point.date);
        let mut previous = None;
        for point in values {
            let value = previous.and_then(|prior: f64| {
                (prior > 0.0 && point.close_adjusted > 0.0)
                    .then_some(point.close_adjusted / prior - 1.0)
            });
            if let Some(&row) = date_index.get(&point.date) {
                let index = row * assets.len() + asset;
                realized[index] = value.filter(|value| value.is_finite());
                if membership[index] {
                    open[index] = Some(point.open_adjusted);
                    high[index] = Some(point.high_adjusted);
                    low[index] = Some(point.low_adjusted);
                    close[index] = Some(point.close_adjusted);
                    volume[index] = Some(point.volume);
                }
            }
            previous = Some(point.close_adjusted);
        }
    }
    let mut signal = vec![None; length];
    for index in 0..length {
        if membership[index] {
            signal[index] = realized[index];
        }
    }

    let timestamps: Vec<_> = dates.iter().map(|date| yyyymmdd(*date)).collect();
    let date_rows = timestamps
        .iter()
        .enumerate()
        .map(|(index, timestamp)| (timestamp.to_string(), index))
        .collect();
    let dataset = ColumnarDataset {
        schema: 1,
        dataset_id: config.dataset_id.clone(),
        timestamps,
        assets,
        fields: BTreeMap::from([
            ("close".to_owned(), NumericColumn { values: close }),
            ("high".to_owned(), NumericColumn { values: high }),
            ("low".to_owned(), NumericColumn { values: low }),
            ("open".to_owned(), NumericColumn { values: open }),
            ("returns".to_owned(), NumericColumn { values: signal }),
            ("volume".to_owned(), NumericColumn { values: volume }),
        ]),
        realized_returns: NumericColumn { values: realized },
        groups: BTreeMap::new(),
        source_label: format!(
            "local Sharadar Parquet; dynamic same-day marketcap TOP{}; {}..{}",
            config.universe_size, config.start_date, config.end_date
        ),
        preprocessing: vec![
            "each date independently selects positive finite same-day marketcap".to_owned(),
            "returns are closeadj percentage changes per ticker".to_owned(),
            "OHLC are adjusted by closeadj/close; volume is the source daily share volume"
                .to_owned(),
            "all signal fields are null outside daily universe membership".to_owned(),
            "realized returns remain available for strictly future targets".to_owned(),
            "no forward fill and no static classification labels".to_owned(),
        ],
    };
    dataset
        .validate()
        .map_err(|error| ParquetInputError::Config(error.to_string()))?;
    Ok(PreparedPanel { dataset, date_rows })
}

fn validate_config(config: &SharadarPanelConfig) -> Result<(), ParquetInputError> {
    if config.start_date > config.end_date
        || config.universe_size < 2
        || config.dataset_id.trim().is_empty()
    {
        return Err(ParquetInputError::Config(
            "require ordered dates, universe size >= 2, and a dataset id".to_owned(),
        ));
    }
    Ok(())
}

fn partition_files(root: &Path, table: &str, year: i32) -> Result<Vec<PathBuf>, ParquetInputError> {
    let directory = root.join(table).join(format!("partition_year={year}"));
    let entries = std::fs::read_dir(&directory).map_err(|source| ParquetInputError::Io {
        path: directory.clone(),
        source,
    })?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| ParquetInputError::Io {
            path: directory.clone(),
            source,
        })?;
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "parquet")
        {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(ParquetInputError::Config(format!(
            "no Parquet files found in {}",
            directory.display()
        )));
    }
    Ok(paths)
}

fn read_daily(
    path: &Path,
    minimum_date: i32,
    maximum_date: i32,
    output: &mut BTreeMap<i32, Vec<(String, f64)>>,
) -> Result<(), ParquetInputError> {
    let reader = parquet_reader(path)?;
    for batch in reader {
        let batch = batch.map_err(|source| ParquetInputError::Parquet {
            path: path.to_owned(),
            source: source.into(),
        })?;
        let ticker = string_column(path, &batch, "ticker")?;
        let date = date_column(path, &batch, "date")?;
        let marketcap = float_column(path, &batch, "marketcap")?;
        for row in 0..batch.num_rows() {
            if ticker.is_null(row) || date.is_null(row) || marketcap.is_null(row) {
                continue;
            }
            let date = date.value(row);
            let value = marketcap.value(row);
            if (minimum_date..=maximum_date).contains(&date) && value.is_finite() && value > 0.0 {
                output
                    .entry(date)
                    .or_default()
                    .push((ticker.value(row).to_owned(), value));
            }
        }
    }
    Ok(())
}

fn read_stocks(
    path: &Path,
    minimum_date: i32,
    maximum_date: i32,
    selected: &BTreeSet<String>,
    output: &mut BTreeMap<String, Vec<PricePoint>>,
) -> Result<(), ParquetInputError> {
    let reader = parquet_reader(path)?;
    for batch in reader {
        let batch = batch.map_err(|source| ParquetInputError::Parquet {
            path: path.to_owned(),
            source: source.into(),
        })?;
        let ticker = string_column(path, &batch, "ticker")?;
        let date = date_column(path, &batch, "date")?;
        let close = float_column(path, &batch, "closeadj")?;
        let close_unadjusted = float_column(path, &batch, "close")?;
        let open = float_column(path, &batch, "open")?;
        let high = float_column(path, &batch, "high")?;
        let low = float_column(path, &batch, "low")?;
        let volume = float_column(path, &batch, "volume")?;
        for row in 0..batch.num_rows() {
            if ticker.is_null(row)
                || date.is_null(row)
                || close.is_null(row)
                || close_unadjusted.is_null(row)
                || open.is_null(row)
                || high.is_null(row)
                || low.is_null(row)
                || volume.is_null(row)
            {
                continue;
            }
            let ticker = ticker.value(row);
            let date = date.value(row);
            let close_adjusted = close.value(row);
            let close_unadjusted = close_unadjusted.value(row);
            let adjustment = close_adjusted / close_unadjusted;
            let open_adjusted = open.value(row) * adjustment;
            let high_adjusted = high.value(row) * adjustment;
            let low_adjusted = low.value(row) * adjustment;
            let volume = volume.value(row);
            if selected.contains(ticker)
                && (minimum_date..=maximum_date).contains(&date)
                && [
                    close_adjusted,
                    close_unadjusted,
                    adjustment,
                    open_adjusted,
                    high_adjusted,
                    low_adjusted,
                    volume,
                ]
                .iter()
                .all(|value| value.is_finite())
                && close_adjusted > 0.0
                && close_unadjusted > 0.0
                && open_adjusted > 0.0
                && high_adjusted > 0.0
                && low_adjusted > 0.0
                && volume >= 0.0
            {
                output
                    .entry(ticker.to_owned())
                    .or_default()
                    .push(PricePoint {
                        date,
                        open_adjusted,
                        high_adjusted,
                        low_adjusted,
                        close_adjusted,
                        volume,
                    });
            }
        }
    }
    Ok(())
}

fn parquet_reader(
    path: &Path,
) -> Result<parquet::arrow::arrow_reader::ParquetRecordBatchReader, ParquetInputError> {
    let file = File::open(path).map_err(|source| ParquetInputError::Io {
        path: path.to_owned(),
        source,
    })?;
    ParquetRecordBatchReaderBuilder::try_new(file)
        .and_then(|builder| builder.with_batch_size(65_536).build())
        .map_err(|source| ParquetInputError::Parquet {
            path: path.to_owned(),
            source,
        })
}

fn string_column<'a>(
    path: &Path,
    batch: &'a arrow_array::RecordBatch,
    name: &str,
) -> Result<&'a StringArray, ParquetInputError> {
    column(path, batch, name)
}

fn date_column<'a>(
    path: &Path,
    batch: &'a arrow_array::RecordBatch,
    name: &str,
) -> Result<&'a Date32Array, ParquetInputError> {
    column(path, batch, name)
}

fn float_column<'a>(
    path: &Path,
    batch: &'a arrow_array::RecordBatch,
    name: &str,
) -> Result<&'a Float64Array, ParquetInputError> {
    column(path, batch, name)
}

fn column<'a, T: 'static>(
    path: &Path,
    batch: &'a arrow_array::RecordBatch,
    name: &str,
) -> Result<&'a T, ParquetInputError> {
    let index = batch
        .schema()
        .index_of(name)
        .map_err(|_| ParquetInputError::Column {
            path: path.to_owned(),
            column: name.to_owned(),
        })?;
    batch
        .column(index)
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| ParquetInputError::Column {
            path: path.to_owned(),
            column: name.to_owned(),
        })
}

fn date32(date: NaiveDate) -> i32 {
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("Unix epoch is a valid date");
    i32::try_from((date - epoch).num_days()).expect("supported dates fit Arrow Date32")
}

fn yyyymmdd(value: i32) -> i64 {
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("Unix epoch is a valid date");
    let date = epoch + chrono::Duration::days(i64::from(value));
    i64::from(date.year()) * 10_000 + i64::from(date.month()) * 100 + i64::from(date.day())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, RecordBatch};
    use parquet::arrow::ArrowWriter;

    use super::*;

    #[test]
    fn date32_round_trip_is_stable() {
        for date in [
            NaiveDate::from_ymd_opt(1970, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2019, 1, 2).unwrap(),
            NaiveDate::from_ymd_opt(2024, 2, 29).unwrap(),
        ] {
            assert_eq!(
                yyyymmdd(date32(date)),
                date.format("%Y%m%d").to_string().parse::<i64>().unwrap()
            );
        }
    }

    #[test]
    fn adapter_builds_dynamic_membership_without_hiding_realized_returns() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let daily_directory = root.join("daily/partition_year=2020");
        let stocks_directory = root.join("stocks/partition_year=2020");
        std::fs::create_dir_all(&daily_directory).unwrap();
        std::fs::create_dir_all(&stocks_directory).unwrap();
        let dates = [
            NaiveDate::from_ymd_opt(2020, 1, 2).unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 3).unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 6).unwrap(),
            NaiveDate::from_ymd_opt(2020, 1, 7).unwrap(),
        ];
        let tickers: Vec<_> = dates.iter().flat_map(|_| ["A", "B", "C"]).collect();
        let date_values: Vec<_> = dates.iter().flat_map(|date| [date32(*date); 3]).collect();
        write_batch(
            &daily_directory.join("data_0.parquet"),
            &RecordBatch::try_from_iter(vec![
                (
                    "ticker",
                    Arc::new(StringArray::from(tickers.clone())) as ArrayRef,
                ),
                (
                    "date",
                    Arc::new(Date32Array::from(date_values.clone())) as ArrayRef,
                ),
                (
                    "marketcap",
                    Arc::new(Float64Array::from(vec![
                        300.0, 200.0, 100.0, 300.0, 200.0, 400.0, 300.0, 200.0, 100.0, 300.0,
                        200.0, 100.0,
                    ])) as ArrayRef,
                ),
            ])
            .unwrap(),
        );
        let prices = vec![
            100.0, 100.0, 100.0, 101.0, 102.0, 103.0, 102.0, 104.0, 106.0, 103.0, 106.0, 109.0,
        ];
        write_batch(
            &stocks_directory.join("data_0.parquet"),
            &RecordBatch::try_from_iter(vec![
                ("ticker", Arc::new(StringArray::from(tickers)) as ArrayRef),
                ("date", Arc::new(Date32Array::from(date_values)) as ArrayRef),
                (
                    "closeadj",
                    Arc::new(Float64Array::from(prices.clone())) as ArrayRef,
                ),
                (
                    "close",
                    Arc::new(Float64Array::from(prices.clone())) as ArrayRef,
                ),
                (
                    "open",
                    Arc::new(Float64Array::from(prices.clone())) as ArrayRef,
                ),
                (
                    "high",
                    Arc::new(Float64Array::from(prices.clone())) as ArrayRef,
                ),
                ("low", Arc::new(Float64Array::from(prices)) as ArrayRef),
                (
                    "volume",
                    Arc::new(Float64Array::from(vec![1_000.0; 12])) as ArrayRef,
                ),
            ])
            .unwrap(),
        );
        let prepared = prepare_sharadar_panel(&SharadarPanelConfig {
            parquet_root: root.to_owned(),
            start_date: dates[0],
            end_date: dates[3],
            universe_size: 2,
            dataset_id: "synthetic-parquet-panel-v1".to_owned(),
        })
        .unwrap();
        assert_eq!(
            prepared.dataset.timestamps,
            vec![20_200_102, 20_200_103, 20_200_106, 20_200_107]
        );
        assert_eq!(prepared.dataset.assets, vec!["A", "B", "C"]);
        let row = 1;
        let b = 1;
        let index = row * 3 + b;
        assert!(prepared.dataset.realized_returns.values[index].is_some());
        assert!(prepared.dataset.fields["returns"].values[index].is_none());
        assert_eq!(prepared.date_rows["20200106"], 2);
    }

    fn write_batch(path: &Path, batch: &RecordBatch) {
        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
    }
}
