use std::{fmt, path::PathBuf, process};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use segment_cache_store::{
    CommitStats, OpenOptions, Result as StoreResult, Store, StoreMetadata, StoreStorageStats,
};

mod hex;

use hex::{HexParseExt, HexPreviewExt};

fn main() {
    if let Err(error) = Cli::parse().run() {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

#[derive(Parser)]
#[command(name = "scs", about = "Inspect and maintain segment-cache-store roots")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

impl Cli {
    fn run(self) -> Result<()> {
        self.command.run()
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Print logical and file-size statistics for a store.
    #[command(visible_alias = "inspect")]
    Stats {
        /// Store root.
        root: PathBuf,
    },
    /// Fetch and preview one value by hex-encoded key.
    Get {
        /// Store root.
        root: PathBuf,
        /// Hex key, with optional 0x prefix and separators.
        key: String,
        /// Maximum value bytes to print as hex and UTF-8 preview.
        #[arg(long, default_value_t = 256)]
        value_limit: usize,
    },
    /// Atomically merge a compatible source store into a destination store.
    Merge {
        /// Destination store root to mutate.
        destination: PathBuf,
        /// Source store root to read.
        source: PathBuf,
    },
    /// Fold live patch segments into the main tier.
    Normalize {
        /// Store root.
        root: PathBuf,
    },
    /// Normalize and then garbage-collect unreferenced segment files.
    Compact {
        /// Store root.
        root: PathBuf,
    },
    /// Garbage-collect unreferenced segment files.
    Gc {
        /// Store root.
        root: PathBuf,
    },
}

impl Commands {
    fn run(self) -> Result<()> {
        match self {
            Self::Stats { root } => StoreRoot::new(root).print_stats(),
            Self::Get {
                root,
                key,
                value_limit,
            } => StoreRoot::new(root).print_value(
                &key.parse_hex_bytes().context("parse hex key")?,
                value_limit,
            ),
            Self::Merge {
                destination,
                source,
            } => StoreRoot::new(destination).merge_from(&StoreRoot::new(source)),
            Self::Normalize { root } => StoreRoot::new(root).normalize(),
            Self::Compact { root } => StoreRoot::new(root).compact(),
            Self::Gc { root } => StoreRoot::new(root).garbage_collect(),
        }
    }
}

#[derive(Clone, Debug)]
struct StoreRoot {
    path: PathBuf,
}

impl StoreRoot {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn print_stats(&self) -> Result<()> {
        let opened = self.open_read_only()?;
        let logical = LogicalStats::collect(opened.store()).with_context(|| {
            format!("collect logical stats from store `{}`", self.path.display())
        })?;
        let storage = opened.store().storage_stats().with_context(|| {
            format!("collect storage stats from store `{}`", self.path.display())
        })?;

        println!("root: {}", self.path.display());
        opened.print_identity();
        logical.print();
        StorageStatsView(storage).print(logical.total_bytes());
        if let Some(key) = logical.min_key.as_deref() {
            println!("min_key_hex: {}", key.preview(key.len()));
        }
        if let Some(key) = logical.max_key.as_deref() {
            println!("max_key_hex: {}", key.preview(key.len()));
        }
        Ok(())
    }

    fn print_value(&self, key: &[u8], value_limit: usize) -> Result<()> {
        let opened = self.open_read_only()?;
        let value = opened
            .store()
            .fetch_one(key)
            .with_context(|| format!("fetch key from store `{}`", self.path.display()))?;

        println!("root: {}", self.path.display());
        println!("key_hex: {}", key.preview(key.len()));
        ValuePreview::new(value, value_limit).print();
        Ok(())
    }

    fn merge_from(&self, source: &Self) -> Result<()> {
        let destination_store = self.open_writer()?;
        let source_store = source.open_read_only()?;
        let stats = destination_store
            .store()
            .merge_from(source_store.store())
            .with_context(|| {
                format!(
                    "merge source store `{}` into destination store `{}`",
                    source.path.display(),
                    self.path.display()
                )
            })?;

        println!("destination: {}", self.path.display());
        println!("source: {}", source.path.display());
        println!("{}", CommitStatsView(&stats));
        Ok(())
    }

    fn normalize(&self) -> Result<()> {
        let opened = self.open_writer()?;
        let stats = opened
            .store()
            .normalize()
            .with_context(|| format!("normalize store `{}`", self.path.display()))?;

        println!("root: {}", self.path.display());
        println!("{}", CommitStatsView(&stats));
        Ok(())
    }

    fn compact(&self) -> Result<()> {
        let opened = self.open_writer()?;
        let stats = opened
            .store()
            .normalize()
            .with_context(|| format!("normalize store `{}`", self.path.display()))?;
        opened
            .store()
            .garbage_collect()
            .with_context(|| format!("garbage-collect store `{}`", self.path.display()))?;

        println!("root: {}", self.path.display());
        println!("{}", CommitStatsView(&stats));
        println!("garbage_collect: done");
        Ok(())
    }

    fn garbage_collect(&self) -> Result<()> {
        let opened = self.open_writer()?;
        opened
            .store()
            .garbage_collect()
            .with_context(|| format!("garbage-collect store `{}`", self.path.display()))?;

        println!("root: {}", self.path.display());
        println!("garbage_collect: done");
        Ok(())
    }

    fn open_read_only(&self) -> Result<OpenedStore> {
        self.open(OpenMode::ReadOnly)
    }

    fn open_writer(&self) -> Result<OpenedStore> {
        self.open(OpenMode::Writer)
    }

    fn open(&self, mode: OpenMode) -> Result<OpenedStore> {
        let info = Store::inspect(&self.path)
            .with_context(|| format!("inspect store `{}`", self.path.display()))?;
        let options = mode.open_options(info.metadata);
        Ok(OpenedStore {
            store: Store::open(&self.path, options).with_context(|| {
                format!(
                    "open store `{}`{}",
                    self.path.display(),
                    mode.description_suffix()
                )
            })?,
        })
    }
}

#[derive(Clone, Copy)]
enum OpenMode {
    ReadOnly,
    Writer,
}

impl OpenMode {
    fn open_options(self, metadata: StoreMetadata) -> OpenOptions {
        match self {
            Self::ReadOnly => OpenOptions::read_only(metadata),
            Self::Writer => OpenOptions::read_write(metadata),
        }
    }

    fn description_suffix(self) -> &'static str {
        match self {
            Self::ReadOnly => " read-only",
            Self::Writer => " for writing",
        }
    }
}

struct OpenedStore {
    store: Store,
}

impl OpenedStore {
    fn store(&self) -> &Store {
        &self.store
    }

    fn print_identity(&self) {
        println!("metadata: {}", MetadataView(self.store.metadata()));
        println!("key_len: {}", self.store.key_len());
        println!("value_layout: {}", ValueLayoutView(&self.store));
        println!("block_checksum: {}", self.store.block_checksum().name());
        println!(
            "value_payload_compression: {}",
            self.store.value_payload_compression().name()
        );
    }
}

struct MetadataView<'a>(&'a StoreMetadata);

impl std::fmt::Display for MetadataView<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes = self.0.as_bytes();
        write!(formatter, "hex:{}", bytes.preview(bytes.len()))?;
        if let Ok(text) = std::str::from_utf8(bytes)
            && !text.is_empty()
        {
            write!(formatter, " text:{text}")?;
        }
        Ok(())
    }
}

struct ValueLayoutView<'a>(&'a Store);

impl std::fmt::Display for ValueLayoutView<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0.value_layout().fixed_value_len() {
            Some(len) => write!(formatter, "fixed({})", len.get()),
            None => formatter.write_str("variable"),
        }
    }
}

#[derive(Default)]
struct LogicalStats {
    records: usize,
    key_bytes: usize,
    value_bytes: usize,
    min_key: Option<Vec<u8>>,
    max_key: Option<Vec<u8>>,
}

impl LogicalStats {
    fn collect(store: &Store) -> StoreResult<Self> {
        let mut stats = Self::default();
        store.visit_all(|key, value| stats.record(key, value))?;
        Ok(stats)
    }

    fn record(&mut self, key: &[u8], value: &[u8]) {
        self.records += 1;
        self.key_bytes += key.len();
        self.value_bytes += value.len();
        if self.min_key.is_none() {
            self.min_key = Some(key.to_vec());
        }
        match &mut self.max_key {
            Some(max_key) => {
                max_key.clear();
                max_key.extend_from_slice(key);
            }
            None => self.max_key = Some(key.to_vec()),
        }
    }

    fn total_bytes(&self) -> usize {
        self.key_bytes + self.value_bytes
    }

    fn print(&self) {
        println!("visible_records: {}", self.records);
        println!("visible_key_bytes: {}", self.key_bytes);
        println!("visible_value_bytes: {}", self.value_bytes);
        println!("visible_logical_bytes: {}", self.total_bytes());
    }
}

struct StorageStatsView(StoreStorageStats);

impl StorageStatsView {
    fn print(&self, logical_bytes: usize) {
        println!("segment_files: {}", self.0.segment_files);
        println!("segment_file_bytes: {}", ByteCount(self.0.segment_bytes));
        println!("all_files: {}", self.0.total_files);
        println!("all_file_bytes: {}", ByteCount(self.0.total_bytes));
        println!(
            "segment_space_amplification: {}",
            SpaceRatio::new(self.0.segment_bytes, logical_bytes)
        );
        println!(
            "total_space_amplification: {}",
            SpaceRatio::new(self.0.total_bytes, logical_bytes)
        );
    }
}

struct ValuePreview {
    value: Option<Vec<u8>>,
    limit: usize,
}

impl ValuePreview {
    fn new(value: Option<Vec<u8>>, limit: usize) -> Self {
        Self { value, limit }
    }

    fn print(self) {
        match self.value {
            Some(value) => Self::print_hit(&value, self.limit),
            None => println!("hit: false"),
        }
    }

    fn print_hit(value: &[u8], limit: usize) {
        let shown = value.len().min(limit);
        println!("hit: true");
        println!("value_len: {}", value.len());
        println!("value_hex: {}", value.preview(limit));
        if shown > 0
            && let Ok(text) = std::str::from_utf8(&value[..shown])
        {
            println!("value_utf8_preview: {text}");
        }
        if shown < value.len() {
            println!("value_truncated: true");
        }
    }
}

struct CommitStatsView<'a>(&'a CommitStats);

impl fmt::Display for CommitStatsView<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "input_records: {}", self.0.input_records)?;
        writeln!(formatter, "input_bytes: {}", self.0.input_bytes)?;
        writeln!(
            formatter,
            "segments_published: {}",
            self.0.segments_published
        )?;
        writeln!(formatter, "segments_retired: {}", self.0.segments_retired)?;
        write!(formatter, "output_records: {}", self.0.output_records)
    }
}

struct ByteCount(u64);

impl std::fmt::Display for ByteCount {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
        let mut value = self.0 as f64;
        let mut unit = UNITS[0];
        for next_unit in &UNITS[1..] {
            if value < 1024.0 {
                break;
            }
            value /= 1024.0;
            unit = next_unit;
        }
        if unit == "B" {
            write!(formatter, "{} B", self.0)
        } else {
            write!(formatter, "{value:.2} {unit} ({} B)", self.0)
        }
    }
}

struct SpaceRatio {
    bytes: u64,
    logical_bytes: usize,
}

impl SpaceRatio {
    fn new(bytes: u64, logical_bytes: usize) -> Self {
        Self {
            bytes,
            logical_bytes,
        }
    }
}

impl std::fmt::Display for SpaceRatio {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.logical_bytes == 0 {
            return formatter.write_str("n/a");
        }
        write!(
            formatter,
            "{:.3}x",
            self.bytes as f64 / self.logical_bytes as f64
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use segment_cache_store::{BlockChecksumKind, CreateOptions, WriteBatch};

    use super::*;

    const KEY_ONE: [u8; 4] = 1u32.to_be_bytes();
    const KEY_TWO: [u8; 4] = 2u32.to_be_bytes();
    const KEY_MISS: [u8; 4] = 99u32.to_be_bytes();

    #[test]
    fn cli_commands_operate_on_real_stores() -> Result<()> {
        let destination = tempfile::tempdir()?;
        let source = tempfile::tempdir()?;
        create_store(destination.path(), &[(KEY_ONE, b"alpha".as_slice())])?;
        create_store(source.path(), &[(KEY_TWO, b"beta".as_slice())])?;
        let destination_arg = path_arg(destination.path());
        let source_arg = path_arg(source.path());

        run_cli(["scs", "stats", destination_arg.as_str()])?;
        run_cli([
            "scs",
            "get",
            destination_arg.as_str(),
            "00:00:00:01",
            "--value-limit",
            "2",
        ])?;
        StoreRoot::new(destination.path().to_path_buf()).print_value(&KEY_MISS, 8)?;
        run_cli([
            "scs",
            "merge",
            destination_arg.as_str(),
            source_arg.as_str(),
        ])?;
        run_cli(["scs", "normalize", destination_arg.as_str()])?;
        run_cli(["scs", "gc", destination_arg.as_str()])?;
        run_cli(["scs", "compact", destination_arg.as_str()])?;

        let reopened = Store::open(destination.path(), OpenOptions::read_only(test_metadata()))?;
        assert_eq!(reopened.fetch_one(&KEY_ONE)?, Some(b"alpha".to_vec()));
        assert_eq!(reopened.fetch_one(&KEY_TWO)?, Some(b"beta".to_vec()));
        Ok(())
    }

    #[test]
    fn store_root_open_modes_and_stats_cover_empty_store() -> Result<()> {
        let tempdir = tempfile::tempdir()?;
        create_store(tempdir.path(), &[])?;
        let root = StoreRoot::new(tempdir.path().to_path_buf());

        root.print_stats()?;
        let read_only = root.open_read_only()?;
        assert_eq!(read_only.store().iter_all()?.count(), 0);
        drop(read_only);

        let writer = root.open_writer()?;
        writer.store().garbage_collect()?;
        Ok(())
    }

    #[test]
    fn logical_stats_collect_visible_records() -> Result<()> {
        let tempdir = tempfile::tempdir()?;
        create_store(tempdir.path(), &[
            (KEY_ONE, b"alpha".as_slice()),
            (KEY_TWO, b"beta".as_slice()),
        ])?;
        let opened = StoreRoot::new(tempdir.path().to_path_buf()).open_read_only()?;

        let stats = LogicalStats::collect(opened.store())?;

        assert_eq!(stats.records, 2);
        assert_eq!(stats.key_bytes, KEY_ONE.len() + KEY_TWO.len());
        assert_eq!(stats.value_bytes, b"alpha".len() + b"beta".len());
        assert_eq!(stats.min_key.as_deref(), Some(KEY_ONE.as_slice()));
        assert_eq!(stats.max_key.as_deref(), Some(KEY_TWO.as_slice()));
        Ok(())
    }

    fn run_cli<const N: usize>(args: [&str; N]) -> Result<()> {
        Cli::try_parse_from(args)?.run()
    }

    fn create_store(root: &Path, entries: &[([u8; 4], &[u8])]) -> Result<()> {
        let store = Store::create(root, create_options())?;
        let mut batch = WriteBatch::new();
        for (key, value) in entries {
            batch.push(key, value);
        }
        store.commit_batch(batch)?;
        Ok(())
    }

    fn create_options() -> CreateOptions {
        CreateOptions::new(4, test_metadata(), BlockChecksumKind::None)
            .expect("test key length should be valid")
    }

    fn test_metadata() -> StoreMetadata {
        StoreMetadata::from_text("scs-test")
    }

    fn path_arg(path: &Path) -> String {
        path.display().to_string()
    }
}
