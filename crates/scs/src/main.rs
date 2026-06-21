use std::{
    fmt, fs,
    path::{Path, PathBuf},
    process,
};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use segment_cache_store::{CommitStats, OpenOptions, Result as StoreResult, Store, StoreMetadata};

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
        let files = FileStats::collect(self).with_context(|| {
            format!(
                "collect file stats from store root `{}`",
                self.path.display()
            )
        })?;

        println!("root: {}", self.path.display());
        opened.print_identity();
        logical.print();
        files.print(logical.total_bytes());
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
        let metadata = self.read_metadata()?;
        let options = OpenOptions::new(metadata).with_read_only(mode.is_read_only());
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

    fn read_metadata(&self) -> Result<StoreMetadata> {
        let store_file = self.path.join("STORE");
        let text = fs::read_to_string(&store_file)
            .with_context(|| format!("read store descriptor `{}`", store_file.display()))?;
        for line in text.lines() {
            if let Some(hex) = line.strip_prefix("metadata=") {
                return Ok(StoreMetadata::from_bytes(
                    hex.parse_hex_bytes().with_context(|| {
                        format!("parse metadata field in `{}`", store_file.display())
                    })?,
                ));
            }
        }
        Err(anyhow!(
            "{} does not contain a metadata field",
            store_file.display()
        ))
    }
}

#[derive(Clone, Copy)]
enum OpenMode {
    ReadOnly,
    Writer,
}

impl OpenMode {
    fn is_read_only(self) -> bool {
        matches!(self, Self::ReadOnly)
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
        for record in store.iter_all()? {
            let (key, value) = record?;
            stats.records += 1;
            stats.key_bytes += key.len();
            stats.value_bytes += value.len();
            if stats.min_key.is_none() {
                stats.min_key = Some(key.clone());
            }
            stats.max_key = Some(key);
        }
        Ok(stats)
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

#[derive(Default)]
struct FileStats {
    files: usize,
    bytes: u64,
    segment_files: usize,
    segment_bytes: u64,
}

impl FileStats {
    fn collect(root: &StoreRoot) -> std::io::Result<Self> {
        let mut stats = Self::default();
        stats.collect_recurse(&root.path, &root.path)?;
        Ok(stats)
    }

    fn collect_recurse(&mut self, root: &Path, path: &Path) -> std::io::Result<()> {
        let metadata = fs::symlink_metadata(path)?;
        if metadata.is_file() {
            self.files += 1;
            self.bytes += metadata.len();
            if Self::is_segment_file(root, path) {
                self.segment_files += 1;
                self.segment_bytes += metadata.len();
            }
            return Ok(());
        }
        if metadata.is_dir() {
            for entry in fs::read_dir(path)? {
                self.collect_recurse(root, &entry?.path())?;
            }
        }
        Ok(())
    }

    fn is_segment_file(root: &Path, path: &Path) -> bool {
        let Ok(relative) = path.strip_prefix(root) else {
            return false;
        };
        let mut components = relative.components();
        let Some(first) = components.next() else {
            return false;
        };
        if first.as_os_str() != "segments" || components.next().is_none() {
            return false;
        }
        path.extension().is_some_and(|extension| extension == "seg")
    }

    fn print(&self, logical_bytes: usize) {
        println!("segment_files: {}", self.segment_files);
        println!("segment_file_bytes: {}", ByteCount(self.segment_bytes));
        println!("all_files: {}", self.files);
        println!("all_file_bytes: {}", ByteCount(self.bytes));
        println!(
            "segment_space_amplification: {}",
            SpaceRatio::new(self.segment_bytes, logical_bytes)
        );
        println!(
            "total_space_amplification: {}",
            SpaceRatio::new(self.bytes, logical_bytes)
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
        writeln!(formatter, "records: {}", self.0.records)?;
        writeln!(formatter, "bytes: {}", self.0.bytes)?;
        writeln!(
            formatter,
            "segments_published: {}",
            self.0.segments_published
        )?;
        writeln!(formatter, "segments_retired: {}", self.0.segments_retired)?;
        write!(formatter, "merged_records: {}", self.0.merged_records)
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
