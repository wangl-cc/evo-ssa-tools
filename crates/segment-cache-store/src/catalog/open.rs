//! Store open options and implementation.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use parking_lot::{Mutex, RwLock};

use crate::{
    block::{BlockChecksumKind, ValuePayloadCompressionKind},
    catalog::{
        descriptor::StoreDescriptor, io::WriterLock, manifest::StoreManifest, paths::StorePaths,
    },
    error::{CatalogMismatch, OptionsError, Result},
    schema::{StoreGeometry, StoreMetadata, ValueLayout},
    segment::{
        file::{OpenedSegment, SegmentOpenOptions},
        state::SegmentState,
    },
    store::{
        Store,
        state::{StoreInner, StoreState},
    },
};

/// Options consumed only when opening an existing store root.
///
/// Persistent geometry such as key length and value layout is read from
/// `STORE`. Open options are read-time policy and compatibility checks, not
/// store creation metadata.
#[derive(Clone, Debug)]
pub struct OpenOptions {
    /// Required caller metadata for this namespace.
    expected_metadata: StoreMetadata,
    /// Whether data-block checksums are verified on read.
    verify_block_checksums: bool,
    /// Whether this open is read-only.
    ///
    /// A read-only open takes no writer lock and runs no garbage collection, so
    /// multiple read-only opens of one root can coexist with a writer.
    read_only: bool,
}

/// Persistent store identity read from a root's `STORE` descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct StoreInfo {
    /// Caller-defined compatibility metadata for this namespace.
    pub metadata: StoreMetadata,
    /// Fixed key length in bytes.
    pub key_len: usize,
    /// Value layout shared by all visible segments.
    pub value_layout: ValueLayout,
    /// Block checksum kind persisted for all visible segments.
    pub block_checksum: BlockChecksumKind,
    /// Value-payload compression kind persisted for all visible segments.
    pub value_payload_compression: ValuePayloadCompressionKind,
}

impl OpenOptions {
    /// Creates options for a writable open that verifies block checksums.
    pub fn read_write(expected_metadata: StoreMetadata) -> Self {
        Self {
            expected_metadata,
            verify_block_checksums: true,
            read_only: false,
        }
    }

    /// Creates options for a read-only open that verifies block checksums.
    ///
    /// A read-only handle takes no writer lock, runs no garbage collection,
    /// and creates no directories. Mutating operations return
    /// [`crate::InputError::ReadOnlyStore`].
    pub fn read_only(expected_metadata: StoreMetadata) -> Self {
        Self {
            expected_metadata,
            verify_block_checksums: true,
            read_only: true,
        }
    }

    /// Enables or disables per-block checksum verification on reads.
    ///
    /// Disabling verification is only accepted for read-only opens; writable
    /// handles must keep verification enabled so corrupt bytes cannot be merged
    /// into freshly checksummed replacement segments. Open-time catalog,
    /// segment-structure, and manifest fingerprint checks still run either way.
    pub fn with_block_checksum_verification(mut self, verify: bool) -> Self {
        self.verify_block_checksums = verify;
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if !self.read_only && !self.verify_block_checksums {
            return Err(OptionsError::WritableStoreRequiresBlockChecksumVerification.into());
        }
        Ok(())
    }
}

impl StoreInfo {
    fn from_descriptor(descriptor: StoreDescriptor) -> Result<Self> {
        descriptor.validate_structure()?;
        Ok(Self {
            metadata: descriptor.metadata,
            key_len: descriptor.key_len,
            value_layout: descriptor.value_layout,
            block_checksum: resolve_block_checksum(descriptor.block_checksum_id)?,
            value_payload_compression: resolve_value_payload_compression(
                descriptor.value_payload_compression_id,
            )?,
        })
    }
}

impl Store {
    /// Reads persistent store identity from `STORE` without opening segment files.
    ///
    /// This does not acquire the writer lock, validate `MANIFEST`, or run any
    /// catalog housekeeping. It is intended for tools that need to discover the
    /// metadata required by [`OpenOptions::read_write`] or [`OpenOptions::read_only`].
    pub fn inspect(root: impl AsRef<Path>) -> Result<StoreInfo> {
        let paths = StorePaths::new(root);
        let descriptor = paths
            .load_descriptor()?
            .ok_or(CatalogMismatch::MissingStore)?;
        StoreInfo::from_descriptor(descriptor)
    }

    /// Opens an existing store rooted at `root`.
    pub fn open(root: impl Into<PathBuf>, options: OpenOptions) -> Result<Self> {
        open_existing(root.into(), options, None)
    }
}

pub(super) fn open_existing(
    root: PathBuf,
    options: OpenOptions,
    pre_acquired_writer_lock: Option<WriterLock>,
) -> Result<Store> {
    options.validate()?;
    let paths = StorePaths::new(&root);
    let descriptor = paths
        .load_descriptor()?
        .ok_or(CatalogMismatch::MissingStore)?;
    descriptor.validate_structure()?;
    if descriptor.metadata != options.expected_metadata {
        return Err(CatalogMismatch::Metadata.into());
    }
    let block_checksum = resolve_block_checksum(descriptor.block_checksum_id)?;
    let value_payload_compression =
        resolve_value_payload_compression(descriptor.value_payload_compression_id)?;

    // A writer takes the advisory lock before reading the manifest so that
    // any later commit runs under the same lock without a gap a second writer
    // could slip into.
    let writer_lock = if options.read_only {
        None
    } else if let Some(writer_lock) = pre_acquired_writer_lock {
        Some(writer_lock)
    } else {
        Some(WriterLock::acquire(paths.lock_file())?)
    };

    let manifest = paths
        .load_manifest()?
        .ok_or(CatalogMismatch::MissingManifest)?;
    manifest.validate_structure(descriptor.key_len)?;

    // Only a writer mutates catalog housekeeping on open. Segment garbage
    // collection is explicit so it cannot race a read-only open that already
    // read an older manifest but has not opened all referenced segments yet.
    if writer_lock.is_some() {
        paths.ensure_dirs()?;
        paths.remove_stale_catalog_temps();
    }

    build_store(
        paths,
        descriptor,
        manifest,
        block_checksum,
        value_payload_compression,
        options.verify_block_checksums,
        writer_lock,
    )
}

/// Builds the shared runtime state from validated catalog data, opening every
/// referenced segment file.
///
/// A referenced segment that fails to open (missing, corrupt, or
/// footer-mismatched) is a dead entry: invisible now, dropped at the next
/// manifest publication.
fn build_store(
    paths: StorePaths,
    descriptor: StoreDescriptor,
    manifest: StoreManifest,
    block_checksum: BlockChecksumKind,
    value_payload_compression: ValuePayloadCompressionKind,
    verify_block_checksums: bool,
    writer_lock: Option<WriterLock>,
) -> Result<Store> {
    let geometry = StoreGeometry {
        key_len: descriptor.key_len,
        value_layout: descriptor.value_layout,
        block_checksum,
        value_payload_compression,
    };
    let mut main_segment_states = Vec::new();
    let mut patch_segment_states = Vec::new();
    for entry in &manifest.segments {
        let path = paths.final_segment(entry.segment_id);
        let segment = match OpenedSegment::open(path, SegmentOpenOptions {
            expected_key_len: geometry.key_len,
            expected_value_layout: geometry.value_layout,
            expected_block_checksum: geometry.block_checksum,
            expected_value_payload_compression: geometry.value_payload_compression,
            expected_fingerprint: entry.fingerprint,
        })? {
            Some(segment)
                if entry.matches_segment(
                    &segment.min_key,
                    &segment.max_key,
                    segment.fingerprint,
                ) =>
            {
                segment
            }
            _ => continue,
        };
        let segment_state = Arc::new(SegmentState::from_opened(entry.segment_id, segment));
        if entry.is_main() {
            main_segment_states.push(segment_state);
        } else {
            patch_segment_states.push(segment_state);
        }
    }

    Ok(Store {
        inner: Arc::new(StoreInner {
            paths,
            metadata: descriptor.metadata.clone(),
            geometry,
            verify_block_checksums,
            commit_lock: Mutex::new(()),
            state: RwLock::new(StoreState::new(
                manifest,
                main_segment_states,
                patch_segment_states,
            )),
            writer_lock,
        }),
    })
}

fn resolve_block_checksum(format_id: u32) -> Result<BlockChecksumKind> {
    BlockChecksumKind::from_format_id(format_id)
        .ok_or(CatalogMismatch::UnsupportedBlockChecksum { format_id }.into())
}

fn resolve_value_payload_compression(format_id: u32) -> Result<ValuePayloadCompressionKind> {
    ValuePayloadCompressionKind::from_format_id(format_id)
        .ok_or(CatalogMismatch::UnsupportedValuePayloadCompression { format_id }.into())
}
