//! Store open options and implementation.

use std::sync::Arc;

use super::{
    Catalog, CatalogMismatch, StoreManifest, StorePaths, VisibleSnapshot, WriterLock,
    descriptor::StoreDescriptor, metadata::StoreMetadata,
};
use crate::{
    block::{BlockChecksumKind, ValuePayloadCompressionKind},
    error::{OptionsError, Result},
    segment::{Segment, SegmentGeometry, SegmentOpenOptions},
    value::ValueLayout,
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

/// Fully validated persistent state ready to become an in-process store handle.
pub(crate) struct OpenedStore {
    pub(crate) paths: StorePaths,
    pub(crate) metadata: StoreMetadata,
    pub(crate) geometry: SegmentGeometry,
    pub(crate) verify_block_checksums: bool,
    pub(crate) snapshot: VisibleSnapshot,
    pub(crate) writer_lock: Option<WriterLock>,
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

impl Catalog {
    pub(crate) fn inspect(&self) -> Result<StoreInfo> {
        let paths = &self.paths;
        let descriptor = paths
            .load_descriptor()?
            .ok_or(CatalogMismatch::MissingStore)?;
        StoreInfo::from_descriptor(descriptor)
    }

    pub(crate) fn open(self, options: OpenOptions) -> Result<OpenedStore> {
        self.open_existing(options, None)
    }

    pub(super) fn open_existing(
        self,
        options: OpenOptions,
        pre_acquired_writer_lock: Option<WriterLock>,
    ) -> Result<OpenedStore> {
        options.validate()?;
        let paths = self.paths;
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

        Self::build_inner(
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
    /// A referenced segment that fails to open (missing, corrupt, or mismatched
    /// with its manifest identity) is a dead entry: invisible now, dropped at
    /// the next manifest publication.
    fn build_inner(
        paths: StorePaths,
        descriptor: StoreDescriptor,
        manifest: StoreManifest,
        block_checksum: BlockChecksumKind,
        value_payload_compression: ValuePayloadCompressionKind,
        verify_block_checksums: bool,
        writer_lock: Option<WriterLock>,
    ) -> Result<OpenedStore> {
        let geometry = SegmentGeometry {
            key_len: descriptor.key_len,
            value_layout: descriptor.value_layout,
            block_checksum,
            value_payload_compression,
        };
        let mut main_segment_states = Vec::new();
        let mut patch_segment_states = Vec::new();
        for entry in &manifest.segments {
            let path = paths.final_segment(entry.segment_id);
            let segment = match Segment::open(entry.segment_id, path, SegmentOpenOptions {
                geometry,
                expected_fingerprint: entry.fingerprint,
                expected_min_key: &entry.min_key,
                expected_max_key: &entry.max_key,
            })? {
                Some(segment) => segment,
                _ => continue,
            };
            let segment = Arc::new(segment);
            if entry.is_main() {
                main_segment_states.push(segment);
            } else {
                patch_segment_states.push(segment);
            }
        }

        Ok(OpenedStore {
            paths,
            metadata: descriptor.metadata,
            geometry,
            verify_block_checksums,
            snapshot: VisibleSnapshot::new(manifest, main_segment_states, patch_segment_states),
            writer_lock,
        })
    }
}

fn resolve_block_checksum(format_id: u8) -> Result<BlockChecksumKind> {
    BlockChecksumKind::from_format_id(format_id)
        .ok_or(CatalogMismatch::UnsupportedBlockChecksum { format_id }.into())
}

fn resolve_value_payload_compression(format_id: u8) -> Result<ValuePayloadCompressionKind> {
    ValuePayloadCompressionKind::from_format_id(format_id)
        .ok_or(CatalogMismatch::UnsupportedValuePayloadCompression { format_id }.into())
}
