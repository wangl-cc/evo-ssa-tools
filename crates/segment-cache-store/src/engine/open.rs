//! Store open options and implementation.

use std::{path::PathBuf, sync::Arc};

use parking_lot::{Mutex, RwLock};

use crate::{
    engine::{
        gc::garbage_collect_unreferenced,
        io::WriterLock,
        paths::{self, StorePaths},
        runtime::{SegmentState, StoreGeometry, StoreInner, StoreState},
        segment_file::{OpenedSegment, SegmentOpenOptions},
    },
    error::{OptionsError, Result},
    format::{
        CatalogMismatch, StoreMetadata, manifest::StoreManifest, store_file::StoreDescriptor,
    },
    store::Store,
};

/// Options consumed only when opening an existing store root.
///
/// Persistent geometry such as key length and value layout is read from
/// `STORE`. Open options are read-time policy and compatibility checks, not
/// store creation metadata.
#[derive(Clone, Debug)]
pub struct OpenOptions {
    /// Required caller metadata for this namespace.
    pub expected_metadata: StoreMetadata,
    /// Whether data-block CRC32C checksums are verified on read.
    pub verify_block_checksums: bool,
    /// Whether this open is read-only.
    ///
    /// A read-only open takes no writer lock and runs no garbage collection, so
    /// multiple read-only opens of one root can coexist with a writer.
    pub read_only: bool,
}

impl OpenOptions {
    /// Creates open options that require the persisted store metadata to match.
    pub fn new(expected_metadata: StoreMetadata) -> Self {
        Self {
            expected_metadata,
            verify_block_checksums: true,
            read_only: false,
        }
    }

    /// Enables or disables block checksum verification on reads.
    ///
    /// Disabling verification is only accepted for read-only opens; writable
    /// handles must keep verification enabled so corrupt bytes cannot be merged
    /// into freshly checksummed replacement segments.
    pub fn with_block_checksum_verification(mut self, verify: bool) -> Self {
        self.verify_block_checksums = verify;
        self
    }

    /// Opens the store without acquiring the writer lock or running GC.
    ///
    /// A read-only handle never touches the filesystem beyond reads: it takes
    /// no lock, runs no GC, and creates no directories. Commits on a read-only
    /// handle are rejected with [`crate::InputError::ReadOnlyStore`].
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if !self.read_only && !self.verify_block_checksums {
            return Err(OptionsError::WritableStoreRequiresBlockChecksumVerification.into());
        }
        Ok(())
    }
}

impl Store {
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
    let descriptor = paths::load_descriptor(&paths)?.ok_or(CatalogMismatch::MissingStore)?;
    descriptor.validate_structure()?;
    if descriptor.metadata != options.expected_metadata {
        return Err(CatalogMismatch::Metadata.into());
    }

    // A writer takes the advisory lock before reading the manifest so that
    // garbage collection and any later commit run under the same lock without
    // a gap a second writer could slip into.
    let writer_lock = if options.read_only {
        None
    } else if let Some(writer_lock) = pre_acquired_writer_lock {
        Some(writer_lock)
    } else {
        Some(WriterLock::acquire(paths.lock_file())?)
    };

    let manifest = paths::load_manifest(&paths)?.ok_or(CatalogMismatch::MissingManifest)?;
    manifest.validate_structure(descriptor.key_len)?;

    // Only a writer mutates the filesystem on open. A read-only open must
    // work on a read-only mount; a missing `segments/` directory just means
    // every manifest entry is dead.
    if writer_lock.is_some() {
        paths.ensure_dirs()?;
        paths.remove_stale_catalog_temps();
        garbage_collect_unreferenced(&paths, &manifest);
    }

    build_store(
        paths,
        descriptor,
        manifest,
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
    verify_block_checksums: bool,
    writer_lock: Option<WriterLock>,
) -> Result<Store> {
    let geometry = StoreGeometry::from_descriptor(&descriptor);
    let mut main_segment_states = Vec::new();
    let mut patch_segment_states = Vec::new();
    for entry in &manifest.segments {
        let path = paths.final_segment(entry.segment_id);
        let segment = match OpenedSegment::open(path, SegmentOpenOptions {
            expected_key_len: geometry.key_len,
            expected_value_layout: geometry.value_layout,
        })? {
            Some(segment) if entry.matches_segment_footer(&segment.min_key, &segment.max_key) => {
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
