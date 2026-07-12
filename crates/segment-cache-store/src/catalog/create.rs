//! Store creation options and implementation.

use std::{num::NonZeroU32, path::PathBuf};

use super::OpenOptions;
use crate::{
    block::{BlockChecksumKind, ValuePayloadCompressionKind},
    catalog::{
        descriptor::StoreDescriptor, io::WriterLock, manifest::StoreManifest, open::open_existing,
        paths::StorePaths,
    },
    error::{InputError, OptionsError, Result},
    schema::{StoreMetadata, ValueLayout},
    store::Store,
};

/// Options consumed only when creating a new store root.
///
/// These fields become part of the persistent namespace contract in `STORE`.
/// They are not re-supplied on ordinary open; `Store::open` reads them from
/// `STORE` and checks the caller metadata supplied to
/// `OpenOptions`.
#[derive(Clone, Debug)]
pub struct CreateOptions {
    /// Fixed key length in bytes.
    key_len: usize,
    /// Value layout shared by all visible segments.
    value_layout: ValueLayout,
    /// Block checksum kind persisted for all visible segments.
    block_checksum: BlockChecksumKind,
    /// Value-payload compression kind persisted for all visible segments.
    value_payload_compression: ValuePayloadCompressionKind,
    /// Opaque caller compatibility metadata for this namespace.
    metadata: StoreMetadata,
}

impl CreateOptions {
    /// Creates options for a store with fixed-width keys and an explicit block checksum.
    ///
    /// # Errors
    ///
    /// Returns [`OptionsError::KeyLenZero`] when `key_len` is zero or
    /// [`OptionsError::KeyLenTooLarge`] when it does not fit the v1 format.
    pub fn new(
        key_len: usize,
        metadata: StoreMetadata,
        block_checksum: BlockChecksumKind,
    ) -> std::result::Result<Self, OptionsError> {
        if key_len == 0 {
            return Err(OptionsError::KeyLenZero);
        }
        if key_len > u32::MAX as usize {
            return Err(OptionsError::KeyLenTooLarge);
        }
        Ok(Self {
            key_len,
            value_layout: ValueLayout::VARIABLE,
            block_checksum,
            value_payload_compression: ValuePayloadCompressionKind::DEFAULT,
            metadata,
        })
    }

    /// Selects the physical value layout used by all visible segments.
    pub fn with_value_layout(mut self, value_layout: ValueLayout) -> Self {
        self.value_layout = value_layout;
        self
    }

    /// Enables fixed-value layout and rejects future writes with any other value length.
    pub fn with_fixed_value_len(mut self, value_len: NonZeroU32) -> Self {
        self.value_layout = ValueLayout::fixed(value_len);
        self
    }

    /// Selects the block-level value-payload compression kind.
    pub fn with_value_payload_compression(
        mut self,
        compression: ValuePayloadCompressionKind,
    ) -> Self {
        self.value_payload_compression = compression;
        self
    }
}

impl Store {
    /// Creates a new empty store rooted at `root`.
    ///
    /// Creation fails if persistent store state already exists. Use
    /// `Store::open` for an existing store.
    ///
    /// Creation takes the same stable writer lock used by ordinary writer opens,
    /// so two cooperating creators cannot publish different `STORE` files for
    /// one root. A creator racing an existing writer fails with `WriterLocked`;
    /// once the writer is gone, creating over an existing root fails with
    /// `StoreAlreadyExists`.
    pub fn create(root: impl Into<PathBuf>, options: CreateOptions) -> Result<Self> {
        let root = root.into();
        let paths = StorePaths::new(&root);
        paths.ensure_root()?;
        let writer_lock = WriterLock::acquire(paths.lock_file())?;
        // `STORE` is the creation completion marker, so its presence alone means
        // the root exists. A leftover `MANIFEST` from an aborted creation is
        // overwritten below.
        if paths.store_file().exists() {
            return Err(InputError::StoreAlreadyExists.into());
        }
        paths.ensure_dirs()?;
        let descriptor = StoreDescriptor::new(
            options.metadata.clone(),
            options.key_len,
            options.value_layout,
            options.block_checksum.format_id(),
            options.value_payload_compression.format_id(),
        );
        // Write `MANIFEST` first and `STORE` last. A crash between the two leaves
        // a root with no `STORE`, which `create` can safely re-create and `open`
        // rejects as missing.
        let manifest = StoreManifest::new(options.key_len);
        paths.publish_manifest(&manifest)?;
        paths.publish_descriptor(&descriptor)?;
        open_existing(
            root,
            OpenOptions::read_write(options.metadata),
            Some(writer_lock),
        )
    }
}
