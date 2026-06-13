//! Store creation options and implementation.

use std::{num::NonZeroU32, path::PathBuf};

use super::OpenOptions;
use crate::{
    engine::{
        io::WriterLock,
        open::open_existing,
        paths::{self, StorePaths},
    },
    error::{InputError, OptionsError, Result},
    format::{StoreMetadata, ValueLayout, manifest::StoreManifest, store_file::StoreDescriptor},
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
    pub key_len: usize,
    /// Value layout shared by all visible segments.
    pub value_layout: ValueLayout,
    /// Opaque caller compatibility metadata for this namespace.
    pub metadata: StoreMetadata,
}

impl CreateOptions {
    /// Creates default creation options for a store with fixed-width keys.
    pub fn new(key_len: usize, metadata: StoreMetadata) -> Self {
        Self {
            key_len,
            value_layout: ValueLayout::VARIABLE,
            metadata,
        }
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

    pub(crate) fn validate(&self) -> Result<()> {
        if self.key_len == 0 {
            return Err(OptionsError::KeyLenZero.into());
        }
        if self.key_len > u32::MAX as usize {
            return Err(OptionsError::KeyLenTooLarge.into());
        }
        Ok(())
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
        options.validate()?;
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
        );
        // Write `MANIFEST` first and `STORE` last. A crash between the two leaves
        // a root with no `STORE`, which `create` can safely re-create and `open`
        // rejects as missing.
        let manifest = StoreManifest::new(options.key_len);
        paths::publish_manifest(&paths, &manifest)?;
        paths::publish_descriptor(&paths, &descriptor)?;
        open_existing(root, OpenOptions::new(options.metadata), Some(writer_lock))
    }
}
