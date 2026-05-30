//! Store migration utilities.
//!
//! Use these helpers when you need to move cached data between crate-provided storage locations or
//! transcode it from one codec to another.
//!
//! # Choosing the right function
//!
//! - [`copy`]: same codec, new storage location (e.g., `Fjall3Store` → new keyspace).
//! - [`copy_transcoded`]: new codec, same or new storage location (e.g., `Bitcode06` → `Postcard`).

use super::{codec::CodecEngine, storage};
use crate::error::Result;

/// Stores that support full-scan iteration over all cached entries.
///
/// This trait is the foundation for store migration.
pub trait IterableStore {
    /// Call `f` for every `(key, encoded_value)` pair in the store.
    ///
    /// Iteration stops early if `f` returns `Err`.
    fn iter_encoded<F>(&self, f: F) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]) -> Result<()>;
}

#[cfg(feature = "fjall3")]
impl IterableStore for super::storage::Fjall3Store {
    fn iter_encoded<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]) -> Result<()>,
    {
        for guard in self.handle.iter() {
            let (key, value) = guard.into_inner().map_err(storage::StorageError::from)?;
            f(key.as_ref(), value.as_ref())?;
        }
        Ok(())
    }
}

/// Statistics returned by [`copy`] and [`copy_transcoded`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MigrateStats {
    /// Number of entries successfully written to the destination store.
    pub migrated: usize,
    /// Number of entries skipped because re-encoding was rejected by the destination engine.
    ///
    /// Always zero for [`copy`] (raw bytes are never re-encoded).
    pub skipped: usize,
}

/// Copy all raw encoded entries from `src` to `dst` without any transcoding.
///
/// Use this when migrating to a new storage location while keeping the same codec engine.
/// Entries already present in `dst` are overwritten.
///
/// # Unsupported in-place migration
///
/// `src` and `dst` must be distinct stores. Passing the same store for both is unsupported:
/// backends that hold a read lock for the entire iteration may deadlock on any write within the
/// closure.
///
/// # Example
///
/// ```no_run
/// use ssa_workflow::{cache::migrate::copy, prelude::*};
///
/// // Open src and dst storage locations, e.g. Fjall3Store::open(...).
/// // let stats = copy(&src, &dst).unwrap();
/// ```
pub fn copy<Src, Dst>(src: &Src, dst: &Dst) -> Result<MigrateStats>
where
    Src: IterableStore,
    Dst: storage::EncodedStorage,
{
    let mut stats = MigrateStats::default();
    src.iter_encoded(|key, encoded| {
        dst.store_encoded(key, encoded)?;
        stats.migrated += 1;
        Ok(())
    })?;
    Ok(stats)
}

/// Copy all entries from `src` to `dst`, decoding with `from_engine` and re-encoding with
/// `to_engine`.
///
/// This is the primary migration path when changing codec engines (e.g., `Bitcode06` →
/// `Postcard`). Entries that fail to re-encode are skipped with a warning, matching the
/// behavior of [`Cache::store`](super::Cache::store). Entries that fail to decode
/// abort the migration with an error.
///
/// # Unsupported in-place migration
///
/// `src` and `dst` must be distinct stores. Passing the same store for both is unsupported:
/// backends that hold a read lock for the entire iteration may deadlock on any write within the
/// closure.
///
/// In-place transcoding is intentionally not provided: a failure partway through would leave
/// the store in a mixed-codec state that cannot be rolled back.
///
/// # Example
///
/// ```no_run
/// use ssa_workflow::{cache::migrate::copy_transcoded, prelude::*};
///
/// // Open src and dst using any `IterableStore` + `EncodedStorage` implementations,
/// // e.g. Fjall3Store::open(...).
/// // let stats = copy_transcoded::<MyType, _, _, _, _>(&src, &mut from_engine, &dst, &mut to_engine).unwrap();
/// ```
pub fn copy_transcoded<T, Src, Dst, FromCE, ToCE>(
    src: &Src,
    from_engine: &mut FromCE,
    dst: &Dst,
    to_engine: &mut ToCE,
) -> Result<MigrateStats>
where
    Src: IterableStore,
    Dst: storage::EncodedStorage,
    FromCE: CodecEngine<T>,
    ToCE: CodecEngine<T>,
{
    let mut stats = MigrateStats::default();
    src.iter_encoded(|key, encoded| {
        let value = from_engine.decode(encoded)?;
        match to_engine.encode(&value) {
            Ok(re_encoded) => {
                dst.store_encoded(key, re_encoded)?;
                stats.migrated += 1;
            }
            Err(reason) => {
                warn!("[ssa-workflow] skipping migration entry: {reason}");
                stats.skipped += 1;
            }
        }
        Ok(())
    })?;
    Ok(stats)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::cache::{
        Cache, CloneShared, EncodedCache,
        codec::{CodecEngine, Error as CodecError, SkipReason, fixtures::FixtureEngine},
        storage::{EncodedStorage, StorageResult},
    };

    /// Minimal in-memory store that implements both `EncodedStorage` and `IterableStore`.
    #[derive(Default, Clone)]
    struct TestStore(Arc<Mutex<std::collections::HashMap<Vec<u8>, Vec<u8>>>>);

    impl EncodedStorage for TestStore {
        type Encoded<'a>
            = Vec<u8>
        where
            Self: 'a;

        fn fetch_encoded(&self, key: &[u8]) -> StorageResult<Option<Self::Encoded<'_>>> {
            Ok(self.0.lock().unwrap().get(key).cloned())
        }

        fn store_encoded(&self, key: &[u8], encoded: &[u8]) -> StorageResult<()> {
            self.0
                .lock()
                .unwrap()
                .insert(key.to_owned(), encoded.to_vec());
            Ok(())
        }
    }

    impl IterableStore for TestStore {
        fn iter_encoded<F>(&self, mut f: F) -> Result<()>
        where
            F: FnMut(&[u8], &[u8]) -> Result<()>,
        {
            for (k, v) in self.0.lock().unwrap().iter() {
                f(k, v)?;
            }
            Ok(())
        }
    }

    fn make_store(entries: &[(&[u8], u32)]) -> TestStore {
        let store = TestStore::default();
        let mut cache = EncodedCache::new(store.clone(), FixtureEngine::default());
        for &(key, value) in entries {
            cache.store(key, &value).unwrap();
        }
        store
    }

    // -- copy ---------------------------------------------------------------

    #[test]
    fn copy_copies_all_entries() -> Result<()> {
        let src = make_store(&[(b"k1", 1), (b"k2", 2), (b"k3", 3)]);
        let dst = TestStore::default();
        let stats = copy(&src, &dst)?;
        assert_eq!(stats.migrated, 3);
        assert_eq!(stats.skipped, 0);

        let mut cache = EncodedCache::new(dst.clone(), FixtureEngine::default());
        assert_eq!(cache.fetch(b"k1")?, Some(1u32));
        assert_eq!(cache.fetch(b"k2")?, Some(2u32));
        assert_eq!(cache.fetch(b"k3")?, Some(3u32));
        Ok(())
    }

    #[test]
    fn copy_empty_source() -> Result<()> {
        let src = TestStore::default();
        let dst = TestStore::default();
        let stats = copy(&src, &dst)?;
        assert_eq!(stats, MigrateStats::default());
        Ok(())
    }

    // -- copy_transcoded ----------------------------------------------------

    #[test]
    fn copy_transcoded_roundtrips() -> Result<()> {
        let src = make_store(&[(b"k1", 1), (b"k2", 2), (b"k3", 3)]);
        let dst = TestStore::default();
        let stats = copy_transcoded::<u32, _, _, _, _>(
            &src,
            &mut FixtureEngine::default(),
            &dst,
            &mut FixtureEngine::default(),
        )?;
        assert_eq!(stats.migrated, 3);
        assert_eq!(stats.skipped, 0);

        let mut cache = EncodedCache::new(dst.clone(), FixtureEngine::default());
        assert_eq!(cache.fetch(b"k1")?, Some(1u32));
        assert_eq!(cache.fetch(b"k2")?, Some(2u32));
        assert_eq!(cache.fetch(b"k3")?, Some(3u32));
        Ok(())
    }

    #[test]
    fn copy_transcoded_aborts_on_decode_error() {
        let src = TestStore::default();
        src.store_encoded(b"bad", b"not valid fixture encoding")
            .unwrap();
        let dst = TestStore::default();
        let result = copy_transcoded::<u32, _, _, _, _>(
            &src,
            &mut FixtureEngine::default(),
            &dst,
            &mut FixtureEngine::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn copy_transcoded_skips_on_encode_rejection() -> Result<()> {
        #[derive(Default)]
        struct AlwaysSkipEngine;

        impl crate::cache::codec::CloneFresh for AlwaysSkipEngine {
            fn clone_fresh(&self) -> Self {
                Self
            }
        }

        impl CodecEngine<u32> for AlwaysSkipEngine {
            const VALUE_FORMAT: crate::cache::codec::ValueFormat =
                crate::cache::codec::ValueFormat::new("test-always-skip-u32-v1");

            fn encode(&mut self, _: &u32) -> std::result::Result<&[u8], SkipReason> {
                Err(SkipReason::EncodedValueTooLarge {
                    encoded_len: 100,
                    max_len: 0,
                })
            }

            fn decode(&mut self, _: &[u8]) -> std::result::Result<u32, CodecError> {
                unreachable!()
            }
        }

        let src = make_store(&[(b"k1", 1), (b"k2", 2), (b"k3", 3)]);
        let dst = TestStore::default();
        let stats = copy_transcoded::<u32, _, _, _, _>(
            &src,
            &mut FixtureEngine::default(),
            &dst,
            &mut AlwaysSkipEngine,
        )?;
        assert_eq!(stats.migrated, 0);
        assert_eq!(stats.skipped, 3);
        Ok(())
    }

    #[test]
    fn copy_transcoded_empty_source() -> Result<()> {
        let src = TestStore::default();
        let dst = TestStore::default();
        let stats = copy_transcoded::<u32, _, _, _, _>(
            &src,
            &mut FixtureEngine::default(),
            &dst,
            &mut FixtureEngine::default(),
        )?;
        assert_eq!(stats, MigrateStats::default());
        Ok(())
    }

    // -- IterableStore: fjall3 ----------------------------------------------

    #[cfg(feature = "fjall3")]
    #[test]
    fn fjall3_iter_encoded_roundtrips() -> Result<()> {
        use storage::{Fjall3Store, StorageError};

        let tmp = tempfile::tempdir().unwrap();
        let db = fjall3::Database::builder(&tmp)
            .open()
            .map_err(StorageError::from)?;
        let src = Fjall3Store::open(db, "src")?;
        let mut src_cache = EncodedCache::new(src.clone_shared(), FixtureEngine::default());
        src_cache.store(b"k1", &1u32)?;
        src_cache.store(b"k2", &2u32)?;

        let tmp2 = tempfile::tempdir().unwrap();
        let db2 = fjall3::Database::builder(&tmp2)
            .open()
            .map_err(StorageError::from)?;
        let dst = Fjall3Store::open(db2, "dst")?;
        let stats = copy(&src, &dst)?;
        assert_eq!(stats.migrated, 2);

        let mut dst_cache = EncodedCache::new(dst, FixtureEngine::default());
        assert_eq!(dst_cache.fetch(b"k1")?, Some(1u32));
        assert_eq!(dst_cache.fetch(b"k2")?, Some(2u32));
        Ok(())
    }

    #[cfg(feature = "fjall3")]
    #[test]
    fn fjall3_iter_encoded_empty() -> Result<()> {
        use storage::{Fjall3Store, StorageError};

        let tmp = tempfile::tempdir().unwrap();
        let db = fjall3::Database::builder(&tmp)
            .open()
            .map_err(StorageError::from)?;
        let store = Fjall3Store::open(db, "empty")?;
        let mut count = 0usize;
        store.iter_encoded(|_, _| {
            count += 1;
            Ok(())
        })?;
        assert_eq!(count, 0);
        Ok(())
    }
}
