use std::num::NonZeroU32;

#[cfg(feature = "value-compression")]
use segment_cache_store::{CompressionPolicyError, ValuePayloadCompressionPolicy};
use segment_cache_store::{
    CreateOptions, Error, InputError, OpenOptions, OptionsError, Result, Store, ValueLayout,
    WriteBatch,
};

use crate::support::api::{
    commit_entries, create_options, create_options_with_key_len, create_store, create_store_with,
    make_key, make_value, metadata, open_options, reopen_store, test_block_checksum,
};

#[test]
fn wrong_length_keys_are_rejected() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let mut batch = WriteBatch::new();
    batch.push(b"short", b"value");
    let error = match store.commit_batch(batch) {
        Ok(_) => panic!("wrong-length key should be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        Error::Input(InputError::WrongKeyLength { .. })
    ));
    Ok(())
}

#[test]
fn invalid_create_options_are_rejected() {
    assert!(matches!(
        CreateOptions::new(0, metadata(), test_block_checksum()),
        Err(OptionsError::KeyLenZero)
    ));
    assert!(matches!(
        CreateOptions::new(usize::MAX, metadata(), test_block_checksum(),),
        Err(OptionsError::KeyLenTooLarge)
    ));
}

#[test]
fn fixed_value_layout_rejects_lengths_above_64_mib() {
    let tempdir = tempfile::tempdir().expect("tempdir should be created");
    let options = create_options().with_fixed_value_len(non_zero_u32(64 * 1024 * 1024 + 1));
    let error = match create_store_with(&tempdir, options) {
        Ok(_) => panic!("oversized fixed values should be rejected"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        Error::Input(InputError::InvalidOptions(
            OptionsError::FixedValueLenTooLarge
        ))
    ));
}

#[cfg(feature = "value-compression")]
#[test]
fn compression_policy_rejects_invalid_saved_percentage() {
    let error = ValuePayloadCompressionPolicy::new(64, 101)
        .expect_err("saved percentage above 100 must be rejected");
    assert_eq!(error, CompressionPolicyError::MinSavedPercentTooLarge);
    assert_eq!(
        ValuePayloadCompressionPolicy::DEFAULT.with_min_saved_percent(101),
        Err(CompressionPolicyError::MinSavedPercentTooLarge)
    );
    assert!(matches!(
        Error::from(error),
        Error::Input(InputError::InvalidOptions(OptionsError::CompressionPolicy(
            CompressionPolicyError::MinSavedPercentTooLarge
        )))
    ));

    let policy =
        ValuePayloadCompressionPolicy::new(64, 20).expect("compression policy should be valid");
    assert_eq!(policy.min_try_len(), 64);
    assert_eq!(policy.min_saved_percent(), 20);
}

#[test]
fn writable_open_requires_block_checksum_verification() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    drop(store);

    let error = match Store::open(
        tempdir.path(),
        open_options().with_block_checksum_verification(false),
    ) {
        Ok(_) => panic!("writable unchecked open should be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        Error::Input(InputError::InvalidOptions(
            OptionsError::WritableStoreRequiresBlockChecksumVerification
        ))
    ));

    let reader = Store::open(
        tempdir.path(),
        OpenOptions::read_only(metadata()).with_block_checksum_verification(false),
    )?;
    assert_eq!(reader.iter_all()?.count(), 0);
    Ok(())
}

#[test]
fn default_options_support_short_keys() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store_with(&tempdir, create_options_with_key_len(8))?;
    let key = 7u64.to_be_bytes().to_vec();
    let value = make_value(7, 8);
    commit_entries(&store, &[(key.clone(), value.clone())])?;

    assert_eq!(store.fetch_one(&key)?, Some(value));
    Ok(())
}

#[test]
fn fixed_value_layout_round_trips_all_read_paths() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store_with(
        &tempdir,
        create_options().with_value_layout(ValueLayout::fixed(non_zero_u32(32))),
    )?;
    let entries = vec![
        (make_key(1, 0, 0), make_value(1, 32)),
        (make_key(1, 0, 1), make_value(2, 32)),
        (make_key(1, 0, 2), make_value(3, 32)),
    ];
    commit_entries(&store, &entries)?;

    assert_eq!(store.fetch_one(&entries[1].0)?, Some(entries[1].1.clone()));
    let key_refs = entries
        .iter()
        .map(|(key, _)| key.as_slice())
        .collect::<Vec<_>>();
    assert_eq!(
        store.contains_many_ordered(key_refs.iter().copied())?,
        vec![true, true, true]
    );
    assert_eq!(
        store.fetch_many_ordered(key_refs.iter().copied())?,
        entries
            .iter()
            .map(|(_, value)| Some(value.clone()))
            .collect::<Vec<_>>()
    );
    let scanned: Result<Vec<_>> = store.iter_all()?.collect();
    assert_eq!(scanned?, entries);
    drop(store);

    let reopened = reopen_store(&tempdir)?;
    assert_eq!(reopened.iter_all()?.collect::<Result<Vec<_>>>()?, entries);
    Ok(())
}

#[test]
fn fixed_value_layout_rejects_wrong_value_length() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store_with(
        &tempdir,
        create_options().with_fixed_value_len(non_zero_u32(32)),
    )?;
    let error = commit_entries(&store, &[
        (make_key(1, 0, 0), make_value(1, 32)),
        (make_key(1, 0, 1), make_value(2, 31)),
    ])
    .unwrap_err();
    assert!(matches!(
        error,
        Error::Input(InputError::WrongValueLength { .. })
    ));
    assert_eq!(store.iter_all()?.count(), 0);
    Ok(())
}

fn non_zero_u32(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).expect("test value length is non-zero")
}

#[test]
fn duplicate_keys_inside_batch_are_rejected() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let key = make_key(1, 1, 1);
    let entries = vec![(key.clone(), make_value(1, 8)), (key, make_value(2, 8))];
    let error = commit_entries(&store, &entries).unwrap_err();
    assert!(matches!(
        error,
        Error::Input(InputError::DuplicateKeyInBatch)
    ));
    Ok(())
}
