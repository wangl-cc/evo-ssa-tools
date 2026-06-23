use std::num::NonZeroU32;

use segment_cache_store::{
    CommitOptions, Error, InputError, OptionsError, Result, Store, ValueLayout,
    ValuePayloadCompressionPolicy,
};

use crate::support::api::{
    commit_entries, create_options, create_options_with_key_len, create_store, create_store_with,
    make_key, make_value, open_options, reopen_store,
};

#[test]
fn wrong_length_keys_are_rejected() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let mut batch = store.begin_batch();
    batch.push(b"short", b"value")?;
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
fn invalid_store_options_are_rejected() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let invalid_options = vec![
        create_options_with_key_len(0),
        create_options_with_key_len(usize::MAX),
    ];
    for invalid in invalid_options {
        let error = match Store::create(tempdir.path(), invalid) {
            Ok(_) => panic!("invalid store options should be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            Error::Input(InputError::InvalidOptions(
                OptionsError::KeyLenZero | OptionsError::KeyLenTooLarge
            ))
        ));
    }

    for invalid in [
        CommitOptions::default().with_flush_threshold_records(0),
        CommitOptions::default().with_flush_threshold_bytes(0),
        CommitOptions::default().with_value_payload_compression_policy(
            ValuePayloadCompressionPolicy::DEFAULT.with_min_saved_percent(101),
        ),
    ] {
        let store = create_store(&tempfile::tempdir()?)?;
        let error = store
            .commit_batch_with_options(store.begin_batch(), &invalid)
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Input(InputError::InvalidOptions(
                OptionsError::FlushThresholdRecordsZero
                    | OptionsError::FlushThresholdBytesZero
                    | OptionsError::CompressionMinSavedPercentTooLarge
            ))
        ));
    }

    if let Some(oversized_target_block_size) = usize::try_from(u32::MAX)
        .ok()
        .and_then(|max| max.checked_add(1))
    {
        let store = create_store(&tempfile::tempdir()?)?;
        let error = store
            .commit_batch_with_options(
                store.begin_batch(),
                &CommitOptions::default().with_target_block_size(oversized_target_block_size),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Input(InputError::InvalidOptions(
                OptionsError::TargetBlockSizeTooLarge
            ))
        ));
    }
    Ok(())
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
        open_options()
            .with_block_checksum_verification(false)
            .with_read_only(true),
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
    commit_entries(&store, &[(key.clone(), value.clone())], true)?;

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
    commit_entries(&store, &entries, true)?;

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
    let error = commit_entries(
        &store,
        &[
            (make_key(1, 0, 0), make_value(1, 32)),
            (make_key(1, 0, 1), make_value(2, 31)),
        ],
        true,
    )
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
    let error = commit_entries(&store, &entries, true).unwrap_err();
    assert!(matches!(
        error,
        Error::Input(InputError::DuplicateKeyInBatch)
    ));
    Ok(())
}

#[test]
fn non_overlapping_gap_insert_is_allowed() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;

    let early = vec![
        (make_key(1, 0, 0), make_value(1, 8)),
        (make_key(1, 0, 1), make_value(2, 8)),
    ];
    let late = vec![
        (make_key(1, 0, 4), make_value(4, 8)),
        (make_key(1, 0, 5), make_value(5, 8)),
    ];
    let middle = vec![
        (make_key(1, 0, 2), make_value(6, 8)),
        (make_key(1, 0, 3), make_value(7, 8)),
    ];

    commit_entries(&store, &early, true)?;
    commit_entries(&store, &late, true)?;
    commit_entries(&store, &middle, true)?;

    let mut expected = early;
    expected.extend(middle);
    expected.extend(late);
    assert_eq!(store.iter_all()?.collect::<Result<Vec<_>>>()?, expected);
    Ok(())
}

#[test]
fn interleaving_commit_rebuilds_the_intersecting_region() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let first = vec![
        (make_key(1, 1, 0), make_value(1, 8)),
        (make_key(1, 1, 10), make_value(2, 8)),
    ];
    // A key landing inside the published [0, 10] range no longer rejects; the
    // commit rebuilds that region by merging the batch with the segment.
    let interleaving = vec![(make_key(1, 1, 5), make_value(3, 8))];

    commit_entries(&store, &first, true)?;
    commit_entries(&store, &interleaving, true)?;

    assert_eq!(store.fetch_one(&make_key(1, 1, 0))?, Some(make_value(1, 8)));
    assert_eq!(store.fetch_one(&make_key(1, 1, 5))?, Some(make_value(3, 8)));
    assert_eq!(
        store.fetch_one(&make_key(1, 1, 10))?,
        Some(make_value(2, 8))
    );
    assert_eq!(store.iter_all()?.collect::<Result<Vec<_>>>()?, vec![
        (make_key(1, 1, 0), make_value(1, 8)),
        (make_key(1, 1, 5), make_value(3, 8)),
        (make_key(1, 1, 10), make_value(2, 8)),
    ]);
    Ok(())
}
