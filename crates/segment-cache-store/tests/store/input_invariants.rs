use crate::common::*;

#[test]
fn wrong_length_keys_are_rejected() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let mut batch = store.begin_batch();
    batch.push(b"short", b"value")?;
    let error = match store.commit_batch(batch) {
        Ok(_) => panic!("wrong-length key should be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::WrongKeyLength { .. }));
    Ok(())
}

#[test]
fn invalid_store_options_are_rejected() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let mut invalid_options = vec![
        StoreOptions::new(tempdir.path(), 16).with_shard_count(0),
        StoreOptions::new(tempdir.path(), 16).with_shard_count(usize::MAX),
        StoreOptions::new(tempdir.path(), 0),
        StoreOptions::new(tempdir.path(), usize::MAX),
        StoreOptions::new(tempdir.path(), 16).with_fixed_value_len(0),
        StoreOptions::new(tempdir.path(), 16).with_fixed_value_len(usize::MAX),
        StoreOptions::new(tempdir.path(), 16).with_shard_key_offset(17),
        StoreOptions::new(tempdir.path(), 16).with_flush_threshold_records(0),
        StoreOptions::new(tempdir.path(), 16).with_flush_threshold_bytes(0),
    ];
    if let Some(oversized_target_block_size) = usize::try_from(u32::MAX)
        .ok()
        .and_then(|max| max.checked_add(1))
    {
        invalid_options.push(
            StoreOptions::new(tempdir.path(), 16)
                .with_target_block_size(oversized_target_block_size),
        );
    }
    for invalid in invalid_options {
        let error = match Store::open(invalid) {
            Ok(_) => panic!("invalid store options should be rejected"),
            Err(error) => error,
        };
        assert!(matches!(error, Error::InvalidOptions { .. }));
    }
    Ok(())
}

#[test]
fn default_options_support_short_keys() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(StoreOptions::new(tempdir.path(), 8).with_shard_count(1))?;
    let key = 7u64.to_be_bytes().to_vec();
    let value = make_value(7, 8);
    commit_entries(&store, &[(key.clone(), value.clone())], true)?;

    assert_eq!(store.fetch_one(&key)?, Some(value));
    Ok(())
}

#[test]
fn fixed_value_layout_round_trips_all_read_paths() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let options = options(&tempdir).with_value_layout(ValueLayout::Fixed { value_len: 32 });
    let store = Store::open(options.clone())?;
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
    assert_eq!(store.probe_ordered(key_refs.iter().copied())?, vec![
        true, true, true
    ]);
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

    let reopened = Store::open(options)?;
    assert_eq!(reopened.iter_all()?.collect::<Result<Vec<_>>>()?, entries);
    Ok(())
}

#[test]
fn fixed_value_layout_rejects_wrong_value_length() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir).with_fixed_value_len(32))?;
    let error = commit_entries(
        &store,
        &[
            (make_key(1, 0, 0), make_value(1, 32)),
            (make_key(1, 0, 1), make_value(2, 31)),
        ],
        true,
    )
    .unwrap_err();
    assert!(matches!(error, Error::WrongValueLength { .. }));
    assert_eq!(store.iter_all()?.count(), 0);
    Ok(())
}

#[test]
fn duplicate_keys_inside_batch_are_rejected() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let key = make_key(1, 1, 1);
    let entries = vec![(key.clone(), make_value(1, 8)), (key, make_value(2, 8))];
    let error = commit_entries(&store, &entries, true).unwrap_err();
    assert!(matches!(error, Error::DuplicateKeyInBatch));
    Ok(())
}

#[test]
fn shard_local_out_of_order_append_is_rejected() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir).with_shard_count(1))?;
    let first = make_key(1, 1, 10);
    let second = make_key(1, 1, 1);

    commit_entries(&store, &[(first, make_value(1, 8))], true)?;
    let error = commit_entries(&store, &[(second, make_value(2, 8))], true).unwrap_err();
    assert!(matches!(error, Error::OutOfOrderAppend { .. }));
    Ok(())
}
