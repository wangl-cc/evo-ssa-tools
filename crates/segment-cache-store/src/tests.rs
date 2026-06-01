use std::{
    fs,
    fs::OpenOptions,
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
};

use crate::{
    CommitStats, Error, Result, Store, StoreOptions, ValueLayout,
    format::{FOOTER_MAGIC, ShardPolicy},
    manifest::StorePaths,
};

fn options(tempdir: &tempfile::TempDir) -> StoreOptions {
    StoreOptions::new(tempdir.path(), 16)
        .with_shard_count(4)
        .with_target_block_size(256)
}

fn make_key(a: u32, b: u32, rep: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(16);
    key.extend_from_slice(&a.to_be_bytes());
    key.extend_from_slice(&b.to_be_bytes());
    key.extend_from_slice(&rep.to_be_bytes());
    key
}

fn make_value(tag: u8, len: usize) -> Vec<u8> {
    vec![tag; len]
}

fn commit_entries(
    store: &Store,
    entries: &[(Vec<u8>, Vec<u8>)],
    sorted: bool,
) -> Result<CommitStats> {
    let mut batch = store.begin_batch();
    for (key, value) in entries {
        batch.push(key, value)?;
    }
    if sorted {
        batch = batch.mark_sorted();
    }
    store.commit_batch(batch)
}

#[test]
fn round_trip_batch_commit_then_fetch() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries = vec![
        (make_key(1, 1, 0), make_value(1, 8)),
        (make_key(1, 2, 0), make_value(2, 128)),
        (make_key(2, 1, 0), make_value(3, 2048)),
    ];
    let stats = commit_entries(&store, &entries, true)?;
    assert_eq!(stats.records, entries.len());

    for (key, value) in &entries {
        assert_eq!(store.fetch_one(key)?, Some(value.clone()));
    }
    Ok(())
}

#[test]
fn sorted_and_unsorted_batches_publish_same_results() -> Result<()> {
    let tempdir_a = tempfile::tempdir()?;
    let tempdir_b = tempfile::tempdir()?;
    let store_a = Store::open(options(&tempdir_a))?;
    let store_b = Store::open(options(&tempdir_b))?;
    let sorted = vec![
        (make_key(1, 0, 0), make_value(1, 8)),
        (make_key(1, 0, 1), make_value(2, 8)),
        (make_key(2, 0, 0), make_value(3, 8)),
    ];
    let mut unsorted = sorted.clone();
    unsorted.swap(0, 2);

    commit_entries(&store_a, &sorted, true)?;
    commit_entries(&store_b, &unsorted, false)?;

    for (key, value) in &sorted {
        assert_eq!(store_a.fetch_one(key)?, Some(value.clone()));
        assert_eq!(store_b.fetch_one(key)?, Some(value.clone()));
    }
    Ok(())
}

#[test]
fn owned_batch_entries_round_trip() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries = vec![
        (make_key(1, 0, 0), make_value(1, 8)),
        (make_key(1, 0, 1), make_value(2, 16)),
    ];
    let mut batch = store.begin_batch().mark_sorted();
    for (key, value) in entries.clone() {
        batch.push_owned(key, value)?;
    }
    store.commit_batch(batch)?;

    let scanned = store.iter_all()?.collect::<Result<Vec<_>>>()?;
    assert_eq!(scanned, entries);
    Ok(())
}

#[test]
fn ordered_probe_matches_fetch_hit_pattern() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries = vec![
        (make_key(1, 0, 0), make_value(1, 8)),
        (make_key(1, 0, 1), make_value(2, 8)),
        (make_key(1, 0, 3), make_value(3, 8)),
    ];
    commit_entries(&store, &entries, true)?;
    let keys = [
        make_key(1, 0, 0),
        make_key(1, 0, 1),
        make_key(1, 0, 2),
        make_key(1, 0, 3),
    ];
    let key_refs: Vec<_> = keys.iter().map(Vec::as_slice).collect();
    let probe = store.probe_ordered(key_refs.iter().copied())?;
    let fetch = store.fetch_many_ordered(key_refs.iter().copied())?;
    assert_eq!(
        probe,
        fetch
            .iter()
            .map(|entry| entry.is_some())
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn ordered_lookup_session_can_restart_from_earlier_block() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries: Vec<_> = (0..64u64)
        .map(|rep| (make_key(1, 0, rep), make_value(rep as u8, 96)))
        .collect();
    commit_entries(&store, &entries, true)?;
    let key_refs: Vec<_> = entries.iter().map(|(key, _)| key.as_slice()).collect();
    let mut lookup = store.lookup_session();

    let first = lookup.fetch_many(key_refs.iter().copied())?;
    let second = lookup.fetch_many(key_refs.iter().copied())?;

    assert_eq!(first, second);
    assert_eq!(second.into_iter().flatten().count(), entries.len());
    Ok(())
}

#[test]
fn range_iteration_returns_globally_sorted_records() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries = vec![
        (make_key(1, 0, 0), make_value(1, 8)),
        (make_key(1, 0, 1), make_value(2, 8)),
        (make_key(2, 0, 0), make_value(3, 8)),
        (make_key(3, 0, 0), make_value(4, 8)),
    ];
    commit_entries(&store, &entries, true)?;

    let range: Result<Vec<_>> = store
        .range(&make_key(1, 0, 1), &make_key(3, 0, 0))?
        .collect();
    let range = range?;
    assert_eq!(range.len(), 2);
    assert!(range.windows(2).all(|window| window[0].0 < window[1].0));
    Ok(())
}

#[test]
fn iter_all_returns_all_records_exactly_once() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries: Vec<_> = (0..32u64)
        .map(|rep| (make_key(1, (rep % 4) as u32, rep), make_value(rep as u8, 8)))
        .collect();
    commit_entries(&store, &entries, true)?;
    let all: Result<Vec<_>> = store.iter_all()?.collect();
    let all = all?;
    assert_eq!(all.len(), entries.len());
    assert!(all.windows(2).all(|window| window[0].0 < window[1].0));
    Ok(())
}

#[test]
fn visit_all_matches_iter_all_order() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries: Vec<_> = (0..32u64)
        .map(|rep| (make_key(1, (rep % 4) as u32, rep), make_value(rep as u8, 8)))
        .collect();
    commit_entries(&store, &entries, true)?;

    let iterated = store.iter_all()?.collect::<Result<Vec<_>>>()?;
    let mut visited = Vec::new();
    store.visit_all(|key, value| visited.push((key.to_vec(), value.to_vec())))?;

    assert_eq!(visited, iterated);
    Ok(())
}

#[test]
fn visit_many_ordered_slice_matches_fetch_many() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries: Vec<_> = (0..16u64)
        .map(|rep| (make_key(1, 0, rep), make_value(rep as u8, 8)))
        .collect();
    commit_entries(&store, &entries, true)?;

    let mut visited = Vec::new();
    let keys = entries
        .iter()
        .map(|(key, _)| key.clone())
        .collect::<Vec<_>>();
    store.visit_many_ordered_slice(&keys, |_, value| {
        visited.push(value.map(ToOwned::to_owned));
    })?;

    assert_eq!(
        visited,
        entries
            .iter()
            .map(|(_, value)| Some(value.clone()))
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn changing_target_block_size_can_reopen_existing_segments() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let first_options = options(&tempdir).with_target_block_size(128);
    let store = Store::open(first_options.clone())?;
    let first_key = make_key(1, 0, 0);
    commit_entries(&store, &[(first_key.clone(), make_value(1, 32))], true)?;
    drop(store);

    let second_options = first_options.with_target_block_size(512);
    let reopened = Store::open(second_options)?;

    assert_eq!(reopened.fetch_one(&first_key)?, Some(make_value(1, 32)));
    Ok(())
}

#[test]
fn manifest_is_line_oriented_v1_text() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    commit_entries(&store, &[(make_key(1, 0, 0), make_value(1, 8))], true)?;

    let manifest = fs::read_to_string(StorePaths::new(tempdir.path()).manifest())?;

    assert!(manifest.starts_with("segment-cache-store manifest v1\n"));
    assert!(manifest.contains("version=1\n"));
    assert!(manifest.contains("value_layout=variable\n"));
    assert!(manifest.contains("[shard "));
    assert!(manifest.contains("segment\tsegment-"));
    assert!(!manifest.contains('{'));
    Ok(())
}

#[test]
fn malformed_manifest_is_rejected() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    fs::write(
        StorePaths::new(tempdir.path()).manifest(),
        "not a segment-cache manifest\n",
    )?;

    let error = match Store::open(options(&tempdir)) {
        Ok(_) => panic!("malformed manifest should be rejected"),
        Err(error) => error,
    };

    assert!(matches!(error, Error::ManifestParse { .. }));
    Ok(())
}

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
    let error = match Store::open(StoreOptions::new(tempdir.path(), 16).with_shard_count(0)) {
        Ok(_) => panic!("zero shards should be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::InvalidOptions { .. }));

    let error = match Store::open(StoreOptions::new(tempdir.path(), 0)) {
        Ok(_) => panic!("zero-length keys should be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::InvalidOptions { .. }));

    let error = match Store::open(StoreOptions::new(tempdir.path(), 16).with_fixed_value_len(0)) {
        Ok(_) => panic!("zero-length fixed values should be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::InvalidOptions { .. }));
    Ok(())
}

#[test]
fn fixed_value_layout_round_trips_all_read_paths() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store =
        Store::open(options(&tempdir).with_value_layout(ValueLayout::Fixed { value_len: 32 }))?;
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
    let store = Store::open(options(&tempdir))?;
    let first = make_key(1, 1, 10);
    let second = make_key(1, 1, 1);
    let shard_policy = ShardPolicy::new(
        store.inner.options.shard_count,
        store.inner.options.shard_key_offset,
    );
    let first_shard = shard_policy.shard_for_key(&first);
    let mut candidate = second.clone();
    while shard_policy.shard_for_key(&candidate) != first_shard {
        let rep = u64::from_be_bytes(candidate[8..16].try_into().expect("rep bytes")) + 1;
        candidate[8..16].copy_from_slice(&rep.to_be_bytes());
    }

    commit_entries(&store, &[(first, make_value(1, 8))], true)?;
    let error = commit_entries(&store, &[(candidate, make_value(2, 8))], true).unwrap_err();
    assert!(matches!(error, Error::OutOfOrderAppend { .. }));
    Ok(())
}

fn first_segment_path(store: &Store) -> PathBuf {
    let state = store.inner.state.read();
    let shard_id = state
        .shards
        .iter()
        .position(|shard| !shard.segments.is_empty())
        .expect("segment should exist");
    state.shards[shard_id].segments[0].path.clone()
}

#[test]
fn corrupted_block_checksum_becomes_miss() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 16))], true)?;
    let path = first_segment_path(&store);
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    file.seek(SeekFrom::Start(24))?;
    file.write_all(&[0xFF])?;
    file.sync_all()?;

    let reopened = Store::open(options(&tempdir))?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    Ok(())
}

#[test]
fn block_checksum_verification_can_be_disabled_for_benchmarks() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let options = options(&tempdir);
    let store = Store::open(options.clone())?;
    let key = make_key(1, 1, 0);
    let value = make_value(9, 16);
    commit_entries(&store, &[(key.clone(), value.clone())], true)?;
    let path = first_segment_path(&store);
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    let value_offset = 16 + key.len() + 4 + 4;
    file.seek(SeekFrom::Start(value_offset as u64))?;
    file.write_all(&[0xFF])?;
    file.sync_all()?;

    let checked = Store::open(options.clone())?;
    assert_eq!(checked.fetch_one(&key)?, None);

    let unchecked = Store::open(options.with_block_checksum_verification(false))?;
    let unchecked_value = unchecked
        .fetch_one(&key)?
        .expect("unchecked read should return corrupted bytes");
    assert_ne!(unchecked_value, value);
    Ok(())
}

#[test]
fn codec_version_mismatch_becomes_miss() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let options = options(&tempdir);
    let store = Store::open(options.clone().with_codec_version(1))?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 16))], true)?;
    drop(store);

    let reopened = Store::open(options.with_codec_version(2))?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    assert_eq!(reopened.probe_ordered([key.as_slice()])?, vec![false]);
    let scanned: Result<Vec<_>> = reopened.iter_all()?.collect();
    assert!(scanned?.is_empty());
    Ok(())
}

#[test]
fn flush_thresholds_split_one_batch_into_multiple_segments() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let options = options(&tempdir).with_flush_threshold_records(2);
    let store = Store::open(options)?;
    let entries = vec![
        (make_key(1, 1, 0), make_value(1, 8)),
        (make_key(1, 1, 1), make_value(2, 8)),
        (make_key(1, 1, 2), make_value(3, 8)),
    ];
    let stats = commit_entries(&store, &entries, true)?;
    assert_eq!(stats.segments_published, 2);
    assert_eq!(store.iter_all()?.count(), entries.len());
    Ok(())
}

#[test]
fn malformed_block_becomes_miss_in_all_read_paths() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries = vec![
        (make_key(1, 1, 0), make_value(1, 16)),
        (make_key(1, 1, 1), make_value(2, 16)),
    ];
    commit_entries(&store, &entries, true)?;
    let path = first_segment_path(&store);
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    file.seek(SeekFrom::Start(4))?;
    file.write_all(&u32::MAX.to_le_bytes())?;
    file.sync_all()?;

    let reopened = Store::open(options(&tempdir))?;
    let key_refs = entries
        .iter()
        .map(|(key, _)| key.as_slice())
        .collect::<Vec<_>>();
    assert_eq!(reopened.fetch_one(&entries[0].0)?, None);
    assert_eq!(
        reopened.fetch_many_ordered(key_refs.iter().copied())?,
        vec![None, None]
    );
    assert_eq!(reopened.probe_ordered(key_refs.iter().copied())?, vec![
        false, false
    ]);
    let scanned: Result<Vec<_>> = reopened.iter_all()?.collect();
    assert!(scanned?.is_empty());
    Ok(())
}

#[test]
fn corrupted_footer_hides_whole_segment() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 16))], true)?;
    let path = first_segment_path(&store);
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    file.seek(SeekFrom::End(
        -4 - i64::try_from(FOOTER_MAGIC.len()).expect("magic len"),
    ))?;
    file.write_all(&0u32.to_le_bytes())?;
    file.sync_all()?;

    let reopened = Store::open(options(&tempdir))?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    Ok(())
}

#[test]
fn missing_manifest_segment_remains_reserved_for_future_appends() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let options = options(&tempdir);
    let store = Store::open(options.clone())?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(1, 8))], true)?;
    fs::remove_file(first_segment_path(&store))?;
    drop(store);

    let reopened = Store::open(options)?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    let error = commit_entries(&reopened, &[(key, make_value(2, 8))], true).unwrap_err();
    assert!(matches!(error, Error::OutOfOrderAppend { .. }));
    Ok(())
}

#[test]
fn incomplete_temp_segment_is_ignored_on_reopen() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let options = options(&tempdir);
    let store = Store::open(options.clone())?;
    commit_entries(&store, &[(make_key(1, 1, 0), make_value(1, 8))], true)?;

    let tmp_path = tempdir.path().join("tmp").join("orphan.seg.tmp");
    fs::write(&tmp_path, b"incomplete")?;

    let reopened = Store::open(options)?;
    let all: Result<Vec<_>> = reopened.iter_all()?.collect();
    assert_eq!(all?.len(), 1);
    Ok(())
}

#[test]
fn orphan_segment_is_ignored_until_manifest_references_it() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let options = options(&tempdir);
    let store = Store::open(options.clone())?;
    commit_entries(&store, &[(make_key(1, 1, 0), make_value(1, 8))], true)?;

    let orphan_dir = StorePaths::new(&options.root).segment_dir(0);
    let orphan_path = orphan_dir.join("segment-orphan.seg");
    fs::write(orphan_path, b"not a real segment")?;

    let reopened = Store::open(options)?;
    let all: Result<Vec<_>> = reopened.iter_all()?.collect();
    assert_eq!(all?.len(), 1);
    Ok(())
}
