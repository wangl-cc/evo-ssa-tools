use segment_cache_store::{CommitStats, Result};

use crate::support::api::{commit_entries, create_store, make_key, make_value};

#[test]
fn round_trip_batch_commit_then_fetch() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
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
    let store_a = create_store(&tempdir_a)?;
    let store_b = create_store(&tempdir_b)?;
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
    let store = create_store(&tempdir)?;
    let entries = vec![
        (make_key(1, 0, 0), make_value(1, 8)),
        (make_key(1, 0, 1), make_value(2, 16)),
    ];
    let mut batch = store.begin_batch();
    for (key, value) in entries.clone() {
        batch.push_owned(key, value)?;
    }
    store.commit_batch(batch)?;

    let scanned = store.iter_all()?.collect::<Result<Vec<_>>>()?;
    assert_eq!(scanned, entries);
    Ok(())
}

#[test]
fn empty_batch_commit_and_batch_len_are_noops() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let mut batch = store.begin_batch();
    assert_eq!(batch.len(), 0);
    batch.push(&make_key(1, 0, 0), &make_value(1, 8))?;
    assert_eq!(batch.len(), 1);

    let empty_stats = store.commit_batch(store.begin_batch())?;
    assert_eq!(empty_stats, CommitStats::default());
    assert_eq!(store.iter_all()?.count(), 0);
    Ok(())
}
