use segment_cache_store::{CommitStats, Result, WriteBatch};

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
    let stats = commit_entries(&store, &entries)?;
    assert_eq!(stats.input_records, entries.len());

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

    commit_entries(&store_a, &sorted)?;
    commit_entries(&store_b, &unsorted)?;

    for (key, value) in &sorted {
        assert_eq!(store_a.fetch_one(key)?, Some(value.clone()));
        assert_eq!(store_b.fetch_one(key)?, Some(value.clone()));
    }
    Ok(())
}

#[test]
fn batch_entries_round_trip() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let entries = vec![
        (make_key(1, 0, 0), make_value(1, 8)),
        (make_key(1, 0, 1), make_value(2, 16)),
    ];
    let mut batch = WriteBatch::new();
    for (key, value) in &entries {
        batch.push(key, value);
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
    let mut batch = WriteBatch::new();
    assert_eq!(batch.len(), 0);
    batch.push(&make_key(1, 0, 0), &make_value(1, 8));
    assert_eq!(batch.len(), 1);

    let empty_stats = store.commit_batch(WriteBatch::new())?;
    assert_eq!(empty_stats, CommitStats::default());
    assert_eq!(store.iter_all()?.count(), 0);
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

    commit_entries(&store, &early)?;
    commit_entries(&store, &late)?;
    commit_entries(&store, &middle)?;

    let mut expected = early;
    expected.extend(middle);
    expected.extend(late);
    assert_eq!(store.iter_all()?.collect::<Result<Vec<_>>>()?, expected);
    Ok(())
}

#[test]
fn interleaving_commit_preserves_existing_records() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let first = vec![
        (make_key(1, 1, 0), make_value(1, 8)),
        (make_key(1, 1, 10), make_value(2, 8)),
    ];
    let interleaving = vec![(make_key(1, 1, 5), make_value(3, 8))];

    commit_entries(&store, &first)?;
    commit_entries(&store, &interleaving)?;

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
