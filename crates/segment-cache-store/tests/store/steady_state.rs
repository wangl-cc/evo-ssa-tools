//! Steady-state write design: replacing commits, dead-entry dropping, the
//! advisory writer lock, open-time GC, and the lookup-across-commit fix.

use crate::common::*;

#[test]
fn second_writer_open_fails_fast_while_writer_is_alive() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;

    let error = match reopen_store(&tempdir) {
        Ok(_) => panic!("a second writer should fail fast"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::Input(InputError::WriterLocked)));

    // A read-only handle coexists with the live writer.
    let reader = reopen_store_read_only(&tempdir)?;
    assert_eq!(reader.iter_all()?.count(), 0);
    drop(reader);
    drop(store);

    // Once the writer drops, the lock is released for the next writer.
    let reopened = reopen_store(&tempdir)?;
    commit_entries(&reopened, &[(make_key(1, 0, 0), make_value(1, 8))], true)?;
    Ok(())
}

#[test]
fn interleaving_commit_spans_two_segments_and_a_tail() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    commit_entries(
        &store,
        &[
            (make_key(1, 0, 0), make_value(1, 8)),
            (make_key(1, 0, 2), make_value(2, 8)),
        ],
        true,
    )?;
    commit_entries(
        &store,
        &[
            (make_key(2, 0, 0), make_value(3, 8)),
            (make_key(2, 0, 2), make_value(4, 8)),
        ],
        true,
    )?;
    // This batch interleaves both published segments and extends past the last.
    let stats = commit_entries(
        &store,
        &[
            (make_key(1, 0, 1), make_value(5, 8)),
            (make_key(2, 0, 1), make_value(6, 8)),
            (make_key(3, 0, 0), make_value(7, 8)),
        ],
        true,
    )?;
    // Stats expose the rewrite amplification: 3 input records forced both
    // 2-record segments to be rebuilt into one 7-record replacement.
    assert_eq!(stats.records, 3);
    assert_eq!(stats.merged_records, 7);
    assert_eq!(stats.segments_retired, 2);
    assert_eq!(stats.segments_published, 1);

    let keys: Vec<_> = store
        .iter_all()?
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .map(|(key, _)| key)
        .collect();
    assert_eq!(
        keys,
        vec![
            make_key(1, 0, 0),
            make_key(1, 0, 1),
            make_key(1, 0, 2),
            make_key(2, 0, 0),
            make_key(2, 0, 1),
            make_key(2, 0, 2),
            make_key(3, 0, 0),
        ]
    );
    Ok(())
}

#[test]
fn duplicate_key_commit_keeps_lexicographically_smallest_value() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let key = make_key(1, 0, 0);

    commit_entries(&store, &[(key.clone(), make_value(5, 8))], true)?;
    // A smaller-byte copy wins.
    commit_entries(&store, &[(key.clone(), make_value(1, 8))], true)?;
    assert_eq!(store.fetch_one(&key)?, Some(make_value(1, 8)));
    // A larger-byte copy does not displace the existing winner.
    commit_entries(&store, &[(key.clone(), make_value(9, 8))], true)?;
    assert_eq!(store.fetch_one(&key)?, Some(make_value(1, 8)));
    Ok(())
}

#[test]
fn lookup_session_sees_data_committed_after_first_use() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    commit_entries(&store, &[(make_key(2, 0, 0), make_value(2, 8))], true)?;

    let mut session = store.lookup_session();
    assert_eq!(
        session.fetch_many([make_key(2, 0, 0).as_slice()])?,
        vec![Some(make_value(2, 8))]
    );

    // Insert a segment *before* the cached one, shifting segment indices. A
    // session that cached stale indices/blocks would now miss.
    commit_entries(&store, &[(make_key(1, 0, 0), make_value(1, 8))], true)?;

    assert_eq!(
        session.fetch_many([make_key(1, 0, 0).as_slice(), make_key(2, 0, 0).as_slice()])?,
        vec![Some(make_value(1, 8)), Some(make_value(2, 8))]
    );
    Ok(())
}

#[test]
fn unrelated_commit_drops_a_dead_manifest_entry() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    commit_entries(&store, &[(make_key(1, 0, 0), make_value(1, 8))], true)?;
    commit_entries(&store, &[(make_key(5, 0, 0), make_value(5, 8))], true)?;

    // The lowest-id segment holds key (1, 0, 0); deleting it leaves a dead entry.
    fs::remove_file(first_segment_path(tempdir.path())?)?;
    drop(store);

    let store = reopen_store(&tempdir)?;
    assert_eq!(store.fetch_one(&make_key(1, 0, 0))?, None);

    // A commit far from the dead range still drops the dead entry...
    commit_entries(&store, &[(make_key(9, 0, 0), make_value(9, 8))], true)?;
    // ...so the lost range is writable again.
    commit_entries(&store, &[(make_key(1, 0, 0), make_value(2, 8))], true)?;
    assert_eq!(store.fetch_one(&make_key(1, 0, 0))?, Some(make_value(2, 8)));
    assert_eq!(store.fetch_one(&make_key(5, 0, 0))?, Some(make_value(5, 8)));
    Ok(())
}

#[test]
fn open_time_gc_deletes_unreferenced_files() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    commit_entries(&store, &[(make_key(1, 0, 0), make_value(1, 8))], true)?;
    drop(store);

    let seg_dir = tempdir.path().join("segments");
    let orphan = seg_dir.join("segment-0000000099.seg");
    let leftover_tmp = seg_dir.join("segment-0000000100.seg.tmp");
    fs::write(&orphan, b"orphan")?;
    fs::write(&leftover_tmp, b"tmp")?;

    // A writer reopen runs GC under the writer lock.
    let store = reopen_store(&tempdir)?;
    assert!(!orphan.exists());
    assert!(!leftover_tmp.exists());
    assert_eq!(store.iter_all()?.count(), 1);
    Ok(())
}

#[test]
fn read_only_handle_rejects_commit() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    commit_entries(&store, &[(make_key(1, 0, 0), make_value(1, 8))], true)?;

    let reader = reopen_store_read_only(&tempdir)?;
    let error = match commit_entries(&reader, &[(make_key(2, 0, 0), make_value(2, 8))], true) {
        Ok(_) => panic!("a read-only handle must not publish"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::Input(InputError::ReadOnlyStore)));

    // Nothing was published and the reader still reads.
    assert_eq!(reader.fetch_one(&make_key(2, 0, 0))?, None);
    assert_eq!(
        reader.fetch_one(&make_key(1, 0, 0))?,
        Some(make_value(1, 8))
    );
    Ok(())
}

#[test]
fn commit_time_gc_deletes_unreferenced_files() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    commit_entries(&store, &[(make_key(1, 0, 0), make_value(1, 8))], true)?;

    let seg_dir = tempdir.path().join("segments");
    let orphan = seg_dir.join("segment-0000000099.seg");
    fs::write(&orphan, b"orphan")?;

    // A long-running writer reclaims orphans at its next publication, without
    // needing a reopen.
    commit_entries(&store, &[(make_key(5, 0, 0), make_value(5, 8))], true)?;
    assert!(!orphan.exists());
    assert_eq!(store.iter_all()?.count(), 2);
    Ok(())
}

#[test]
fn create_recovers_from_leftover_manifest_without_store() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    drop(store);

    // Simulate a creation that crashed after MANIFEST but before STORE.
    fs::remove_file(tempdir.path().join("STORE"))?;
    let error = match reopen_store(&tempdir) {
        Ok(_) => panic!("open without STORE should be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        Error::Catalog(CatalogError::Mismatch(CatalogMismatch::MissingStore))
    ));

    // Creation succeeds because STORE is absent; the leftover MANIFEST is
    // overwritten.
    let store = create_store(&tempdir)?;
    commit_entries(&store, &[(make_key(1, 0, 0), make_value(1, 8))], true)?;
    assert_eq!(store.fetch_one(&make_key(1, 0, 0))?, Some(make_value(1, 8)));
    Ok(())
}
