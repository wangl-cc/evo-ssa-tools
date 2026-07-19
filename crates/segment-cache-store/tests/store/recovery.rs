use std::{fs, fs::OpenOptions as FsOpenOptions, io::Write};

use segment_cache_store::Result;

use crate::support::{
    api::{
        commit_entries, create_store, make_key, make_value, reopen_store, reopen_store_read_only,
    },
    segment_file::first_segment_path,
};

#[test]
fn missing_manifest_segment_is_dropped_and_range_becomes_writable() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(1, 8))])?;
    fs::remove_file(first_segment_path(tempdir.path())?)?;
    drop(store);

    let reopened = reopen_store(&tempdir)?;
    // The dead entry is invisible immediately.
    assert_eq!(reopened.fetch_one(&key)?, None);
    // It is dropped at the next publication, so the lost range is writable again
    // instead of staying reserved forever.
    commit_entries(&reopened, &[(key.clone(), make_value(2, 8))])?;
    assert_eq!(reopened.fetch_one(&key)?, Some(make_value(2, 8)));
    Ok(())
}

#[test]
fn missing_manifest_segment_allows_non_overlapping_future_appends() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let old_key = make_key(1, 1, 0);
    let new_key = make_key(2, 1, 0);
    commit_entries(&store, &[(old_key.clone(), make_value(1, 8))])?;
    fs::remove_file(first_segment_path(tempdir.path())?)?;
    drop(store);

    let reopened = reopen_store(&tempdir)?;
    commit_entries(&reopened, &[(new_key.clone(), make_value(2, 8))])?;

    assert_eq!(reopened.fetch_one(&old_key)?, None);
    assert_eq!(reopened.fetch_one(&new_key)?, Some(make_value(2, 8)));
    assert_eq!(reopened.iter_all()?.collect::<Result<Vec<_>>>()?, vec![(
        new_key,
        make_value(2, 8)
    )]);
    Ok(())
}

#[test]
fn segment_with_trailing_garbage_is_ignored_on_reopen() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(1, 8))])?;
    let path = first_segment_path(tempdir.path())?;
    FsOpenOptions::new()
        .append(true)
        .open(path)?
        .write_all(&[0])?;
    drop(store);

    let reopened = reopen_store(&tempdir)?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    Ok(())
}

#[test]
fn incomplete_temp_segment_is_ignored_on_reopen() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    commit_entries(&store, &[(make_key(1, 1, 0), make_value(1, 8))])?;

    let tmp_path = tempdir.path().join("segments").join("orphan.seg.tmp");
    fs::write(&tmp_path, b"incomplete")?;

    let reopened = reopen_store_read_only(&tempdir)?;
    let all: Result<Vec<_>> = reopened.iter_all()?.collect();
    assert_eq!(all?.len(), 1);
    Ok(())
}

#[test]
fn orphan_segment_is_ignored_until_manifest_references_it() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    commit_entries(&store, &[(make_key(1, 1, 0), make_value(1, 8))])?;

    let orphan_dir = tempdir.path().join("segments");
    let orphan_path = orphan_dir.join("segment-orphan.seg");
    fs::write(orphan_path, b"not a real segment")?;

    let reopened = reopen_store_read_only(&tempdir)?;
    let all: Result<Vec<_>> = reopened.iter_all()?.collect();
    assert_eq!(all?.len(), 1);
    Ok(())
}

#[test]
fn orphan_segment_at_next_id_is_overwritten_by_commit() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let orphan_path = tempdir
        .path()
        .join("segments")
        .join("segment-0000000000.seg");
    fs::write(orphan_path, b"orphan")?;

    // An unreferenced file is provably dead, so the commit publishes its fresh
    // segment over it instead of wedging on an id collision.
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(1, 8))])?;
    assert_eq!(store.fetch_one(&key)?, Some(make_value(1, 8)));
    Ok(())
}
