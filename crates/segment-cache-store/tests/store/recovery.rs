use crate::common::*;

#[test]
fn missing_manifest_segment_remains_reserved_for_future_appends() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let options = options(&tempdir);
    let store = Store::open(options.clone())?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(1, 8))], true)?;
    fs::remove_file(first_segment_path(tempdir.path())?)?;
    drop(store);

    let reopened = Store::open(options)?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    let error = commit_entries(&reopened, &[(key, make_value(2, 8))], true).unwrap_err();
    assert!(matches!(error, Error::OutOfOrderAppend { .. }));
    Ok(())
}

#[test]
fn segment_file_length_mismatch_is_ignored_on_reopen() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let options = options(&tempdir);
    let store = Store::open(options.clone())?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(1, 8))], true)?;
    let path = first_segment_path(tempdir.path())?;
    OpenOptions::new()
        .append(true)
        .open(path)?
        .write_all(&[0])?;
    drop(store);

    let reopened = Store::open(options)?;
    assert_eq!(reopened.fetch_one(&key)?, None);
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

    let orphan_dir = options.root.join("shards").join("0").join("segments");
    let orphan_path = orphan_dir.join("segment-orphan.seg");
    fs::write(orphan_path, b"not a real segment")?;

    let reopened = Store::open(options)?;
    let all: Result<Vec<_>> = reopened.iter_all()?.collect();
    assert_eq!(all?.len(), 1);
    Ok(())
}

#[test]
fn orphan_segment_does_not_get_overwritten_by_commit() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let options = options(&tempdir).with_shard_count(1);
    let store = Store::open(options)?;
    let orphan_path = tempdir
        .path()
        .join("shards")
        .join("0")
        .join("segments")
        .join("segment-00000000000000000000.seg");
    fs::write(orphan_path, b"orphan")?;

    let error = commit_entries(&store, &[(make_key(1, 1, 0), make_value(1, 8))], true).unwrap_err();
    assert!(matches!(error, Error::SegmentFileAlreadyExists { .. }));
    Ok(())
}
