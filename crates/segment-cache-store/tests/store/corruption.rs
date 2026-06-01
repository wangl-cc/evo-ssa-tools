use crate::common::*;

#[test]
fn corrupted_block_checksum_becomes_miss() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 16))], true)?;
    let path = first_segment_path(tempdir.path())?;
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
    let path = first_segment_path(tempdir.path())?;
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
fn single_shard_store_round_trips_without_shard_prefix() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir).with_shard_count(1))?;
    let entries = vec![
        (make_key(1, 0, 0), make_value(1, 8)),
        (make_key(2, 0, 0), make_value(2, 8)),
    ];
    commit_entries(&store, &entries, true)?;

    assert_eq!(store.iter_all()?.collect::<Result<Vec<_>>>()?, entries);
    Ok(())
}

#[test]
fn truncated_segment_file_is_ignored_on_reopen() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 16))], true)?;
    let path = first_segment_path(tempdir.path())?;
    OpenOptions::new()
        .write(true)
        .open(path)?
        .set_len(u64::try_from(FOOTER_MAGIC.len() - 1).expect("magic len"))?;

    let reopened = Store::open(options(&tempdir))?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    assert_eq!(reopened.iter_all()?.count(), 0);
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
    let path = first_segment_path(tempdir.path())?;
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
fn corrupted_middle_block_only_loses_that_block() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let options = options(&tempdir).with_target_block_size(96);
    let store = Store::open(options.clone())?;
    let entries: Vec<_> = (0..3u64)
        .map(|rep| (make_key(1, 1, rep), make_value(rep as u8, 160)))
        .collect();
    commit_entries(&store, &entries, true)?;
    let path = first_segment_path(tempdir.path())?;
    let second_block_offset = block_offset(&path, 1)?;
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    file.seek(SeekFrom::Start(second_block_offset + 24))?;
    file.write_all(&[0xFF])?;
    file.sync_all()?;

    let reopened = Store::open(options)?;
    let key_refs = entries
        .iter()
        .map(|(key, _)| key.as_slice())
        .collect::<Vec<_>>();
    assert_eq!(
        reopened.fetch_many_ordered(key_refs.iter().copied())?,
        vec![Some(entries[0].1.clone()), None, Some(entries[2].1.clone())]
    );
    assert_eq!(reopened.iter_all()?.collect::<Result<Vec<_>>>()?, vec![
        entries[0].clone(),
        entries[2].clone()
    ]);
    Ok(())
}

#[test]
fn corrupted_block_index_hides_whole_segment() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 16))], true)?;
    let path = first_segment_path(tempdir.path())?;
    let block_index_offset = block_index_offset(&path)?;
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    file.seek(SeekFrom::Start(block_index_offset + 4))?;
    file.write_all(&[0xFF])?;
    file.sync_all()?;

    let reopened = Store::open(options(&tempdir))?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    assert_eq!(reopened.iter_all()?.count(), 0);
    Ok(())
}

#[test]
fn corrupted_footer_hides_whole_segment() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 16))], true)?;
    let path = first_segment_path(tempdir.path())?;
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
