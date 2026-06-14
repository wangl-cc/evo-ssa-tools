use crate::common::*;

#[test]
fn corrupted_block_checksum_becomes_miss() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 16))], true)?;
    let path = first_segment_path(tempdir.path())?;
    corrupt_block_value_payload(&path, 0)?;

    let reopened = reopen_store_read_only(&tempdir)?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    Ok(())
}

#[test]
fn block_checksum_verification_can_be_disabled_for_benchmarks() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let key = make_key(1, 1, 0);
    let value = make_value(9, 16);
    commit_entries(&store, &[(key.clone(), value.clone())], true)?;
    let path = first_segment_path(tempdir.path())?;
    corrupt_block_value_payload(&path, 0)?;

    let checked = reopen_store_read_only(&tempdir)?;
    assert_eq!(checked.fetch_one(&key)?, None);

    let unchecked = Store::open(
        tempdir.path(),
        StoreOpenOptions::new(metadata())
            .with_block_checksum_verification(false)
            .with_read_only(true),
    )?;
    let unchecked_value = unchecked
        .fetch_one(&key)?
        .expect("unchecked read should return corrupted bytes");
    assert_ne!(unchecked_value, value);
    Ok(())
}

#[test]
fn store_round_trips_with_global_segments() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
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
    let store = create_store(&tempdir)?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 16))], true)?;
    let path = first_segment_path(tempdir.path())?;
    FsOpenOptions::new()
        .write(true)
        .open(path)?
        .set_len(FOOTER_TRAILER_LEN - 1)?;

    let reopened = reopen_store_read_only(&tempdir)?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    assert_eq!(reopened.iter_all()?.count(), 0);
    Ok(())
}

#[test]
fn metadata_mismatch_rejects_open() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 16))], true)?;
    drop(store);

    let error = match Store::open(
        tempdir.path(),
        StoreOpenOptions::new(StoreMetadata::from_text("different")),
    ) {
        Ok(_) => panic!("metadata mismatch should reject open"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        Error::Catalog(CatalogError::Mismatch(CatalogMismatch::Metadata))
    ));
    Ok(())
}

#[test]
fn flush_thresholds_split_one_batch_into_multiple_segments() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let entries = vec![
        (make_key(1, 1, 0), make_value(1, 8)),
        (make_key(1, 1, 1), make_value(2, 8)),
        (make_key(1, 1, 2), make_value(3, 8)),
    ];
    let stats = commit_entries_with_options(
        &store,
        &entries,
        true,
        &CommitOptions::default()
            .with_target_block_size(256)
            .with_flush_threshold_records(2),
    )?;
    assert_eq!(stats.segments_published, 2);
    assert_eq!(store.iter_all()?.count(), entries.len());
    Ok(())
}

#[test]
fn malformed_block_becomes_miss_in_all_read_paths() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let entries = vec![
        (make_key(1, 1, 0), make_value(1, 16)),
        (make_key(1, 1, 1), make_value(2, 16)),
    ];
    commit_entries(&store, &entries, true)?;
    let path = first_segment_path(tempdir.path())?;
    let mut file = FsOpenOptions::new().read(true).write(true).open(&path)?;
    file.seek(SeekFrom::Start(block_offset(&path, 0)? + 4))?;
    file.write_all(&u32::MAX.to_le_bytes())?;
    file.sync_all()?;

    let reopened = reopen_store_read_only(&tempdir)?;
    let key_refs = entries
        .iter()
        .map(|(key, _)| key.as_slice())
        .collect::<Vec<_>>();
    assert_eq!(reopened.fetch_one(&entries[0].0)?, None);
    assert_eq!(
        reopened.fetch_many_ordered(key_refs.iter().copied())?,
        vec![None, None]
    );
    assert_eq!(
        reopened.contains_many_ordered(key_refs.iter().copied())?,
        vec![false, false]
    );
    let scanned: Result<Vec<_>> = reopened.iter_all()?.collect();
    assert!(scanned?.is_empty());
    Ok(())
}

#[test]
fn corrupted_middle_block_only_loses_that_block() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let entries: Vec<_> = (0..3u64)
        .map(|rep| (make_key(1, 1, rep), make_value(rep as u8, 160)))
        .collect();
    commit_entries_with_options(
        &store,
        &entries,
        true,
        &CommitOptions::default().with_target_block_size(96),
    )?;
    let path = first_segment_path(tempdir.path())?;
    corrupt_block_value_payload(&path, 1)?;

    let reopened = reopen_store_read_only(&tempdir)?;
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
fn corrupted_block_key_ordering_becomes_miss_in_ordered_reads() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let entries = vec![
        (make_key(1, 1, 0), make_value(1, 16)),
        (make_key(1, 1, 1), make_value(2, 16)),
    ];
    commit_entries(&store, &entries, true)?;
    let path = first_segment_path(tempdir.path())?;
    let key_len = entries[0].0.len();
    mutate_block_metadata(&path, 0, |metadata| {
        let prefix_len = key_len - 1;
        let suffix_start = 4 + prefix_len;
        let first_suffix = suffix_start;
        let second_suffix = suffix_start + 1;
        metadata[second_suffix] = metadata[first_suffix];
    })?;

    let reopened = reopen_store_read_only(&tempdir)?;
    let key_refs = entries
        .iter()
        .map(|(key, _)| key.as_slice())
        .collect::<Vec<_>>();
    assert_eq!(
        reopened.fetch_many_ordered(key_refs.iter().copied())?,
        vec![None, None]
    );
    assert_eq!(
        reopened.contains_many_ordered(key_refs.iter().copied())?,
        vec![false, false]
    );
    assert!(reopened.iter_all()?.collect::<Result<Vec<_>>>()?.is_empty());
    Ok(())
}

#[test]
fn malformed_footer_block_index_metadata_hides_whole_segment() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 16))], true)?;
    let path = first_segment_path(tempdir.path())?;
    mutate_footer_payload(&path, |payload| {
        let block_count_offset = 8 + key.len() * 2;
        payload[block_count_offset..block_count_offset + 4].copy_from_slice(&0u32.to_le_bytes());
    })?;

    let reopened = reopen_store_read_only(&tempdir)?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    assert_eq!(reopened.contains_many_ordered([key.as_slice()])?, vec![
        false
    ]);
    assert_eq!(reopened.iter_all()?.count(), 0);
    Ok(())
}

#[test]
fn corrupted_header_hides_whole_segment() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 16))], true)?;
    let path = first_segment_path(tempdir.path())?;
    let mut file = FsOpenOptions::new().read(true).write(true).open(path)?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(b"badmagic")?;
    file.sync_all()?;

    let reopened = reopen_store_read_only(&tempdir)?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    assert_eq!(reopened.iter_all()?.count(), 0);
    Ok(())
}

#[test]
fn corrupted_block_index_hides_whole_segment() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 16))], true)?;
    let path = first_segment_path(tempdir.path())?;
    let block_index_offset = block_index_offset(&path)?;
    let mut file = FsOpenOptions::new().read(true).write(true).open(path)?;
    file.seek(SeekFrom::Start(block_index_offset + 4))?;
    file.write_all(&[0xFF])?;
    file.sync_all()?;

    let reopened = reopen_store_read_only(&tempdir)?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    assert_eq!(reopened.iter_all()?.count(), 0);
    Ok(())
}

#[test]
fn corrupted_footer_hides_whole_segment() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 16))], true)?;
    let path = first_segment_path(tempdir.path())?;
    let mut file = FsOpenOptions::new().read(true).write(true).open(path)?;
    file.seek(SeekFrom::End(-4))?;
    file.write_all(&0u32.to_le_bytes())?;
    file.sync_all()?;

    let reopened = reopen_store_read_only(&tempdir)?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    Ok(())
}
