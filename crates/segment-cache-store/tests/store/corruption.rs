use std::{
    fs,
    fs::OpenOptions as FsOpenOptions,
    io::{Seek, SeekFrom, Write},
};

#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
use segment_cache_store::ValuePayloadCompressionPolicy;
use segment_cache_store::{
    BlockChecksumKind, CatalogError, CatalogMismatch, CommitOptions, Error,
    OpenOptions as StoreOpenOptions, Result, Store, StoreMetadata,
};

#[cfg(any(feature = "checksum-crc32c", feature = "checksum-rapidhash"))]
use crate::support::api::metadata;
#[cfg(feature = "checksum-rapidhash")]
use crate::support::api::reopen_store;
#[cfg(any(feature = "value-compression-lz4", feature = "value-compression-zstd"))]
use crate::support::segment_file::corrupt_block_value_frame_start;
use crate::support::{
    api::{
        commit_entries, commit_entries_with_options, create_options, create_store,
        create_store_with, make_key, make_value, reopen_store_read_only,
    },
    segment_file::{
        FOOTER_TRAILER_LEN, block_index_offset, block_offset, corrupt_block_value_payload,
        first_segment_path, mutate_block_metadata, mutate_footer_payload,
        truncate_first_block_to_declared_len,
    },
};

#[cfg(any(feature = "checksum-crc32c", feature = "checksum-rapidhash"))]
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

#[cfg(any(feature = "checksum-crc32c", feature = "checksum-rapidhash"))]
#[test]
fn sparse_ordered_lookup_corrupted_payload_becomes_miss() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let entries: Vec<_> = (0..8u64)
        .map(|rep| (make_key(1, 1, rep), make_value(rep as u8, 32)))
        .collect();
    commit_entries_with_options(
        &store,
        &entries,
        true,
        &CommitOptions::default()
            .with_target_block_size(4096)
            .with_flush_threshold_records(128),
    )?;
    let path = first_segment_path(tempdir.path())?;
    corrupt_block_value_payload(&path, 0)?;
    drop(store);

    let reopened = reopen_store_read_only(&tempdir)?;
    let key = make_key(1, 1, 4);
    assert_eq!(reopened.fetch_many_ordered([key.as_slice()])?, vec![None]);
    assert_eq!(reopened.contains_many_ordered([key.as_slice()])?, vec![
        false
    ]);
    Ok(())
}

#[test]
#[cfg(feature = "checksum-rapidhash")]
fn rapidhash_block_checksum_round_trips() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store_with(
        &tempdir,
        create_options().with_block_checksum(BlockChecksumKind::RapidHashV3_64),
    )?;
    let entries = vec![
        (make_key(1, 1, 0), make_value(1, 16)),
        (make_key(1, 1, 1), make_value(2, 16)),
    ];
    commit_entries(&store, &entries, true)?;
    drop(store);

    let reopened = reopen_store(&tempdir)?;
    assert_eq!(reopened.iter_all()?.collect::<Result<Vec<_>>>()?, entries);
    Ok(())
}

#[cfg(feature = "value-compression-lz4")]
#[test]
fn lz4_value_payload_compression_round_trips() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store_with(
        &tempdir,
        create_options()
            .with_value_payload_compression(segment_cache_store::ValuePayloadCompressionKind::Lz4),
    )?;
    let entries = vec![
        (make_key(1, 1, 0), make_value(1, 96 * 1024)),
        (make_key(1, 1, 1), make_value(2, 96 * 1024)),
    ];
    commit_entries(&store, &entries, true)?;

    assert_eq!(store.fetch_one(&entries[0].0)?, Some(entries[0].1.clone()));
    assert_eq!(
        store.fetch_many_ordered(entries.iter().map(|(key, _)| key.as_slice()))?,
        entries
            .iter()
            .map(|(_, value)| Some(value.clone()))
            .collect::<Vec<_>>()
    );
    assert_eq!(store.iter_all()?.collect::<Result<Vec<_>>>()?, entries);

    let segment_len = fs::metadata(first_segment_path(tempdir.path())?)?.len();
    assert!(
        segment_len < 64 * 1024,
        "compressible payloads should be stored smaller than their raw bytes"
    );
    Ok(())
}

#[cfg(feature = "value-compression-zstd")]
#[test]
fn zstd_value_payload_compression_round_trips() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store_with(
        &tempdir,
        create_options().with_value_payload_compression(
            segment_cache_store::ValuePayloadCompressionKind::ZstdLevel1,
        ),
    )?;
    let entries = vec![
        (make_key(1, 1, 0), make_value(1, 96 * 1024)),
        (make_key(1, 1, 1), make_value(2, 96 * 1024)),
    ];
    commit_entries(&store, &entries, true)?;

    assert_eq!(store.fetch_one(&entries[0].0)?, Some(entries[0].1.clone()));
    assert_eq!(
        store.fetch_many_ordered(entries.iter().map(|(key, _)| key.as_slice()))?,
        entries
            .iter()
            .map(|(_, value)| Some(value.clone()))
            .collect::<Vec<_>>()
    );
    assert_eq!(store.iter_all()?.collect::<Result<Vec<_>>>()?, entries);

    let segment_len = fs::metadata(first_segment_path(tempdir.path())?)?.len();
    assert!(
        segment_len < 64 * 1024,
        "compressible payloads should be stored smaller than their raw bytes"
    );
    Ok(())
}

#[cfg(feature = "value-compression-lz4")]
#[test]
fn corrupted_lz4_value_payload_becomes_miss() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store_with(
        &tempdir,
        create_options()
            .with_value_payload_compression(segment_cache_store::ValuePayloadCompressionKind::Lz4),
    )?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 96 * 1024))], true)?;
    let path = first_segment_path(tempdir.path())?;
    corrupt_block_value_frame_start(&path, 0)?;
    drop(store);

    let reopened = reopen_store_read_only(&tempdir)?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    Ok(())
}

#[cfg(feature = "value-compression-lz4")]
#[test]
fn sparse_lz4_frame_header_corruption_becomes_miss() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store_with(
        &tempdir,
        create_options()
            .with_value_payload_compression(segment_cache_store::ValuePayloadCompressionKind::Lz4),
    )?;
    let entries: Vec<_> = (0..4u64)
        .map(|rep| (make_key(1, 2, rep), make_value(rep as u8 + 1, 16 * 1024)))
        .collect();
    commit_entries_with_options(
        &store,
        &entries,
        true,
        &CommitOptions::default()
            .with_target_block_size(256 * 1024)
            .with_flush_threshold_records(128)
            .with_value_payload_compression_policy(
                ValuePayloadCompressionPolicy::new(1, 0).expect("compression policy is valid"),
            ),
    )?;
    let path = first_segment_path(tempdir.path())?;
    corrupt_block_value_frame_start(&path, 0)?;
    drop(store);

    let reopened = reopen_store_read_only(&tempdir)?;
    let key = make_key(1, 2, 2);
    assert_eq!(reopened.fetch_many_ordered([key.as_slice()])?, vec![None]);
    assert_eq!(reopened.contains_many_ordered([key.as_slice()])?, vec![
        false
    ]);
    Ok(())
}

#[cfg(feature = "value-compression-zstd")]
#[test]
fn corrupted_zstd_value_payload_becomes_miss() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store_with(
        &tempdir,
        create_options().with_value_payload_compression(
            segment_cache_store::ValuePayloadCompressionKind::ZstdLevel1,
        ),
    )?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 96 * 1024))], true)?;
    let path = first_segment_path(tempdir.path())?;
    corrupt_block_value_frame_start(&path, 0)?;
    drop(store);

    let reopened = reopen_store_read_only(&tempdir)?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    Ok(())
}

#[cfg(feature = "value-compression-zstd")]
#[test]
fn sparse_zstd_frame_header_corruption_becomes_miss() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store_with(
        &tempdir,
        create_options().with_value_payload_compression(
            segment_cache_store::ValuePayloadCompressionKind::ZstdLevel1,
        ),
    )?;
    let entries: Vec<_> = (0..4u64)
        .map(|rep| (make_key(1, 3, rep), make_value(rep as u8 + 1, 16 * 1024)))
        .collect();
    commit_entries_with_options(
        &store,
        &entries,
        true,
        &CommitOptions::default()
            .with_target_block_size(256 * 1024)
            .with_flush_threshold_records(128)
            .with_value_payload_compression_policy(
                ValuePayloadCompressionPolicy::new(1, 0).expect("compression policy is valid"),
            ),
    )?;
    let path = first_segment_path(tempdir.path())?;
    corrupt_block_value_frame_start(&path, 0)?;
    drop(store);

    let reopened = reopen_store_read_only(&tempdir)?;
    let key = make_key(1, 3, 2);
    assert_eq!(reopened.fetch_many_ordered([key.as_slice()])?, vec![None]);
    assert_eq!(reopened.contains_many_ordered([key.as_slice()])?, vec![
        false
    ]);
    Ok(())
}

#[test]
fn no_checksum_open_handle_does_not_detect_payload_corruption() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store_with(
        &tempdir,
        create_options().with_block_checksum(BlockChecksumKind::None),
    )?;
    let key = make_key(1, 1, 0);
    let value = make_value(9, 16);
    commit_entries(&store, &[(key.clone(), value.clone())], true)?;
    let path = first_segment_path(tempdir.path())?;
    corrupt_block_value_payload(&path, 0)?;

    let corrupted = store
        .fetch_one(&key)?
        .expect("open no-checksum handle cannot detect corruption");
    assert_ne!(corrupted, value);
    drop(store);

    let reopened = reopen_store_read_only(&tempdir)?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    Ok(())
}

#[cfg(any(feature = "checksum-crc32c", feature = "checksum-rapidhash"))]
#[test]
fn block_checksum_verification_can_be_disabled_for_benchmarks() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let key = make_key(1, 1, 0);
    let value = make_value(9, 16);
    commit_entries(&store, &[(key.clone(), value.clone())], true)?;
    let path = first_segment_path(tempdir.path())?;
    let unchecked = Store::open(
        tempdir.path(),
        StoreOpenOptions::new(metadata())
            .with_block_checksum_verification(false)
            .with_read_only(true),
    )?;
    corrupt_block_value_payload(&path, 0)?;

    let unchecked_value = unchecked
        .fetch_one(&key)?
        .expect("unchecked read should return corrupted bytes");
    assert_ne!(unchecked_value, value);

    let checked = reopen_store_read_only(&tempdir)?;
    assert_eq!(checked.fetch_one(&key)?, None);
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
fn manifest_fingerprint_rejects_same_range_replacement_segment() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(1, 16))], true)?;
    let original_path = first_segment_path(tempdir.path())?;

    let replacement_dir = tempfile::tempdir()?;
    let replacement = create_store(&replacement_dir)?;
    commit_entries(&replacement, &[(key.clone(), make_value(2, 16))], true)?;
    let replacement_path = first_segment_path(replacement_dir.path())?;
    drop(store);
    drop(replacement);

    fs::copy(replacement_path, original_path)?;

    let reopened = reopen_store_read_only(&tempdir)?;
    assert_eq!(reopened.fetch_one(&key)?, None);
    assert_eq!(reopened.iter_all()?.count(), 0);
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
fn footer_length_past_file_hides_whole_segment() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let key = make_key(1, 1, 0);
    commit_entries(&store, &[(key.clone(), make_value(9, 16))], true)?;
    let path = first_segment_path(tempdir.path())?;
    let mut file = FsOpenOptions::new().read(true).write(true).open(path)?;
    file.seek(SeekFrom::End(
        -i64::try_from(FOOTER_TRAILER_LEN).expect("trailer len fits"),
    ))?;
    file.write_all(&u32::MAX.to_le_bytes())?;
    file.sync_all()?;

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
fn variable_value_index_must_start_at_zero_without_block_checksums() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store_with(
        &tempdir,
        create_options().with_block_checksum(BlockChecksumKind::None),
    )?;
    let entries = vec![
        (make_key(1, 1, 0), make_value(1, 16)),
        (make_key(1, 1, 1), make_value(2, 16)),
    ];
    commit_entries(&store, &entries, true)?;
    let path = first_segment_path(tempdir.path())?;
    let key_len = entries[0].0.len();
    let record_count = entries.len();
    mutate_block_metadata(&path, 0, |metadata| {
        let prefix_len = u32::from_le_bytes(
            metadata[..4]
                .try_into()
                .expect("prefix length field should exist"),
        ) as usize;
        let suffix_len = key_len - prefix_len;
        let value_index_offset = 4 + prefix_len + record_count * suffix_len;
        metadata[value_index_offset..value_index_offset + 4].copy_from_slice(&1u32.to_le_bytes());
    })?;

    let key_refs = entries
        .iter()
        .map(|(key, _)| key.as_slice())
        .collect::<Vec<_>>();
    assert_eq!(store.fetch_many_ordered(key_refs.iter().copied())?, vec![
        None, None
    ]);
    assert_eq!(
        store.contains_many_ordered(key_refs.iter().copied())?,
        vec![false, false]
    );
    assert!(store.iter_all()?.collect::<Result<Vec<_>>>()?.is_empty());
    Ok(())
}

#[test]
fn block_last_key_must_stay_within_segment_bounds_without_block_checksums() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store_with(
        &tempdir,
        create_options().with_block_checksum(BlockChecksumKind::None),
    )?;
    let entries = vec![
        (make_key(1, 1, 0), make_value(1, 16)),
        (make_key(1, 1, 1), make_value(2, 16)),
    ];
    commit_entries(&store, &entries, true)?;
    let path = first_segment_path(tempdir.path())?;
    let key_len = entries[0].0.len();
    let replacement_key = make_key(1, 1, 2);
    mutate_block_metadata(&path, 0, |metadata| {
        let prefix_len = u32::from_le_bytes(
            metadata[..4]
                .try_into()
                .expect("prefix length field should exist"),
        ) as usize;
        let suffix_len = key_len - prefix_len;
        let last_suffix_offset = 4 + prefix_len + suffix_len;
        metadata[last_suffix_offset..last_suffix_offset + suffix_len]
            .copy_from_slice(&replacement_key[prefix_len..]);
    })?;

    let key_refs = entries
        .iter()
        .map(|(key, _)| key.as_slice())
        .collect::<Vec<_>>();
    assert_eq!(store.fetch_many_ordered(key_refs.iter().copied())?, vec![
        None, None
    ]);
    assert_eq!(
        store.contains_many_ordered(key_refs.iter().copied())?,
        vec![false, false]
    );
    assert!(store.iter_all()?.collect::<Result<Vec<_>>>()?.is_empty());
    Ok(())
}

#[cfg(any(feature = "checksum-crc32c", feature = "checksum-rapidhash"))]
#[test]
fn corrupted_middle_block_only_loses_that_block_for_open_handle() -> Result<()> {
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
    let live_reader = reopen_store_read_only(&tempdir)?;
    corrupt_block_value_payload(&path, 1)?;

    let key_refs = entries
        .iter()
        .map(|(key, _)| key.as_slice())
        .collect::<Vec<_>>();
    assert_eq!(
        live_reader.fetch_many_ordered(key_refs.iter().copied())?,
        vec![Some(entries[0].1.clone()), None, Some(entries[2].1.clone())]
    );
    assert_eq!(live_reader.iter_all()?.collect::<Result<Vec<_>>>()?, vec![
        entries[0].clone(),
        entries[2].clone()
    ]);
    drop(live_reader);

    let reopened = reopen_store_read_only(&tempdir)?;
    assert_eq!(
        reopened.fetch_many_ordered(key_refs.iter().copied())?,
        vec![None, None, None]
    );
    assert!(reopened.iter_all()?.collect::<Result<Vec<_>>>()?.is_empty());
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
fn sparse_ordered_lookup_short_block_metadata_becomes_miss() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let entries: Vec<_> = (0..8u64)
        .map(|rep| (make_key(1, 4, rep), make_value(rep as u8, 32)))
        .collect();
    commit_entries_with_options(
        &store,
        &entries,
        true,
        &CommitOptions::default()
            .with_target_block_size(4096)
            .with_flush_threshold_records(128),
    )?;
    let path = first_segment_path(tempdir.path())?;
    truncate_first_block_to_declared_len(&path, 4)?;
    drop(store);

    let reopened = reopen_store_read_only(&tempdir)?;
    let key = make_key(1, 4, 4);
    assert_eq!(reopened.fetch_many_ordered([key.as_slice()])?, vec![None]);
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
