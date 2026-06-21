use crate::common::*;

#[test]
fn merge_from_imports_disjoint_source_records() -> Result<()> {
    let destination_dir = tempfile::tempdir()?;
    let source_dir = tempfile::tempdir()?;
    let destination = create_store(&destination_dir)?;
    let source = create_store(&source_dir)?;

    let destination_entries = vec![
        (make_key(1, 0, 0), make_value(10, 8)),
        (make_key(1, 0, 3), make_value(13, 8)),
    ];
    let source_entries = vec![
        (make_key(1, 0, 1), make_value(11, 8)),
        (make_key(1, 0, 2), make_value(12, 8)),
    ];
    commit_entries(&destination, &destination_entries, true)?;
    commit_entries(&source, &source_entries, true)?;

    let stats = destination.merge_from_with_options(&source, &commit_options())?;

    assert_eq!(stats.records, 2);
    assert_eq!(stats.bytes, 48);
    assert_eq!(destination.iter_all()?.collect::<Result<Vec<_>>>()?, {
        let mut expected = destination_entries;
        expected.extend(source_entries);
        expected.sort_by(|left, right| left.0.cmp(&right.0));
        expected
    });
    Ok(())
}

#[test]
fn merge_from_resolves_duplicate_keys_by_smallest_value() -> Result<()> {
    let destination_dir = tempfile::tempdir()?;
    let source_dir = tempfile::tempdir()?;
    let destination = create_store(&destination_dir)?;
    let source = create_store(&source_dir)?;
    let smaller_from_source = make_key(2, 0, 0);
    let smaller_from_destination = make_key(2, 0, 1);
    let source_only = make_key(2, 0, 2);

    commit_entries(
        &destination,
        &[
            (smaller_from_source.clone(), make_value(9, 8)),
            (smaller_from_destination.clone(), make_value(1, 8)),
        ],
        true,
    )?;
    commit_entries(
        &source,
        &[
            (smaller_from_source.clone(), make_value(1, 8)),
            (smaller_from_destination.clone(), make_value(9, 8)),
            (source_only.clone(), make_value(5, 8)),
        ],
        true,
    )?;

    let stats = destination.merge_from_with_options(
        &source,
        &commit_options()
            .with_patch_segment_limit(0)
            .with_flush_threshold_records(64),
    )?;

    assert_eq!(stats.records, 3);
    assert_eq!(stats.merged_records, 3);
    assert_eq!(stats.segments_retired, 1);
    assert_eq!(
        destination.fetch_one(&smaller_from_source)?,
        Some(make_value(1, 8))
    );
    assert_eq!(
        destination.fetch_one(&smaller_from_destination)?,
        Some(make_value(1, 8))
    );
    assert_eq!(destination.fetch_one(&source_only)?, Some(make_value(5, 8)));
    Ok(())
}

#[test]
fn merge_from_rejects_incompatible_source_metadata() -> Result<()> {
    let destination_dir = tempfile::tempdir()?;
    let source_dir = tempfile::tempdir()?;
    let destination = create_store(&destination_dir)?;
    let source = Store::create(
        source_dir.path(),
        CreateOptions::new_with_block_checksum(
            16,
            StoreMetadata::from_text("different-cache-namespace"),
            test_block_checksum(),
        ),
    )?;

    let error = destination
        .merge_from(&source)
        .expect_err("merge rejects metadata mismatch");

    assert!(matches!(
        error,
        Error::Input(InputError::SourceMetadataMismatch)
    ));
    Ok(())
}

#[test]
fn merge_from_rejects_read_only_destination() -> Result<()> {
    let destination_dir = tempfile::tempdir()?;
    let source_dir = tempfile::tempdir()?;
    let destination = create_store(&destination_dir)?;
    let source = create_store(&source_dir)?;
    drop(destination);
    let read_only_destination = reopen_store_read_only(&destination_dir)?;

    let error = read_only_destination
        .merge_from(&source)
        .expect_err("merge rejects read-only destination");

    assert!(matches!(error, Error::Input(InputError::ReadOnlyStore)));
    Ok(())
}

#[cfg(any(feature = "checksum-crc32c", feature = "checksum-rapidhash"))]
#[test]
fn merge_from_skips_corrupt_source_blocks_when_forcing_verification() -> Result<()> {
    let destination_dir = tempfile::tempdir()?;
    let source_dir = tempfile::tempdir()?;
    let destination = create_store(&destination_dir)?;
    let source = create_store(&source_dir)?;
    commit_entries(&source, &[(make_key(3, 0, 0), make_value(7, 32))], true)?;
    corrupt_block_value_payload(&first_segment_path(source_dir.path())?, 0)?;
    drop(source);
    let source = Store::open(
        source_dir.path(),
        open_options()
            .with_read_only(true)
            .with_block_checksum_verification(false),
    )?;

    let stats = destination.merge_from(&source)?;

    assert_eq!(stats, CommitStats::default());
    assert_eq!(destination.iter_all()?.count(), 0);
    Ok(())
}
