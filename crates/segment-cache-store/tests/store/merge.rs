use std::num::{NonZeroU32, NonZeroUsize};

#[cfg(any(feature = "checksum-crc32c", feature = "checksum-rapidhash"))]
use segment_cache_store::OpenOptions;
use segment_cache_store::{
    CommitStats, CreateOptions, Error, InputError, Result, Store, StoreMetadata,
};

use crate::support::api::{
    commit_entries, commit_options, create_options, create_options_with_key_len, create_store,
    create_store_with, make_key, make_value, reopen_store_read_only, test_block_checksum,
};
#[cfg(any(feature = "checksum-crc32c", feature = "checksum-rapidhash"))]
use crate::support::{
    api::metadata,
    segment_file::{corrupt_block_value_payload, first_segment_path},
};

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
    commit_entries(&destination, &destination_entries)?;
    commit_entries(&source, &source_entries)?;

    let stats = destination.merge_from_with_options(&source, &commit_options())?;

    assert_eq!(stats.input_records, 2);
    assert_eq!(stats.input_bytes, 48);
    assert_eq!(stats.segments_retired, 1);
    assert_eq!(stats.output_records, 4);
    assert_eq!(destination.iter_all()?.collect::<Result<Vec<_>>>()?, {
        let mut expected = destination_entries;
        expected.extend(source_entries);
        expected.sort_by(|left, right| left.0.cmp(&right.0));
        expected
    });
    assert_eq!(destination.normalize()?, CommitStats::default());
    Ok(())
}

#[test]
fn merge_from_collapses_identical_records() -> Result<()> {
    let destination_dir = tempfile::tempdir()?;
    let source_dir = tempfile::tempdir()?;
    let destination = create_store(&destination_dir)?;
    let source = create_store(&source_dir)?;
    let shared = make_key(2, 0, 0);
    let destination_only = make_key(2, 0, 1);
    let source_only = make_key(2, 0, 2);

    commit_entries(&destination, &[
        (shared.clone(), make_value(1, 8)),
        (destination_only.clone(), make_value(2, 8)),
    ])?;
    commit_entries(&source, &[
        (shared.clone(), make_value(1, 8)),
        (source_only.clone(), make_value(3, 8)),
    ])?;

    let stats = destination.merge_from_with_options(
        &source,
        &commit_options()
            .with_flush_threshold_records(NonZeroUsize::new(64).expect("non-zero literal")),
    )?;

    assert_eq!(stats.input_records, 2);
    assert_eq!(stats.output_records, 3);
    assert_eq!(stats.segments_retired, 1);
    assert_eq!(destination.fetch_one(&shared)?, Some(make_value(1, 8)));
    assert_eq!(
        destination.fetch_one(&destination_only)?,
        Some(make_value(2, 8))
    );
    assert_eq!(destination.fetch_one(&source_only)?, Some(make_value(3, 8)));
    assert_eq!(destination.normalize()?, CommitStats::default());
    Ok(())
}

#[test]
fn merge_from_rejects_conflicting_values_atomically() -> Result<()> {
    let destination_dir = tempfile::tempdir()?;
    let source_dir = tempfile::tempdir()?;
    let destination = create_store(&destination_dir)?;
    let source = create_store(&source_dir)?;
    let shared = make_key(2, 0, 0);
    let source_only = make_key(2, 0, 1);

    commit_entries(&destination, &[(shared.clone(), make_value(1, 8))])?;
    commit_entries(&source, &[
        (shared.clone(), make_value(9, 8)),
        (source_only.clone(), make_value(2, 8)),
    ])?;

    let error = destination
        .merge_from(&source)
        .expect_err("different values for one key must reject the merge");

    assert!(matches!(error, Error::Input(InputError::KeyConflict)));
    assert_eq!(destination.fetch_one(&shared)?, Some(make_value(1, 8)));
    assert_eq!(destination.fetch_one(&source_only)?, None);
    assert_eq!(destination.iter_all()?.count(), 1);
    Ok(())
}

#[test]
fn merge_from_normalizes_source_patch() -> Result<()> {
    let destination_dir = tempfile::tempdir()?;
    let source_dir = tempfile::tempdir()?;
    let destination = create_store(&destination_dir)?;
    let source = create_store(&source_dir)?;
    let first_main = make_key(3, 0, 0);
    let patch_only = make_key(3, 0, 1);
    let main_only = make_key(3, 0, 2);

    commit_entries(&source, &[
        (first_main.clone(), make_value(1, 8)),
        (main_only.clone(), make_value(2, 8)),
    ])?;
    commit_entries(&source, &[(patch_only.clone(), make_value(3, 8))])?;

    let stats = destination.merge_from_with_options(&source, &commit_options())?;

    assert_eq!(stats.input_records, 3);
    assert_eq!(stats.input_bytes, 72);
    assert_eq!(destination.iter_all()?.collect::<Result<Vec<_>>>()?, vec![
        (first_main, make_value(1, 8)),
        (patch_only, make_value(3, 8)),
        (main_only, make_value(2, 8)),
    ]);
    assert_eq!(destination.normalize()?, CommitStats::default());
    Ok(())
}

#[test]
fn merge_from_rejects_incompatible_source_metadata() -> Result<()> {
    let destination_dir = tempfile::tempdir()?;
    let source_dir = tempfile::tempdir()?;
    let destination = create_store(&destination_dir)?;
    let source = Store::create(
        source_dir.path(),
        CreateOptions::new(
            16,
            StoreMetadata::from_text("different-cache-namespace"),
            test_block_checksum(),
        )?,
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
fn merge_from_rejects_incompatible_source_geometry() -> Result<()> {
    let destination_dir = tempfile::tempdir()?;
    let short_key_source_dir = tempfile::tempdir()?;
    let fixed_value_source_dir = tempfile::tempdir()?;
    let destination = create_store(&destination_dir)?;
    let short_key_source =
        create_store_with(&short_key_source_dir, create_options_with_key_len(8))?;
    let fixed_value_source = create_store_with(
        &fixed_value_source_dir,
        create_options().with_fixed_value_len(NonZeroU32::new(8).expect("non-zero")),
    )?;

    let error = destination
        .merge_from(&short_key_source)
        .expect_err("merge rejects key length mismatch");
    assert!(matches!(
        error,
        Error::Input(InputError::SourceKeyLengthMismatch {
            expected: 16,
            actual: 8
        })
    ));

    let error = destination
        .merge_from(&fixed_value_source)
        .expect_err("merge rejects value layout mismatch");
    assert!(matches!(
        error,
        Error::Input(InputError::SourceValueLayoutMismatch)
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
fn merge_from_rejects_corrupt_source_blocks_when_forcing_verification() -> Result<()> {
    let destination_dir = tempfile::tempdir()?;
    let source_dir = tempfile::tempdir()?;
    let destination = create_store(&destination_dir)?;
    let source = create_store(&source_dir)?;
    let destination_key = make_key(3, 0, 0);
    let source_key = make_key(3, 0, 1);
    let destination_value = make_value(1, 32);
    commit_entries(&destination, &[(
        destination_key.clone(),
        destination_value.clone(),
    )])?;
    commit_entries(&source, &[(source_key.clone(), make_value(7, 32))])?;
    corrupt_block_value_payload(&first_segment_path(source_dir.path())?, 0)?;
    drop(source);
    let source = Store::open(
        source_dir.path(),
        OpenOptions::read_only(metadata()).with_block_checksum_verification(false),
    )?;

    let error = destination
        .merge_from(&source)
        .expect_err("strict merge must reject source corruption");

    assert!(matches!(error, Error::Corruption(_)));
    assert_eq!(
        destination.fetch_one(&destination_key)?,
        Some(destination_value)
    );
    assert_eq!(destination.fetch_one(&source_key)?, None);
    Ok(())
}
