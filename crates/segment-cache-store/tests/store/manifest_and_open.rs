use crc32c::crc32c;

use crate::common::*;

#[test]
fn changing_target_block_size_can_reopen_existing_segments() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let first_key = make_key(1, 0, 0);
    commit_entries_with_options(
        &store,
        &[(first_key.clone(), make_value(1, 32))],
        true,
        &CommitOptions::default().with_target_block_size(128),
    )?;
    drop(store);

    let reopened = reopen_store(&tempdir)?;
    commit_entries_with_options(
        &reopened,
        &[(make_key(9, 0, 0), make_value(2, 32))],
        true,
        &CommitOptions::default().with_target_block_size(512),
    )?;

    assert_eq!(reopened.fetch_one(&first_key)?, Some(make_value(1, 32)));
    Ok(())
}

#[test]
fn manifest_is_binary_v1_snapshot() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    commit_entries(&store, &[(make_key(1, 0, 0), make_value(1, 8))], true)?;

    let store_file = fs::read_to_string(tempdir.path().join("STORE"))?;
    let manifest = fs::read(tempdir.path().join("MANIFEST"))?;

    assert!(store_file.starts_with("segment-cache-store store v1\n"));
    assert!(store_file.contains("version=1\n"));
    assert!(store_file.contains("metadata=7365676d656e742d63616368652d73746f72652d74657374\n"));
    assert!(store_file.contains("key_len=16\n"));
    assert!(store_file.contains("value_len=0\n"));
    assert!(!store_file.contains('{'));

    assert_eq!(&manifest[..4], b"SCSM");
    assert_eq!(
        u32::from_le_bytes(manifest[4..8].try_into().expect("version")),
        1
    );
    assert_eq!(
        u32::from_le_bytes(manifest[8..12].try_into().expect("key len")),
        16
    );
    assert_eq!(
        u32::from_le_bytes(manifest[12..16].try_into().expect("next segment id")),
        1
    );
    assert_eq!(
        u32::from_le_bytes(manifest[16..20].try_into().expect("segment count")),
        1
    );
    assert_eq!(manifest.len(), 20 + 4 + 16 + 16 + 4);
    Ok(())
}

#[test]
fn missing_store_file_is_rejected() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    fs::write(tempdir.path().join("MANIFEST"), "not enough\n")?;

    let error = match Store::open(tempdir.path(), open_options()) {
        Ok(_) => panic!("missing store file should be rejected"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        Error::Catalog(CatalogError::Mismatch(CatalogMismatch::MissingStore))
    ));
    Ok(())
}

#[test]
fn malformed_store_file_is_rejected() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    drop(store);

    fs::write(tempdir.path().join("STORE"), "not a segment-cache store\n")?;

    let error = match Store::open(tempdir.path(), open_options()) {
        Ok(_) => panic!("malformed store should be rejected"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        Error::Catalog(CatalogError::MalformedStore { .. })
    ));
    Ok(())
}

#[test]
fn malformed_manifest_is_rejected() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    drop(store);

    fs::write(
        tempdir.path().join("MANIFEST"),
        "not a segment-cache manifest\n",
    )?;

    let error = match Store::open(tempdir.path(), open_options()) {
        Ok(_) => panic!("malformed manifest should be rejected"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        Error::Catalog(CatalogError::MalformedManifest { .. })
    ));
    Ok(())
}

#[test]
fn manifest_metadata_mismatch_is_rejected() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    commit_entries(&store, &[(make_key(1, 0, 0), make_value(1, 8))], true)?;
    drop(store);

    let error = match Store::open(
        tempdir.path(),
        StoreOpenOptions::new(StoreMetadata::from_text("different")),
    ) {
        Ok(_) => panic!("mismatched manifest metadata should be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        Error::Catalog(CatalogError::Mismatch(CatalogMismatch::Metadata))
    ));

    assert_eq!(reopen_store(&tempdir)?.iter_all()?.count(), 1);
    Ok(())
}

#[test]
fn manifest_rejects_reused_next_segment_id() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    commit_entries(&store, &[(make_key(1, 0, 0), make_value(1, 8))], true)?;
    drop(store);

    let manifest_path = tempdir.path().join("MANIFEST");
    let mut manifest = fs::read(&manifest_path)?;
    manifest[12..16].copy_from_slice(&0u32.to_le_bytes());
    let crc_offset = manifest.len() - 4;
    let crc = crc32c(&manifest[..crc_offset]);
    manifest[crc_offset..].copy_from_slice(&crc.to_le_bytes());
    fs::write(&manifest_path, manifest)?;

    let error = match Store::open(tempdir.path(), open_options()) {
        Ok(_) => panic!("manifest should reject reused next_segment_id"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        Error::Catalog(CatalogError::Mismatch(CatalogMismatch::NextSegmentId))
    ));
    Ok(())
}

#[test]
fn manifest_footer_mismatch_hides_wrong_segment_file() -> Result<()> {
    let tempdir_a = tempfile::tempdir()?;
    let tempdir_b = tempfile::tempdir()?;
    let store_a = create_store(&tempdir_a)?;
    let store_b = create_store(&tempdir_b)?;
    let key_a = make_key(1, 0, 0);
    let key_b = make_key(9, 0, 0);
    commit_entries(&store_a, &[(key_a.clone(), make_value(1, 8))], true)?;
    commit_entries(&store_b, &[(key_b.clone(), make_value(2, 8))], true)?;

    let path_a = first_segment_path(tempdir_a.path())?;
    let path_b = first_segment_path(tempdir_b.path())?;
    drop(store_a);
    fs::copy(path_b, path_a)?;

    let reopened = reopen_store(&tempdir_a)?;
    assert_eq!(reopened.fetch_one(&key_a)?, None);
    assert_eq!(reopened.fetch_one(&key_b)?, None);
    Ok(())
}
