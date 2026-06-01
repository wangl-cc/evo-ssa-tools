use crate::common::*;

#[test]
fn changing_target_block_size_can_reopen_existing_segments() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let first_options = options(&tempdir).with_target_block_size(128);
    let store = Store::open(first_options.clone())?;
    let first_key = make_key(1, 0, 0);
    commit_entries(&store, &[(first_key.clone(), make_value(1, 32))], true)?;
    drop(store);

    let second_options = first_options.with_target_block_size(512);
    let reopened = Store::open(second_options)?;

    assert_eq!(reopened.fetch_one(&first_key)?, Some(make_value(1, 32)));
    Ok(())
}

#[test]
fn manifest_is_line_oriented_v1_text() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    commit_entries(&store, &[(make_key(1, 0, 0), make_value(1, 8))], true)?;

    let manifest = fs::read_to_string(tempdir.path().join("MANIFEST"))?;

    assert!(manifest.starts_with("segment-cache-store manifest v1\n"));
    assert!(manifest.contains("version=1\n"));
    assert!(manifest.contains("value_layout=variable\n"));
    assert!(manifest.contains("[shard "));
    assert!(manifest.contains("segment\tsegment-"));
    assert!(!manifest.contains('{'));
    let segment_line = manifest
        .lines()
        .find(|line| line.starts_with("segment\t"))
        .expect("manifest should contain a segment entry");
    assert_eq!(segment_line.split('\t').count(), 8);
    Ok(())
}

#[test]
fn malformed_manifest_is_rejected() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    fs::write(
        tempdir.path().join("MANIFEST"),
        "not a segment-cache manifest\n",
    )?;

    let error = match Store::open(options(&tempdir)) {
        Ok(_) => panic!("malformed manifest should be rejected"),
        Err(error) => error,
    };

    assert!(matches!(error, Error::ManifestParse { .. }));
    Ok(())
}

#[test]
fn manifest_option_mismatches_are_rejected() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let base_options = options(&tempdir);
    let store = Store::open(base_options.clone())?;
    commit_entries(&store, &[(make_key(1, 0, 0), make_value(1, 8))], true)?;
    drop(store);

    for mismatched in [
        StoreOptions::new(tempdir.path(), 8)
            .with_shard_count(4)
            .with_shard_key_offset(8),
        options(&tempdir).with_fixed_value_len(8),
        options(&tempdir).with_shard_count(2),
        options(&tempdir).with_shard_key_offset(1),
    ] {
        let error = match Store::open(mismatched) {
            Ok(_) => panic!("mismatched manifest options should be rejected"),
            Err(error) => error,
        };
        assert!(
            matches!(
                error,
                Error::ManifestMismatch { .. } | Error::UnsupportedFormatVersion { .. }
            ),
            "unexpected error: {error:?}"
        );
    }

    assert_eq!(Store::open(base_options)?.iter_all()?.count(), 1);
    Ok(())
}

#[test]
fn manifest_rejects_reused_next_segment_id() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    commit_entries(&store, &[(make_key(1, 0, 0), make_value(1, 8))], true)?;
    drop(store);

    let manifest_path = tempdir.path().join("MANIFEST");
    let manifest = fs::read_to_string(&manifest_path)?;
    fs::write(
        &manifest_path,
        manifest.replace("next_segment_id=1", "next_segment_id=0"),
    )?;

    let error = match Store::open(options(&tempdir)) {
        Ok(_) => panic!("manifest should reject reused next_segment_id"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::ManifestMismatch {
        reason: "next_segment_id"
    }));
    Ok(())
}

#[test]
fn manifest_footer_mismatch_hides_wrong_segment_file() -> Result<()> {
    let tempdir_a = tempfile::tempdir()?;
    let tempdir_b = tempfile::tempdir()?;
    let options_a = options(&tempdir_a);
    let options_b = options(&tempdir_b);
    let store_a = Store::open(options_a.clone())?;
    let store_b = Store::open(options_b)?;
    let key_a = make_key(1, 0, 0);
    let key_b = make_key(9, 0, 0);
    commit_entries(&store_a, &[(key_a.clone(), make_value(1, 8))], true)?;
    commit_entries(&store_b, &[(key_b.clone(), make_value(2, 8))], true)?;

    let path_a = first_segment_path(tempdir_a.path())?;
    let path_b = first_segment_path(tempdir_b.path())?;
    drop(store_a);
    fs::copy(path_b, path_a)?;

    let reopened = Store::open(options_a)?;
    assert_eq!(reopened.fetch_one(&key_a)?, None);
    assert_eq!(reopened.fetch_one(&key_b)?, None);
    Ok(())
}
