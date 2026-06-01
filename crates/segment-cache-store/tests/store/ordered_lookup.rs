use crate::common::*;

#[test]
fn probe_matches_fetch_hit_pattern() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries = vec![
        (make_key(1, 0, 0), make_value(1, 8)),
        (make_key(1, 0, 1), make_value(2, 8)),
        (make_key(1, 0, 3), make_value(3, 8)),
    ];
    commit_entries(&store, &entries, true)?;
    let keys = [
        make_key(1, 0, 0),
        make_key(1, 0, 1),
        make_key(1, 0, 2),
        make_key(1, 0, 3),
    ];
    let key_refs: Vec<_> = keys.iter().map(Vec::as_slice).collect();
    let probe = store.probe_ordered(key_refs.iter().copied())?;
    let fetch = store.fetch_many_ordered(key_refs.iter().copied())?;
    assert_eq!(
        probe,
        fetch
            .iter()
            .map(|entry| entry.is_some())
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn rejects_bad_key_streams() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let first = make_key(1, 0, 0);
    let second = make_key(1, 0, 1);

    let error = store
        .fetch_many_ordered([second.as_slice(), first.as_slice()])
        .unwrap_err();
    assert!(matches!(error, Error::UnsortedLookupKeys));

    let short = b"short".as_slice();
    let error = store.probe_ordered([short]).unwrap_err();
    assert!(matches!(error, Error::WrongKeyLength { .. }));
    Ok(())
}

#[test]
fn visit_many_matches_owned_fetch() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries: Vec<_> = (0..8u64)
        .map(|rep| (make_key(1, 0, rep), make_value(rep as u8, 24)))
        .collect();
    commit_entries(&store, &entries, true)?;
    let key_refs = entries
        .iter()
        .map(|(key, _)| key.as_slice())
        .collect::<Vec<_>>();

    let mut visited = vec![None; key_refs.len()];
    store.visit_many_ordered(key_refs.iter().copied(), |index, value| {
        visited[index] = value.map(ToOwned::to_owned);
    })?;

    assert_eq!(visited, store.fetch_many_ordered(key_refs.iter().copied())?);
    Ok(())
}

#[test]
fn session_can_restart_from_earlier_block() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries: Vec<_> = (0..64u64)
        .map(|rep| (make_key(1, 0, rep), make_value(rep as u8, 96)))
        .collect();
    commit_entries(&store, &entries, true)?;
    let key_refs: Vec<_> = entries.iter().map(|(key, _)| key.as_slice()).collect();
    let mut lookup = store.lookup_session();

    let first = lookup.fetch_many(key_refs.iter().copied())?;
    let second = lookup.fetch_many(key_refs.iter().copied())?;

    assert_eq!(first, second);
    assert_eq!(second.into_iter().flatten().count(), entries.len());
    Ok(())
}

#[test]
fn reports_misses_before_between_and_after_segments() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir).with_flush_threshold_records(1))?;
    let entries = vec![
        (make_key(1, 0, 1), make_value(1, 8)),
        (make_key(1, 0, 3), make_value(3, 8)),
    ];
    commit_entries(&store, &entries, true)?;
    let keys = [
        make_key(1, 0, 0),
        make_key(1, 0, 1),
        make_key(1, 0, 2),
        make_key(1, 0, 3),
        make_key(1, 0, 4),
    ];

    assert_eq!(store.probe_ordered(keys.iter().map(Vec::as_slice))?, vec![
        false, true, false, true, false
    ]);
    Ok(())
}
