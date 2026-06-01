use crate::common::*;

#[test]
fn range_iteration_returns_globally_sorted_records() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries = vec![
        (make_key(1, 0, 0), make_value(1, 8)),
        (make_key(1, 0, 1), make_value(2, 8)),
        (make_key(2, 0, 0), make_value(3, 8)),
        (make_key(3, 0, 0), make_value(4, 8)),
    ];
    commit_entries(&store, &entries, true)?;

    let range: Result<Vec<_>> = store
        .range(&make_key(1, 0, 1), &make_key(3, 0, 0))?
        .collect();
    let range = range?;
    assert_eq!(range.len(), 2);
    assert!(range.windows(2).all(|window| window[0].0 < window[1].0));
    Ok(())
}

#[test]
fn visit_range_matches_owned_range() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries: Vec<_> = (0..8u64)
        .map(|rep| (make_key(1, 0, rep), make_value(rep as u8, 8)))
        .collect();
    commit_entries(&store, &entries, true)?;
    let start = make_key(1, 0, 2);
    let end = make_key(1, 0, 6);
    let owned = store.range(&start, &end)?.collect::<Result<Vec<_>>>()?;
    let mut visited = Vec::new();
    store.visit_range(&start, &end, |key, value| {
        visited.push((key.to_vec(), value.to_vec()));
    })?;

    assert_eq!(visited, owned);
    Ok(())
}

#[test]
fn range_outside_all_segments_returns_empty() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    commit_entries(
        &store,
        &[
            (make_key(2, 0, 0), make_value(1, 8)),
            (make_key(3, 0, 0), make_value(2, 8)),
        ],
        true,
    )?;

    assert_eq!(
        store.range(&make_key(0, 0, 0), &make_key(1, 0, 0))?.count(),
        0
    );
    assert_eq!(
        store.range(&make_key(4, 0, 0), &make_key(5, 0, 0))?.count(),
        0
    );
    Ok(())
}

#[test]
fn overlapping_shard_ranges_iterate_with_k_way_merge() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir).with_shard_key_offset(4))?;
    let entries = vec![
        (make_key(1, 0, 0), make_value(1, 8)),
        (make_key(1, u32::MAX, 0), make_value(2, 8)),
        (make_key(2, 0, 0), make_value(3, 8)),
        (make_key(2, u32::MAX, 0), make_value(4, 8)),
    ];
    commit_entries(&store, &entries, true)?;

    let iterated = store.iter_all()?.collect::<Result<Vec<_>>>()?;
    assert_eq!(iterated, entries);
    Ok(())
}

#[test]
fn iter_all_returns_all_records_exactly_once() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries: Vec<_> = (0..32u64)
        .map(|rep| (make_key(1, (rep % 4) as u32, rep), make_value(rep as u8, 8)))
        .collect();
    commit_entries(&store, &entries, true)?;
    let all: Result<Vec<_>> = store.iter_all()?.collect();
    let all = all?;
    assert_eq!(all.len(), entries.len());
    assert!(all.windows(2).all(|window| window[0].0 < window[1].0));
    Ok(())
}

#[test]
fn visit_all_matches_iter_all_order() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries: Vec<_> = (0..32u64)
        .map(|rep| (make_key(1, (rep % 4) as u32, rep), make_value(rep as u8, 8)))
        .collect();
    commit_entries(&store, &entries, true)?;

    let iterated = store.iter_all()?.collect::<Result<Vec<_>>>()?;
    let mut visited = Vec::new();
    store.visit_all(|key, value| visited.push((key.to_vec(), value.to_vec())))?;

    assert_eq!(visited, iterated);
    Ok(())
}

#[test]
fn visit_many_ordered_slice_matches_fetch_many() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = Store::open(options(&tempdir))?;
    let entries: Vec<_> = (0..16u64)
        .map(|rep| (make_key(1, 0, rep), make_value(rep as u8, 8)))
        .collect();
    commit_entries(&store, &entries, true)?;

    let mut visited = Vec::new();
    let keys = entries
        .iter()
        .map(|(key, _)| key.clone())
        .collect::<Vec<_>>();
    store.visit_many_ordered_slice(&keys, |_, value| {
        visited.push(value.map(ToOwned::to_owned));
    })?;

    assert_eq!(
        visited,
        entries
            .iter()
            .map(|(_, value)| Some(value.clone()))
            .collect::<Vec<_>>()
    );
    Ok(())
}
