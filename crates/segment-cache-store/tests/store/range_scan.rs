use segment_cache_store::Result;

use crate::support::api::{commit_entries, create_store, make_key, make_value};

#[test]
fn range_iteration_returns_globally_sorted_records() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let entries = vec![
        (make_key(1, 0, 0), make_value(1, 8)),
        (make_key(1, 0, 1), make_value(2, 8)),
        (make_key(2, 0, 0), make_value(3, 8)),
        (make_key(3, 0, 0), make_value(4, 8)),
    ];
    commit_entries(&store, &entries)?;

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
    let store = create_store(&tempdir)?;
    let entries: Vec<_> = (0..8u64)
        .map(|rep| (make_key(1, 0, rep), make_value(rep as u8, 8)))
        .collect();
    commit_entries(&store, &entries)?;
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
    let store = create_store(&tempdir)?;
    commit_entries(&store, &[
        (make_key(2, 0, 0), make_value(1, 8)),
        (make_key(3, 0, 0), make_value(2, 8)),
    ])?;

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
fn iter_all_returns_all_records_exactly_once() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let entries: Vec<_> = (0..32u64)
        .map(|rep| (make_key(1, (rep % 4) as u32, rep), make_value(rep as u8, 8)))
        .collect();
    commit_entries(&store, &entries)?;
    let all: Result<Vec<_>> = store.iter_all()?.collect();
    let all = all?;
    assert_eq!(all.len(), entries.len());
    assert!(all.windows(2).all(|window| window[0].0 < window[1].0));
    Ok(())
}

#[test]
fn visit_all_matches_iter_all_order() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    let entries: Vec<_> = (0..32u64)
        .map(|rep| (make_key(1, (rep % 4) as u32, rep), make_value(rep as u8, 8)))
        .collect();
    commit_entries(&store, &entries)?;

    let iterated = store.iter_all()?.collect::<Result<Vec<_>>>()?;
    let mut visited = Vec::new();
    store.visit_all(|key, value| visited.push((key.to_vec(), value.to_vec())))?;

    assert_eq!(visited, iterated);
    Ok(())
}

#[test]
fn range_cursors_merge_visible_patch_winners() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let store = create_store(&tempdir)?;
    commit_entries(&store, &[
        (make_key(1, 0, 0), make_value(0, 8)),
        (make_key(1, 0, 2), make_value(2, 8)),
        (make_key(1, 0, 5), make_value(9, 8)),
    ])?;
    commit_entries(&store, &[
        (make_key(1, 0, 1), make_value(1, 8)),
        (make_key(1, 0, 5), make_value(3, 8)),
        (make_key(1, 0, 6), make_value(6, 8)),
    ])?;
    let expected = vec![
        (make_key(1, 0, 0), make_value(0, 8)),
        (make_key(1, 0, 1), make_value(1, 8)),
        (make_key(1, 0, 2), make_value(2, 8)),
        (make_key(1, 0, 5), make_value(3, 8)),
        (make_key(1, 0, 6), make_value(6, 8)),
    ];

    assert_eq!(store.iter_all()?.collect::<Result<Vec<_>>>()?, expected);
    let mut visited = Vec::new();
    store.visit_all(|key, value| visited.push((key.to_vec(), value.to_vec())))?;
    assert_eq!(visited, expected);

    let range = store
        .range(&make_key(1, 0, 1), &make_key(1, 0, 6))?
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(range, expected[1..4]);
    Ok(())
}
