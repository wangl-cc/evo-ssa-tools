use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::profile::{KEY_LEN, ValueProfile};

#[derive(Clone)]
pub(crate) struct Dataset {
    pub(crate) entries: Vec<(Vec<u8>, Vec<u8>)>,
    pub(crate) ordered_keys: Vec<Vec<u8>>,
}

#[derive(Clone)]
pub(crate) struct MiddleInsertDataset {
    pub(crate) old_entries: Vec<(Vec<u8>, Vec<u8>)>,
    pub(crate) new_entries: Vec<(Vec<u8>, Vec<u8>)>,
    pub(crate) new_keys: Vec<Vec<u8>>,
    pub(crate) inserted_entries: Vec<(Vec<u8>, Vec<u8>)>,
}

fn make_key(axis_0: u64, axis_1: u64, axis_2: u64, axis_3: u64, axis_4: u64, rep: u64) -> Vec<u8> {
    // Wide, ordered keys with a mostly fixed prefix are a closer fit for
    // canonical-encoded parameter sweeps than short random keys. The first
    // fields model stable namespace/schema/model choices; later fields model
    // parameter axes, with repetition as the fastest-changing suffix.
    let fields = [
        0x5353475f43414348_u64,
        0x0000_0000_0000_0001,
        0x0000_0000_0000_0010,
        0x0000_0000_0000_0020,
        0x0000_0000_0000_0030,
        0x0000_0000_0000_0040,
        axis_0,
        axis_1,
        axis_2,
        axis_3,
        axis_4,
        0,
        0,
        0,
        0,
        rep,
    ];
    let mut key = Vec::with_capacity(KEY_LEN);
    for field in fields {
        key.extend_from_slice(&field.to_be_bytes());
    }
    debug_assert_eq!(key.len(), KEY_LEN);
    key
}

fn make_grid_key(x: u32, y: u32, rep: u64) -> Vec<u8> {
    make_key(u64::from(x), u64::from(y), 0, 0, 0, rep)
}

fn make_value(rng: &mut StdRng, profile: ValueProfile) -> Vec<u8> {
    let base = profile.base_len();
    let jitter = profile.jitter();
    let spread = jitter.saturating_mul(2) + 1;
    let offset = usize::from(rng.random::<u16>()) % spread;
    let len = base + offset - jitter;
    (0..len).map(|_| rng.random()).collect()
}

fn make_grid_value(profile: ValueProfile, x: u32, y: u32, rep: u64) -> Vec<u8> {
    let seed = 1_000_003u64
        .wrapping_mul(u64::from(x) + 1)
        .wrapping_add(97_003u64.wrapping_mul(u64::from(y) + 1))
        .wrapping_add(7_919u64.wrapping_mul(rep + 1))
        .wrapping_add(u64::try_from(profile.base_len()).expect("profile base len should fit"));
    let mut rng = StdRng::seed_from_u64(seed);
    make_value(&mut rng, profile)
}

pub(crate) fn build_dataset(n: usize, profile: ValueProfile) -> Dataset {
    let mut rng = StdRng::seed_from_u64(
        42 + u64::try_from(profile.base_len()).expect("profile base len should fit"),
    );
    let mut entries = Vec::with_capacity(n);
    for index in 0..n {
        let axis_0 = (index / 4096) as u64;
        let axis_1 = ((index / 1024) % 4) as u64;
        let axis_2 = ((index / 256) % 4) as u64;
        let axis_3 = ((index / 16) % 16) as u64;
        let axis_4 = 0;
        let rep = (index % 16) as u64;
        entries.push((
            make_key(axis_0, axis_1, axis_2, axis_3, axis_4, rep),
            make_value(&mut rng, profile),
        ));
    }
    entries.sort_by(|left, right| left.0.cmp(&right.0));

    let ordered_keys = entries
        .iter()
        .map(|(key, _)| key.clone())
        .collect::<Vec<_>>();
    Dataset {
        entries,
        ordered_keys,
    }
}

pub(crate) fn build_middle_insert_dataset(profile: ValueProfile) -> MiddleInsertDataset {
    const X_COUNT: u32 = 512;
    const REP_COUNT: u64 = 16;
    const OLD_Y: &[u32] = &[0, 2];
    const NEW_Y: &[u32] = &[0, 1, 2];
    const INSERTED_Y: u32 = 1;

    let mut old_entries = Vec::with_capacity(
        usize::try_from(X_COUNT).expect("x count should fit") * OLD_Y.len() * REP_COUNT as usize,
    );
    let mut new_entries = Vec::with_capacity(
        usize::try_from(X_COUNT).expect("x count should fit") * NEW_Y.len() * REP_COUNT as usize,
    );
    let mut inserted_entries = Vec::with_capacity(
        usize::try_from(X_COUNT).expect("x count should fit") * REP_COUNT as usize,
    );

    for x in 0..X_COUNT {
        for &y in OLD_Y {
            for rep in 0..REP_COUNT {
                old_entries.push((
                    make_grid_key(x, y, rep),
                    make_grid_value(profile, x, y, rep),
                ));
            }
        }
        for &y in NEW_Y {
            for rep in 0..REP_COUNT {
                let entry = (
                    make_grid_key(x, y, rep),
                    make_grid_value(profile, x, y, rep),
                );
                if y == INSERTED_Y {
                    inserted_entries.push(entry.clone());
                }
                new_entries.push(entry);
            }
        }
    }

    old_entries.sort_by(|left, right| left.0.cmp(&right.0));
    new_entries.sort_by(|left, right| left.0.cmp(&right.0));
    inserted_entries.sort_by(|left, right| left.0.cmp(&right.0));
    let new_keys = new_entries
        .iter()
        .map(|(key, _)| key.clone())
        .collect::<Vec<_>>();

    MiddleInsertDataset {
        old_entries,
        new_entries,
        new_keys,
        inserted_entries,
    }
}
