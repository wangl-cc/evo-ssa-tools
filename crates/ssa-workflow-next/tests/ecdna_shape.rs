//! ecDNA-shaped chain: early SSA (stochastic root) -> top-tumor ODE (deterministic, shared
//! operator) -> surgery (stochastic) -> regrowth ODE. The numerics are stand-ins; the tests check
//! the shape of the API: demand-first lookups, stage-by-stage execution, per-stage pools,
//! borrowing, worker-state reuse, linkage checks and schedule-independent results.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use parking_lot::Mutex;
use rand::RngExt;
use ssa_workflow_next::{Block, Error, Exec, Kernel, MemoryStore, Result, Stage, Streams};

type Log = Arc<Mutex<Vec<&'static str>>>;

/// Exact SSA stand-in. Param = initial cells.
struct EarlySsa(Log);

impl Kernel for EarlySsa {
    const ID: &'static str = "early-ssa-v1";
    type Param = u32;
    type Input = ();
    type Output = Vec<u64>;
    type State = ();

    fn compute(
        &self,
        streams: &mut Streams<'_>,
        _: &mut (),
        cells: &u32,
        _: &(),
    ) -> Result<Vec<u64>> {
        self.0.lock().push(Self::ID);
        let rng = streams.rng();
        Ok((0..6)
            .map(|_| rng.random_range(0..=u64::from(*cells)))
            .collect())
    }
}

/// A different root with the same output type, used to test linkage checks.
struct OtherRoot;

impl Kernel for OtherRoot {
    const ID: &'static str = "other-root-v1";
    type Param = u32;
    type Input = ();
    type Output = Vec<u64>;
    type State = ();

    fn compute(&self, _: &mut Streams<'_>, _: &mut (), cells: &u32, _: &()) -> Result<Vec<u64>> {
        Ok(vec![u64::from(*cells)])
    }
}

/// Per-worker cache of an "assembled operator", keyed by the model it was built for.
#[derive(Default)]
struct OdeWorkspace {
    operator: Option<(usize, Vec<f64>)>,
}

impl OdeWorkspace {
    fn operator(&mut self, model: usize, builds: &AtomicUsize) -> &[f64] {
        if self.operator.as_ref().map(|(key, _)| *key) != Some(model) {
            builds.fetch_add(1, Ordering::Relaxed);
            let operator = (0..model).map(|i| 1.0 + i as f64 / 10.0).collect();
            self.operator = Some((model, operator));
        }
        &self.operator.as_ref().expect("set above").1
    }
}

/// Deterministic growth stand-in. Param = target size.
struct TopOde {
    log: Log,
    builds: AtomicUsize,
}

impl Kernel for TopOde {
    const ID: &'static str = "top-ode-v1";
    type Param = f64;
    type Input = Vec<u64>;
    type Output = Vec<f64>;
    type State = OdeWorkspace;

    fn compute(
        &self,
        _: &mut Streams<'_>,
        ws: &mut OdeWorkspace,
        target: &f64,
        early: &Vec<u64>,
    ) -> Result<Vec<f64>> {
        self.log.lock().push(Self::ID);
        let operator = ws.operator(early.len(), &self.builds);
        let total: f64 = early.iter().map(|&n| n as f64).sum::<f64>().max(1.0);
        Ok(early
            .iter()
            .zip(operator)
            .map(|(&n, w)| n as f64 * w * target / total)
            .collect())
    }
}

/// Multinomial surgery stand-in. Param = residual size.
struct Surgery(Log);

impl Kernel for Surgery {
    const ID: &'static str = "surgery-v1";
    type Param = u32;
    type Input = Vec<f64>;
    type Output = Vec<u64>;
    type State = ();

    fn compute(
        &self,
        streams: &mut Streams<'_>,
        _: &mut (),
        residual: &u32,
        top: &Vec<f64>,
    ) -> Result<Vec<u64>> {
        self.0.lock().push(Self::ID);
        let total: f64 = top.iter().sum();
        let rng = streams.rng();
        let mut counts = vec![0u64; top.len()];
        for _ in 0..*residual {
            let mut u = rng.random::<f64>() * total;
            let pick = top.iter().position(|&w| {
                u -= w;
                u <= 0.0
            });
            counts[pick.unwrap_or(top.len() - 1)] += 1;
        }
        Ok(counts)
    }
}

/// Deterministic regrowth stand-in. Param = final size.
struct RegrowOde {
    log: Log,
    builds: AtomicUsize,
}

impl Kernel for RegrowOde {
    const ID: &'static str = "regrow-ode-v1";
    type Param = f64;
    type Input = Vec<u64>;
    type Output = Vec<f64>;
    type State = OdeWorkspace;

    fn compute(
        &self,
        _: &mut Streams<'_>,
        ws: &mut OdeWorkspace,
        target: &f64,
        residual: &Vec<u64>,
    ) -> Result<Vec<f64>> {
        self.log.lock().push(Self::ID);
        let operator = ws.operator(residual.len(), &self.builds);
        let total: f64 = residual.iter().map(|&n| n as f64).sum::<f64>().max(1.0);
        Ok(residual
            .iter()
            .zip(operator)
            .map(|(&n, w)| n as f64 * w * target / total)
            .collect())
    }
}

struct Stages<'s> {
    early: Stage<'s, EarlySsa, MemoryStore>,
    top: Stage<'s, TopOde, MemoryStore>,
    surgery: Stage<'s, Surgery, MemoryStore>,
    regrow: Stage<'s, RegrowOde, MemoryStore>,
}

fn stages<'s>(store: &'s MemoryStore, log: &Log) -> Stages<'s> {
    let early = Stage::root(EarlySsa(log.clone()), store);
    let top = Stage::after(
        TopOde {
            log: log.clone(),
            builds: AtomicUsize::new(0),
        },
        store,
        &early,
    );
    let surgery = Stage::after(Surgery(log.clone()), store, &top);
    let regrow = Stage::after(
        RegrowOde {
            log: log.clone(),
            builds: AtomicUsize::new(0),
        },
        store,
        &surgery,
    );
    Stages {
        early,
        top,
        surgery,
        regrow,
    }
}

/// The caller-driven loop from the design: the caller picks block size and per-stage pools.
fn run_final(
    st: &Stages<'_>,
    reps: std::ops::Range<u64>,
    block: usize,
    ssa: Exec,
    ode: Exec,
) -> Result<Vec<Vec<f64>>> {
    let requests: Vec<(u32, u64)> = reps.map(|rep| (50u32, rep)).collect();
    let mut out = Vec::new();
    for chunk in requests.chunks(block) {
        let e = st.early.request_root(chunk).on(ssa);
        let t = st.top.request(&e, 1e6)?.on(ode);
        let r = st.surgery.request(&t, 200)?;
        let f = st.regrow.request(&r, 1e9)?.on(ode);
        out.extend(f.materialize()?.into_values());
    }
    Ok(out)
}

#[test]
fn cold_run_is_stage_by_stage_within_each_block() -> Result<()> {
    let store = MemoryStore::default();
    let log = Log::default();
    let st = stages(&store, &log);
    let finals = run_final(&st, 0..8, 4, Exec::threads(4), Exec::threads(2))?;
    assert_eq!(finals.len(), 8);

    // Within each block of 4: all early, then all top, then all surgery, then all regrowth.
    let log = log.lock();
    assert_eq!(log.len(), 32);
    for block in log.chunks(16) {
        let stages: Vec<&str> = block
            .chunks(4)
            .map(|group| {
                assert!(
                    group.iter().all(|id| *id == group[0]),
                    "stage interleaved: {group:?}"
                );
                group[0]
            })
            .collect();
        assert_eq!(
            stages,
            ["early-ssa-v1", "top-ode-v1", "surgery-v1", "regrow-ode-v1"]
        );
    }
    Ok(())
}

#[test]
fn all_hit_rerun_never_touches_upstream() -> Result<()> {
    let store = MemoryStore::default();
    let log = Log::default();
    let first = run_final(
        &stages(&store, &log),
        0..8,
        4,
        Exec::default(),
        Exec::default(),
    )?;
    log.lock().clear();
    let contains_before = store.contains_calls.load(Ordering::Relaxed);

    let st = stages(&store, &log);
    let second = run_final(&st, 0..8, 4, Exec::default(), Exec::default())?;

    assert_eq!(first, second);
    assert!(log.lock().is_empty(), "nothing recomputed");
    assert_eq!(st.regrow.stats().loaded, 8);
    for stats in [st.early.stats(), st.top.stats(), st.surgery.stats()] {
        assert_eq!(
            (stats.loaded, stats.computed),
            (0, 0),
            "upstream never read"
        );
    }
    // One ordered lookup per block, only on the final stage.
    assert_eq!(
        store.contains_calls.load(Ordering::Relaxed) - contains_before,
        2
    );
    Ok(())
}

#[test]
fn extending_repetitions_computes_only_new_items() -> Result<()> {
    let store = MemoryStore::default();
    let log = Log::default();
    run_final(
        &stages(&store, &log),
        0..4,
        4,
        Exec::default(),
        Exec::default(),
    )?;

    let st = stages(&store, &log);
    run_final(&st, 0..8, 8, Exec::default(), Exec::default())?;
    assert_eq!(
        st.regrow.stats(),
        ssa_workflow_next::StageStats {
            computed: 4,
            loaded: 4,
            states: st.regrow.stats().states
        }
    );
    assert_eq!(st.early.stats().computed, 4);
    assert_eq!(
        st.early.stats().loaded,
        0,
        "upstream asked only for the 4 missing items"
    );
    Ok(())
}

#[test]
fn results_do_not_depend_on_block_size_or_threads() -> Result<()> {
    let store_a = MemoryStore::default();
    let a = run_final(
        &stages(&store_a, &Log::default()),
        0..16,
        1,
        Exec::serial(),
        Exec::serial(),
    )?;
    let store_b = MemoryStore::default();
    let b = run_final(
        &stages(&store_b, &Log::default()),
        0..16,
        16,
        Exec::threads(4),
        Exec::threads(4),
    )?;
    assert_eq!(a, b);
    Ok(())
}

#[test]
fn worker_state_survives_across_blocks() -> Result<()> {
    let store = MemoryStore::default();
    let st = stages(&store, &Log::default());
    run_final(&st, 0..32, 4, Exec::default(), Exec::threads(2))?;
    // 32 items in 8 blocks, but never more states than workers (a late worker may reuse a state
    // an early one already returned), and the operator is assembled at most once per state.
    let builds = st.top.kernel().builds.load(Ordering::Relaxed);
    let states = st.top.stats().states;
    assert!(
        (1..=2).contains(&states) && builds <= states,
        "builds={builds}, states={states}"
    );
    Ok(())
}

#[test]
fn staged_bundle_borrows_every_stage() -> Result<()> {
    let store = MemoryStore::default();
    let log = Log::default();
    let st = stages(&store, &log);

    let e: Block<Vec<u64>> = st.early.request_repeated(&50, 0..4).materialize()?;
    let t = st.top.request(&e, 1e6)?.materialize()?;
    let r = st.surgery.request(&t, 200)?.materialize()?;
    let f = st.regrow.request(&r, 1e9)?.materialize()?;

    // Four blocks, each computed once; downstream borrowed upstream values without clones.
    assert_eq!((e.len(), t.len(), r.len(), f.len()), (4, 4, 4, 4));
    assert_eq!(log.lock().len(), 16);
    // Same keys and values as the demand-driven chain.
    let finals = run_final(&st, 0..4, 4, Exec::default(), Exec::default())?;
    assert_eq!(f.into_values(), finals);
    Ok(())
}

#[test]
fn two_branches_share_one_early_block() -> Result<()> {
    let store = MemoryStore::default();
    let log = Log::default();
    let st = stages(&store, &log);
    let e = st.early.request_repeated(&50, 0..4).materialize()?;

    let small = st.top.request(&e, 1e5)?.materialize()?;
    let large = st.top.request(&e, 1e7)?.materialize()?;
    assert_ne!(small.values(), large.values());
    assert_eq!(
        st.early.stats().computed,
        4,
        "early computed once for both branches"
    );
    Ok(())
}

#[test]
fn upstream_from_another_stage_is_rejected() -> Result<()> {
    let store = MemoryStore::default();
    let st = stages(&store, &Log::default());
    let other = Stage::root(OtherRoot, &store);
    let wrong = other.request_repeated(&50, 0..1).materialize()?;

    // Same Rust type (Block<Vec<u64>>), different computation path.
    match st.top.request(&wrong, 1e6) {
        Err(Error::WrongUpstream { .. }) => Ok(()),
        Err(error) => panic!("unexpected error: {error}"),
        Ok(_) => panic!("wrong upstream accepted"),
    }
}

#[test]
fn fan_out_pulls_each_upstream_item_once() -> Result<()> {
    let store = MemoryStore::default();
    let log = Log::default();
    let st = stages(&store, &log);
    let targets = [1e5, 1e6, 1e7];

    // One simulation chunk analysed under three parameters, never materializing early.
    let e = st.early.request_repeated(&50, 0..4);
    let t = st.top.request_fan(&e, &targets)?;
    let r = st.surgery.request(&t, 200)?;
    let f = st.regrow.request(&r, 1e9)?.materialize()?;

    assert_eq!(f.len(), 12);
    assert_eq!(st.early.stats().computed, 4, "early asked once per item");
    assert_eq!(st.top.stats().computed, 12);

    // Parameter-major: chunk p holds every replicate under targets[p], with the same keys and
    // values as a separate request per parameter.
    let fresh = MemoryStore::default();
    let alone = stages(&fresh, &Log::default());
    for (target, chunk) in targets.iter().zip(f.values().chunks(4)) {
        let e = alone.early.request_repeated(&50, 0..4);
        let t = alone.top.request(&e, *target)?;
        let r = alone.surgery.request(&t, 200)?;
        let f = alone.regrow.request(&r, 1e9)?.materialize()?;
        assert_eq!(f.values(), chunk);
    }
    Ok(())
}

#[test]
fn adding_a_parameter_computes_only_its_items() -> Result<()> {
    let store = MemoryStore::default();
    let st = stages(&store, &Log::default());
    let e = st.early.request_repeated(&50, 0..4);
    st.top.request_fan(&e, &[1e5, 1e6])?.materialize()?;

    let st = stages(&store, &Log::default());
    let e = st.early.request_repeated(&50, 0..4);
    st.top.request_fan(&e, &[1e5, 1e6, 1e7])?.materialize()?;
    assert_eq!((st.top.stats().computed, st.top.stats().loaded), (4, 8));
    assert_eq!(
        (st.early.stats().computed, st.early.stats().loaded),
        (0, 4),
        "early read once per item for the new parameter only"
    );
    Ok(())
}

/// Tracks how many items run at once and fails on one chosen item.
#[derive(Default)]
struct Probe {
    running: AtomicUsize,
    peak: AtomicUsize,
    fail_on: Option<u32>,
}

impl Kernel for Probe {
    const ID: &'static str = "probe-v1";
    type Param = u32;
    type Input = ();
    type Output = u64;
    type State = ();

    fn compute(&self, streams: &mut Streams<'_>, _: &mut (), param: &u32, _: &()) -> Result<u64> {
        let now = self.running.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(5));
        self.running.fetch_sub(1, Ordering::SeqCst);
        if Some(*param) == self.fail_on {
            return Err(Error::Kernel {
                path: Self::ID.into(),
                message: format!("item {param} failed"),
            });
        }
        Ok(streams.rng().random())
    }
}

#[test]
fn at_most_n_items_in_flight() -> Result<()> {
    let store = MemoryStore::default();
    let stage = Stage::root(Probe::default(), &store);
    let items: Vec<(u32, u64)> = (0..24).map(|rep| (0, rep)).collect();
    stage
        .request_root(&items)
        .on(Exec::threads(3))
        .materialize()?;
    let peak = stage.kernel().peak.load(Ordering::SeqCst);
    assert!((1..=3).contains(&peak), "peak={peak}");
    assert!(stage.stats().states <= 3);
    Ok(())
}

#[test]
fn kernel_error_stops_the_block() {
    let store = MemoryStore::default();
    let stage = Stage::root(
        Probe {
            fail_on: Some(5),
            ..Probe::default()
        },
        &store,
    );
    let items: Vec<(u32, u64)> = (0..64).map(|param| (param, 0)).collect();
    match stage
        .request_root(&items)
        .on(Exec::threads(4))
        .materialize()
    {
        Err(Error::Kernel { message, .. }) => assert_eq!(message, "item 5 failed"),
        other => panic!("expected kernel error, got {:?}", other.map(|b| b.len())),
    }
    assert!(
        store.puts.load(Ordering::Relaxed) < 64,
        "workers stopped early"
    );
}
