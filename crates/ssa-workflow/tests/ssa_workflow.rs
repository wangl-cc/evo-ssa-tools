use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use rand::RngExt;
use ssa_workflow::{error::Result, prelude::*};

const INITIAL_CELLS: u32 = 25;
const MAX_EVENTS: u32 = 64;

#[derive(Debug, Clone, PartialEq)]
struct SsaSummary {
    final_cells: u32,
    events: u32,
    elapsed_time: f64,
    population_grew: bool,
}

#[test]
fn birth_death_ssa_transform_is_reproducible_and_cached() -> Result<()> {
    let ssa_calls = Arc::new(AtomicUsize::new(0));
    let ssa_calls_clone = Arc::clone(&ssa_calls);
    let summary_calls = Arc::new(AtomicUsize::new(0));
    let summary_calls_clone = Arc::clone(&summary_calls);

    let transform = StochasticTask::builder("birth-death-ssa-integration-v1")
        .streams(["waiting_time", "reaction_choice"])
        .function(move |streams, (initial_cells, max_events): (u32, u32)| {
            ssa_calls_clone.fetch_add(1, Ordering::SeqCst);

            let [waiting_time_rng, reaction_choice_rng] = streams.as_mut();
            let birth_rate = 0.8;
            let death_rate = 0.4;
            let mut cells = initial_cells.max(1);
            let mut time = 0.0;
            let mut events = 0;

            for _ in 0..max_events {
                let birth_propensity = birth_rate * cells as f64;
                let death_propensity = death_rate * cells as f64;
                let total_propensity = birth_propensity + death_propensity;
                if total_propensity == 0.0 {
                    break;
                }

                let u = waiting_time_rng
                    .random::<f64>()
                    .clamp(f64::MIN_POSITIVE, 1.0);
                time += -u.ln() / total_propensity;

                let reaction_threshold = reaction_choice_rng.random::<f64>() * total_propensity;
                if reaction_threshold < birth_propensity {
                    cells = cells.saturating_add(1);
                } else {
                    cells = cells.saturating_sub(1);
                }
                events += 1;
            }

            Ok((cells, events, time))
        })
        .cache(ManagedHashCache::<(u32, u32, f64)>::default())
        .build()?
        .transform("birth-death-ssa-summary-v1")
        .function(
            move |(final_cells, events, elapsed_time): (u32, u32, f64)| {
                summary_calls_clone.fetch_add(1, Ordering::SeqCst);
                Ok(SsaSummary {
                    final_cells,
                    events,
                    elapsed_time,
                    population_grew: final_cells >= INITIAL_CELLS,
                })
            },
        )
        .cache(ManagedHashCache::<SsaSummary>::default())
        .build()?;

    let inputs: Vec<_> = (0..16)
        .map(|repetition| StochasticInput::new((INITIAL_CELLS, MAX_EVENTS), repetition))
        .collect();
    let input_count = inputs.len();

    let first_run: Vec<SsaSummary> = transform.with_inputs(inputs.clone()).collect()?;

    assert_eq!(first_run.len(), input_count);
    assert!(first_run.iter().all(|summary| summary.events <= MAX_EVENTS));
    assert!(
        first_run
            .iter()
            .all(|summary| summary.elapsed_time.is_finite() && summary.elapsed_time >= 0.0)
    );
    assert_eq!(ssa_calls.load(Ordering::SeqCst), input_count);
    assert_eq!(summary_calls.load(Ordering::SeqCst), input_count);

    let second_run: Vec<SsaSummary> = transform.with_inputs(inputs).collect()?;

    assert_eq!(first_run, second_run);
    assert_eq!(ssa_calls.load(Ordering::SeqCst), input_count);
    assert_eq!(summary_calls.load(Ordering::SeqCst), input_count);
    Ok(())
}
