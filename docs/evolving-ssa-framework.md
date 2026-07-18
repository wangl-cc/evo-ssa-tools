# Evolving SSA Framework Research Notes

This document records the research and design path. The implemented crate contract is documented in [`crates/evo-ssa/docs/design.md`](../crates/evo-ssa/docs/design.md).

## Problem Shape

This repository is aimed at evolutionary stochastic simulations rather than fixed chemical reaction networks. The important difference is that the set of species is part of the simulated state: a reaction can create a new species, remove the last individual of an existing species, or alter which concrete reaction instances are enabled. The reaction families themselves can still be fixed. For example, a model may always have birth, death, mutation, and interaction families, while each family contains many concrete instances such as `birth(species_i)`, `death(species_i)`, or `mutation(species_i -> species_j)`. A useful framework therefore should not ask model authors to pre-enumerate all possible concrete instances over all possible species. It should let them describe fixed reaction families and let the engine maintain the currently active concrete instances under those families.

The `pheno-geno` implementation in `/Users/loong/Repos/Projects/pheno-geno/crates/core/src/ssa` is a good compact baseline. It separates model parameters, reactions, population state, and output callbacks. It also supports reaction-specific waiting time logic through `Reaction::tau`, which is more general than a constant-propensity direct method. The main limitations for this repository are that reactions are stored as a fixed tuple of reaction families, the execution loop knows only one concrete clock per family, the population layout assumes two cell classes, and every step recomputes every family clock. Those are fine for five aggregate reactions, but evolutionary models need fixed families with many active concrete instances and cached hazards or propensities per instance.

## Design Goal

The framework should own scheduling, random stream usage, event-loop invariants, and reusable Gillespie method implementations. Models should own biological or evolutionary semantics: how species are represented, which local channels are possible, how propensities are computed, and how firing a channel mutates state. The core performance invariant should be that one event costs roughly `O(log M + U)` or `O(U log M)` where `M` is the number of active concrete channels and `U` is the number of channels whose propensities actually change after the event, not `O(number_of_possible_species)` and not necessarily `O(M)`.

## Core Vocabulary

- `SpeciesId`: a compact stable runtime identifier for an extant or previously seen species.
- `SpeciesMeta`: model-defined immutable or slowly changing data for a species, such as genotype, phenotype, parent lineage, fitness class, or cached mutation-neighborhood information.
- `PopulationState`: sparse counts and model-owned per-species metadata. It should support adding a species, changing a count, removing inactive species from active indices, and sampling individuals within a species when needed.
- `ReactionFamily`: a fixed model-level rule template such as birth, death, mutation, competition, or migration. Most models should have a small fixed set of families.
- `ChannelKey`: a stable key for one active concrete instance under a reaction family, for example `(family_id, species)` for unary birth/death, `(family_id, source_species, target_species)` for explicit mutation, or `(family_id, species_a, species_b)` for pair interactions.
- `Channel`: cached scheduler-facing data for one concrete instance: key, current propensity or hazard state, dependency metadata, and any compact payload needed by the family to fire it.
- `Event`: the result of firing a channel: time, channel key, population deltas, new species, removed species, changed environment state, and optional model-specific observation data.
- `Scheduler`: the Gillespie method implementation. It chooses the next channel/time from active channels and consumes local channel updates after each event.

## Proposed Trait Split

The existing `pheno-geno` `Reaction` trait combines three concerns: propensity calculation, waiting-time generation, and state mutation. For a framework with multiple methods, these should be split so the method can own the clock mathematics.

```rust
pub trait EvolvingModel {
    type Species;
    type State;
    type ChannelKey: Copy + Eq + std::hash::Hash;
    type ChannelPayload;
    type Event;

    fn initial_state(&self) -> Self::State;
    fn initialize_channels(&self, state: &Self::State, out: &mut ChannelEditor<Self>);
    fn propensity(&self, state: &Self::State, key: Self::ChannelKey, payload: &Self::ChannelPayload, time: f64) -> f64;
    fn fire(&self, state: &mut Self::State, key: Self::ChannelKey, payload: &Self::ChannelPayload, rng: &mut impl rand::Rng) -> Self::Event;
    fn refresh_after_event(&self, state: &Self::State, event: &Self::Event, out: &mut ChannelEditor<Self>);
}
```

`ChannelEditor` is the engine-owned mutation surface for channels. It can insert a new channel, update an existing channel, mark a channel inactive, or request recomputation of a channel's propensity. This keeps the model from knowing which scheduler is installed while still letting it express locality.

```rust
pub enum ChannelUpdate<K, P> {
    Upsert { key: K, payload: P },
    Recompute { key: K },
    Remove { key: K },
}
```

This contract also makes dynamic species natural while keeping reaction families fixed: if a mutation creates a genotype that has never existed before, `fire` inserts it into `State` and `refresh_after_event` emits concrete channels involving that new `SpeciesId` under existing families. If a species count falls to zero, the model can remove unary channels and any indexed interactions involving that species.

## Cached Hazards And Clocks

The framework should avoid resampling or recomputing every concrete instance on every event. There are two related but different caches:

- Propensity cache: Direct-style schedulers store the current `a_i(t)` for each active concrete instance and maintain a total or grouped total. After an event, only affected instances are recomputed and their weights are updated.
- Hazard or clock cache: Next-Reaction-style schedulers store each instance's internal integrated hazard progress and next firing threshold. After an event, unaffected instances keep their clocks; affected instances adjust their scheduled time from cached hazard state rather than drawing every clock again.

For a constant propensity instance, a Next Reaction scheduler can store:

```rust
struct Clock {
    integrated_hazard: f64, // T_i
    threshold: f64,         // P_i
    propensity: f64,        // a_i
    scheduled_time: f64,
}
```

Between events at `t_prev` and `t`, the scheduler advances `T_i += a_i * (t - t_prev)` for updated or fired channels. The next firing time is `t + (P_i - T_i) / a_i` when `a_i > 0`. When a channel fires, set `P_i += Exp(1)` and reschedule it. When a channel's propensity changes, keep `T_i` and `P_i`, update `a_i`, and recompute `scheduled_time`. This is the cleanest way to cache hazard without biasing the process. For pure piecewise-constant hazards, the equivalent Gibson-Bruck residual update is `t + (old_a / new_a) * (old_scheduled_time - t)` for non-fired channels whose propensity changes from `old_a` to `new_a`.

This suggests the scheduler API should expose channel updates as changes to cached concrete instances, not as a request to rebuild all clocks for a reaction family. For fixed reaction families, family-separated schedulers should keep updates in `(family, local_channel)` form instead of flattening them through one global channel id and mapping them back on every event. Hot static schedulers should receive dependency updates through a typed callback sink so the compiler can monomorphize the update writer; a caller-selected inline capacity can keep known small dependency fanout off the heap while reusable spill storage handles larger events.

## Scheduler Interface

Schedulers should be generic over active channels and receive only explicit updates. A sketch:

```rust
pub trait SsaScheduler<M: EvolvingModel> {
    fn initialize(&mut self, model: &M, state: &M::State, channels: impl IntoIterator<Item = ChannelUpdate<M::ChannelKey, M::ChannelPayload>>);
    fn next_event(&mut self, rng: &mut impl rand::Rng, model: &M, state: &M::State, time: f64) -> Option<ScheduledEvent<M::ChannelKey>>;
    fn apply_updates(&mut self, model: &M, state: &M::State, time: f64, updates: impl IntoIterator<Item = ChannelUpdate<M::ChannelKey, M::ChannelPayload>>);
}
```

The outer engine is then small and stable: initialize state and channels, ask the scheduler for the next event, advance time, call `model.fire`, ask the model for local channel updates, and feed those updates back to the scheduler. Observers and termination conditions can be orthogonal traits.

## Gillespie Methods To Support

### Direct Method

Direct Method is the best baseline API because it is easy to validate and works well when the active concrete instance count is moderate. The scheduler stores cached propensities in an indexed weighted sampler such as a Fenwick tree or segment tree. Each step samples `dt ~ Exp(total_propensity)` and then samples one concrete instance by cumulative propensity. Dynamic updates are `insert`, `remove`, and `set_weight`, each `O(log M)`. This should be the first implementation because it gives simple correctness tests against fixed small systems and `pheno-geno`-style models. It caches propensities, but it does not cache per-instance exponential clocks.

### First Reaction Method

First Reaction Method samples one waiting time per active channel and fires the minimum. A naive implementation is `O(M)` per event and is mainly useful as a reference implementation or for very small channel sets. It is also the conceptual bridge to the current `pheno-geno` design, where each reaction has a custom `tau`. For this framework, it should probably be a testing scheduler or a specialized scheduler for nonhomogeneous hazards, not the main large-system path.

### Next Reaction Method

Next Reaction Method is the most important exact method for large sparse evolving systems when per-instance hazard caching matters. It stores one scheduled firing time per active concrete instance in an indexed min-heap and uses a dependency graph to update only affected instances after an event. Unaffected instances keep their scheduled times. For changing propensities, use the Gibson-Bruck residual update or Anderson internal-time formulation so rate changes do not bias the process. Dynamic instances are natural: inserted instances get a fresh internal threshold, removed instances are deleted from the heap, and updated instances adjust their next time from cached hazard state.

### Composition-Rejection Or Grouped Direct Method

For hundreds of thousands of active channels, a grouped sampler may beat a Fenwick tree when propensities have structure. Channels can be grouped by rule, species class, or propensity magnitude. The top level samples a group; the second level samples inside the group. This is especially useful if many channels share a formula such as `birth_rate(species) * count(species)`. This should be a later scheduler because it needs benchmarks and a good grouping API.

### Partial-Propensity Methods

Partial-propensity methods can reduce cost for reaction networks with many bimolecular mass-action channels by factorizing propensities. They are powerful but less general. The framework can support them as a specialized scheduler/model adapter once the base `ReactionRule` contract can expose reactant factors and dependency neighborhoods explicitly.

### Tau-Leaping And Hybrid Methods

Tau-leaping is approximate and changes the model contract because multiple firings happen in one step. It should not be mixed into the first exact SSA engine. A later hybrid layer can reuse the same `PopulationState` and rule definitions but needs separate validity checks, nonnegative count handling, and observer semantics.

## State Representation For Many Species

A good default state layout is a sparse registry plus dense active arrays:

- `Vec<SpeciesRecord<S>>` stores metadata and count by `SpeciesId`.
- `Vec<SpeciesId>` stores active species with nonzero count.
- `HashMap<Species, SpeciesId>` or model-provided interner maps newly generated species values to ids.
- Optional per-rule indices maintain neighborhoods, such as phenotype buckets, genotype Hamming-neighborhood candidates, or spatial bins.

The framework should not require storing one individual per cell. For very large populations and many species, counts are the primitive state. If lineage tracking is needed, it can be a model-specific marker layer, borrowing ideas from `evo-marker`, but it should not be required by the SSA engine. A model can choose individual storage only when events need individual-level sampling.

## Dependency Updates

The hardest abstraction is not firing a reaction; it is knowing what changed. The model should return a compact event delta:

```rust
pub struct StateDelta {
    pub count_changes: Vec<(SpeciesId, i64)>,
    pub new_species: Vec<SpeciesId>,
    pub removed_species: Vec<SpeciesId>,
    pub environment_changed: bool,
}
```

Rules then translate the delta into channel updates. For unary reactions, only channels for changed species need updates. For mutation, the birth channel of the parent and unary channels of the child may change. For pair interactions, the rule needs an index that can enumerate affected partners without scanning all species unless the model genuinely has all-to-all interactions. If a model has all-to-all competition among 100,000 species, the framework should make that cost visible rather than hiding it behind a generic fallback.

## Time-Dependent Hazards

The `pheno-geno` inhibition reaction is an important warning: some hazards are functions of time, not just state. Exact support can be split into tiers:

- Tier 1: piecewise-constant propensities between events. This supports Direct Method and most evolutionary birth/death/mutation models.
- Tier 2: exogenous scheduled environment changes. The engine treats environment changes as events that update affected propensities, preserving exactness for piecewise-constant schedules.
- Tier 3: channel-specific nonhomogeneous hazards with integrated hazard inversion. This matches `pheno-geno::Inhibition::tau` and fits First Reaction or modified Next Reaction schedulers.

The first production framework should implement Tier 1 and design the clock traits so Tier 3 can be added without rewriting model rules.

## Integration With Existing Crates

`ssa-workflow` should stay at the workflow/cache/reproducibility layer. An SSA simulation built with the new framework can be wrapped in `StochasticTask`, using stable stream names such as `clock`, `channel`, and `mutation`. This preserves reproducible sweeps and caching without forcing the SSA engine to know about cache storage.

`frequency` is useful for analysis outputs and for building initial sparse species counts from large samples. It should not be part of the scheduler core.

`evo-marker` is useful for optional lineage metadata. The new state layer should allow marker-bearing species or individuals, but the scheduler should only need counts, channel keys, and propensities.

## Implemented Crate Layout

```text
crates/evo-ssa/
  src/lib.rs
  src/model.rs              # EvolvingModel, ChannelUpdate, ChannelEditor
  src/engine.rs             # Dynamic and shared static-family simulation loops
  src/random.rs             # Independent clock, selection, and event RNG streams
  src/scheduler/mod.rs      # Scheduler contracts
  src/scheduler/family.rs   # Arbitrary-length static family lists and typed ids
  src/scheduler/direct.rs   # Dynamic Direct and family-separated Direct
  src/scheduler/nrm.rs      # Generic and statically dispatched family NRM
  src/scheduler/update.rs   # Typed dependency-update sinks
  src/scheduler/weighted.rs # Cached segment-tree sampler
  src/scheduler/nrm_clock.rs
```

Keep `evo-ssa` independent of `ssa-workflow`. Then add examples or integration tests showing how to wrap an `evo-ssa` simulation as an `ssa-workflow::StochasticTask`.

## Implementation Status and Next Steps

The current crate implements cached dynamic Direct, two-level family Direct, generic family NRM, and statically dispatched family NRM. It includes dynamic species creation, arbitrary-length typed family lists, fail-closed dependency updates, independent random streams, analytical distribution tests, and handwritten benchmark baselines.

The next validation milestones are a randomized internal-time oracle for NRM, event-throughput benchmarks at 1,000 to 100,000 active species, and explicit measurements across dependency fanout. A nonhomogeneous clock API and approximate tau-leaping remain separate future layers.

The prototype should avoid pairwise all-to-all interactions at first. Unary birth/death/mutation is enough to validate dynamic species creation, active channel bookkeeping, and scheduler independence. Pair interactions should be the second milestone because they force the dependency-index design to become explicit.

## Main Risks

- If the model API asks rules to enumerate too much, large systems will accidentally become `O(S)` per event. The update contract must force locality.
- If the scheduler owns too much model semantics, adding new evolutionary models will require scheduler changes. Channel payloads and rule-owned refresh logic avoid that.
- If time-dependent hazards are baked into the initial Direct Method API, the framework will become complicated before the constant-propensity path is reliable. Keep the exact constant-propensity engine small, and leave an explicit clock extension point.
- If species ids are reused too aggressively, cached channel keys and heap entries can become invalid. Prefer generational ids or never reuse ids during one simulation.

## Recommendation

Yes, this is a good fit for a framework, but the framework should be a dynamic reaction-channel engine rather than a model hierarchy. The first useful version should implement exact constant-propensity SSA with dynamic channel maintenance, not every Gillespie variant at once. Once Direct Method and the model/channel/update contract are solid, Next Reaction Method becomes the main scalability path for the evolutionary setting.
