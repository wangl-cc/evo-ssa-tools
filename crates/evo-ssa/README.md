# evo-ssa

`evo-ssa` implements exact stochastic simulation algorithms for evolutionary systems with piecewise-constant propensities. A model has a fixed set of reaction families, while every family may contain a large and changing number of concrete channels as species appear, disappear, or reuse dense scheduler slots.

## Model Surfaces

- `EvolvingModel` supports arbitrary hashable channel keys and explicit `Upsert`, `Recompute`, and `Remove` updates. `DirectScheduler` maps those keys onto reusable dense slots.
- `StaticFamilyModel` and `StaticReactionFamily` provide the performance API. `reaction_families!` declares named, bundle-branded family ids, and `family_list!` constructs an arbitrary-length statically dispatched family list.

Static reactions return a compact model event. The model then translates that event into dependency updates through `ChannelRecomputeSink`; schedulers never infer biological dependencies. Invalid recomputes fail with a structured error instead of leaving a stale propensity or clock in the cache.

## Schedulers

- `FamilyDirect` caches one propensity tree per family and samples in two levels.
- `FamilyNrm` caches one indexed clock heap per family in a `Vec`-backed readable implementation.
- `StaticFamilyNrm<M, Families, INLINE_CAP>` stores the same NRM clock structure recursively in the family type. The first `INLINE_CAP` dependency updates stay inline and larger fanout uses reusable heap spill space.

All three family schedulers use the same monomorphized `FamilySimulation` lifecycle and return model events from `step`. New suffix channels are initialized incrementally, unaffected NRM clocks remain cached, and `reinitialize` gives a reused slot a fresh NRM clock.

## Random Streams

`SsaRngs` keeps clock, channel-selection, and model-event randomness independent. This prevents scheduler implementation changes from shifting mutation draws while preserving caller control over seeding and stream ownership.

Run the dynamic birth/death example with:

```bash
cargo run -p evo-ssa --example birth_death
```

See [`docs/design.md`](docs/design.md) for ownership boundaries, cache invariants, error semantics, tests, and benchmark scope.
