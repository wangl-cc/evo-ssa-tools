# evo-ssa

`evo-ssa` is an experimental stochastic simulation engine for evolutionary systems with fixed reaction families and a dynamic number of concrete reaction channels.

The first implementation includes cached Direct Method schedulers, a family-separated Next Reaction Method scheduler, and a statically dispatched tuple-family NRM fast path. Models can initialize any number of concrete channels, then update only the affected channels after each event. Static family schedulers report dependencies through a typed update sink, so hot paths such as `StaticFamilyNrm<M, Families, UPDATE_CAP>` can write `(family, local_channel)` updates into a fixed buffer without a global channel-id round trip.

See [`docs/design.md`](docs/design.md) for the current architecture, scheduler contracts, and benchmark interpretation.
