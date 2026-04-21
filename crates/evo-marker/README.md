# Evo-Marker

When running stochastic simulations, we often need to track cell lineages or use genetic markers to understand the evolutionary history of a population. This crate provides a `Marker` trait that represents a general tracking marker for these purposes.

## Implemented Markers

- `NoMarker`: A marker that doesn't track any information and should be optimized away by the compiler, which is useful as a placeholder for simulations that don't require marker tracking.
- `LineageNode`: A marker that tracks cell lineages, allowing reconstruction of evolutionary history and calculation of statistics such as single cell mutation burden distribution (scMBD), site frequency spectrum (SFS), and pairwise genetic distance.
- `EcDna`: A marker that tracks ecDNA copy number, replicates `N` copies to `2N` before division, and then partitions the `2N` copies between two daughters by sampling one daughter from `Binomial(2N, 0.5)` and giving the other daughter the remainder.

## Features

- `bitcode`: Enables serialization and deserialization for markers using the `bitcode` crate.

## Tips

- If your workload spends a lot of time simulating `LineageNode` trees or building `LineageTree`s, enabling `mimalloc` in the final binary or benchmark target may improve performance noticeably.
