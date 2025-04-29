# Evo-Marker

When running stochastic simulations, we often need to track cell lineages or use genetic markers to understand the evolutionary history of a population. This crate provides a `Marker` trait that represents a general tracking marker for these purposes.

## Implemented Markers

- `NoMarker`: A marker that doesn't track any information and should be optimized away by the compiler, which is useful as a placeholder for simulations that don't require marker tracking.
- `LineageNode`: A marker that tracks cell lineages, allowing reconstruction of evolutionary history and calculation of statistics such as single cell mutation burden distribution (scMBD), site frequency spectrum (SFS), and pairwise genetic distance.

## Features

- `bitcode`: Enables serialization and deserialization for markers using the `bitcode` crate.
