# Evo-Marker

A Rust crate for evolutionary marker tracking, which allows for the efficient tracking of cell lineages and other genetic markers in evolutionary stochastic simulations.

## Overview

When running stochastic simulations, we often need to track cell lineages or use genetic markers to understand the evolutionary history of a population. This crate provides a `Marker` trait that represents a general tracking marker for these purposes.

## Implemented Markers

- `NoMarker`: A marker that doesn't track any information and is optimized away by the compiler, which is useful as a placeholder or for simulations that don't require marker tracking.
- `LineageNode`: A marker that tracks cell lineages, allowing reconstruction of evolutionary history and calculation of statistics such as mutation burden distribution and site frequency spectrum.

## Features

- `bitcode`: Enables serialization and deserialization for markers using the `bitcode` crate.
