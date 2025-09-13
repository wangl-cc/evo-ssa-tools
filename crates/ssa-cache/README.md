# SSA Cache

A high-performance caching library for expensive computations with support for multiple storage backends and parallel execution.

## Overview

`ssa-cache` provides a flexible caching system designed to cache the results of expensive computations. It supports both deterministic (pure) and stochastic computations, with built-in serialization and multiple storage backends.

## Compute Types

The library provides three main computation types, each designed for different use cases:

### PureCompute

For deterministic computations where the same input always produces the same output.

```rust
let compute = PureCompute::new(|x: i32| x * x);
```

### StochasticCompute  

For computations involving randomness that need to be reproducible.

```rust
let mut compute = StochasticCompute::<i32, i32, _, SmallRng>::new(
    |rng, x| x + rng.gen_range(0..100)
);
```

### ExpAnalysis

For two-stage computations with expensive experiment and analysis phases.

**Use cases**: Scientific simulations with post-processing;
**Caching strategy**: Two-level caching for intermediate and final results;
**Parallelization**: Parallel execution with intelligent cache reuse;
**Efficiency**: Avoids recomputing expensive intermediate results;

```rust
let exp_analysis = ExpAnalysis::<Input, Intermediate, Output, _, _, SmallRng>::new(
    |rng, input| expensive_experiment(rng, input),    // Stage 1: Experiment  
    |intermediate| expensive_analysis(intermediate),   // Stage 2: Analysis
);
```

## Cache Backends

The library supports multiple storage backends through the `CacheStore` trait:

### HashMapStore

An in-memory storage backend, suitable for development, testing, and short-lived computations. All data is stored in memory and lost when the program exits.

```rust
use ssa_cache::HashMapStore;
use std::collections::hash_map::RandomState;

let cache = HashMapStore::<RandomState>::default();
```

### Fjall Database

A persistent embedded key-value store, suitable for long-running computations and cross-session caching.

```rust
let db = fjall::Config::new("/path/to/cache").open()?;
let cache = db.open_partition("computations", Default::default())?;
```

### No-Op Cache

`()` can be used as a no-op cache backend for testing, benchmarking without cache effects.

### Custom Backends

Implement `CacheStore` trait for custom storage solutions:

```rust
impl CacheStore for MyCustomStore {
    fn fetch<T: Cacheable>(&self, sig: &[u8], buffer: &mut T::Buffer) -> Result<Option<T>> {
        // Custom fetch logic
    }
    
    fn store<T: Cacheable>(&self, sig: &[u8], buffer: &mut T::Buffer, value: &T) -> Result<()> {
        // Custom store logic  
    }
}
```

## Parallel Execution and Interruption

```rust
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::Duration;

let interrupted = Arc::new(AtomicBool::new(false));
let interrupted_clone = interrupted.clone();

// Start computation in background
let handle = thread::spawn(move || {
    compute.execute_many(
        &cache,
        interrupted,
        (0..1000).into_par_iter(),
    )?.collect::<Result<Vec<_>>>()
});

// Cancel after some time
thread::sleep(Duration::from_secs(1));
interrupted_clone.store(true, Ordering::Relaxed);

let result = handle.join().unwrap();
// Will return Err(Error::Interrupted) if cancelled
```

## Error Handling

The crate defines a comprehensive error type:

```rust
use ssa_cache::{Error, Result};

match result {
    Ok(value) => println!("Success: {:?}", value),
    Err(Error::Codec(e)) => println!("Serialization error: {}", e),
    Err(Error::Db(e)) => println!("Database error: {}", e),
    Err(Error::Interrupted) => println!("Computation was cancelled"),
    Err(Error::CacheOutofIndex { total, want }) => {
        println!("Invalid cache index {} (max {})", want, total)
    }
}
```

## License

This crate is part of the `evo-ssa-tools` project and shares the same license.
