# Frequency

`frequency` is a small crate for counting how many times each value appears in an iterator. It supports fixed-range counting with arrays, histogram-style binning for continuous values, and general-purpose counting with hash maps, plus weighted and parallel variants of the same APIs.

## Choosing An Approach

Most users only need to answer one question before picking an API: can your values be mapped into a small dense range such as `0..=255` or `0..=4096`?

### Bounded Frequency

Use bounded counting when your values fit into a known range and that range is not too sparse. This is usually the fastest option because it counts directly into a `Vec` instead of doing hash lookups.

```rust
use frequency::prelude::*;

let dice_rolls: Vec<usize> = vec![1, 6, 3, 4, 6, 6];
let frequencies: Vec<usize> = dice_rolls
    .iter()
    .map(|&x| x - 1)
    .into_bounded_iter(5)
    .freq();
assert_eq!(frequencies, vec![1, 0, 1, 1, 0, 3]);
```

Use this path when you can supply an upper bound and when converting each item into a `usize` is straightforward.

### Hash Frequency

Use hash counting when the value space is large, sparse, or not naturally indexable. This is the more flexible option and works well for strings, enums, and custom key types.

```rust
use frequency::prelude::*;
use std::collections::HashMap;

let fruits = vec!["apple", "banana", "apple", "orange", "banana"];
let frequencies: HashMap<&str, usize> = fruits.into_hash_iter().freq();
assert_eq!(frequencies["apple"], 2);
assert_eq!(frequencies["banana"], 2);
assert_eq!(frequencies["orange"], 1);
```

The examples use `std::collections::HashMap` for simplicity, but the standard library default hasher is usually not a good fit for this crate's performance-oriented use cases. For integer keys, prefer `nohash_hasher::IntMap`. For general keys, prefer a faster non-cryptographic hasher such as `rapidhash`.

### Binned Frequency

Use binned counting when your input values are continuous but you still want a fixed-size `Vec` result. This is useful for histogram-style summaries over a known range such as distances, probabilities, or normalized scores.

```rust
use frequency::prelude::*;

let distances = vec![0.02, 0.10, 0.31, 0.50, 0.74, 0.99];
let frequencies: Vec<usize> = distances.into_binned_iter(4, 0.0, 1.0).freq();
assert_eq!(frequencies, vec![2, 1, 2, 1]);
```

This path divides `[min, max]` into equal-width bins. Values below `min` are placed into the first bin, and values above `max` are placed into the last bin.
`n_bins` must be greater than zero, and the range must satisfy `max > min`.

## Weighted Frequency

The default `freq()` API adds `1` for every item. When each item should contribute a different amount, use `weighted_freq()` instead.

```rust
use frequency::prelude::*;

let weighted_data: Vec<(usize, f64)> = vec![(0, 0.5), (2, 1.5), (1, 2.0), (2, 1.0)];
let frequencies = weighted_data
    .iter()
    .cloned()
    .into_bounded_iter(2)
    .weighted_freq();
assert_eq!(frequencies, vec![0.5, 2.0, 2.5]);
```

The weighted APIs follow the same split as the normal ones: use bounded iterators when keys fit a dense range, binned iterators when your inputs are continuous values over a known range, and hash iterators otherwise.

## Parallel Iterator

If you already use `rayon`, or your input is large enough to benefit from parallelism, enable the `parallel` feature. In practice you will usually import both `rayon::prelude::*` for parallel iteration and `frequency::prelude::*` for the counting adapters.

Parallel counting has two independent dimensions:

1. Counting strategy: `bounded` or `hash`
2. Input shape: indexed input such as `&[T]` / `Vec<T>`, or a more general `ParallelIterator`

That gives four parallel adapters:

1. Indexed + bounded: use `into_bounded_indexed_par_iter()` or `into_unchecked_bounded_indexed_par_iter()`
2. General + bounded: use `into_bounded_par_iter()` or `into_unchecked_bounded_par_iter()`
3. Indexed + hash: use `into_hash_indexed_par_iter()`
4. General + hash: use `into_hash_per_iter()`

The examples below focus on the bounded cases, because that is where the indexed and general variants differ the most in performance behavior.

### Bounded Frequency for Indexed Inputs

If your input is indexed, such as a slice or `Vec`, and your values are bounded, start with `into_bounded_indexed_par_iter()`. This is the best default for dense bounded counting on parallel data.

```rust
# #[cfg(feature = "parallel")]
# {
use frequency::prelude::*;
use rayon::prelude::*;

let rolls: Vec<u16> = [0u16, 1, 2, 1, 2, 2, 3, 1].repeat(512);
let frequencies: Vec<usize> = rolls
    .par_iter()
    .copied()
    .into_bounded_indexed_par_iter(3)
    .freq();
assert_eq!(frequencies, vec![512, 1536, 1536, 512]);
# }
```

This path applies a coarser default split strategy before counting. That matters because bounded frequency counting allocates one local frequency table per Rayon job, and very small jobs can spend more time on allocation and reduction than on counting.

### Bounded Frequency for General Parallel Iterators

Use `into_bounded_par_iter()` or `into_unchecked_bounded_par_iter()` when your values are bounded but you are not using the indexed bounded convenience path.

```rust
# #[cfg(feature = "parallel")]
# {
use frequency::prelude::*;
use rayon::prelude::*;

let rolls: Vec<u16> = [0u16, 1, 2, 1, 2, 2, 3, 1].repeat(512);
let frequencies: Vec<usize> = rolls.par_iter().copied().into_bounded_par_iter(3).freq();
assert_eq!(frequencies, vec![512, 1536, 1536, 512]);
# }
```

If you can estimate a reasonable minimum chunk size for your workload, you can guide Rayon manually with `with_min_len()` before converting into a bounded adapter:

```rust
# #[cfg(feature = "parallel")]
# {
use frequency::prelude::*;
use rayon::prelude::*;

let rolls: Vec<u16> = [0u16, 1, 2, 1, 2, 2, 3, 1].repeat(512);
let frequencies: Vec<usize> = rolls
    .par_iter()
    .copied()
    .with_min_len(256)
    .into_bounded_par_iter(3)
    .freq();
assert_eq!(frequencies, vec![512, 1536, 1536, 512]);
# }
```

This can still reduce the number of tiny jobs and recover much of the same benefit when you have a good chunk-size hint but cannot use the indexed bounded convenience constructor.

For hash-based parallel counting, indexed inputs can use `into_hash_indexed_par_iter()` while general parallel iterators use `into_hash_per_iter()`. The indexed form applies the same coarser split strategy as the bounded indexed path, which can reduce repeated map allocation and merge overhead on dense indexed workloads. The same hasher advice as the sequential case still applies there: prefer `nohash_hasher::IntMap` for integer keys and a faster hasher such as `rapidhash` for general keys when throughput matters.

**NOTE**: Due to the overhead of parallelization, this approach may not always be faster than the sequential version. Benchmarking is recommended to determine the optimal approach for your specific use case.
