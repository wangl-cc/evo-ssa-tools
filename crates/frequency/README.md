# frequency

A crate for counting occurrences (frequency) of unique elements in iterators.

## Calculation Approach

To efficiently calculate frequencies, there are two main approaches:

1. *Bounded Frequency Calculation*: Utilizes a fixed-size array to store counts for each unique element within a known range. This method is useful when the range is small or dense distributed.

Example:

```rust
use frequency::prelude::*;

let dice_rolls: Vec<usize> = vec![1, 6, 3, 4, 6, 6];
// Shift values to make them start from 0 and upper bound 5
let frequencies: Vec<usize> = dice_rolls
    .iter()
    .map(|&x| x - 1)
    .into_bounded_iter(5)
    .freq();
assert_eq!(frequencies, vec![1, 0, 1, 1, 0, 3]);
```

2. *Hash Frequency Calculation*: Uses a hash map to store counts for each unique element. This method is suitable for larger ranges or when elements cannot be easily converted to indices. Choose a high performance hasher for better performance.

```rust
use frequency::prelude::*;
use std::collections::HashMap;

let fruits = vec!["apple", "banana", "apple", "orange", "banana"];
let frequencies: HashMap<&str, usize> = fruits.into_hash_iter().freq();
assert_eq!(frequencies["apple"], 2);
assert_eq!(frequencies["banana"], 2);
assert_eq!(frequencies["orange"], 1);
```

## Frequency Trait

The crate provides two main traits for frequency calculation:

### [`Frequency`]

Normal frequency calculation where items have the same weight.

### [`WeightedFrequency`]

Calculates the frequency where items have different weights.

Example:
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

## Parallel Iterator

When you work with `rayon` parallel iterators, or your dataset is very large. We have a parallel version of implementations that can be used to efficiently calculate frequencies in parallel. To use the parallel version, you need to enable the `parallel` feature and then use replace the `into_*_iter` methods with `into_*_par_iter`.

Example:

```rust
use frequency::prelude::*;
use std::collections::HashMap;

let fruits = vec!["apple", "banana", "apple", "orange", "banana"];
let frequencies: HashMap<&str, usize> = fruits.into_hash_per_iter().freq();
assert_eq!(frequencies["apple"], 2);
assert_eq!(frequencies["banana"], 2);
assert_eq!(frequencies["orange"], 1);
```

**NOTE**: Due to the overhead of parallelization, this approach may not always be faster than the sequential version. Benchmarking is recommended to determine the optimal approach for your specific use case.
