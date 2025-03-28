//! Benchmarking to compare different methods of calculating frequency for elements in a range
//!
//! ## Benchmark Results Summary (Apple M1)
//!
//! - BoundedIterator implementation: Approximately 2% slower than the manual implementation.
//! - Unchecked variants: Approximately 10% faster than their checked counterparts.
//! - BoundedParallelIterator performance characteristics:
//!   - Small datasets (2^16 elements): Significantly slower than serial implementations
//!   - Medium datasets (2^24 elements): Comparable performance to serial implementations
//!   - Large datasets (2^30 elements): Faster than serial implementations
//!
//! The parallel implementation's performance is affected by:
//!
//! 1. Parallel execution overhead.
//! 2. Cache sharing between threads, which may increase cache misses.
//!
//! Note: In real-world scenarios with "cold" data (not cached), parallel implementations
//! may demonstrate better relative performance than shown in these benchmarks due to
//! reduced impact of cache-related bottlenecks.

mod manual;

use criterion::{Criterion, criterion_group, criterion_main};
use frequency::prelude::*;
use manual::*;
use rand::{distr::Uniform, prelude::*, rngs::SmallRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

fn compare_bounded_freq<T, C>(c: &mut Criterion, size: usize, max_value: usize, data: &[T])
where
    T: ToUsize + Sync + Send + Copy,
    C: Count + Send,
{
    let mut group = c.benchmark_group(format!(
        "Compare Methods (T: {}, U: {}, Size: {}, Max Value: {})",
        std::any::type_name::<T>(),
        std::any::type_name::<C>(),
        size,
        max_value,
    ));

    group.bench_with_input("Manual", &(&data, max_value), |b, &(data, max_value)| {
        b.iter(|| bounded_freq::<T, C>(data, max_value))
    });

    group.bench_with_input(
        "Manual (Unchecked)",
        &(&data, max_value),
        |b, &(data, max_value)| {
            b.iter(|| unsafe { unchecked_bounded_freq::<T, C>(data, max_value) })
        },
    );

    group.bench_with_input(
        "BoundedIterator",
        &(&data, max_value),
        |b, &(data, max_value)| {
            b.iter(|| {
                let freq: Vec<C> = data.iter().copied().into_bounded_iter(max_value).freq();
                freq
            })
        },
    );

    group.bench_with_input(
        "BoundedIterator (Unchecked)",
        &(&data, max_value),
        |b, &(data, max_value)| {
            b.iter(|| {
                let freq: Vec<C> = unsafe {
                    data.iter()
                        .copied()
                        .into_unchecked_bounded_iter(max_value)
                        .freq()
                };
                freq
            })
        },
    );

    #[cfg(feature = "parallel")]
    group.bench_with_input(
        "BoundedParallelIterator",
        &(&data, max_value),
        |b, &(&data, max_value)| {
            b.iter(|| {
                let freq: Vec<C> = data
                    .par_iter()
                    .copied()
                    .into_bounded_par_iter(max_value)
                    .freq();
                freq
            })
        },
    );

    #[cfg(feature = "parallel")]
    group.bench_with_input(
        "BoundedParallelIterator (Unchecked)",
        &(&data, max_value),
        |b, &(&data, max_value)| {
            b.iter(|| {
                let freq: Vec<C> = unsafe {
                    data.par_iter()
                        .copied()
                        .into_unchecked_bounded_par_iter(max_value)
                        .freq()
                };
                freq
            })
        },
    );

    group.finish();
}

fn bench_methods(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let max_value = 4096;
    for size in [1 << 16, 1 << 24] {
        let data: Vec<_> = Uniform::new(0, max_value as u16)
            .unwrap()
            .sample_iter(&mut rng)
            .take(size)
            .collect();
        compare_bounded_freq::<_, usize>(c, size, max_value, &data);
    }
}

criterion_group!(benches, bench_methods);

criterion_main!(benches);
