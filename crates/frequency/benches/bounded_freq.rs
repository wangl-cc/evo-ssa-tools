mod baseline;
mod common;

use std::{hint::black_box, time::Duration};

use baseline::*;
use common::{BenchData, MAX_VALUE, SIZES};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use frequency::prelude::*;

fn bench_bounded_freq<T>(c: &mut Criterion)
where
    T: BenchData + ToUsize + Eq + std::hash::Hash + nohash_hasher::IsEnabled + Copy + Send + Sync,
{
    let mut group = c.benchmark_group(format!("bounded/{}", std::any::type_name::<T>()));

    for &size in SIZES {
        let data = T::data(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(BenchmarkId::new("baseline", size), |b| {
            b.iter(|| {
                black_box(bounded_freq::<T, usize>(
                    black_box(data),
                    black_box(MAX_VALUE),
                ))
            })
        });

        group.bench_function(BenchmarkId::new("baseline_unchecked", size), |b| {
            b.iter(|| unsafe {
                black_box(bounded_freq_unchecked::<T, usize>(
                    black_box(data),
                    black_box(MAX_VALUE),
                ))
            })
        });

        group.bench_function(BenchmarkId::new("bounded_iter", size), |b| {
            b.iter(|| {
                let freq: Vec<usize> = black_box(data)
                    .iter()
                    .copied()
                    .into_bounded_iter(black_box(MAX_VALUE))
                    .freq();
                black_box(freq)
            })
        });

        group.bench_function(BenchmarkId::new("bounded_iter_unchecked", size), |b| {
            b.iter(|| unsafe {
                let freq: Vec<usize> = black_box(data)
                    .iter()
                    .copied()
                    .into_unchecked_bounded_iter(black_box(MAX_VALUE))
                    .freq();
                black_box(freq)
            })
        });

        #[cfg(feature = "parallel")]
        {
            use baseline::parallel::*;
            #[cfg(feature = "parallel")]
            use rayon::prelude::*;

            let n_threads =
                std::thread::available_parallelism().expect("Failed to get available parallelism");

            group.bench_function(BenchmarkId::new("par_baseline_par", size), |b| {
                b.iter(|| {
                    black_box(par_bounded_freq::<T, usize>(
                        black_box(data),
                        black_box(MAX_VALUE),
                        n_threads,
                    ))
                })
            });

            group.bench_function(BenchmarkId::new("par_baseline_unchecked", size), |b| {
                b.iter(|| unsafe {
                    black_box(par_bounded_freq_unchecked::<T, usize>(
                        black_box(data),
                        black_box(MAX_VALUE),
                        n_threads,
                    ))
                })
            });

            group.bench_function(BenchmarkId::new("par_bounded_iter", size), |b| {
                b.iter(|| {
                    let freq: Vec<usize> = black_box(data)
                        .par_iter()
                        .copied()
                        .into_bounded_par_iter(black_box(MAX_VALUE))
                        .freq();
                    black_box(freq)
                })
            });

            group.bench_function(BenchmarkId::new("par_bounded_iter_unchecked", size), |b| {
                b.iter(|| unsafe {
                    let freq: Vec<usize> = black_box(data)
                        .par_iter()
                        .copied()
                        .into_unchecked_bounded_par_iter(black_box(MAX_VALUE))
                        .freq();
                    black_box(freq)
                })
            });

            group.bench_function(BenchmarkId::new("par_bounded_indexed_iter", size), |b| {
                b.iter(|| {
                    let freq: Vec<usize> = black_box(data)
                        .par_iter()
                        .copied()
                        .into_bounded_indexed_par_iter(black_box(MAX_VALUE))
                        .freq();
                    black_box(freq)
                })
            });

            group.bench_function(
                BenchmarkId::new("par_bounded_indexed_iter_unchecked", size),
                |b| {
                    b.iter(|| unsafe {
                        let freq: Vec<usize> = black_box(data)
                            .par_iter()
                            .copied()
                            .into_unchecked_bounded_indexed_par_iter(black_box(MAX_VALUE))
                            .freq();
                        black_box(freq)
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_bounded_freq::<u16>, bench_bounded_freq::<u32>,
}
criterion_main!(benches);
