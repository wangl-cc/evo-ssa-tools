use std::{collections::HashMap, hint::black_box, sync::LazyLock, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use frequency::prelude::*;
use nohash_hasher::IntMap;
use rand::{
    distr::{Uniform, uniform::SampleUniform},
    prelude::*,
    rngs::SmallRng,
};

const SIZES: &[usize] = &[1 << 16, 1 << 20, 1 << 24];
const KEY_SPACE: usize = 4096;

trait Sampleable: SampleUniform + Count {
    fn from_usize(value: usize) -> Self;
}

impl Sampleable for u32 {
    fn from_usize(value: usize) -> Self {
        value as Self
    }
}

fn gen_data<T: Sampleable>(size: usize) -> Vec<T> {
    Uniform::new_inclusive(T::ZERO, T::from_usize(KEY_SPACE - 1))
        .unwrap()
        .sample_iter(&mut SmallRng::seed_from_u64(42))
        .take(size)
        .collect()
}

fn canonical_u32_data(size: usize) -> &'static [u32] {
    static DATA: LazyLock<HashMap<usize, Vec<u32>>> = LazyLock::new(|| {
        SIZES
            .iter()
            .copied()
            .map(|size| (size, gen_data::<u32>(size)))
            .collect()
    });

    DATA.get(&size).unwrap().as_slice()
}

fn str_pool() -> &'static [&'static str] {
    static POOL: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
        (0..KEY_SPACE)
            .map(|index| Box::leak(format!("key-{index}").into_boxed_str()) as &'static str)
            .collect()
    });

    POOL.as_slice()
}

fn str_data(size: usize) -> &'static [&'static str] {
    static DATA: LazyLock<HashMap<usize, Vec<&'static str>>> = LazyLock::new(|| {
        SIZES
            .iter()
            .copied()
            .map(|size| {
                (
                    size,
                    canonical_u32_data(size)
                        .iter()
                        .copied()
                        .map(|index| str_pool()[index as usize])
                        .collect(),
                )
            })
            .collect()
    });

    DATA.get(&size).unwrap().as_slice()
}

fn bench_hash_freq_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("u32");

    for &size in SIZES {
        let data = canonical_u32_data(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(BenchmarkId::new("hash_iter_std", size), |b| {
            b.iter(|| {
                let freq: HashMap<u32, usize> =
                    black_box(data).iter().copied().into_hash_iter().freq();
                black_box(freq)
            })
        });

        group.bench_function(BenchmarkId::new("hash_iter_nohash", size), |b| {
            b.iter(|| {
                let freq: IntMap<u32, usize> =
                    black_box(data).iter().copied().into_hash_iter().freq();
                black_box(freq)
            })
        });

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            group.bench_function(BenchmarkId::new("par_hash_iter_std", size), |b| {
                b.iter(|| {
                    let freq: HashMap<u32, usize> = black_box(data)
                        .par_iter()
                        .copied()
                        .into_hash_per_iter()
                        .freq();
                    black_box(freq)
                })
            });

            group.bench_function(BenchmarkId::new("par_hash_indexed_iter_std", size), |b| {
                b.iter(|| {
                    let freq: HashMap<u32, usize> = black_box(data)
                        .par_iter()
                        .copied()
                        .into_hash_indexed_par_iter()
                        .freq();
                    black_box(freq)
                })
            });

            group.bench_function(BenchmarkId::new("par_hash_iter_nohash", size), |b| {
                b.iter(|| {
                    let freq: IntMap<u32, usize> = black_box(data)
                        .par_iter()
                        .copied()
                        .into_hash_per_iter()
                        .freq();
                    black_box(freq)
                })
            });

            group.bench_function(
                BenchmarkId::new("par_hash_indexed_iter_nohash", size),
                |b| {
                    b.iter(|| {
                        let freq: IntMap<u32, usize> = black_box(data)
                            .par_iter()
                            .copied()
                            .into_hash_indexed_par_iter()
                            .freq();
                        black_box(freq)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_hash_freq_str(c: &mut Criterion) {
    let mut group = c.benchmark_group("str");

    for &size in SIZES {
        let data = str_data(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(BenchmarkId::new("hash_iter_std", size), |b| {
            b.iter(|| {
                let freq: HashMap<&'static str, usize> =
                    black_box(data).iter().copied().into_hash_iter().freq();
                black_box(freq)
            })
        });

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            group.bench_function(BenchmarkId::new("par_hash_iter_std", size), |b| {
                b.iter(|| {
                    let freq: HashMap<&'static str, usize> = black_box(data)
                        .par_iter()
                        .copied()
                        .into_hash_per_iter()
                        .freq();
                    black_box(freq)
                })
            });

            group.bench_function(BenchmarkId::new("par_hash_indexed_iter_std", size), |b| {
                b.iter(|| {
                    let freq: HashMap<&'static str, usize> = black_box(data)
                        .par_iter()
                        .copied()
                        .into_hash_indexed_par_iter()
                        .freq();
                    black_box(freq)
                })
            });
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_hash_freq_u32, bench_hash_freq_str,
}
criterion_main!(benches);
