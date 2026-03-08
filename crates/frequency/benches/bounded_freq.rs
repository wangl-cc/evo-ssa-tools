mod baseline;

use std::{collections::HashMap, hint::black_box, sync::LazyLock, time::Duration};

use baseline::*;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use frequency::prelude::*;
use rand::{
    distr::{Uniform, uniform::SampleUniform},
    prelude::*,
    rngs::SmallRng,
};

const SIZES: &[usize] = &[1 << 16, 1 << 20, 1 << 24];
const MAX_VALUE: usize = 4096;

trait Sampleable: SampleUniform + Count {
    fn from_usize(value: usize) -> Self;
}

trait BenchData: Sampleable + Copy + Send + Sync + 'static {
    fn from_canonical(value: u32) -> Self;
    fn data(size: usize) -> &'static [Self];
}

macro_rules! impl_sampleable {
    ($($ty:ty),*) => {
        $(
            impl Sampleable for $ty {
                fn from_usize(value: usize) -> Self {
                    value as $ty
                }
            }
        )*
    };
}

impl_sampleable!(u8, u16, u32, u64, usize);

fn canonical_data(size: usize) -> &'static [u32] {
    static DATA: LazyLock<HashMap<usize, Vec<u32>>> = LazyLock::new(|| {
        SIZES
            .iter()
            .copied()
            .map(|size| (size, gen_data::<u32>(size)))
            .collect()
    });

    DATA.get(&size).unwrap().as_slice()
}

impl BenchData for u16 {
    fn from_canonical(value: u32) -> Self {
        value as u16
    }

    fn data(size: usize) -> &'static [Self] {
        static DATA: LazyLock<HashMap<usize, Vec<u16>>> = LazyLock::new(|| {
            SIZES
                .iter()
                .copied()
                .map(|size| {
                    (
                        size,
                        canonical_data(size)
                            .iter()
                            .copied()
                            .map(u16::from_canonical)
                            .collect(),
                    )
                })
                .collect()
        });

        DATA.get(&size).unwrap().as_slice()
    }
}

impl BenchData for u32 {
    fn from_canonical(value: u32) -> Self {
        value
    }

    fn data(size: usize) -> &'static [Self] {
        canonical_data(size)
    }
}

fn gen_data<T: Sampleable>(size: usize) -> Vec<T> {
    Uniform::new_inclusive(T::ZERO, T::from_usize(MAX_VALUE))
        .unwrap()
        .sample_iter(&mut SmallRng::seed_from_u64(42))
        .take(size)
        .collect()
}

fn bench_bounded_freq<T>(c: &mut Criterion)
where
    T: BenchData + ToUsize + Eq + std::hash::Hash + nohash_hasher::IsEnabled + Copy + Send + Sync,
{
    let mut group = c.benchmark_group(std::any::type_name::<T>());

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
