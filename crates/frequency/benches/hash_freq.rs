mod common;

use std::{hint::black_box, time::Duration};

use common::{MAX_VALUE, SIZES, canonical_u32_data, str_data};
use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use frequency::prelude::*;
use nohash_hasher::IntMap;
use rapidhash::RapidHashMap;

fn bench_hash_freq_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash/u32");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in SIZES {
        let data = canonical_u32_data(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(BenchmarkId::new("hash_iter_rapid", size), |b| {
            b.iter(|| {
                let freq: RapidHashMap<u32, usize> =
                    black_box(data).iter().copied().into_hash_iter().freq();
                debug_assert!(freq.len() <= MAX_VALUE + 1);
                black_box(freq)
            })
        });

        group.bench_function(BenchmarkId::new("hash_iter_nohash", size), |b| {
            b.iter(|| {
                let freq: IntMap<u32, usize> =
                    black_box(data).iter().copied().into_hash_iter().freq();
                debug_assert!(freq.len() <= MAX_VALUE + 1);
                black_box(freq)
            })
        });

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            group.bench_function(BenchmarkId::new("par_hash_iter_rapid", size), |b| {
                b.iter(|| {
                    let freq: RapidHashMap<u32, usize> = black_box(data)
                        .par_iter()
                        .copied()
                        .into_hash_per_iter()
                        .freq();
                    debug_assert!(freq.len() <= MAX_VALUE + 1);
                    black_box(freq)
                })
            });

            group.bench_function(BenchmarkId::new("par_hash_indexed_iter_rapid", size), |b| {
                b.iter(|| {
                    let freq: RapidHashMap<u32, usize> = black_box(data)
                        .par_iter()
                        .copied()
                        .into_hash_indexed_par_iter()
                        .freq();
                    debug_assert!(freq.len() <= MAX_VALUE + 1);
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
                    debug_assert!(freq.len() <= MAX_VALUE + 1);
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
                        debug_assert!(freq.len() <= MAX_VALUE + 1);
                        black_box(freq)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_hash_freq_str(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash/str");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in SIZES {
        let data = str_data(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(BenchmarkId::new("hash_iter_rapid", size), |b| {
            b.iter(|| {
                let freq: RapidHashMap<&'static str, usize> =
                    black_box(data).iter().copied().into_hash_iter().freq();
                black_box(freq)
            })
        });

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            group.bench_function(BenchmarkId::new("par_hash_iter_rapid", size), |b| {
                b.iter(|| {
                    let freq: RapidHashMap<&'static str, usize> = black_box(data)
                        .par_iter()
                        .copied()
                        .into_hash_per_iter()
                        .freq();
                    black_box(freq)
                })
            });

            group.bench_function(BenchmarkId::new("par_hash_indexed_iter_rapid", size), |b| {
                b.iter(|| {
                    let freq: RapidHashMap<&'static str, usize> = black_box(data)
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
