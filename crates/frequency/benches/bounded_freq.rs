//! Benchmarking to compare different methods of calculating frequency for elements in a range

mod baseline;

use std::{collections::HashMap, sync::LazyLock};

use baseline::*;
use divan::Bencher;
use frequency::prelude::*;
use rand::{
    distr::{Uniform, uniform::SampleUniform},
    prelude::*,
    rngs::SmallRng,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zerocopy::{FromBytes, Immutable, IntoBytes};

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[1 << 16, 1 << 20, 1 << 24];

trait Sampleable: SampleUniform + Count {
    fn from_usize(value: usize) -> Self;
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

fn prepare_data<T>(size: usize) -> (Vec<T>, usize)
where
    T: SampleUniform + IntoBytes + FromBytes + Immutable + Copy,
{
    const MAX_VALUE: usize = 4096;

    fn cast_vec<A, B>(src: &[A]) -> Vec<B>
    where
        A: IntoBytes + Immutable,
        B: FromBytes + Immutable + Clone,
    {
        let transmute: &[B] = zerocopy::transmute_ref!(src);
        transmute.to_vec()
    }

    fn gen_data<T: Sampleable>(size: usize) -> Vec<T> {
        Uniform::new_inclusive(T::ZERO, T::from_usize(MAX_VALUE))
            .unwrap()
            .sample_iter(&mut SmallRng::seed_from_u64(42))
            .take(size)
            .collect()
    }

    // Use precomputed data to avoid generating data for each benchmark run
    static BENCH_DATA: LazyLock<HashMap<(usize, &'static str), Vec<u8>>> = LazyLock::new(|| {
        let mut data = HashMap::new();

        for &size in SIZES {
            // data.insert((size, "u8"), cast_vec(&gen_data::<u8>(size)));
            data.insert((size, "u16"), cast_vec(&gen_data::<u16>(size)));
            data.insert((size, "u32"), cast_vec(&gen_data::<u32>(size)));
        }

        data
    });

    (
        cast_vec(BENCH_DATA.get(&(size, std::any::type_name::<T>())).unwrap()),
        MAX_VALUE,
    )
}

#[divan::bench_group(sample_count = 100, sample_size = 10)]
mod compare_methods {

    use super::*;

    macro_rules! bench_bounded_freq {
        ($name:ident, $func:expr) => {
            #[divan::bench(args = SIZES, types = [u16, u32])]
            fn $name<T>(bencher: Bencher, size: usize)
            where
                T: SampleUniform
                    + ToUsize
                    + Eq
                    + std::hash::Hash
                    + nohash_hasher::IsEnabled
                    + IntoBytes
                    + FromBytes
                    + Immutable
                    + Copy
                    + Send
                    + Sync,
            {
                bencher
                    .with_inputs(|| prepare_data::<T>(size))
                    .bench_local_refs($func)
            }
        };
    }

    bench_bounded_freq!(baseline, |(data, max_value)| bounded_freq::<T, usize>(
        data, *max_value
    ));
    bench_bounded_freq!(baseline_unchecked, |(data, max_value)| unsafe {
        bounded_freq_unchecked::<T, usize>(data, *max_value)
    });

    bench_bounded_freq!(par_baseline_par, |(data, max_value)| {
        par_bounded_freq::<T, usize>(
            data,
            *max_value,
            std::thread::available_parallelism().expect("Failed to get available parallelism"),
        )
    });
    bench_bounded_freq!(par_baseline_unchecked, |(data, max_value)| unsafe {
        par_bounded_freq_unchecked::<T, usize>(
            data,
            *max_value,
            std::thread::available_parallelism().expect("Failed to get available parallelism"),
        )
    });

    bench_bounded_freq!(bounded_iter, |(data, max_value)| {
        let freq: Vec<usize> = data.iter().copied().into_bounded_iter(*max_value).freq();
        freq
    });

    bench_bounded_freq!(bounded_iter_unchecked, |(data, max_value)| unsafe {
        let freq: Vec<usize> = data
            .iter()
            .copied()
            .into_unchecked_bounded_iter(*max_value)
            .freq();
        freq
    });

    // 10x slower than other implementations, so commented out to decrease benchmarking time.
    // bench_bounded_freq!(hash_iter, |(data, _)| {
    //     let freq: nohash_hasher::IntMap<T, usize> = data.iter().copied().into_hash_iter().freq();
    //     freq
    // });

    #[cfg(feature = "parallel")]
    bench_bounded_freq!(par_bounded_iter, |(data, max_value)| {
        let freq: Vec<usize> = data
            .par_iter()
            .copied()
            .into_bounded_par_iter(*max_value)
            .freq();
        freq
    });

    #[cfg(feature = "parallel")]
    bench_bounded_freq!(par_bounded_iter_unchecked, |(data, max_value)| unsafe {
        let freq: Vec<usize> = data
            .par_iter()
            .copied()
            .into_unchecked_bounded_par_iter(*max_value)
            .freq();
        freq
    });
}
