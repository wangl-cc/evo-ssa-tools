//! Benchmarking to compare different methods of calculating frequency for elements in a range

/*
Results on my machine (Apple M1, lto=fat, target-cpu=native):

bounded_freq                      fastest       │ slowest       │ median        │ mean          │ samples │ iters
╰─ compare_methods                              │               │               │               │         │
   ├─ bounded_iter                              │               │               │               │         │
   │  ├─ u16                                    │               │               │               │         │
   │  │  ├─ 65536                 23.53 µs      │ 29.51 µs      │ 23.99 µs      │ 23.98 µs      │ 100     │ 1000
   │  │  ├─ 1048576               370.8 µs      │ 431.3 µs      │ 374.4 µs      │ 378.1 µs      │ 100     │ 1000
   │  │  ╰─ 16777216              5.945 ms      │ 7.714 ms      │ 6.066 ms      │ 6.075 ms      │ 100     │ 1000
   │  ╰─ u32                                    │               │               │               │         │
   │     ├─ 65536                 23.68 µs      │ 30.82 µs      │ 25.71 µs      │ 25.74 µs      │ 100     │ 1000
   │     ├─ 1048576               374 µs        │ 648.4 µs      │ 382.5 µs      │ 390.7 µs      │ 100     │ 1000
   │     ╰─ 16777216              6.018 ms      │ 9.644 ms      │ 6.11 ms       │ 6.141 ms      │ 100     │ 1000
   ├─ bounded_iter_unchecked                    │               │               │               │         │
   │  ├─ u16                                    │               │               │               │         │
   │  │  ├─ 65536                 21.56 µs      │ 29.58 µs      │ 22.38 µs      │ 23.16 µs      │ 100     │ 1000
   │  │  ├─ 1048576               339.7 µs      │ 373.5 µs      │ 350.5 µs      │ 350.3 µs      │ 100     │ 1000
   │  │  ╰─ 16777216              5.536 ms      │ 12.63 ms      │ 5.588 ms      │ 5.774 ms      │ 100     │ 1000
   │  ╰─ u32                                    │               │               │               │         │
   │     ├─ 65536                 21.89 µs      │ 38.37 µs      │ 24.27 µs      │ 24.07 µs      │ 100     │ 1000
   │     ├─ 1048576               343.8 µs      │ 384.5 µs      │ 354.2 µs      │ 356.5 µs      │ 100     │ 1000
   │     ╰─ 16777216              5.584 ms      │ 6.081 ms      │ 5.646 ms      │ 5.652 ms      │ 100     │ 1000
   ├─ bounded_par_iter                          │               │               │               │         │
   │  ├─ u16                                    │               │               │               │         │
   │  │  ├─ 65536                 1.084 ms      │ 3.363 ms      │ 1.905 ms      │ 1.975 ms      │ 100     │ 1000
   │  │  ├─ 1048576               2.488 ms      │ 6.545 ms      │ 3.965 ms      │ 4.112 ms      │ 100     │ 1000
   │  │  ╰─ 16777216              5.019 ms      │ 8.188 ms      │ 6.116 ms      │ 6.158 ms      │ 100     │ 1000
   │  ╰─ u32                                    │               │               │               │         │
   │     ├─ 65536                 1.501 ms      │ 5.362 ms      │ 3.392 ms      │ 3.449 ms      │ 100     │ 1000
   │     ├─ 1048576               2.511 ms      │ 5.578 ms      │ 3.816 ms      │ 3.843 ms      │ 100     │ 1000
   │     ╰─ 16777216              4.067 ms      │ 9.609 ms      │ 6.092 ms      │ 6.428 ms      │ 100     │ 1000
   ├─ bounded_par_iter_unchecked                │               │               │               │         │
   │  ├─ u16                                    │               │               │               │         │
   │  │  ├─ 65536                 1.965 ms      │ 5.068 ms      │ 2.891 ms      │ 2.993 ms      │ 100     │ 1000
   │  │  ├─ 1048576               2.48 ms       │ 7.702 ms      │ 3.723 ms      │ 3.871 ms      │ 100     │ 1000
   │  │  ╰─ 16777216              4.83 ms       │ 7.904 ms      │ 5.67 ms       │ 5.776 ms      │ 100     │ 1000
   │  ╰─ u32                                    │               │               │               │         │
   │     ├─ 65536                 1.512 ms      │ 4.548 ms      │ 2.462 ms      │ 2.574 ms      │ 100     │ 1000
   │     ├─ 1048576               2.283 ms      │ 7.259 ms      │ 3.104 ms      │ 3.233 ms      │ 100     │ 1000
   │     ╰─ 16777216              4.726 ms      │ 7.604 ms      │ 5.779 ms      │ 5.84 ms       │ 100     │ 1000
   ├─ hash_iter                                 │               │               │               │         │
   │  ├─ u16                                    │               │               │               │         │
   │  │  ├─ 65536                 294.7 µs      │ 456.1 µs      │ 303.6 µs      │ 309.4 µs      │ 100     │ 1000
   │  │  ├─ 1048576               3.441 ms      │ 5.822 ms      │ 3.651 ms      │ 3.751 ms      │ 100     │ 1000
   │  │  ╰─ 16777216              53.64 ms      │ 61.51 ms      │ 54.35 ms      │ 55.24 ms      │ 100     │ 1000
   │  ╰─ u32                                    │               │               │               │         │
   │     ├─ 65536                 296.3 µs      │ 761.6 µs      │ 307.2 µs      │ 321.7 µs      │ 100     │ 1000
   │     ├─ 1048576               3.435 ms      │ 4.913 ms      │ 3.647 ms      │ 3.698 ms      │ 100     │ 1000
   │     ╰─ 16777216              53.67 ms      │ 62.74 ms      │ 55.93 ms      │ 56.27 ms      │ 100     │ 1000
   ├─ manual                                    │               │               │               │         │
   │  ├─ u16                                    │               │               │               │         │
   │  │  ├─ 65536                 24 µs         │ 34.92 µs      │ 26.77 µs      │ 27.08 µs      │ 100     │ 1000
   │  │  ├─ 1048576               378.6 µs      │ 604.6 µs      │ 400 µs        │ 417.8 µs      │ 100     │ 1000
   │  │  ╰─ 16777216              6.039 ms      │ 7.464 ms      │ 6.089 ms      │ 6.161 ms      │ 100     │ 1000
   │  ╰─ u32                                    │               │               │               │         │
   │     ├─ 65536                 24.14 µs      │ 31.54 µs      │ 25.82 µs      │ 25.66 µs      │ 100     │ 1000
   │     ├─ 1048576               381.8 µs      │ 636.7 µs      │ 398 µs        │ 407.3 µs      │ 100     │ 1000
   │     ╰─ 16777216              6.119 ms      │ 8.899 ms      │ 6.394 ms      │ 6.546 ms      │ 100     │ 1000
   ╰─ manual_unchecked                          │               │               │               │         │
      ├─ u16                                    │               │               │               │         │
      │  ├─ 65536                 22.31 µs      │ 82.83 µs      │ 23.41 µs      │ 27.5 µs       │ 100     │ 1000
      │  ├─ 1048576               349.8 µs      │ 945.8 µs      │ 369.8 µs      │ 401.9 µs      │ 100     │ 1000
      │  ╰─ 16777216              5.572 ms      │ 9.203 ms      │ 5.702 ms      │ 5.817 ms      │ 100     │ 1000
      ╰─ u32                                    │               │               │               │         │
         ├─ 65536                 22.27 µs      │ 33.61 µs      │ 24.6 µs       │ 24.75 µs      │ 100     │ 1000
         ├─ 1048576               352.3 µs      │ 397.6 µs      │ 359.6 µs      │ 363.3 µs      │ 100     │ 1000
         ╰─ 16777216              5.639 ms      │ 8.409 ms      │ 5.779 ms      │ 5.908 ms      │ 100     │ 1000
*/

mod manual;

use std::{collections::HashMap, sync::LazyLock};

use divan::Bencher;
use frequency::prelude::*;
use manual::*;
use rand::{
    distr::{Uniform, uniform::SampleUniform},
    prelude::*,
    rngs::SmallRng,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[1 << 16, 1 << 20, 1 << 24];

fn prepare_data<T>(size: usize) -> (Vec<T>, usize)
where
    T: num_traits::PrimInt + SampleUniform + bytemuck::AnyBitPattern,
{
    const MAX_VALUE: usize = 4096;

    fn cast_vec<A, B>(src: &[A]) -> Vec<B>
    where
        A: bytemuck::NoUninit,
        B: bytemuck::AnyBitPattern,
    {
        bytemuck::cast_slice(src).to_vec()
    }

    fn gen_data<T>(size: usize) -> Vec<T>
    where
        T: num_traits::PrimInt + SampleUniform,
    {
        Uniform::new_inclusive(T::zero(), T::from(MAX_VALUE).unwrap_or(T::max_value()))
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
                    + num_traits::PrimInt
                    + bytemuck::AnyBitPattern
                    + Send
                    + Sync,
            {
                bencher
                    .with_inputs(|| prepare_data::<T>(size))
                    .bench_local_refs($func)
            }
        };
    }

    bench_bounded_freq!(manual, |(data, max_value)| bounded_freq::<T, usize>(
        data, *max_value
    ));
    bench_bounded_freq!(manual_unchecked, |(data, max_value)| unsafe {
        unchecked_bounded_freq::<T, usize>(data, *max_value)
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
    bench_bounded_freq!(bounded_par_iter, |(data, max_value)| {
        let freq: Vec<usize> = data
            .par_iter()
            .copied()
            .into_bounded_par_iter(*max_value)
            .freq();
        freq
    });

    #[cfg(feature = "parallel")]
    bench_bounded_freq!(bounded_par_iter_unchecked, |(data, max_value)| unsafe {
        let freq: Vec<usize> = data
            .par_iter()
            .copied()
            .into_unchecked_bounded_par_iter(*max_value)
            .freq();
        freq
    });
}
