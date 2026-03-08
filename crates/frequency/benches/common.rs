#![allow(dead_code)]

use std::{collections::HashMap, sync::LazyLock};

use frequency::prelude::*;
use rand::{
    distr::{Uniform, uniform::SampleUniform},
    prelude::*,
    rngs::SmallRng,
};

pub const SIZES: &[usize] = &[1 << 16, 1 << 20, 1 << 24];
pub const MAX_VALUE: usize = 4096;

pub trait Sampleable: SampleUniform + Count {
    fn from_usize(value: usize) -> Self;
}

pub trait BenchData: Sampleable + Copy + Send + Sync + 'static {
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

pub fn gen_data<T: Sampleable>(size: usize) -> Vec<T> {
    Uniform::new_inclusive(T::ZERO, T::from_usize(MAX_VALUE))
        .unwrap()
        .sample_iter(&mut SmallRng::seed_from_u64(42))
        .take(size)
        .collect()
}

pub fn canonical_u32_data(size: usize) -> &'static [u32] {
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
                        canonical_u32_data(size)
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
        canonical_u32_data(size)
    }
}

fn str_pool() -> &'static [&'static str] {
    static POOL: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
        (0..=MAX_VALUE)
            .map(|index| Box::leak(format!("key-{index}").into_boxed_str()) as &'static str)
            .collect()
    });

    POOL.as_slice()
}

pub fn str_data(size: usize) -> &'static [&'static str] {
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
