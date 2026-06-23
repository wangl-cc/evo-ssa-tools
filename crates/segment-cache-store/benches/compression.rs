use criterion::{criterion_group, criterion_main};

#[path = "workload/backends.rs"]
mod backends;
#[path = "workload/compression.rs"]
mod compression;
#[path = "workload/data.rs"]
mod data;
#[path = "workload/profile.rs"]
mod profile;

criterion_group!(benches, compression::workload);
criterion_main!(benches);
