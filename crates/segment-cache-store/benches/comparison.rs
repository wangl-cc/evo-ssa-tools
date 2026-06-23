use criterion::{criterion_group, criterion_main};

#[path = "workload/backends.rs"]
mod backends;
#[path = "workload/comparison.rs"]
mod comparison;
#[path = "workload/data.rs"]
mod data;
#[path = "workload/profile.rs"]
mod profile;

criterion_group!(benches, comparison::workload);
criterion_main!(benches);
