use criterion::{criterion_group, criterion_main};

#[path = "workload/backends.rs"]
mod backends;
#[path = "workload/data.rs"]
mod data;
#[path = "workload/parameter_evolution.rs"]
mod parameter_evolution;
#[path = "workload/profile.rs"]
mod profile;

criterion_group!(benches, parameter_evolution::workload);
criterion_main!(benches);
