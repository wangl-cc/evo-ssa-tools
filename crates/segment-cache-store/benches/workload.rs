use criterion::{criterion_group, criterion_main};

#[path = "workload/backends.rs"]
mod backends;
#[path = "workload/data.rs"]
mod data;
#[path = "workload/profile.rs"]
mod profile;
#[path = "workload/scenarios.rs"]
mod scenarios;

criterion_group!(benches, scenarios::workload);
criterion_main!(benches);
