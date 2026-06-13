use criterion::{criterion_group, criterion_main};

#[path = "workload/backends.rs"]
mod backends;
#[path = "workload/data.rs"]
mod data;
#[path = "workload/ordered_lookup.rs"]
mod ordered_lookup;
#[path = "workload/profile.rs"]
mod profile;

criterion_group!(benches, ordered_lookup::workload);
criterion_main!(benches);
