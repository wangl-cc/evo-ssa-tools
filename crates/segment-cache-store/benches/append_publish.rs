use criterion::{criterion_group, criterion_main};

#[path = "workload/append_publish.rs"]
mod append_publish;
#[path = "workload/backends.rs"]
mod backends;
#[path = "workload/data.rs"]
mod data;
#[path = "workload/profile.rs"]
mod profile;

criterion_group!(benches, append_publish::workload);
criterion_main!(benches);
