use criterion::{criterion_group, criterion_main};

#[path = "workload/physical_format.rs"]
mod physical_format;

criterion_group!(benches, physical_format::workload);
criterion_main!(benches);
