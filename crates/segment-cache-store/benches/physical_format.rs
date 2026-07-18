use criterion::{criterion_group, criterion_main};

#[path = "workload/physical_format.rs"]
mod physical_format;
#[path = "workload/segment_sizing.rs"]
mod segment_sizing;

criterion_group!(benches, physical_format::workload, segment_sizing::workload);
criterion_main!(benches);
