#[path = "common/alloc_benches.rs"]
mod alloc_benches;

use criterion::{criterion_group, criterion_main};

fn bench(c: &mut criterion::Criterion) {
    alloc_benches::bench_all(c, "default");
}

criterion_group!(benches, bench);
criterion_main!(benches);
