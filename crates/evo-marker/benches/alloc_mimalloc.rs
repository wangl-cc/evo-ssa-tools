#[path = "common/alloc_benches.rs"]
mod alloc_benches;

use criterion::{criterion_group, criterion_main};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn bench(c: &mut criterion::Criterion) {
    alloc_benches::bench_all(c, "mimalloc");
}

criterion_group!(benches, bench);
criterion_main!(benches);
