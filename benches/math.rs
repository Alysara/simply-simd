use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use simply_simd::StaticSimd;

fn math_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp");
    let values = black_box(StaticSimd::splat(3.3_f32));
    let values_array = black_box(values.to_array());

    group.throughput(Throughput::Elements(StaticSimd::<f32>::LANES as u64));
    group.bench_function("scalar exp", |b| {
        b.iter(|| {
            values_array.iter().for_each(|x| {
                black_box(x.exp());
            })
        });
    });
    group.bench_function("simd checked exp", |b| {
        b.iter(|| black_box(values.exp()));
    });
    group.bench_function("simd unchecked exp", |b| {
        b.iter(|| {
            unsafe { black_box(values.exp_unchecked()) };
        });
    });
    group.finish();
}

criterion_group!(benches, math_bench);
criterion_main!(benches);
