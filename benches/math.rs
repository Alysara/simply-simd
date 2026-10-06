use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use num_traits::Pow;
use simply_simd::StaticSimd;

fn math_bench(c: &mut Criterion) {
    let values1 = black_box(StaticSimd::splat(3.3_f32));
    let values2 = black_box(StaticSimd::splat(2.3_f32));
    let values1_array = black_box(values1.to_array());
    let values2_array = black_box(values2.to_array());

    // ---------- exp -----------
    let mut group = c.benchmark_group("exp");
    group.throughput(Throughput::Elements(StaticSimd::<f32>::LANES as u64));
    group.bench_function("scalar exp", |b| {
        b.iter(|| {
            values1_array.iter().for_each(|x| {
                black_box(black_box(x).exp());
            })
        });
    });
    group.bench_function("simd checked exp", |b| {
        b.iter(|| black_box(black_box(values1).exp()));
    });
    group.bench_function("simd unchecked exp", |b| {
        b.iter(|| {
            unsafe { black_box(black_box(values1).exp_unchecked()) };
        });
    });
    group.finish();

    // ---------- exp2 -----------
    let mut group = c.benchmark_group("exp2");
    group.throughput(Throughput::Elements(StaticSimd::<f32>::LANES as u64));
    group.bench_function("scalar exp2", |b| {
        b.iter(|| {
            values1_array.iter().for_each(|x| {
                black_box(black_box(x).exp2());
            })
        });
    });
    group.bench_function("simd checked exp2", |b| {
        b.iter(|| black_box(black_box(values1).exp2()));
    });
    group.bench_function("simd unchecked exp2", |b| {
        b.iter(|| {
            unsafe { black_box(black_box(values1).exp2_unchecked()) };
        });
    });
    group.finish();

    // ---------- log2 -----------
    let mut group = c.benchmark_group("log2");
    group.throughput(Throughput::Elements(StaticSimd::<f32>::LANES as u64));
    group.bench_function("scalar log2", |b| {
        b.iter(|| {
            values1_array.iter().for_each(|x| {
                black_box(black_box(x).log2());
            })
        });
    });
    group.bench_function("simd checked log2", |b| {
        b.iter(|| black_box(black_box(values1).log2()));
    });
    group.bench_function("simd unchecked log2", |b| {
        b.iter(|| {
            unsafe { black_box(black_box(values1).log2_unchecked()) };
        });
    });
    group.finish();

    // ---------- pow -----------
    let mut group = c.benchmark_group("pow");
    group.throughput(Throughput::Elements(StaticSimd::<f32>::LANES as u64));
    group.bench_function("scalar pow", |b| {
        b.iter(|| {
            values1_array
                .iter()
                .zip(values2_array.iter())
                .for_each(|(x, y)| {
                    black_box(black_box(x).pow(black_box(*y)));
                })
        });
    });
    group.bench_function("simd checked pow", |b| {
        b.iter(|| black_box(black_box(values1).pow(black_box(values2))));
    });
    group.bench_function("simd unchecked pow", |b| {
        b.iter(|| {
            unsafe { black_box(black_box(values1).pow_unchecked(black_box(values2))) };
        });
    });
    group.finish();
}

criterion_group!(benches, math_bench);
criterion_main!(benches);
