use std::hint::black_box;

use simply_simd::StaticSimd;

fn main() {
    let a = StaticSimd::iota(0u32);
    let b = StaticSimd::iota(12382u32);

    let c = black_box(a).interleave_even_8(black_box(b));

    black_box(c);
}
