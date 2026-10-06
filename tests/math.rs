use num_traits::Pow;
use simply_simd::StaticSimd;

fn ord(f: f32) -> i64 {
    let i = f.to_bits() as i32;
    (if i < 0 { i32::MIN.wrapping_sub(i) } else { i }) as i64
}

fn ulp_diff(a: f32, b: f32) -> u64 {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => return 0,
        (false, false) => {}
        _ => return u64::MAX,
    }
    if a == b {
        return 0;
    };
    (ord(a) - ord(b)).unsigned_abs()
}

#[test]
fn exp_test() {
    let mut i = -87.0;
    while i < 88.0 {
        let vals = StaticSimd::<f32>::splat(i);
        let simd_result = vals.exp().to_array()[0];
        let true_result = i.exp();

        let ulp = ulp_diff(simd_result, true_result);
        assert!(ulp <= 20, "exp ulp is {ulp} for input {i}!");

        i += 2.3546324;
    }
}

#[test]
fn exp2_test() {
    let mut i = -126.0;
    while i < 127.4 {
        let vals = StaticSimd::<f32>::splat(i);
        let simd_result = vals.exp2().to_array()[0];
        let true_result = i.exp2();

        let ulp = ulp_diff(simd_result, true_result);
        assert!(ulp <= 20, "exp2 ulp is {ulp} for input {i}!");

        i += 2.3546324;
    }
}

#[test]
fn log2_test() {
    let mut i = -1000.0;
    while i < 1000.0 {
        let vals = StaticSimd::<f32>::splat(i);
        let simd_result = vals.log2().to_array()[0];
        let true_result = i.log2();

        let ulp = ulp_diff(simd_result, true_result);
        assert!(ulp <= 4, "log2 ulp is {ulp} for input {i}!");

        i += 2.3546324;
    }
}

#[test]
fn pow_test() {
    let mut i: f32 = -100.0;
    while i < 100.0 {
        let bits = i.to_bits() % 100;
        let j = (bits as f32) / 9.239874;

        let vals = StaticSimd::<f32>::splat(i);
        let exps = StaticSimd::<f32>::splat(j);
        let simd_result = vals.pow(exps).to_array()[0];
        let true_result = i.pow(j);

        let ulp = ulp_diff(simd_result, true_result);
        assert!(ulp <= 30, "pow ulp is {ulp} for input {i} ^ {j}!");

        i += 2.3546324;
    }
}
