use num_traits::NumCast;

use crate::architectures::interface::*;
use crate::register::Simd;
use crate::simd_types::*;

const EXP_COEFFS_F32: [f32; 8] = [
    0.0001984127,
    0.0013888889,
    0.008333334,
    0.041666668,
    0.16666667,
    0.5,
    1.0,
    1.0,
];

// ln2^n / n!
const EXP2_COEFFS_F32: [f32; 8] = [
    1.525_273_4e-5,
    1.540_353e-4,
    1.333_355_8e-3,
    9.618_129e-3,
    5.550_411e-2,
    2.402_265e-1,
    std::f32::consts::LN_2,
    1.0,
];

const LOG2_E_F32: f32 = std::f32::consts::LOG2_E;
const LN2_HI_F32: f32 = 6.931_457_5e-1;
const LN2_LO_F32: f32 = 1.428_606_8e-6;
const MAGIC_F32: f32 = 12582912.0;

const EXP_COEFFS_F64: [f64; 14] = [
    1.6059043836821613e-10,
    2.08767569878681e-9,
    2.505210838544172e-8,
    2.755731922398589e-7,
    2.7557319223985893e-6,
    2.48015873015873e-5,
    0.0001984126984126984,
    0.001388888888888889,
    0.008333333333333333,
    0.041666666666666664,
    0.16666666666666666,
    0.5,
    1.0,
    1.0,
];

const EXP2_COEFFS_F64: [f64; 14] = [
    1.3691488853904e-12,
    2.567843599348e-11,
    4.445538271870811e-10,
    7.054911620801123e-9,
    1.0178086009239699e-7,
    1.3215486790144307e-6,
    1.525273380405984e-5,
    1.5403530393381606e-4,
    1.3333558146428443e-3,
    9.618129107628477e-3,
    5.550410866482158e-2,
    0.24022650695910072,
    std::f64::consts::LN_2,
    1.0,
];

const LOG2_E_F64: f64 = std::f64::consts::LOG2_E;
const LN2_HI_F64: f64 = 6.931471803691238e-1;
const LN2_LO_F64: f64 = 1.908214929270588e-10;
const MAGIC_F64: f64 = 6755399441055744.0;

// Q(f) ~= log2(1+f)/f on [sqrt(0.5)-1, sqrt(2)-1], degree 9, max rel err 4.7e-9
const LOG2_Q_F32: [f32; 10] = [
    -1.0749789e-01,
    1.8476033e-01,
    -1.9138838e-01,
    2.0485765e-01,
    -2.3960812e-01,
    2.8855255e-01,
    -3.6069655e-01,
    4.8089853e-01,
    -7.2134733e-01,
    std::f32::consts::LOG2_E,
];

const fn log2_coeffs<const N: usize>() -> [f64; N] {
    let mut out = [0.0; N];
    let mut n = 0;
    while n < N {
        out[N - 1 - n] = 2.0 * std::f64::consts::LOG2_E / (2 * n + 1) as f64;
        n += 1;
    }
    out
}

const LOG2_COEFFS_F64: [f64; 11] = log2_coeffs::<11>();
// const LOG2_COEFFS_F32: [f32; 5] = {
//     let c = log2_coeffs::<5>();
//     let mut out = [0.0f32; 5];
//     let mut i = 0;
//     while i < 5 {
//         out[i] = c[i] as f32;
//         i += 1;
//     }
//     out
// };

// TODO: Explore using a LUT?
impl<T: SimdFloat, A: Arch> Simd<T, A> {
    #[inline]
    pub fn exp(self) -> Self {
        match T::TYPE {
            SimdType::F64 => {
                let log2_e = Simd::splat(NumCast::from(LOG2_E_F64).unwrap());
                let ln2_hi = Simd::splat(NumCast::from(LN2_HI_F64).unwrap());
                let ln2_lo = Simd::splat(NumCast::from(LN2_LO_F64).unwrap());
                let magic = Simd::splat(NumCast::from(MAGIC_F64).unwrap());

                let lower_bound = Simd::splat(NumCast::from(-708.0).unwrap());
                let upper_bound = Simd::splat(NumCast::from(709.0).unwrap());

                let k = (self * log2_e).round();

                let r = (-k).mul_add(ln2_hi, self);
                let r = (-k).mul_add(ln2_lo, r);

                let mut p = Simd::splat(NumCast::from(EXP_COEFFS_F64[0]).unwrap());
                for x in EXP_COEFFS_F64.iter().skip(1) {
                    let x = Simd::splat(NumCast::from(*x).unwrap());
                    p = p.mul_add(r, x)
                }

                let add_bits = Simd::splat(NumCast::from(1023).unwrap());
                let kb: Simd<T::UintType, A> = (k + magic).raw_cast();
                let scale: Self = ((kb + add_bits) << 52).raw_cast();

                let res = p * scale;

                let is_zero = self.simd_lt(lower_bound);
                let is_inf = self.simd_gt(upper_bound);

                let zero = Simd::zero();
                let inf = Simd::splat(NumCast::from(f64::INFINITY).unwrap());

                let res = is_zero.select(zero, res);
                is_inf.select(inf, res)
            }
            SimdType::F32 => {
                let log2_e = Simd::splat(NumCast::from(LOG2_E_F32).unwrap());
                let ln2_hi = Simd::splat(NumCast::from(LN2_HI_F32).unwrap());
                let ln2_lo = Simd::splat(NumCast::from(LN2_LO_F32).unwrap());
                let magic = Simd::splat(NumCast::from(MAGIC_F32).unwrap());

                let lower_bound = Simd::splat(NumCast::from(-87.0).unwrap());
                let upper_bound = Simd::splat(NumCast::from(88.0).unwrap());

                let k = (self * log2_e).round();

                let r = (-k).mul_add(ln2_hi, self);
                let r = (-k).mul_add(ln2_lo, r);

                let mut p = Simd::splat(NumCast::from(EXP_COEFFS_F32[0]).unwrap());
                for x in EXP_COEFFS_F32.iter().skip(1) {
                    let x = Simd::splat(NumCast::from(*x).unwrap());
                    p = p.mul_add(r, x)
                }

                let add_bits = Simd::splat(NumCast::from(127).unwrap());
                let kb: Simd<T::UintType, A> = (k + magic).raw_cast();
                let scale: Self = ((kb + add_bits) << 23).raw_cast();

                let res = p * scale;

                let is_zero = self.simd_lt(lower_bound);
                let is_inf = self.simd_gt(upper_bound);

                let zero = Simd::zero();
                let inf = Simd::splat(NumCast::from(f32::INFINITY).unwrap());

                let res = is_zero.select(zero, res);
                is_inf.select(inf, res)
            }
            _ => unreachable!(),
        }
    }

    /// # Safety
    ///
    /// Does not check for the 0.0 or INF cases. Assumes the given values
    /// do not result in 0.0 or INF.
    #[inline(always)]
    pub unsafe fn exp_unchecked(self) -> Self {
        match T::TYPE {
            SimdType::F64 => {
                let log2_e = Simd::splat(NumCast::from(LOG2_E_F64).unwrap());
                let ln2_hi = Simd::splat(NumCast::from(LN2_HI_F64).unwrap());
                let ln2_lo = Simd::splat(NumCast::from(LN2_LO_F64).unwrap());
                let magic = Simd::splat(NumCast::from(MAGIC_F64).unwrap());

                let k = (self * log2_e).round();

                let r = (-k).mul_add(ln2_hi, self);
                let r = (-k).mul_add(ln2_lo, r);

                let mut p = Simd::splat(NumCast::from(EXP_COEFFS_F64[0]).unwrap());
                for x in EXP_COEFFS_F64.iter().skip(1) {
                    let x = Simd::splat(NumCast::from(*x).unwrap());
                    p = p.mul_add(r, x)
                }

                let add_bits = Simd::splat(NumCast::from(1023).unwrap());
                let kb: Simd<T::UintType, A> = (k + magic).raw_cast();
                let scale: Self = ((kb + add_bits) << 52).raw_cast();

                p * scale
            }
            SimdType::F32 => {
                let log2_e = Simd::splat(NumCast::from(LOG2_E_F32).unwrap());
                let ln2_hi = Simd::splat(NumCast::from(LN2_HI_F32).unwrap());
                let ln2_lo = Simd::splat(NumCast::from(LN2_LO_F32).unwrap());
                let magic = Simd::splat(NumCast::from(MAGIC_F32).unwrap());

                let k = (self * log2_e).round();

                let r = (-k).mul_add(ln2_hi, self);
                let r = (-k).mul_add(ln2_lo, r);

                let mut p = Simd::splat(NumCast::from(EXP_COEFFS_F32[0]).unwrap());
                for x in EXP_COEFFS_F32.iter().skip(1) {
                    let x = Simd::splat(NumCast::from(*x).unwrap());
                    p = p.mul_add(r, x)
                }

                let add_bits = Simd::splat(NumCast::from(127).unwrap());
                let kb: Simd<T::UintType, A> = (k + magic).raw_cast();
                let scale: Self = ((kb + add_bits) << 23).raw_cast();

                p * scale
            }
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn exp2(self) -> Self {
        match T::TYPE {
            SimdType::F64 => {
                let magic = Simd::splat(NumCast::from(MAGIC_F64).unwrap());

                let lower_bound = Simd::splat(NumCast::from(-1022.0).unwrap());
                let upper_bound = Simd::splat(NumCast::from(1024.0).unwrap());

                let k = self.round();

                let r = self - k;

                let mut p = Simd::splat(NumCast::from(EXP2_COEFFS_F64[0]).unwrap());
                for x in EXP2_COEFFS_F64.iter().skip(1) {
                    let x = Simd::splat(NumCast::from(*x).unwrap());
                    p = p.mul_add(r, x)
                }

                let add_bits = Simd::splat(NumCast::from(1023).unwrap());
                let kb: Simd<T::UintType, A> = (k + magic).raw_cast();
                let scale: Self = ((kb + add_bits) << 52).raw_cast();

                let res = p * scale;

                let is_zero = self.simd_lt(lower_bound);
                let is_inf = self.simd_gt(upper_bound);

                let zero = Simd::zero();
                let inf = Simd::splat(NumCast::from(f64::INFINITY).unwrap());

                let res = is_zero.select(zero, res);
                is_inf.select(inf, res)
            }
            SimdType::F32 => {
                let magic = Simd::splat(NumCast::from(MAGIC_F32).unwrap());

                let lower_bound = Simd::splat(NumCast::from(-126.0).unwrap());
                let upper_bound = Simd::splat(NumCast::from(128.0).unwrap());

                let k = self.round();

                let r = self - k;

                let mut p = Simd::splat(NumCast::from(EXP2_COEFFS_F32[0]).unwrap());
                for x in EXP2_COEFFS_F32.iter().skip(1) {
                    let x = Simd::splat(NumCast::from(*x).unwrap());
                    p = p.mul_add(r, x)
                }

                let add_bits = Simd::splat(NumCast::from(127).unwrap());
                let kb: Simd<T::UintType, A> = (k + magic).raw_cast();
                let scale: Self = ((kb + add_bits) << 23).raw_cast();

                let res = p * scale;

                let is_zero = self.simd_lt(lower_bound);
                let is_inf = self.simd_gt(upper_bound);

                let zero = Simd::zero();
                let inf = Simd::splat(NumCast::from(f32::INFINITY).unwrap());

                let res = is_zero.select(zero, res);
                is_inf.select(inf, res)
            }
            _ => unreachable!(),
        }
    }

    /// # Safety
    ///
    /// Does not check for the 0.0 or INF cases. Assumes the given values
    /// do not result in 0.0 or INF.
    #[inline]
    pub unsafe fn exp2_unchecked(self) -> Self {
        match T::TYPE {
            SimdType::F64 => {
                let magic = Simd::splat(NumCast::from(MAGIC_F64).unwrap());

                let k = self.round();

                let r = self - k;

                let mut p = Simd::splat(NumCast::from(EXP2_COEFFS_F64[0]).unwrap());
                for x in EXP2_COEFFS_F64.iter().skip(1) {
                    let x = Simd::splat(NumCast::from(*x).unwrap());
                    p = p.mul_add(r, x)
                }

                let add_bits = Simd::splat(NumCast::from(1023).unwrap());
                let kb: Simd<T::UintType, A> = (k + magic).raw_cast();
                let scale: Self = ((kb + add_bits) << 52).raw_cast();

                p * scale
            }
            SimdType::F32 => {
                let magic = Simd::splat(NumCast::from(MAGIC_F32).unwrap());

                let k = self.round();

                let r = self - k;

                let mut p = Simd::splat(NumCast::from(EXP2_COEFFS_F32[0]).unwrap());
                for x in EXP2_COEFFS_F32.iter().skip(1) {
                    let x = Simd::splat(NumCast::from(*x).unwrap());
                    p = p.mul_add(r, x)
                }

                let add_bits = Simd::splat(NumCast::from(127).unwrap());
                let kb: Simd<T::UintType, A> = (k + magic).raw_cast();
                let scale: Self = ((kb + add_bits) << 23).raw_cast();

                p * scale
            }
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn log2(self) -> Self {
        match T::TYPE {
            SimdType::F32 => {
                let zero = Simd::zero();
                let one = Simd::splat(NumCast::from(1.0f32).unwrap());

                let min_normal = Simd::splat(NumCast::from(f32::MIN_POSITIVE).unwrap());
                let is_sub = self.simd_lt(min_normal);
                let x = is_sub.select(
                    self * Simd::splat(NumCast::from(16777216.0f32).unwrap()),
                    self,
                );

                let bits: Simd<T::UintType, A> = x.raw_cast();
                let b = bits + Simd::splat(NumCast::from(0x004A_FB0Du32).unwrap());

                let e_raw: Self =
                    ((b >> 23) | Simd::splat(NumCast::from(0x4B00_0000u32).unwrap())).raw_cast();
                let e = e_raw - Simd::splat(NumCast::from(8388608.0f32 + 127.0).unwrap());
                let e = is_sub.select(e - Simd::splat(NumCast::from(24.0f32).unwrap()), e);

                let m: Self = ((b & Simd::splat(NumCast::from(0x007F_FFFFu32).unwrap()))
                    + Simd::splat(NumCast::from(0x3F35_04F3u32).unwrap()))
                .raw_cast();

                let f = m - one;
                let mut q = Simd::splat(NumCast::from(LOG2_Q_F32[0]).unwrap());
                for c in LOG2_Q_F32.iter().skip(1) {
                    q = q.mul_add(f, Simd::splat(NumCast::from(*c).unwrap()));
                }
                let res = f.mul_add(q, e);

                let inf = Simd::splat(NumCast::from(f32::INFINITY).unwrap());
                let res = self.simd_eq(zero).select(-inf, res);
                let res = self.simd_eq(inf).select(inf, res);
                let res = self
                    .simd_lt(zero)
                    .select(Simd::splat(NumCast::from(f32::NAN).unwrap()), res);
                self.simd_ne(self).select(self, res)
            }
            SimdType::F64 => {
                let zero = Simd::zero();
                let one = Simd::splat(NumCast::from(1.0f64).unwrap());

                let min_normal = Simd::splat(NumCast::from(f64::MIN_POSITIVE).unwrap());
                let is_sub = self.simd_lt(min_normal);
                let x = is_sub.select(
                    self * Simd::splat(NumCast::from(18014398509481984.0f64).unwrap()),
                    self,
                );

                let bits: Simd<T::UintType, A> = x.raw_cast();

                let e_raw: Self = ((bits >> 52)
                    | Simd::splat(NumCast::from(0x4330_0000_0000_0000u64).unwrap()))
                .raw_cast();
                let e = e_raw - Simd::splat(NumCast::from(4503599627370496.0f64 + 1023.0).unwrap());

                let m: Self = ((bits
                    & Simd::splat(NumCast::from(0x000F_FFFF_FFFF_FFFFu64).unwrap()))
                    | Simd::splat(NumCast::from(0x3FF0_0000_0000_0000u64).unwrap()))
                .raw_cast();

                let big = m.simd_gt(Simd::splat(
                    NumCast::from(std::f64::consts::SQRT_2).unwrap(),
                ));
                let m = big.select(m * Simd::splat(NumCast::from(0.5f64).unwrap()), m);
                let e = big.select(e + one, e);
                let e = is_sub.select(e - Simd::splat(NumCast::from(54.0f64).unwrap()), e);

                let f = m - one;
                let s = f / (f + Simd::splat(NumCast::from(2.0f64).unwrap()));
                let s2 = s * s;

                let mut p = Simd::splat(NumCast::from(LOG2_COEFFS_F64[0]).unwrap());
                for c in LOG2_COEFFS_F64.iter().skip(1) {
                    p = p.mul_add(s2, Simd::splat(NumCast::from(*c).unwrap()));
                }
                let res = s.mul_add(p, e);

                let inf = Simd::splat(NumCast::from(f64::INFINITY).unwrap());
                let res = self.simd_eq(zero).select(-inf, res);
                let res = self.simd_eq(inf).select(inf, res);
                let res = self
                    .simd_lt(zero)
                    .select(Simd::splat(NumCast::from(f64::NAN).unwrap()), res);
                self.simd_ne(self).select(self, res)
            }
            _ => unreachable!(),
        }
    }

    /// # Safety
    ///
    /// Does not check for the -INF, INF, or NaN cases. Assumes the given values
    /// do not result in 0.0 or INF.
    #[inline]
    pub unsafe fn log2_unchecked(self) -> Self {
        match T::TYPE {
            SimdType::F32 => {
                let one = Simd::splat(NumCast::from(1.0f32).unwrap());

                let min_normal = Simd::splat(NumCast::from(f32::MIN_POSITIVE).unwrap());
                let is_sub = self.simd_lt(min_normal);
                let x = is_sub.select(
                    self * Simd::splat(NumCast::from(16777216.0f32).unwrap()),
                    self,
                );

                let bits: Simd<T::UintType, A> = x.raw_cast();
                let b = bits + Simd::splat(NumCast::from(0x004A_FB0Du32).unwrap());

                let e_raw: Self =
                    ((b >> 23) | Simd::splat(NumCast::from(0x4B00_0000u32).unwrap())).raw_cast();
                let e = e_raw - Simd::splat(NumCast::from(8388608.0f32 + 127.0).unwrap());
                let e = is_sub.select(e - Simd::splat(NumCast::from(24.0f32).unwrap()), e);

                let m: Self = ((b & Simd::splat(NumCast::from(0x007F_FFFFu32).unwrap()))
                    + Simd::splat(NumCast::from(0x3F35_04F3u32).unwrap()))
                .raw_cast();

                let f = m - one;
                let mut q = Simd::splat(NumCast::from(LOG2_Q_F32[0]).unwrap());
                for c in LOG2_Q_F32.iter().skip(1) {
                    q = q.mul_add(f, Simd::splat(NumCast::from(*c).unwrap()));
                }
                f.mul_add(q, e)
            }
            SimdType::F64 => {
                let one = Simd::splat(NumCast::from(1.0f64).unwrap());

                let min_normal = Simd::splat(NumCast::from(f64::MIN_POSITIVE).unwrap());
                let is_sub = self.simd_lt(min_normal);
                let x = is_sub.select(
                    self * Simd::splat(NumCast::from(18014398509481984.0f64).unwrap()),
                    self,
                );

                let bits: Simd<T::UintType, A> = x.raw_cast();

                let e_raw: Self = ((bits >> 52)
                    | Simd::splat(NumCast::from(0x4330_0000_0000_0000u64).unwrap()))
                .raw_cast();
                let e = e_raw - Simd::splat(NumCast::from(4503599627370496.0f64 + 1023.0).unwrap());

                let m: Self = ((bits
                    & Simd::splat(NumCast::from(0x000F_FFFF_FFFF_FFFFu64).unwrap()))
                    | Simd::splat(NumCast::from(0x3FF0_0000_0000_0000u64).unwrap()))
                .raw_cast();

                let big = m.simd_gt(Simd::splat(
                    NumCast::from(std::f64::consts::SQRT_2).unwrap(),
                ));
                let m = big.select(m * Simd::splat(NumCast::from(0.5f64).unwrap()), m);
                let e = big.select(e + one, e);
                let e = is_sub.select(e - Simd::splat(NumCast::from(54.0f64).unwrap()), e);

                let f = m - one;
                let s = f / (f + Simd::splat(NumCast::from(2.0f64).unwrap()));
                let s2 = s * s;

                let mut p = Simd::splat(NumCast::from(LOG2_COEFFS_F64[0]).unwrap());
                for c in LOG2_COEFFS_F64.iter().skip(1) {
                    p = p.mul_add(s2, Simd::splat(NumCast::from(*c).unwrap()));
                }
                s.mul_add(p, e)
            }
            _ => unreachable!(),
        }
    }

    pub fn pow(self, y: Self) -> Self {
        let zero = Simd::zero();
        let one = Simd::splat(NumCast::from(1.0f32).unwrap());
        let half = Simd::splat(NumCast::from(0.5f32).unwrap());
        let inf = Simd::splat(NumCast::from(f32::INFINITY).unwrap());
        let nan = Simd::splat(NumCast::from(f32::NAN).unwrap());

        let res = (y * self.abs().log2()).exp2();

        let neg_x = self.simd_lt(zero);
        let is_int = y.simd_eq(y.round());
        let yh = y * half;
        let is_odd = yh.simd_ne(yh.round());
        let signed = is_odd.select(-res, res);
        let res = neg_x.select(is_int.select(signed, nan), res);

        let res = y.simd_eq(zero).select(one, res);
        let res = self.simd_eq(one).select(one, res);
        let abs_x_is_one = self.abs().simd_eq(one);
        let y_is_inf = y.abs().simd_eq(inf);
        abs_x_is_one.select(y_is_inf.select(one, res), res)
    }

    /// # Safety
    ///
    /// - `self` must be positive, normal, and finite.
    /// - `y` must be finite.
    /// - Result must be representable as an f32.
    pub unsafe fn pow_unchecked(self, y: Self) -> Self {
        unsafe { (y * self.log2_unchecked()).exp2_unchecked() }
    }
}
