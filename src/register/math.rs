use num_traits::NumCast;

use crate::architectures::interface::*;
use crate::register::Simd;
use crate::simd_types::*;

const INV_FACT_F32: [f32; 8] = [
    0.0001984127,
    0.0013888889,
    0.008333334,
    0.041666668,
    0.16666667,
    0.5,
    1.0,
    1.0,
];

const LOG2_E_F32: f32 = std::f32::consts::LOG2_E;
const LN2_HI_F32: f32 = 6.931_457_5e-1;
const LN2_LO_F32: f32 = 1.428_606_8e-6;
const MAGIC_F32: f32 = 12582912.0;

const INV_FACT_F64: [f64; 14] = [
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

const LOG2_E_F64: f64 = std::f64::consts::LOG2_E;
const LN2_HI_F64: f64 = 6.931471803691238e-1;
const LN2_LO_F64: f64 = 1.908214929270588e-10;
const MAGIC_F64: f64 = 6755399441055744.0;

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

                let mut p = Simd::splat(NumCast::from(INV_FACT_F64[0]).unwrap());
                for x in INV_FACT_F64.iter().skip(1) {
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
            },
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

                let mut p = Simd::splat(NumCast::from(INV_FACT_F32[0]).unwrap());
                for x in INV_FACT_F32.iter().skip(1) {
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

                let mut p = Simd::splat(NumCast::from(INV_FACT_F64[0]).unwrap());
                for x in INV_FACT_F64.iter().skip(1) {
                    let x = Simd::splat(NumCast::from(*x).unwrap());
                    p = p.mul_add(r, x)
                }

                let add_bits = Simd::splat(NumCast::from(1023).unwrap());
                let kb: Simd<T::UintType, A> = (k + magic).raw_cast();
                let scale: Self = ((kb + add_bits) << 52).raw_cast();

                p * scale
            },
            SimdType::F32 => {
                let log2_e = Simd::splat(NumCast::from(LOG2_E_F32).unwrap());
                let ln2_hi = Simd::splat(NumCast::from(LN2_HI_F32).unwrap());
                let ln2_lo = Simd::splat(NumCast::from(LN2_LO_F32).unwrap());
                let magic = Simd::splat(NumCast::from(MAGIC_F32).unwrap());

                let k = (self * log2_e).round();

                let r = (-k).mul_add(ln2_hi, self);
                let r = (-k).mul_add(ln2_lo, r);

                let mut p = Simd::splat(NumCast::from(INV_FACT_F32[0]).unwrap());
                for x in INV_FACT_F32.iter().skip(1) {
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
}
