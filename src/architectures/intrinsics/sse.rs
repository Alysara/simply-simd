use std::arch::x86_64::*;
use std::mem::{transmute, transmute_copy};

use crate::architectures::interface::*;
use crate::architectures::macros::*;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct SseReg(pub __m128i);
impl SimdArch for SseReg {}
impl MaskArch for SseReg {}

impl SimdAdd for SseReg {
    #[inline(always)]
    unsafe fn f64_add(self, rhs: Self) -> Self {
        self_from_op!(_mm_add_pd, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_add(self, rhs: Self) -> Self {
        self_from_op!(_mm_add_ps, self, rhs)
    }
    #[inline(always)]
    unsafe fn i64_add(self, rhs: Self) -> Self {
        self_from_op!(_mm_add_epi64, self, rhs)
    }
    #[inline(always)]
    unsafe fn i32_add(self, rhs: Self) -> Self {
        self_from_op!(_mm_add_epi32, self, rhs)
    }
    #[inline(always)]
    unsafe fn i16_add(self, rhs: Self) -> Self {
        self_from_op!(_mm_add_epi16, self, rhs)
    }
    #[inline(always)]
    unsafe fn i8_add(self, rhs: Self) -> Self {
        self_from_op!(_mm_add_epi8, self, rhs)
    }
}

impl SimdSub for SseReg {
    #[inline(always)]
    unsafe fn f64_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm_sub_pd, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm_sub_ps, self, rhs)
    }
    #[inline(always)]
    unsafe fn i64_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm_sub_epi64, self, rhs)
    }
    #[inline(always)]
    unsafe fn i32_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm_sub_epi32, self, rhs)
    }
    #[inline(always)]
    unsafe fn i16_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm_sub_epi16, self, rhs)
    }
    #[inline(always)]
    unsafe fn i8_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm_sub_epi8, self, rhs)
    }
}

impl SimdMul for SseReg {
    #[inline(always)]
    unsafe fn f64_mul(self, rhs: Self) -> Self {
        self_from_op!(_mm_mul_pd, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_mul(self, rhs: Self) -> Self {
        self_from_op!(_mm_mul_ps, self, rhs)
    }
    #[inline(always)]
    unsafe fn i32_mul(self, rhs: Self) -> Self {
        self_from_op!(_mm_mullo_epi32, self, rhs)
    }
    #[inline(always)]
    unsafe fn i16_mul(self, rhs: Self) -> Self {
        self_from_op!(_mm_mullo_epi16, self, rhs)
    }
}

impl SimdDiv for SseReg {
    #[inline(always)]
    unsafe fn f64_div(self, rhs: Self) -> Self {
        self_from_op!(_mm_div_pd, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_div(self, rhs: Self) -> Self {
        self_from_op!(_mm_div_ps, self, rhs)
    }
}

impl SimdBitwise for SseReg {
    #[inline(always)]
    unsafe fn and(self, rhs: Self) -> Self {
        self_from_op!(_mm_and_si128, self, rhs)
    }
    #[inline(always)]
    unsafe fn or(self, rhs: Self) -> Self {
        self_from_op!(_mm_or_si128, self, rhs)
    }
    #[inline(always)]
    unsafe fn xor(self, rhs: Self) -> Self {
        self_from_op!(_mm_xor_si128, self, rhs)
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        unsafe { Self(self.xor(Self::splat_32(!0)).0) }
    }
    #[inline(always)]
    unsafe fn and_not(self, rhs: Self) -> Self {
        self_from_op!(_mm_andnot_si128, rhs, self)
    }
}

macro_rules! map_scalar_shift {
    ($func:ident, $lanes:expr, $elem:ty, $uelem:ty, $op:tt) => {
        #[inline(always)]
        unsafe fn $func(self, rhs: Self) -> Self {
            unsafe {
                let mut a = [<$elem>::default(); $lanes];
                let mut b = [<$elem>::default(); $lanes];
                _mm_storeu_si128(a.as_mut_ptr() as *mut __m128i, self.0);
                _mm_storeu_si128(b.as_mut_ptr() as *mut __m128i, rhs.0);
                let width = (std::mem::size_of::<$elem>() * 8) as u32;
                let mut out = [<$elem>::default(); $lanes];
                for i in 0..$lanes {
                    let shift = b[i] as $uelem as u32;
                    out[i] = map_scalar_shift!(@apply $op, a[i], shift, width, $elem, $uelem);
                }
                Self(_mm_loadu_si128(out.as_ptr() as *const __m128i))
            }
        }
    };
    (@apply sll, $val:expr, $shift:expr, $width:expr, $elem:ty, $uelem:ty) => {
        if $shift >= $width { 0 } else { (($val as $uelem) << $shift) as $elem }
    };
    (@apply srl, $val:expr, $shift:expr, $width:expr, $elem:ty, $uelem:ty) => {
        if $shift >= $width { 0 } else { (($val as $uelem) >> $shift) as $elem }
    };
    (@apply sra, $val:expr, $shift:expr, $width:expr, $elem:ty, $uelem:ty) => {
        $val >> $shift.min($width - 1)
    };
}

impl SimdShift for SseReg {
    map_x86_shift!(sll_64, _mm_sll_epi64);
    map_x86_shift!(srl_64, _mm_srl_epi64);
    map_x86_shift!(sll_32, _mm_sll_epi32);
    map_x86_shift!(srl_32, _mm_srl_epi32);
    map_x86_shift!(sra_32, _mm_sra_epi32);
    map_x86_shift!(sll_16, _mm_sll_epi16);
    map_x86_shift!(srl_16, _mm_srl_epi16);
    map_x86_shift!(sra_16, _mm_sra_epi16);
    map_intrinsic!(slli_64, _mm_slli_epi64, const N: i32, self);
    map_intrinsic!(srli_64, _mm_srli_epi64, const N: i32, self);
    map_intrinsic!(slli_32, _mm_slli_epi32, const N: i32, self);
    map_intrinsic!(srli_32, _mm_srli_epi32, const N: i32, self);
    map_x86_conversion_imm_shift!(srai_32, _mm_sra_epi32);
    map_intrinsic!(slli_16, _mm_slli_epi16, const N: i32, self);
    map_intrinsic!(srli_16, _mm_srli_epi16, const N: i32, self);
    map_x86_conversion_imm_shift!(srai_16, _mm_sra_epi16);
    map_scalar_shift!(sllv_64, 2, i64, u64, sll);
    map_scalar_shift!(srlv_64, 2, i64, u64, srl);
    map_scalar_shift!(srav_64, 2, i64, u64, sra);
    map_scalar_shift!(sllv_32, 4, i32, u32, sll);
    map_scalar_shift!(srlv_32, 4, i32, u32, srl);
    map_scalar_shift!(srav_32, 4, i32, u32, sra);
    map_scalar_shift!(sllv_16, 8, i16, u16, sll);
    map_scalar_shift!(srlv_16, 8, i16, u16, srl);
    map_scalar_shift!(srav_16, 8, i16, u16, sra);

    #[inline(always)]
    unsafe fn sra_64(self, shift: u32) -> Self {
        unsafe {
            let shift = shift.min(63);
            let sign_bits = Self::splat_64(i64::MIN as u64 >> shift);
            let srl = self.srl_64(shift);
            srl.xor(sign_bits).i64_sub(sign_bits)
        }
    }

    #[inline(always)]
    unsafe fn srai_64<const N: i32>(self) -> Self {
        unsafe {
            let shift = N.min(63);
            let sign_bits = Self::splat_64(i64::MIN as u64 >> shift);
            let srl = self.srli_64::<N>();
            srl.xor(sign_bits).i64_sub(sign_bits)
        }
    }
}

impl SimdLoad for SseReg {
    type MaskType = Self;

    #[inline(always)]
    unsafe fn load_aligned<T>(ptr: *const T) -> Self {
        self_from_op!(_mm_load_si128, ptr)
    }
    #[inline(always)]
    unsafe fn load_unaligned<T>(ptr: *const T) -> Self {
        self_from_op!(_mm_loadu_si128, ptr)
    }

    #[inline(always)]
    unsafe fn masked_load_64<T>(ptr: *const T, mask: Self::MaskType) -> Self {
        unsafe {
            let mut m = [0i64; 2];
            _mm_storeu_si128(m.as_mut_ptr() as *mut __m128i, mask.0);
            let src = ptr as *const i64;
            let mut out = [0i64; 2];
            for i in 0..2 {
                if (m[i] as u64) & (1 << 63) != 0 {
                    out[i] = *src.add(i);
                }
            }
            Self(_mm_loadu_si128(out.as_ptr() as *const __m128i))
        }
    }

    #[inline(always)]
    unsafe fn masked_load_32<T>(ptr: *const T, mask: Self::MaskType) -> Self {
        unsafe {
            let mut m = [0i32; 4];
            _mm_storeu_si128(m.as_mut_ptr() as *mut __m128i, mask.0);
            let src = ptr as *const i32;
            let mut out = [0i32; 4];
            for i in 0..4 {
                if (m[i] as u32) & (1 << 31) != 0 {
                    out[i] = *src.add(i);
                }
            }
            Self(_mm_loadu_si128(out.as_ptr() as *const __m128i))
        }
    }
}

impl SimdStore for SseReg {
    type MaskType = Self;

    #[inline(always)]
    unsafe fn store_aligned<T>(self, ptr: *mut T) {
        execute_intrinsic!(_mm_store_si128, ptr, self);
    }
    #[inline(always)]
    unsafe fn store_unaligned<T>(self, ptr: *mut T) {
        execute_intrinsic!(_mm_storeu_si128, ptr, self);
    }

    #[inline(always)]
    unsafe fn masked_store_64<T>(self, ptr: *mut T, mask: Self::MaskType) {
        unsafe {
            let mut m = [0i64; 2];
            let mut v = [0i64; 2];
            _mm_storeu_si128(m.as_mut_ptr() as *mut __m128i, mask.0);
            _mm_storeu_si128(v.as_mut_ptr() as *mut __m128i, self.0);
            let dst = ptr as *mut i64;
            for i in 0..2 {
                if (m[i] as u64) & (1 << 63) != 0 {
                    *dst.add(i) = v[i];
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn masked_store_32<T>(self, ptr: *mut T, mask: Self::MaskType) {
        unsafe {
            let mut m = [0i32; 4];
            let mut v = [0i32; 4];
            _mm_storeu_si128(m.as_mut_ptr() as *mut __m128i, mask.0);
            _mm_storeu_si128(v.as_mut_ptr() as *mut __m128i, self.0);
            let dst = ptr as *mut i32;
            for i in 0..4 {
                if (m[i] as u32) & (1 << 31) != 0 {
                    *dst.add(i) = v[i];
                }
            }
        }
    }
}

impl SimdZero for SseReg {
    #[inline(always)]
    unsafe fn zero() -> Self {
        self_from_op!(_mm_setzero_si128,)
    }
}

impl SimdFloatCasts for SseReg {
    #[inline(always)]
    unsafe fn float_to_int_trunc(self) -> Self {
        self_from_op!(_mm_cvttps_epi32, self)
    }
    #[inline(always)]
    unsafe fn float_to_int_round(self) -> Self {
        self_from_op!(_mm_cvtps_epi32, self)
    }
}

impl SimdIntCasts for SseReg {
    #[inline(always)]
    unsafe fn int_to_float(self) -> Self {
        self_from_op!(_mm_cvtepi32_ps, self)
    }
}

impl SimdPermute for SseReg {
    map_intrinsic!(vshuffle_8, _mm_shuffle_epi8, self, rhs: Self);
    map_intrinsic!(shuffle_32, _mm_shuffle_epi32, const N: i32, self);
    map_intrinsic!(shuffle2_64, _mm_shuffle_pd, const N: i32, self, rhs: Self);
    map_intrinsic!(shuffle2_32, _mm_shuffle_ps, const N: i32, self, rhs: Self);

    #[inline(always)]
    unsafe fn vpermute_64(self, rhs: Self) -> Self {
        unsafe {
            let shifted = rhs.slli_64::<3>();

            #[rustfmt::skip]
            let shuf_indices = self_from_op!(_mm_setr_epi8,
                0, 0, 0, 0, 0, 0, 0, 0,
                8, 8, 8, 8, 8, 8, 8, 8
            );
            #[rustfmt::skip]
            let shuf_offsets = self_from_op!(_mm_setr_epi8,
                0, 1, 2, 3, 4, 5, 6, 7,
                0, 1, 2, 3, 4, 5, 6, 7
            );

            let shuffed = shifted.vshuffle_8(shuf_indices);
            let indices = shuffed.i8_add(shuf_offsets);

            self.vshuffle_8(indices)
        }
    }
    #[inline(always)]
    unsafe fn vpermute_32(self, rhs: Self) -> Self {
        unsafe {
            let shifted = rhs.slli_32::<2>();

            #[rustfmt::skip]
            let shuf_indices = self_from_op!(_mm_setr_epi8,
                0, 0, 0, 0, 4, 4, 4, 4,
                8, 8, 8, 8, 12, 12, 12, 12
            );
            #[rustfmt::skip]
            let shuf_offsets = self_from_op!(_mm_setr_epi8,
                0, 1, 2, 3, 0, 1, 2, 3,
                0, 1, 2, 3, 0, 1, 2, 3
            );

            let shuffed = shifted.vshuffle_8(shuf_indices);
            let indices = shuffed.i8_add(shuf_offsets);

            self.vshuffle_8(indices)
        }
    }
    #[inline(always)]
    unsafe fn vshuffle_16(self, rhs: Self) -> Self {
        unsafe {
            let shifted = rhs.slli_16::<1>();

            #[rustfmt::skip]
            let shuf_indices = self_from_op!(_mm_setr_epi8,
                0, 0, 2, 2, 4, 4, 6, 6,
                8, 8, 10, 10, 12, 12, 14, 14
            );
            #[rustfmt::skip]
            let shuf_offsets = self_from_op!(_mm_setr_epi8,
                0, 1, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 0, 1
            );

            let shuffed = shifted.vshuffle_8(shuf_indices);
            let indices = shuffed.i8_add(shuf_offsets);

            self.vshuffle_8(indices)
        }
    }

    #[inline(always)]
    unsafe fn permute_128<const N: i32>(self) -> Self {
        self
    }

    #[inline(always)]
    unsafe fn shuffle_64<const N: i32>(self) -> Self {
        unsafe { self.shuffle2_64::<N>(self) }
    }

    #[inline(always)]
    unsafe fn shuffle_16<const N: i32>(self) -> Self {
        let first_shuf = self_from_const_op!(_mm_shufflelo_epi16, N, self);
        self_from_const_op!(_mm_shufflehi_epi16, N, first_shuf)
    }

    #[inline(always)]
    unsafe fn permute2_half<const N: i32>(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_64::<N>(rhs) }
    }

    #[inline(always)]
    unsafe fn shuffle2_128<const N: i32>(self, _rhs: Self) -> Self {
        self
    }
}

impl SimdVariableBlend for SseReg {
    type VecType = Self;
    #[inline(always)]
    unsafe fn vblend_64(self, true_values: Self::VecType, false_values: Self::VecType) -> Self {
        self_from_op!(_mm_blendv_pd, false_values, true_values, self)
    }
    #[inline(always)]
    unsafe fn vblend_32(self, true_values: Self::VecType, false_values: Self::VecType) -> Self {
        self_from_op!(_mm_blendv_ps, false_values, true_values, self)
    }
    #[inline(always)]
    unsafe fn vblend_16(self, true_values: Self::VecType, false_values: Self::VecType) -> Self {
        self_from_op!(_mm_blendv_epi8, false_values, true_values, self)
    }
    #[inline(always)]
    unsafe fn vblend_8(self, true_values: Self::VecType, false_values: Self::VecType) -> Self {
        self_from_op!(_mm_blendv_epi8, false_values, true_values, self)
    }
}

impl SimdImmediateBlend for SseReg {
    #[inline(always)]
    unsafe fn blend_64<const N: i32>(self, false_values: Self) -> Self {
        self_from_const_op!(_mm_blend_pd, N, false_values, self)
    }
    #[inline(always)]
    unsafe fn blend_32<const N: i32>(self, false_values: Self) -> Self {
        self_from_const_op!(_mm_blend_ps, N, false_values, self)
    }
    #[inline(always)]
    unsafe fn blend_16<const N: i32>(self, false_values: Self) -> Self {
        self_from_const_op!(_mm_blend_epi16, N, false_values, self)
    }
}

impl SimdInterleave for SseReg {
    #[inline(always)]
    unsafe fn interleave_lower_128(self, _rhs: Self) -> Self {
        self
    }

    map_intrinsic!(interleave_lower_64, _mm_unpacklo_epi64, self, rhs: Self);
    map_intrinsic!(interleave_lower_32, _mm_unpacklo_epi32, self, rhs: Self);
    map_intrinsic!(interleave_lower_16, _mm_unpacklo_epi16, self, rhs: Self);
    map_intrinsic!(interleave_lower_8, _mm_unpacklo_epi8, self, rhs: Self);

    #[inline(always)]
    unsafe fn interleave_upper_128(self, _rhs: Self) -> Self {
        self
    }

    map_intrinsic!(interleave_upper_64, _mm_unpackhi_epi64, self, rhs: Self);
    map_intrinsic!(interleave_upper_32, _mm_unpackhi_epi32, self, rhs: Self);
    map_intrinsic!(interleave_upper_16, _mm_unpackhi_epi16, self, rhs: Self);
    map_intrinsic!(interleave_upper_8, _mm_unpackhi_epi8, self, rhs: Self);

    #[inline(always)]
    unsafe fn interleave_even_128(self, _rhs: Self) -> Self {
        self
    }
    #[inline(always)]
    unsafe fn interleave_even_64(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_64::<0>(rhs) }
    }
    #[inline(always)]
    unsafe fn interleave_even_32(self, rhs: Self) -> Self {
        unsafe { rhs.slli_64::<32>().blend_32::<0xA>(self) }
    }
    #[inline(always)]
    unsafe fn interleave_even_16(self, rhs: Self) -> Self {
        unsafe { rhs.slli_32::<16>().blend_16::<0xAA>(self) }
    }
    #[inline(always)]
    unsafe fn interleave_even_8(self, rhs: Self) -> Self {
        unsafe { Self::splat_16(0x00FF).vblend_8(self, rhs.slli_16::<8>()) }
    }
    #[inline(always)]
    unsafe fn interleave_odd_128(self, _rhs: Self) -> Self {
        self
    }
    #[inline(always)]
    unsafe fn interleave_odd_64(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_64::<0b11>(rhs) }
    }
    #[inline(always)]
    unsafe fn interleave_odd_32(self, rhs: Self) -> Self {
        unsafe { self.srli_64::<32>().blend_32::<0x5>(rhs) }
    }
    #[inline(always)]
    unsafe fn interleave_odd_16(self, rhs: Self) -> Self {
        unsafe { self.srli_32::<16>().blend_16::<0x55>(rhs) }
    }
    #[inline(always)]
    unsafe fn interleave_odd_8(self, rhs: Self) -> Self {
        unsafe { Self::splat_16(0x00FF).vblend_8(self.srli_16::<8>(), rhs) }
    }
    #[inline(always)]
    unsafe fn deinterleave_even_128(self, _rhs: Self) -> Self {
        self
    }
    #[inline(always)]
    unsafe fn deinterleave_even_64(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_64::<0>(rhs) }
    }

    #[inline(always)]
    unsafe fn deinterleave_even_32(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_32::<0x88>(rhs) }
    }

    #[inline(always)]
    unsafe fn deinterleave_even_16(self, rhs: Self) -> Self {
        unsafe {
            let mask = Self::splat_32(0x0000FFFF);
            let low = self.and(mask);
            let high = rhs.and(mask);
            self_from_op!(_mm_packus_epi32, low, high)
        }
    }

    #[inline(always)]
    unsafe fn deinterleave_even_8(self, rhs: Self) -> Self {
        unsafe {
            let mask = Self::splat_16(0x00FF);
            let low = self.and(mask);
            let high = rhs.and(mask);
            self_from_op!(_mm_packus_epi16, low, high)
        }
    }

    #[inline(always)]
    unsafe fn deinterleave_odd_128(self, _rhs: Self) -> Self {
        self
    }

    #[inline(always)]
    unsafe fn deinterleave_odd_64(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_64::<0xF>(rhs) }
    }

    #[inline(always)]
    unsafe fn deinterleave_odd_32(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_32::<0xDD>(rhs) }
    }

    #[inline(always)]
    unsafe fn deinterleave_odd_16(self, rhs: Self) -> Self {
        unsafe {
            let low = self.srli_32::<16>();
            let high = rhs.srli_32::<16>();
            self_from_op!(_mm_packus_epi32, low, high)
        }
    }

    #[inline(always)]
    unsafe fn deinterleave_odd_8(self, rhs: Self) -> Self {
        unsafe {
            let low = self.srli_16::<8>();
            let high = rhs.srli_16::<8>();
            self_from_op!(_mm_packus_epi16, low, high)
        }
    }
}

impl SimdMulAdd for SseReg {
    #[inline(always)]
    unsafe fn mul_add_f64(self, mult: Self, add: Self) -> Self {
        self_from_op!(_mm_add_pd, self_from_op!(_mm_mul_pd, self, mult), add)
    }
    #[inline(always)]
    unsafe fn mul_sub_f64(self, mult: Self, sub: Self) -> Self {
        self_from_op!(_mm_sub_pd, self_from_op!(_mm_mul_pd, self, mult), sub)
    }
    #[inline(always)]
    unsafe fn negated_mul_add_f64(self, mult: Self, add: Self) -> Self {
        self_from_op!(_mm_sub_pd, add, self_from_op!(_mm_mul_pd, self, mult))
    }
    #[inline(always)]
    unsafe fn negated_mul_sub_f64(self, mult: Self, sub: Self) -> Self {
        let neg = self_from_op!(_mm_mul_pd, self, mult);
        self_from_op!(
            _mm_sub_pd,
            self_from_op!(_mm_xor_pd, neg, Self::splat_64(-0.0f64).0),
            sub
        )
    }

    #[inline(always)]
    unsafe fn mul_add_f32(self, mult: Self, add: Self) -> Self {
        self_from_op!(_mm_add_ps, self_from_op!(_mm_mul_ps, self, mult), add)
    }
    #[inline(always)]
    unsafe fn mul_sub_f32(self, mult: Self, sub: Self) -> Self {
        self_from_op!(_mm_sub_ps, self_from_op!(_mm_mul_ps, self, mult), sub)
    }
    #[inline(always)]
    unsafe fn negated_mul_add_f32(self, mult: Self, add: Self) -> Self {
        self_from_op!(_mm_sub_ps, add, self_from_op!(_mm_mul_ps, self, mult))
    }
    #[inline(always)]
    unsafe fn negated_mul_sub_f32(self, mult: Self, sub: Self) -> Self {
        let neg = self_from_op!(_mm_mul_ps, self, mult);
        self_from_op!(
            _mm_sub_ps,
            self_from_op!(_mm_xor_ps, neg, Self::splat_32(-0.0f32).0),
            sub
        )
    }
}

impl SimdRound for SseReg {
    #[inline(always)]
    unsafe fn round_f64(self) -> Self {
        self_from_const_op!(
            _mm_round_pd,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            self
        )
    }
    #[inline(always)]
    unsafe fn round_f32(self) -> Self {
        self_from_const_op!(
            _mm_round_ps,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            self
        )
    }
    #[inline(always)]
    unsafe fn floor_f64(self) -> Self {
        self_from_const_op!(
            _mm_round_pd,
            _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC,
            self
        )
    }
    #[inline(always)]
    unsafe fn floor_f32(self) -> Self {
        self_from_const_op!(
            _mm_round_ps,
            _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC,
            self
        )
    }
    #[inline(always)]
    unsafe fn ceil_f64(self) -> Self {
        self_from_const_op!(
            _mm_round_pd,
            _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC,
            self
        )
    }
    #[inline(always)]
    unsafe fn ceil_f32(self) -> Self {
        self_from_const_op!(
            _mm_round_ps,
            _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC,
            self
        )
    }
}

impl SimdPartialOrd for SseReg {
    type MaskType = Self;
    map_intrinsic!(cmp_f64_eq, _mm_cmpeq_pd, self, rhs: Self);
    map_intrinsic!(cmp_f64_lt, _mm_cmplt_pd, self, rhs: Self);
    map_intrinsic!(cmp_f64_le, _mm_cmple_pd, self, rhs: Self);
    map_intrinsic!(cmp_f64_gt, _mm_cmpgt_pd, self, rhs: Self);
    map_intrinsic!(cmp_f64_ge, _mm_cmpge_pd, self, rhs: Self);
    map_intrinsic!(cmp_f64_ne, _mm_cmpneq_pd, self, rhs: Self);
    map_intrinsic!(cmp_f32_eq, _mm_cmpeq_ps, self, rhs: Self);
    map_intrinsic!(cmp_f32_lt, _mm_cmplt_ps, self, rhs: Self);
    map_intrinsic!(cmp_f32_le, _mm_cmple_ps, self, rhs: Self);
    map_intrinsic!(cmp_f32_gt, _mm_cmpgt_ps, self, rhs: Self);
    map_intrinsic!(cmp_f32_ge, _mm_cmpge_ps, self, rhs: Self);
    map_intrinsic!(cmp_f32_ne, _mm_cmpneq_ps, self, rhs: Self);
    map_intrinsic!(cmp_i64_eq, _mm_cmpeq_epi64, self, rhs: Self);
    map_intrinsic!(cmp_i64_gt, _mm_cmpgt_epi64, self, rhs: Self);
    map_intrinsic!(cmp_i32_eq, _mm_cmpeq_epi32, self, rhs: Self);
    map_intrinsic!(cmp_i32_gt, _mm_cmpgt_epi32, self, rhs: Self);
    map_intrinsic!(cmp_i16_eq, _mm_cmpeq_epi16, self, rhs: Self);
    map_intrinsic!(cmp_i16_gt, _mm_cmpgt_epi16, self, rhs: Self);
    map_intrinsic!(cmp_i8_eq, _mm_cmpeq_epi8, self, rhs: Self);
    map_intrinsic!(cmp_i8_gt, _mm_cmpgt_epi8, self, rhs: Self);
    map_intrinsic!(max_f64, _mm_max_pd, self, rhs: Self);
    map_intrinsic!(min_f64, _mm_min_pd, self, rhs: Self);
    map_intrinsic!(max_f32, _mm_max_ps, self, rhs: Self);
    map_intrinsic!(min_f32, _mm_min_ps, self, rhs: Self);
    map_intrinsic!(max_i32, _mm_max_epi32, self, rhs: Self);
    map_intrinsic!(min_i32, _mm_min_epi32, self, rhs: Self);
    map_intrinsic!(max_i16, _mm_max_epi16, self, rhs: Self);
    map_intrinsic!(min_i16, _mm_min_epi16, self, rhs: Self);
    map_intrinsic!(max_i8, _mm_max_epi8, self, rhs: Self);
    map_intrinsic!(min_i8, _mm_min_epi8, self, rhs: Self);
    map_intrinsic!(max_u32, _mm_max_epu32, self, rhs: Self);
    map_intrinsic!(min_u32, _mm_min_epu32, self, rhs: Self);
    map_intrinsic!(max_u16, _mm_max_epu16, self, rhs: Self);
    map_intrinsic!(min_u16, _mm_min_epu16, self, rhs: Self);
    map_intrinsic!(max_u8, _mm_max_epu8, self, rhs: Self);
    map_intrinsic!(min_u8, _mm_min_epu8, self, rhs: Self);

    #[inline(always)]
    unsafe fn cmp_u64_gt(self, rhs: Self) -> Self {
        unsafe {
            let bias = Self::splat_64(1u64 << 63);
            self.xor(bias).cmp_i64_gt(rhs.xor(bias))
        }
    }

    #[inline(always)]
    unsafe fn cmp_u32_gt(self, rhs: Self) -> Self {
        unsafe {
            let bias = Self::splat_32(1u32 << 31);
            self.xor(bias).cmp_i32_gt(rhs.xor(bias))
        }
    }

    #[inline(always)]
    unsafe fn cmp_u16_gt(self, rhs: Self) -> Self {
        unsafe {
            let bias = Self::splat_16(1u16 << 15);
            self.xor(bias).cmp_i16_gt(rhs.xor(bias))
        }
    }

    #[inline(always)]
    unsafe fn cmp_u8_gt(self, rhs: Self) -> Self {
        unsafe {
            let bias = Self::splat_8(1u8 << 7);
            self.xor(bias).cmp_i8_gt(rhs.xor(bias))
        }
    }

    #[inline(always)]
    unsafe fn max_i64(self, rhs: Self) -> Self {
        unsafe { self.cmp_i64_gt(rhs).vblend_8(self, rhs) }
    }
    #[inline(always)]
    unsafe fn min_i64(self, rhs: Self) -> Self {
        unsafe { self.cmp_i64_gt(rhs).vblend_8(rhs, self) }
    }
    #[inline(always)]
    unsafe fn max_u64(self, rhs: Self) -> Self {
        unsafe { self.cmp_u64_gt(rhs).vblend_8(self, rhs) }
    }
    #[inline(always)]
    unsafe fn min_u64(self, rhs: Self) -> Self {
        unsafe { self.cmp_u64_gt(rhs).vblend_8(rhs, self) }
    }
}

// TODO: Make a custom trait for handling this transmutation into i*.
impl SimdSplat for SseReg {
    #[inline(always)]
    unsafe fn splat_64<T>(val: T) -> Self {
        self_from_op!(_mm_set1_epi64x, val)
    }
    #[inline(always)]
    unsafe fn splat_32<T>(val: T) -> Self {
        self_from_op!(_mm_set1_epi32, val)
    }
    #[inline(always)]
    unsafe fn splat_16<T>(val: T) -> Self {
        self_from_op!(_mm_set1_epi16, val)
    }
    #[inline(always)]
    unsafe fn splat_8<T>(val: T) -> Self {
        self_from_op!(_mm_set1_epi8, val)
    }
}

impl SimdGather for SseReg {
    #[inline(always)]
    unsafe fn gather_32_from_32<T, const B: i32>(self, ptr: *const T) -> Self {
        unsafe {
            let mut idx = [0i32; 4];
            _mm_storeu_si128(idx.as_mut_ptr() as *mut __m128i, self.0);
            let base = ptr as *const u8;
            let mut out = [0i32; 4];
            for i in 0..4 {
                let byte_offset = (idx[i] as isize) * (B as isize);
                out[i] = *(base.offset(byte_offset) as *const i32);
            }
            Self(_mm_loadu_si128(out.as_ptr() as *const __m128i))
        }
    }

    #[inline(always)]
    unsafe fn gather_64_from_64<T, const B: i32>(self, ptr: *const T) -> Self {
        unsafe {
            let mut idx = [0i64; 2];
            _mm_storeu_si128(idx.as_mut_ptr() as *mut __m128i, self.0);
            let base = ptr as *const u8;
            let mut out = [0i64; 2];
            for i in 0..2 {
                let byte_offset = (idx[i] as isize) * (B as isize);
                out[i] = *(base.offset(byte_offset) as *const i64);
            }
            Self(_mm_loadu_si128(out.as_ptr() as *const __m128i))
        }
    }
}

impl SimdSqrt for SseReg {
    #[inline(always)]
    unsafe fn sqrt_f64(self) -> Self {
        self_from_op!(_mm_sqrt_pd, self)
    }
    #[inline(always)]
    unsafe fn sqrt_f32(self) -> Self {
        self_from_op!(_mm_sqrt_ps, self)
    }
    #[inline(always)]
    unsafe fn rsqrt_f32(self) -> Self {
        self_from_op!(_mm_rsqrt_ps, self)
    }
}

impl SimdAllBits for SseReg {
    #[inline(always)]
    unsafe fn all_zero(self) -> bool {
        execute_intrinsic!(_mm_testz_si128, self, self) != 0
    }
}

impl SimdNegate for SseReg {
    #[inline(always)]
    unsafe fn negate_f64(self) -> Self {
        unsafe { Self::splat_64(-0.0f64).xor(self) }
    }
    #[inline(always)]
    unsafe fn negate_f32(self) -> Self {
        unsafe { Self::splat_32(-0.0f64).xor(self) }
    }
}

// TODO: THIS IS BACKWARDS COMPARED TO NON-BLOCKED, figure out why and fix.
impl SimdBlockShift for SseReg {
    #[inline(always)]
    unsafe fn block_left_byte_shift<const N: i32>(self) -> Self {
        self_from_const_op!(_mm_bslli_si128, N, self)
    }
    #[inline(always)]
    unsafe fn block_right_byte_shift<const N: i32>(self) -> Self {
        self_from_const_op!(_mm_bsrli_si128, N, self)
    }
}

impl SimdMaskBitConversion for SseReg {
    #[inline(always)]
    unsafe fn to_bits_64(self) -> u64 {
        execute_intrinsic!(_mm_movemask_pd, self) as u64
    }
    #[inline(always)]
    unsafe fn to_bits_32(self) -> u64 {
        execute_intrinsic!(_mm_movemask_ps, self) as u64
    }
    #[inline(always)]
    unsafe fn to_bits_8(self) -> u64 {
        execute_intrinsic!(_mm_movemask_epi8, self) as u64
    }
    #[inline(always)]
    unsafe fn from_bits_64(bitmask: u64) -> Self {
        let mask = self_from_op!(_mm_set_epi64x, 1, 2);
        let bits = self_from_op!(_mm_set1_epi64x, bitmask as i64);
        self_from_op!(_mm_cmpgt_epi64, bits.and(mask), Self::zero()) //TODO: only sse4.2?
    }
    #[inline(always)]
    unsafe fn from_bits_32(bitmask: u64) -> Self {
        let mask = self_from_op!(_mm_set_epi32, 1, 2, 4, 8);
        let bits = self_from_op!(_mm_set1_epi32, bitmask as u32);
        self_from_op!(_mm_cmpgt_epi32, bits.and(mask), Self::zero())
    }
    #[inline(always)]
    unsafe fn from_bits_16(bitmask: u64) -> Self {
        #[rustfmt::skip]
        let mask = self_from_op!(_mm_set_epi16, 1, 2, 4, 8, 16, 32, 64, 128);
        let bits = self_from_op!(_mm_set1_epi16, bitmask as i16);
        unsafe { self_from_op!(_mm_cmpeq_epi16, bits.and(mask), Self::zero()).not() }
    }
    #[inline(always)]
    unsafe fn from_bits_8(bitmask: u64) -> Self {
        let b1 = bitmask as i8;
        let b2 = (bitmask >> 8) as i8;
        #[rustfmt::skip]
        let mask = self_from_op!(_mm_set_epi8,
            1, 2, 4, 8, 16, 32, 64, -128,
            1, 2, 4, 8, 16, 32, 64, -128
        );
        #[rustfmt::skip]
        let bits = self_from_op!(_mm_set_epi8,
            b1, b1, b1, b1, b1, b1, b1, b1,
            b2, b2, b2, b2, b2, b2, b2, b2
        );
        unsafe { self_from_op!(_mm_cmpeq_epi8, bits.and(mask), Self::zero()).not() }
    }
}

impl SimdLaneShift for SseReg {
    #[inline(always)]
    unsafe fn left_lane_shift_32(self, n: u32) -> Self {
        match n {
            0 => self,
            1 => self_from_const_op!(_mm_bsrli_si128, 4, self),
            2 => self_from_const_op!(_mm_bsrli_si128, 8, self),
            3 => self_from_const_op!(_mm_bsrli_si128, 12, self),
            _ => unsafe { Self::zero() },
        }
    }
    #[inline(always)]
    unsafe fn right_lane_shift_32(self, n: u32) -> Self {
        match n {
            0 => self,
            1 => self_from_const_op!(_mm_bslli_si128, 4, self),
            2 => self_from_const_op!(_mm_bslli_si128, 8, self),
            3 => self_from_const_op!(_mm_bslli_si128, 12, self),
            _ => unsafe { Self::zero() },
        }
    }
}
