use std::arch::x86_64::*;
use std::mem::{transmute, transmute_copy};

use crate::architectures::interface::*;
use crate::architectures::macros::*;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Avx512Reg(pub __m512i);
impl SimdArch for Avx512Reg {}

#[derive(Copy, Clone)]
pub struct Avx512Mask(pub __mmask64);
impl MaskArch for Avx512Mask {}

impl SimdAdd for Avx512Reg {
    #[inline(always)]
    unsafe fn f64_add(self, rhs: Self) -> Self {
        self_from_op!(_mm512_add_pd, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_add(self, rhs: Self) -> Self {
        self_from_op!(_mm512_add_ps, self, rhs)
    }
    #[inline(always)]
    unsafe fn i64_add(self, rhs: Self) -> Self {
        self_from_op!(_mm512_add_epi64, self, rhs)
    }
    #[inline(always)]
    unsafe fn i32_add(self, rhs: Self) -> Self {
        self_from_op!(_mm512_add_epi32, self, rhs)
    }
    #[inline(always)]
    unsafe fn i16_add(self, rhs: Self) -> Self {
        self_from_op!(_mm512_add_epi16, self, rhs)
    }
    #[inline(always)]
    unsafe fn i8_add(self, rhs: Self) -> Self {
        self_from_op!(_mm512_add_epi8, self, rhs)
    }
}

impl SimdSub for Avx512Reg {
    #[inline(always)]
    unsafe fn f64_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm512_sub_pd, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm512_sub_ps, self, rhs)
    }
    #[inline(always)]
    unsafe fn i64_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm512_sub_epi64, self, rhs)
    }
    #[inline(always)]
    unsafe fn i32_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm512_sub_epi32, self, rhs)
    }
    #[inline(always)]
    unsafe fn i16_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm512_sub_epi16, self, rhs)
    }
    #[inline(always)]
    unsafe fn i8_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm512_sub_epi8, self, rhs)
    }
}

impl SimdMul for Avx512Reg {
    #[inline(always)]
    unsafe fn f64_mul(self, rhs: Self) -> Self {
        self_from_op!(_mm512_mul_pd, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_mul(self, rhs: Self) -> Self {
        self_from_op!(_mm512_mul_ps, self, rhs)
    }
    #[inline(always)]
    unsafe fn i32_mul(self, rhs: Self) -> Self {
        self_from_op!(_mm512_mullo_epi32, self, rhs)
    }
    #[inline(always)]
    unsafe fn i16_mul(self, rhs: Self) -> Self {
        self_from_op!(_mm512_mullo_epi16, self, rhs)
    }
}

impl SimdDiv for Avx512Reg {
    #[inline(always)]
    unsafe fn f64_div(self, rhs: Self) -> Self {
        self_from_op!(_mm512_div_pd, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_div(self, rhs: Self) -> Self {
        self_from_op!(_mm512_div_ps, self, rhs)
    }
}

impl SimdBitwise for Avx512Reg {
    #[inline(always)]
    unsafe fn and(self, rhs: Self) -> Self {
        self_from_op!(_mm512_and_si512, self, rhs)
    }
    #[inline(always)]
    unsafe fn or(self, rhs: Self) -> Self {
        self_from_op!(_mm512_or_si512, self, rhs)
    }
    #[inline(always)]
    unsafe fn xor(self, rhs: Self) -> Self {
        self_from_op!(_mm512_xor_si512, self, rhs)
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        unsafe { Self(self.xor(Self::splat_32(!0u32)).0) }
    }
    #[inline(always)]
    unsafe fn and_not(self, rhs: Self) -> Self {
        self_from_op!(_mm512_andnot_si512, rhs, self)
    }
}

impl SimdShift for Avx512Reg {
    map_x86_shift!(sll_64, _mm512_sll_epi64);
    map_x86_shift!(srl_64, _mm512_srl_epi64);
    map_x86_shift!(sra_64, _mm512_sra_epi64);
    map_x86_shift!(sll_32, _mm512_sll_epi32);
    map_x86_shift!(srl_32, _mm512_srl_epi32);
    map_x86_shift!(sra_32, _mm512_sra_epi32);
    map_x86_shift!(sll_16, _mm512_sll_epi16);
    map_x86_shift!(srl_16, _mm512_srl_epi16);
    map_x86_shift!(sra_16, _mm512_sra_epi16);
    map_x86_conversion_imm_shift!(slli_64, _mm512_sll_epi64);
    map_x86_conversion_imm_shift!(srli_64, _mm512_srl_epi64);
    map_x86_conversion_imm_shift!(srai_64, _mm512_sra_epi64);
    map_x86_conversion_imm_shift!(slli_32, _mm512_sll_epi32);
    map_x86_conversion_imm_shift!(srli_32, _mm512_srl_epi32);
    map_x86_conversion_imm_shift!(srai_32, _mm512_sra_epi32);
    map_x86_conversion_imm_shift!(slli_16, _mm512_sll_epi16);
    map_x86_conversion_imm_shift!(srli_16, _mm512_srl_epi16);
    map_x86_conversion_imm_shift!(srai_16, _mm512_sra_epi16);
    map_intrinsic!(sllv_64, _mm512_sllv_epi64, self, rhs: Self);
    map_intrinsic!(srlv_64, _mm512_srlv_epi64, self, rhs: Self);
    map_intrinsic!(srav_64, _mm512_srav_epi64, self, rhs: Self);
    map_intrinsic!(sllv_32, _mm512_sllv_epi32, self, rhs: Self);
    map_intrinsic!(srlv_32, _mm512_srlv_epi32, self, rhs: Self);
    map_intrinsic!(srav_32, _mm512_srav_epi32, self, rhs: Self);
    map_intrinsic!(sllv_16, _mm512_sllv_epi16, self, rhs: Self);
    map_intrinsic!(srlv_16, _mm512_srlv_epi16, self, rhs: Self);
    map_intrinsic!(srav_16, _mm512_srav_epi16, self, rhs: Self);
}

impl SimdLoad for Avx512Reg {
    type MaskType = Avx512Mask;
    #[inline(always)]
    unsafe fn load_aligned<T>(ptr: *const T) -> Self {
        self_from_op!(_mm512_load_si512, ptr)
    }
    #[inline(always)]
    unsafe fn load_unaligned<T>(ptr: *const T) -> Self {
        self_from_op!(_mm512_loadu_si512, ptr)
    }
    #[inline(always)]
    unsafe fn masked_load_64<T>(ptr: *const T, mask: Self::MaskType) -> Self {
        self_from_op!(_mm512_mask_loadu_epi64, Self::zero(), mask, ptr)
    }
    #[inline(always)]
    unsafe fn masked_load_32<T>(ptr: *const T, mask: Self::MaskType) -> Self {
        self_from_op!(_mm512_mask_loadu_epi32, Self::zero(), mask, ptr)
    }
}

impl SimdStore for Avx512Reg {
    type MaskType = Avx512Mask;
    #[inline(always)]
    unsafe fn store_aligned<T>(self, ptr: *mut T) {
        execute_intrinsic!(_mm512_store_si512, ptr, self);
    }
    #[inline(always)]
    unsafe fn store_unaligned<T>(self, ptr: *mut T) {
        execute_intrinsic!(_mm512_storeu_si512, ptr, self);
    }
    #[inline(always)]
    unsafe fn masked_store_64<T>(self, ptr: *mut T, mask: Self::MaskType) {
        execute_intrinsic!(_mm512_mask_storeu_epi64, ptr, mask, self);
    }
    #[inline(always)]
    unsafe fn masked_store_32<T>(self, ptr: *mut T, mask: Self::MaskType) {
        execute_intrinsic!(_mm512_mask_storeu_epi32, ptr, mask, self);
    }
}

impl SimdZero for Avx512Reg {
    #[inline(always)]
    unsafe fn zero() -> Self {
        self_from_op!(_mm512_setzero_si512,)
    }
}

impl SimdFloatCasts for Avx512Reg {
    #[inline(always)]
    unsafe fn float_to_int_trunc(self) -> Self {
        self_from_op!(_mm512_cvttps_epi32, self)
    }
    #[inline(always)]
    unsafe fn float_to_int_round(self) -> Self {
        self_from_op!(_mm512_cvtps_epi32, self)
    }
}

impl SimdIntCasts for Avx512Reg {
    #[inline(always)]
    unsafe fn int_to_float(self) -> Self {
        self_from_op!(_mm512_cvtepi32_ps, self)
    }
}

impl SimdPermute for Avx512Reg {
    map_intrinsic!(vpermute_64, _mm512_permutexvar_epi64, self, rhs: Self);
    map_intrinsic!(vpermute_32, _mm512_permutexvar_epi32, self, rhs: Self);

    #[inline(always)]
    unsafe fn vshuffle_16(self, rhs: Self) -> Self {
        unsafe {
            let multiplier = Self::splat_16(0x0202u16);
            let broadcasted = self.i16_mul(multiplier);
            let hi_dword_offset = Self::splat_16(0x0100u16);
            let indices = broadcasted.i32_add(hi_dword_offset);
            rhs.vshuffle_8(indices)
        }
    }

    map_intrinsic!(vshuffle_8, _mm512_shuffle_epi8, self, rhs: Self);
    map_intrinsic!(permute_128, _mm512_shuffle_i32x4, const N: i32, self, self);

    #[inline(always)]
    unsafe fn shuffle_64<const N: i32>(self) -> Self {
        match N & 0b11 {
            0b00 => self_from_const_op!(_mm512_permute_pd, 0b00000000, self),
            0b01 => self_from_const_op!(_mm512_permute_pd, 0b01010101, self),
            0b10 => self_from_const_op!(_mm512_permute_pd, 0b10101010, self),
            0b11 => self_from_const_op!(_mm512_permute_pd, 0b11111111, self),
            _ => unreachable!(),
        }
    }
    map_intrinsic!(shuffle_32, _mm512_shuffle_epi32, const N: i32, self);

    #[inline(always)]
    unsafe fn shuffle_16<const N: i32>(self) -> Self {
        let first_shuf = self_from_const_op!(_mm512_shufflelo_epi16, N, self);
        self_from_const_op!(_mm512_shufflehi_epi16, N, first_shuf)
    }

    #[inline(always)]
    unsafe fn permute2_half<const N: i32>(self, rhs: Self) -> Self {
        match N & 0b11 {
            0b00 => self_from_const_op!(_mm512_shuffle_i32x4, 0x44, self, rhs),
            0b01 => self_from_const_op!(_mm512_shuffle_i32x4, 0x4E, self, rhs),
            0b10 => self_from_const_op!(_mm512_shuffle_i32x4, 0xE4, self, rhs),
            0b11 => self_from_const_op!(_mm512_shuffle_i32x4, 0xEE, self, rhs),
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    unsafe fn shuffle2_128<const N: i32>(self, rhs: Self) -> Self {
        unsafe {
            match N & 0b11 {
                0b10 => self.blend_64::<0xCC>(rhs),
                0b00 => self_from_const_op!(_mm512_mask_permutex_epi64, 0x44, self, 0xCCu8, rhs),
                0b11 => self_from_const_op!(_mm512_mask_permutex_epi64, 0xEE, rhs, 0x33u8, self),
                0b01 => {
                    let indices = _mm512_setr_epi64(2, 3, 8, 9, 6, 7, 12, 13);
                    Self(_mm512_permutex2var_epi64(self.0, indices, rhs.0))
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    unsafe fn shuffle2_64<const N: i32>(self, rhs: Self) -> Self {
        match N & 0b11 {
            0b00 => self_from_const_op!(_mm512_shuffle_pd, 0b00000000, self, rhs),
            0b01 => self_from_const_op!(_mm512_shuffle_pd, 0b01010101, self, rhs),
            0b10 => self_from_const_op!(_mm512_shuffle_pd, 0b10101010, self, rhs),
            0b11 => self_from_const_op!(_mm512_shuffle_pd, 0b11111111, self, rhs),
            _ => unreachable!(),
        }
    }

    map_intrinsic!(shuffle2_32, _mm512_shuffle_ps, const N: i32, self, rhs: Self);
}

impl SimdVariableBlend for Avx512Mask {
    type VecType = Avx512Reg;
    #[inline(always)]
    unsafe fn vblend_64(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> Self::VecType {
        unsafe {
            Avx512Reg(transmute(execute_intrinsic!(
                _mm512_mask_blend_pd,
                self,
                false_values,
                true_values
            )))
        }
    }
    #[inline(always)]
    unsafe fn vblend_32(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> Self::VecType {
        unsafe {
            Avx512Reg(transmute(execute_intrinsic!(
                _mm512_mask_blend_ps,
                self,
                false_values,
                true_values
            )))
        }
    }
    #[inline(always)]
    unsafe fn vblend_16(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> Self::VecType {
        unsafe {
            Avx512Reg(transmute(execute_intrinsic!(
                _mm512_mask_blend_epi16,
                self,
                false_values,
                true_values
            )))
        }
    }
    #[inline(always)]
    unsafe fn vblend_8(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> Self::VecType {
        unsafe {
            Avx512Reg(transmute(execute_intrinsic!(
                _mm512_mask_blend_epi8,
                self,
                false_values,
                true_values
            )))
        }
    }
}

impl SimdImmediateBlend for Avx512Reg {
    #[inline(always)]
    unsafe fn blend_64<const N: i32>(self, false_values: Self) -> Self {
        let mask = Avx512Mask((N as u64 & 0b11) * 0b01010101);
        unsafe { mask.vblend_64(self, false_values) }
    }
    #[inline(always)]
    unsafe fn blend_32<const N: i32>(self, false_values: Self) -> Self {
        let mask = Avx512Mask((N as u64 & 0xF) * 0x1111);
        unsafe { mask.vblend_32(self, false_values) }
    }
    #[inline(always)]
    unsafe fn blend_16<const N: i32>(self, false_values: Self) -> Self {
        let mask = Avx512Mask((N as u64 & 0xFF) * 0x01010101);
        unsafe { mask.vblend_16(self, false_values) }
    }
}

impl SimdInterleave for Avx512Reg {
    #[inline(always)]
    unsafe fn interleave_lower_128(self, rhs: Self) -> Self {
        let indices = self_from_op!(_mm512_setr_epi64, 0, 1, 8, 9, 2, 3, 10, 11);
        self_from_op!(_mm512_permutex2var_epi64, self, indices, rhs)
    }
    #[inline(always)]
    unsafe fn interleave_lower_64(self, rhs: Self) -> Self {
        unsafe {
            let indices = Self(_mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11));
            self_from_op!(_mm512_permutex2var_epi64, self, indices, rhs)
        }
    }
    #[inline(always)]
    unsafe fn interleave_lower_32(self, rhs: Self) -> Self {
        unsafe {
            #[rustfmt::skip]
            let indices = Self(_mm512_setr_epi32(
                0, 16, 1, 17, 2, 18, 3, 19,
                4, 20, 5, 21, 6, 22, 7, 23
            ));
            self_from_op!(_mm512_permutex2var_epi32, self, indices, rhs)
        }
    }
    #[inline(always)]
    unsafe fn interleave_lower_16(self, rhs: Self) -> Self {
        unsafe {
            let lo = self_from_op!(_mm512_unpacklo_epi16, self, rhs);
            let hi = self_from_op!(_mm512_unpackhi_epi16, self, rhs);
            lo.interleave_lower_128(hi)
        }
    }
    #[inline(always)]
    unsafe fn interleave_lower_8(self, rhs: Self) -> Self {
        unsafe {
            let lo = self_from_op!(_mm512_unpacklo_epi8, self, rhs);
            let hi = self_from_op!(_mm512_unpackhi_epi8, self, rhs);
            lo.interleave_lower_128(hi)
        }
    }
    #[inline(always)]
    unsafe fn interleave_upper_128(self, rhs: Self) -> Self {
        unsafe {
            let indices = Self(_mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15));
            self_from_op!(_mm512_permutex2var_epi64, self, indices, rhs)
        }
    }
    #[inline(always)]
    unsafe fn interleave_upper_64(self, rhs: Self) -> Self {
        unsafe {
            let indices = Self(_mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15));
            self_from_op!(_mm512_permutex2var_epi64, self, indices, rhs)
        }
    }
    #[inline(always)]
    unsafe fn interleave_upper_32(self, rhs: Self) -> Self {
        unsafe {
            #[rustfmt::skip]
            let indices = Self(_mm512_setr_epi32(
                8, 24, 9, 25, 10, 26, 11, 27,
                12, 28, 13, 29, 14, 30, 15, 31
            ));
            self_from_op!(_mm512_permutex2var_epi32, self, indices, rhs)
        }
    }
    #[inline(always)]
    unsafe fn interleave_upper_16(self, rhs: Self) -> Self {
        unsafe {
            let lo = self_from_op!(_mm512_unpacklo_epi16, self, rhs);
            let hi = self_from_op!(_mm512_unpackhi_epi16, self, rhs);
            lo.interleave_upper_128(hi)
        }
    }
    #[inline(always)]
    unsafe fn interleave_upper_8(self, rhs: Self) -> Self {
        unsafe {
            let lo = self_from_op!(_mm512_unpacklo_epi8, self, rhs);
            let hi = self_from_op!(_mm512_unpackhi_epi8, self, rhs);
            lo.interleave_upper_128(hi)
        }
    }
    #[inline(always)]
    unsafe fn interleave_even_128(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_128::<0>(rhs) }
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
        unsafe {
            let mask = Avx512Mask::from_bits_64(0x5555_5555_5555_5555);
            mask.vblend_8(self, rhs.slli_16::<8>())
        }
    }
    #[inline(always)]
    unsafe fn interleave_odd_128(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_128::<3>(rhs) }
    }
    #[inline(always)]
    unsafe fn interleave_odd_64(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_64::<3>(rhs) }
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
        unsafe {
            let mask = Avx512Mask::from_bits_64(0x5555_5555_5555_5555);
            mask.vblend_8(self.srli_16::<8>(), rhs)
        }
    }
    #[inline(always)]
    unsafe fn deinterleave_even_128(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_128::<0x6420>(rhs) }
    }

    #[inline(always)]
    unsafe fn deinterleave_even_64(self, rhs: Self) -> Self {
        unsafe {
            let shuf = self.shuffle2_64::<0>(rhs);
            let indices = Self(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7));
            shuf.vpermute_64(indices)
        }
    }

    #[inline(always)]
    unsafe fn deinterleave_even_32(self, rhs: Self) -> Self {
        unsafe {
            let shuf = self.shuffle2_32::<0x88>(rhs);
            let indices = Self(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7));
            shuf.vpermute_64(indices)
        }
    }

    #[inline(always)]
    unsafe fn deinterleave_even_16(self, rhs: Self) -> Self {
        unsafe {
            let mask = Self::splat_32(0x0000FFFFu32);
            let low = self.and(mask);
            let high = rhs.and(mask);
            let packed = self_from_op!(_mm512_packus_epi32, low, high);
            let indices = Self(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7));
            packed.vpermute_64(indices)
        }
    }

    #[inline(always)]
    unsafe fn deinterleave_even_8(self, rhs: Self) -> Self {
        unsafe {
            let mask = Self::splat_16(0x00FFu16);
            let low = self.and(mask);
            let high = rhs.and(mask);
            let packed = self_from_op!(_mm256_packus_epi16, low, high);
            let indices = Self(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7));
            packed.vpermute_64(indices)
        }
    }

    #[inline(always)]
    unsafe fn deinterleave_odd_128(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_128::<0x7531>(rhs) }
    }

    #[inline(always)]
    unsafe fn deinterleave_odd_64(self, rhs: Self) -> Self {
        unsafe {
            let shuf = self.shuffle2_64::<3>(rhs);
            let indices = Self(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7));
            shuf.vpermute_64(indices)
        }
    }

    #[inline(always)]
    unsafe fn deinterleave_odd_32(self, rhs: Self) -> Self {
        unsafe {
            let shuf = self.shuffle2_32::<0xDD>(rhs);
            let indices = Self(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7));
            shuf.vpermute_64(indices)
        }
    }

    #[inline(always)]
    unsafe fn deinterleave_odd_16(self, rhs: Self) -> Self {
        unsafe {
            let low = self.srli_32::<16>();
            let high = rhs.srli_32::<16>();
            let packed = self_from_op!(_mm256_packus_epi32, low, high);
            let indices = Self(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7));
            packed.vpermute_64(indices)
        }
    }

    #[inline(always)]
    unsafe fn deinterleave_odd_8(self, rhs: Self) -> Self {
        unsafe {
            let low = self.srli_16::<8>();
            let high = rhs.srli_16::<8>();
            let packed = self_from_op!(_mm256_packus_epi16, low, high);
            let indices = Self(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7));
            packed.vpermute_64(indices)
        }
    }
}

impl SimdMulAdd for Avx512Reg {
    #[inline(always)]
    unsafe fn mul_add_f64(self, mult: Self, add: Self) -> Self {
        self_from_op!(_mm512_fmadd_pd, self, mult, add)
    }
    #[inline(always)]
    unsafe fn mul_sub_f64(self, mult: Self, sub: Self) -> Self {
        self_from_op!(_mm512_fmsub_pd, self, mult, sub)
    }
    #[inline(always)]
    unsafe fn negated_mul_add_f64(self, mult: Self, add: Self) -> Self {
        self_from_op!(_mm512_fnmadd_pd, self, mult, add)
    }
    #[inline(always)]
    unsafe fn negated_mul_sub_f64(self, mult: Self, sub: Self) -> Self {
        self_from_op!(_mm512_fnmsub_pd, self, mult, sub)
    }
    #[inline(always)]
    unsafe fn mul_add_f32(self, mult: Self, add: Self) -> Self {
        self_from_op!(_mm512_fmadd_ps, self, mult, add)
    }
    #[inline(always)]
    unsafe fn mul_sub_f32(self, mult: Self, sub: Self) -> Self {
        self_from_op!(_mm512_fmsub_ps, self, mult, sub)
    }
    #[inline(always)]
    unsafe fn negated_mul_add_f32(self, mult: Self, add: Self) -> Self {
        self_from_op!(_mm512_fnmadd_ps, self, mult, add)
    }
    #[inline(always)]
    unsafe fn negated_mul_sub_f32(self, mult: Self, sub: Self) -> Self {
        self_from_op!(_mm512_fnmsub_ps, self, mult, sub)
    }
}

impl SimdRound for Avx512Reg {
    #[inline(always)]
    unsafe fn round_f64(self) -> Self {
        self_from_const_op!(
            _mm512_roundscale_pd,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            self
        )
    }
    #[inline(always)]
    unsafe fn round_f32(self) -> Self {
        self_from_const_op!(
            _mm512_roundscale_ps,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            self
        )
    }
    #[inline(always)]
    unsafe fn floor_f64(self) -> Self {
        self_from_const_op!(
            _mm512_roundscale_pd,
            _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC,
            self
        )
    }
    #[inline(always)]
    unsafe fn floor_f32(self) -> Self {
        self_from_const_op!(
            _mm512_roundscale_ps,
            _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC,
            self
        )
    }
    #[inline(always)]
    unsafe fn ceil_f64(self) -> Self {
        self_from_const_op!(
            _mm512_roundscale_pd,
            _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC,
            self
        )
    }
    #[inline(always)]
    unsafe fn ceil_f32(self) -> Self {
        self_from_const_op!(
            _mm512_roundscale_ps,
            _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC,
            self
        )
    }
}

macro_rules! map_fp_cmp {
    ($func:ident, $intrinsic:ident, $op:ident) => {
        #[inline(always)]
        unsafe fn $func(self, rhs: Self) -> Self::MaskType {
            Avx512Mask(execute_const_intrinsic!($intrinsic, $op, self, rhs) as u64)
        }
    };
}

macro_rules! map_int_cmp {
    ($func:ident, $intrinsic:ident) => {
        #[inline(always)]
        unsafe fn $func(self, rhs: Self) -> Self::MaskType {
            Avx512Mask(execute_intrinsic!($intrinsic, self, rhs) as u64)
        }
    };
}

impl SimdPartialOrd for Avx512Reg {
    type MaskType = Avx512Mask;
    map_fp_cmp!(cmp_f64_eq, _mm512_cmp_pd_mask, _CMP_EQ_OQ);
    map_fp_cmp!(cmp_f64_lt, _mm512_cmp_pd_mask, _CMP_LT_OQ);
    map_fp_cmp!(cmp_f64_le, _mm512_cmp_pd_mask, _CMP_LE_OQ);
    map_fp_cmp!(cmp_f64_gt, _mm512_cmp_pd_mask, _CMP_GT_OQ);
    map_fp_cmp!(cmp_f64_ge, _mm512_cmp_pd_mask, _CMP_GE_OQ);
    map_fp_cmp!(cmp_f64_ne, _mm512_cmp_pd_mask, _CMP_NEQ_OQ);
    map_fp_cmp!(cmp_f32_eq, _mm512_cmp_ps_mask, _CMP_EQ_OQ);
    map_fp_cmp!(cmp_f32_lt, _mm512_cmp_ps_mask, _CMP_LT_OQ);
    map_fp_cmp!(cmp_f32_le, _mm512_cmp_ps_mask, _CMP_LE_OQ);
    map_fp_cmp!(cmp_f32_gt, _mm512_cmp_ps_mask, _CMP_GT_OQ);
    map_fp_cmp!(cmp_f32_ge, _mm512_cmp_ps_mask, _CMP_GE_OQ);
    map_fp_cmp!(cmp_f32_ne, _mm512_cmp_ps_mask, _CMP_NEQ_OQ);
    map_int_cmp!(cmp_i64_eq, _mm512_cmpeq_epi64_mask);
    map_int_cmp!(cmp_i64_gt, _mm512_cmpgt_epi64_mask);
    map_int_cmp!(cmp_i32_eq, _mm512_cmpeq_epi32_mask);
    map_int_cmp!(cmp_i32_gt, _mm512_cmpgt_epi32_mask);
    map_int_cmp!(cmp_i16_eq, _mm512_cmpeq_epi16_mask);
    map_int_cmp!(cmp_i16_gt, _mm512_cmpgt_epi16_mask);
    map_int_cmp!(cmp_i8_eq, _mm512_cmpeq_epi8_mask);
    map_int_cmp!(cmp_i8_gt, _mm512_cmpgt_epi8_mask);
    map_int_cmp!(cmp_u64_gt, _mm512_cmpgt_epu64_mask);
    map_int_cmp!(cmp_u32_gt, _mm512_cmpgt_epu32_mask);
    map_int_cmp!(cmp_u16_gt, _mm512_cmpgt_epu16_mask);
    map_int_cmp!(cmp_u8_gt, _mm512_cmpgt_epu8_mask);
    map_intrinsic!(max_f64, _mm512_max_pd, self, rhs: Self);
    map_intrinsic!(min_f64, _mm512_max_pd, self, rhs: Self);
    map_intrinsic!(max_f32, _mm512_max_ps, self, rhs: Self);
    map_intrinsic!(min_f32, _mm512_max_ps, self, rhs: Self);
    map_intrinsic!(max_i64, _mm512_max_epi64, self, rhs: Self);
    map_intrinsic!(min_i64, _mm512_max_epi64, self, rhs: Self);
    map_intrinsic!(max_i32, _mm512_max_epi32, self, rhs: Self);
    map_intrinsic!(min_i32, _mm512_max_epi32, self, rhs: Self);
    map_intrinsic!(max_i16, _mm512_max_epi16, self, rhs: Self);
    map_intrinsic!(min_i16, _mm512_max_epi16, self, rhs: Self);
    map_intrinsic!(max_i8, _mm512_max_epi8, self, rhs: Self);
    map_intrinsic!(min_i8, _mm512_max_epi8, self, rhs: Self);
    map_intrinsic!(max_u64, _mm512_max_epu64, self, rhs: Self);
    map_intrinsic!(min_u64, _mm512_max_epu64, self, rhs: Self);
    map_intrinsic!(max_u32, _mm512_max_epu32, self, rhs: Self);
    map_intrinsic!(min_u32, _mm512_max_epu32, self, rhs: Self);
    map_intrinsic!(max_u16, _mm512_max_epu16, self, rhs: Self);
    map_intrinsic!(min_u16, _mm512_max_epu16, self, rhs: Self);
    map_intrinsic!(max_u8, _mm512_max_epu8, self, rhs: Self);
    map_intrinsic!(min_u8, _mm512_max_epu8, self, rhs: Self);
}

// TODO: Make a custom trait for handling this transmutation into i*.
impl SimdSplat for Avx512Reg {
    #[inline(always)]
    unsafe fn splat_64<T>(val: T) -> Self {
        self_from_op!(_mm512_set1_epi64, val)
    }
    #[inline(always)]
    unsafe fn splat_32<T>(val: T) -> Self {
        self_from_op!(_mm512_set1_epi32, val)
    }
    #[inline(always)]
    unsafe fn splat_16<T>(val: T) -> Self {
        self_from_op!(_mm512_set1_epi16, val)
    }
    #[inline(always)]
    unsafe fn splat_8<T>(val: T) -> Self {
        self_from_op!(_mm512_set1_epi8, val)
    }
}

impl SimdBitwise for Avx512Mask {
    #[inline(always)]
    unsafe fn and(self, rhs: Self) -> Self {
        self_from_op!(_kand_mask64, self, rhs)
    }
    #[inline(always)]
    unsafe fn or(self, rhs: Self) -> Self {
        self_from_op!(_kor_mask64, self, rhs)
    }
    #[inline(always)]
    unsafe fn xor(self, rhs: Self) -> Self {
        self_from_op!(_kxor_mask64, self, rhs)
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        self_from_op!(_knot_mask64, self)
    }
    #[inline(always)]
    unsafe fn and_not(self, rhs: Self) -> Self {
        self_from_op!(_kandn_mask64, rhs, self)
    }
}

impl SimdGather for Avx512Reg {
    #[inline(always)]
    unsafe fn gather_32_from_32<T, const B: i32>(self, ptr: *const T) -> Self {
        self_from_const_op!(_mm512_i32gather_epi32, B, self, ptr)
    }
    #[inline(always)]
    unsafe fn gather_64_from_64<T, const B: i32>(self, ptr: *const T) -> Self {
        self_from_const_op!(_mm512_i64gather_epi64, B, self, ptr)
    }
}

impl SimdSqrt for Avx512Reg {
    #[inline(always)]
    unsafe fn sqrt_f64(self) -> Self {
        self_from_op!(_mm512_sqrt_pd, self)
    }
    #[inline(always)]
    unsafe fn sqrt_f32(self) -> Self {
        self_from_op!(_mm512_sqrt_ps, self)
    }
    #[inline(always)]
    unsafe fn rsqrt_f32(self) -> Self {
        self_from_op!(_mm512_rsqrt14_ps, self)
    }
}

impl SimdAllBits for Avx512Mask {
    #[inline(always)]
    unsafe fn all_zero(self) -> bool {
        self.0 == 0
    }
}

impl SimdNegate for Avx512Reg {
    #[inline(always)]
    unsafe fn negate_f64(self) -> Self {
        unsafe { Self::splat_64(-0.0f64).xor(self) }
    }
    #[inline(always)]
    unsafe fn negate_f32(self) -> Self {
        unsafe { Self::splat_32(-0.0f64).xor(self) }
    }
}

impl SimdBlockShift for Avx512Reg {
    #[inline(always)]
    unsafe fn block_left_byte_shift<const N: i32>(self) -> Self {
        self_from_const_op!(_mm512_bslli_epi128, N, self)
    }
    #[inline(always)]
    unsafe fn block_right_byte_shift<const N: i32>(self) -> Self {
        self_from_const_op!(_mm512_bsrli_epi128, N, self)
    }
}

impl SimdMaskBitConversion for Avx512Mask {
    #[inline(always)]
    unsafe fn to_bits_64(self) -> u64 {
        self.0
    }
    #[inline(always)]
    unsafe fn to_bits_32(self) -> u64 {
        self.0
    }
    #[inline(always)]
    unsafe fn to_bits_8(self) -> u64 {
        self.0
    }
    #[inline(always)]
    unsafe fn from_bits_64(bitmask: u64) -> Self {
        unsafe { transmute(bitmask) }
    }
    #[inline(always)]
    unsafe fn from_bits_32(bitmask: u64) -> Self {
        unsafe { transmute(bitmask) }
    }
    #[inline(always)]
    unsafe fn from_bits_16(bitmask: u64) -> Self {
        unsafe { transmute(bitmask) }
    }
    #[inline(always)]
    unsafe fn from_bits_8(bitmask: u64) -> Self {
        unsafe { transmute(bitmask) }
    }
}

impl SimdLaneShift for Avx512Reg {
    #[inline(always)]
    unsafe fn right_lane_shift_32(self, n: u32) -> Self {
        match n {
            0 => self,
            1 => Self(unsafe { _mm512_alignr_epi32(self.0, _mm512_setzero_si512(), 15) }),
            2 => Self(unsafe { _mm512_alignr_epi32(self.0, _mm512_setzero_si512(), 14) }),
            3 => Self(unsafe { _mm512_alignr_epi32(self.0, _mm512_setzero_si512(), 13) }),
            4 => Self(unsafe { _mm512_alignr_epi32(self.0, _mm512_setzero_si512(), 12) }),
            5 => Self(unsafe { _mm512_alignr_epi32(self.0, _mm512_setzero_si512(), 11) }),
            6 => Self(unsafe { _mm512_alignr_epi32(self.0, _mm512_setzero_si512(), 10) }),
            7 => Self(unsafe { _mm512_alignr_epi32(self.0, _mm512_setzero_si512(), 9) }),
            8 => Self(unsafe { _mm512_alignr_epi32(self.0, _mm512_setzero_si512(), 8) }),
            9 => Self(unsafe { _mm512_alignr_epi32(self.0, _mm512_setzero_si512(), 7) }),
            10 => Self(unsafe { _mm512_alignr_epi32(self.0, _mm512_setzero_si512(), 6) }),
            11 => Self(unsafe { _mm512_alignr_epi32(self.0, _mm512_setzero_si512(), 5) }),
            12 => Self(unsafe { _mm512_alignr_epi32(self.0, _mm512_setzero_si512(), 4) }),
            13 => Self(unsafe { _mm512_alignr_epi32(self.0, _mm512_setzero_si512(), 3) }),
            14 => Self(unsafe { _mm512_alignr_epi32(self.0, _mm512_setzero_si512(), 2) }),
            15 => Self(unsafe { _mm512_alignr_epi32(self.0, _mm512_setzero_si512(), 1) }),
            _ => unsafe { Self::zero() },
        }
    }

    #[inline(always)]
    unsafe fn left_lane_shift_32(self, n: u32) -> Self {
        match n {
            0 => self,
            1 => Self(unsafe { _mm512_alignr_epi32(_mm512_setzero_si512(), self.0, 1) }),
            2 => Self(unsafe { _mm512_alignr_epi32(_mm512_setzero_si512(), self.0, 2) }),
            3 => Self(unsafe { _mm512_alignr_epi32(_mm512_setzero_si512(), self.0, 3) }),
            4 => Self(unsafe { _mm512_alignr_epi32(_mm512_setzero_si512(), self.0, 4) }),
            5 => Self(unsafe { _mm512_alignr_epi32(_mm512_setzero_si512(), self.0, 5) }),
            6 => Self(unsafe { _mm512_alignr_epi32(_mm512_setzero_si512(), self.0, 6) }),
            7 => Self(unsafe { _mm512_alignr_epi32(_mm512_setzero_si512(), self.0, 7) }),
            8 => Self(unsafe { _mm512_alignr_epi32(_mm512_setzero_si512(), self.0, 8) }),
            9 => Self(unsafe { _mm512_alignr_epi32(_mm512_setzero_si512(), self.0, 9) }),
            10 => Self(unsafe { _mm512_alignr_epi32(_mm512_setzero_si512(), self.0, 10) }),
            11 => Self(unsafe { _mm512_alignr_epi32(_mm512_setzero_si512(), self.0, 11) }),
            12 => Self(unsafe { _mm512_alignr_epi32(_mm512_setzero_si512(), self.0, 12) }),
            13 => Self(unsafe { _mm512_alignr_epi32(_mm512_setzero_si512(), self.0, 13) }),
            14 => Self(unsafe { _mm512_alignr_epi32(_mm512_setzero_si512(), self.0, 14) }),
            15 => Self(unsafe { _mm512_alignr_epi32(_mm512_setzero_si512(), self.0, 15) }),
            _ => unsafe { Self::zero() },
        }
    }
}
