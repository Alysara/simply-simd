use std::arch::x86_64::*;
use std::mem::{transmute, transmute_copy};

use crate::architectures::interface::*;
use crate::architectures::macros::*;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Avx2Reg(pub __m256i);
impl SimdArch for Avx2Reg {}
impl MaskArch for Avx2Reg {}

impl SimdAdd for Avx2Reg {
    #[inline(always)]
    unsafe fn f64_add(self, rhs: Self) -> Self {
        self_from_op!(_mm256_add_pd, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_add(self, rhs: Self) -> Self {
        self_from_op!(_mm256_add_ps, self, rhs)
    }
    #[inline(always)]
    unsafe fn i64_add(self, rhs: Self) -> Self {
        self_from_op!(_mm256_add_epi64, self, rhs)
    }
    #[inline(always)]
    unsafe fn i32_add(self, rhs: Self) -> Self {
        self_from_op!(_mm256_add_epi32, self, rhs)
    }
    #[inline(always)]
    unsafe fn i16_add(self, rhs: Self) -> Self {
        self_from_op!(_mm256_add_epi16, self, rhs)
    }
    #[inline(always)]
    unsafe fn i8_add(self, rhs: Self) -> Self {
        self_from_op!(_mm256_add_epi8, self, rhs)
    }
}

impl SimdSub for Avx2Reg {
    #[inline(always)]
    unsafe fn f64_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm256_sub_pd, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm256_sub_ps, self, rhs)
    }
    #[inline(always)]
    unsafe fn i64_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm256_sub_epi64, self, rhs)
    }
    #[inline(always)]
    unsafe fn i32_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm256_sub_epi32, self, rhs)
    }
    #[inline(always)]
    unsafe fn i16_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm256_sub_epi16, self, rhs)
    }
    #[inline(always)]
    unsafe fn i8_sub(self, rhs: Self) -> Self {
        self_from_op!(_mm256_sub_epi8, self, rhs)
    }
}

impl SimdMul for Avx2Reg {
    #[inline(always)]
    unsafe fn f64_mul(self, rhs: Self) -> Self {
        self_from_op!(_mm256_mul_pd, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_mul(self, rhs: Self) -> Self {
        self_from_op!(_mm256_mul_ps, self, rhs)
    }
    #[inline(always)]
    unsafe fn i32_mul(self, rhs: Self) -> Self {
        self_from_op!(_mm256_mullo_epi32, self, rhs)
    }
    #[inline(always)]
    unsafe fn i16_mul(self, rhs: Self) -> Self {
        self_from_op!(_mm256_mullo_epi16, self, rhs)
    }
}

impl SimdDiv for Avx2Reg {
    #[inline(always)]
    unsafe fn f64_div(self, rhs: Self) -> Self {
        self_from_op!(_mm256_div_pd, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_div(self, rhs: Self) -> Self {
        self_from_op!(_mm256_div_ps, self, rhs)
    }
}

impl SimdBitwise for Avx2Reg {
    #[inline(always)]
    unsafe fn and(self, rhs: Self) -> Self {
        self_from_op!(_mm256_and_si256, self, rhs)
    }
    #[inline(always)]
    unsafe fn or(self, rhs: Self) -> Self {
        self_from_op!(_mm256_or_si256, self, rhs)
    }
    #[inline(always)]
    unsafe fn xor(self, rhs: Self) -> Self {
        self_from_op!(_mm256_xor_si256, self, rhs)
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        unsafe { Self(self.xor(Self::splat_32(!0)).0) }
    }
    #[inline(always)]
    unsafe fn and_not(self, rhs: Self) -> Self {
        self_from_op!(_mm256_andnot_si256, rhs, self)
    }
}

impl SimdShift for Avx2Reg {
    map_x86_shift!(sll_64, _mm256_sll_epi64);
    map_x86_shift!(srl_64, _mm256_srl_epi64);
    map_x86_shift!(sll_32, _mm256_sll_epi32);
    map_x86_shift!(srl_32, _mm256_srl_epi32);
    map_x86_shift!(sra_32, _mm256_sra_epi32);
    map_x86_shift!(sll_16, _mm256_sll_epi16);
    map_x86_shift!(srl_16, _mm256_srl_epi16);
    map_x86_shift!(sra_16, _mm256_sra_epi16);
    map_intrinsic!(slli_64, _mm256_slli_epi64, const N: i32, self);
    map_intrinsic!(srli_64, _mm256_srli_epi64, const N: i32, self);
    map_intrinsic!(slli_32, _mm256_slli_epi32, const N: i32, self);
    map_intrinsic!(srli_32, _mm256_srli_epi32, const N: i32, self);
    map_x86_conversion_imm_shift!(srai_32, _mm256_sra_epi32);
    map_intrinsic!(slli_16, _mm256_slli_epi16, const N: i32, self);
    map_intrinsic!(srli_16, _mm256_srli_epi16, const N: i32, self);
    map_x86_conversion_imm_shift!(srai_16, _mm256_sra_epi16);
    map_intrinsic!(sllv_64, _mm256_sllv_epi64, self, rhs: Self);
    map_intrinsic!(srlv_64, _mm256_srlv_epi64, self, rhs: Self);
    map_intrinsic!(sllv_32, _mm256_sllv_epi32, self, rhs: Self);
    map_intrinsic!(srlv_32, _mm256_srlv_epi32, self, rhs: Self);
    map_intrinsic!(srav_32, _mm256_srav_epi32, self, rhs: Self);

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


    #[inline(always)]
    unsafe fn srav_64(self, shift: Self) -> Self {
        unsafe {
            let min_shift = Self::splat_64(63u64);
            let shift = Self::min_u64(min_shift, shift);

            let sign_bits = Self::splat_64(i64::MIN as u64);
            let sign_bits = sign_bits.srlv_64(shift);

            let srl = self.srlv_64(shift);
            srl.xor(sign_bits).i64_sub(sign_bits)
        }
    }

    #[inline(always)]
    unsafe fn sllv_16(self, shift: Self) -> Self {
        unsafe {
            let even_mask = Self::splat_32(0x0000FFFFu32);
            let odd_mask = Self::splat_32(0xFFFF0000u32);

            let evens = self.sllv_32(shift.and(even_mask));
            let odds = self.and(odd_mask).sllv_32(shift.srli_32::<16>());

            evens.blend_16::<0x55>(odds)
        }
    }

    #[inline(always)]
    unsafe fn srlv_16(self, shift: Self) -> Self {
        unsafe {
            let even_mask = Self::splat_32(0x0000FFFF);

            let evens = self.and(even_mask).srlv_32(shift.and(even_mask));
            let odds = self.srlv_32(shift.srli_32::<16>());

            evens.blend_16::<0x55>(odds)
        }
    }

    #[inline(always)]
    unsafe fn srav_16(self, shift: Self) -> Self {
        unsafe {
            let even_mask = Self::splat_32(0x0000FFFF);

            let evens = self
                .slli_32::<16>()
                .srav_32(shift.and(even_mask))
                .srli_32::<16>();

            let odds = self.srav_32(shift.srli_32::<16>());

            evens.blend_16::<0x55>(odds)
        }
    }
}

impl SimdLoad for Avx2Reg {
    type MaskType = Self;
    #[inline(always)]
    unsafe fn load_aligned<T>(ptr: *const T) -> Self {
        self_from_op!(_mm256_load_si256, ptr)
    }
    #[inline(always)]
    unsafe fn load_unaligned<T>(ptr: *const T) -> Self {
        self_from_op!(_mm256_loadu_si256, ptr)
    }
    #[inline(always)]
    unsafe fn masked_load_64<T>(ptr: *const T, mask: Self::MaskType) -> Self {
        self_from_op!(_mm256_maskload_epi64, ptr, mask)
    }
    #[inline(always)]
    unsafe fn masked_load_32<T>(ptr: *const T, mask: Self::MaskType) -> Self {
        self_from_op!(_mm256_maskload_epi32, ptr, mask)
    }
}

impl SimdStore for Avx2Reg {
    type MaskType = Self;
    #[inline(always)]
    unsafe fn store_aligned<T>(self, ptr: *mut T) {
        execute_intrinsic!(_mm256_store_si256, ptr, self);
    }
    #[inline(always)]
    unsafe fn store_unaligned<T>(self, ptr: *mut T) {
        execute_intrinsic!(_mm256_storeu_si256, ptr, self);
    }
    #[inline(always)]
    unsafe fn masked_store_64<T>(self, ptr: *mut T, mask: Self::MaskType) {
        execute_intrinsic!(_mm256_maskstore_epi64, ptr, mask, self);
    }
    #[inline(always)]
    unsafe fn masked_store_32<T>(self, ptr: *mut T, mask: Self::MaskType) {
        execute_intrinsic!(_mm256_maskstore_epi32, ptr, mask, self);
    }
}

impl SimdZero for Avx2Reg {
    #[inline(always)]
    unsafe fn zero() -> Self {
        self_from_op!(_mm256_setzero_si256,)
    }
}

impl SimdFloatCasts for Avx2Reg {
    #[inline(always)]
    unsafe fn float_to_int_trunc(self) -> Self {
        self_from_op!(_mm256_cvttps_epi32, self)
    }
    #[inline(always)]
    unsafe fn float_to_int_round(self) -> Self {
        self_from_op!(_mm256_cvtps_epi32, self)
    }
}

impl SimdIntCasts for Avx2Reg {
    #[inline(always)]
    unsafe fn int_to_float(self) -> Self {
        self_from_op!(_mm256_cvtepi32_ps, self)
    }
}

impl SimdPermute for Avx2Reg {
    #[inline(always)]
    unsafe fn vpermute_64(self, rhs: Self) -> Self {
        unsafe {
            // Broadcast the indices into the high dword from the low dword.
            let shuffed = rhs.shuffle_32::<0b10_10_00_00>();

            // Double them to account for 64 -> 32 index change.
            let doubled_shuffed = shuffed.i32_add(shuffed);

            // Add one for the high dword indices to get the final indices.
            let hi_dword_offset = Self::splat_64::<u64>(0x0000_0001_0000_0000);
            let indices = doubled_shuffed.i32_add(hi_dword_offset);

            // Use native vpermute_32.
            self.vpermute_32(indices)
        }
    }
    map_intrinsic!(vpermute_32, _mm256_permutevar8x32_epi32, self, rhs: Self);

    #[inline(always)]
    unsafe fn vshuffle_16(self, rhs: Self) -> Self {
        unsafe {
            // Broadcast the indices into the high dword from the low dword.
            // Also multiply them by two to adjust the 16 -> 8 indexing change.
            let multiplier = Self::splat_16(0x0202);
            let broadcasted = rhs.i16_mul(multiplier);

            // Add one for the high dword indices to get the final indices.
            let hi_dword_offset = Self::splat_16(0x0100);
            let indices = broadcasted.i32_add(hi_dword_offset);

            // Use native vpermute_32.
            self.vshuffle_8(indices)
        }
    }

    map_intrinsic!(vshuffle_8, _mm256_shuffle_epi8, self, rhs: Self);

    #[inline(always)]
    unsafe fn permute_128<const N: i32>(self) -> Self {
        let lo = (N & 1) != 0;
        let hi = ((N >> 2) & 1) != 0;
        unsafe {
            match (lo, hi) {
                (false, false) => self.shuffle2_128::<0b00>(self),
                (true, false) => self.shuffle2_128::<0b01>(self),
                (false, true) => self,
                (true, true) => self.shuffle2_128::<0b11>(self),
            }
        }
    }

    #[inline(always)]
    unsafe fn shuffle_64<const N: i32>(self) -> Self {
        match N & 0b11 {
            0b00 => self_from_const_op!(_mm256_permute_pd, 0b0000, self),
            0b01 => self_from_const_op!(_mm256_permute_pd, 0b0101, self),
            0b10 => self_from_const_op!(_mm256_permute_pd, 0b1010, self),
            0b11 => self_from_const_op!(_mm256_permute_pd, 0b1111, self),
            _ => unreachable!(),
        }
    }

    map_intrinsic!(shuffle_32, _mm256_shuffle_epi32, const N: i32, self);

    #[inline(always)]
    unsafe fn shuffle_16<const N: i32>(self) -> Self {
        let first_shuf = self_from_const_op!(_mm256_shufflelo_epi16, N, self);
        self_from_const_op!(_mm256_shufflehi_epi16, N, first_shuf)
    }

    #[inline(always)]
    unsafe fn permute2_half<const N: i32>(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_128::<N>(rhs) }
    }

    #[inline(always)]
    unsafe fn shuffle2_128<const N: i32>(self, rhs: Self) -> Self {
        unsafe {
            match N & 0b11 {
                0b00 => self_from_const_op!(_mm256_permute2x128_si256, 0x20, self, rhs),
                0b01 => self_from_const_op!(_mm256_permute2x128_si256, 0x21, self, rhs),
                0b10 => self_from_const_op!(_mm256_permute2x128_si256, 0x30, self, rhs),
                0b11 => self_from_const_op!(_mm256_permute2x128_si256, 0x31, self, rhs),
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    unsafe fn shuffle2_64<const N: i32>(self, rhs: Self) -> Self {
        match N & 0b11 {
            0b00 => self_from_const_op!(_mm256_shuffle_pd, 0b0000, self, rhs),
            0b01 => self_from_const_op!(_mm256_shuffle_pd, 0b0101, self, rhs),
            0b10 => self_from_const_op!(_mm256_shuffle_pd, 0b1010, self, rhs),
            0b11 => self_from_const_op!(_mm256_shuffle_pd, 0b1111, self, rhs),
            _ => unreachable!(),
        }
    }
    map_intrinsic!(shuffle2_32, _mm256_shuffle_ps, const N: i32, self, rhs: Self);
}

impl SimdVariableBlend for Avx2Reg {
    type VecType = Self;
    #[inline(always)]
    unsafe fn vblend_64(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> Self::VecType {
        self_from_op!(_mm256_blendv_pd, false_values, true_values, self)
    }
    #[inline(always)]
    unsafe fn vblend_32(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> Self::VecType {
        self_from_op!(_mm256_blendv_ps, false_values, true_values, self)
    }
    #[inline(always)]
    unsafe fn vblend_16(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> Self::VecType {
        self_from_op!(_mm256_blendv_epi8, false_values, true_values, self)
    }
    #[inline(always)]
    unsafe fn vblend_8(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> Self::VecType {
        self_from_op!(_mm256_blendv_epi8, false_values, true_values, self)
    }
}

impl SimdImmediateBlend for Avx2Reg {
    #[inline(always)]
    unsafe fn blend_64<const N: i32>(self, false_values: Self) -> Self {
        match N & 0b11 {
            0b00 => false_values,
            0b01 => self_from_const_op!(_mm256_blend_pd, 0b0101, false_values, self),
            0b10 => self_from_const_op!(_mm256_blend_pd, 0b1010, false_values, self),
            0b11 => self,
            _ => unreachable!(),
        }
    }
    #[inline(always)]
    unsafe fn blend_32<const N: i32>(self, false_values: Self) -> Self {
        match N & 0xF {
            0b0000 => false_values,
            0b0001 => self_from_const_op!(_mm256_blend_ps, 0b0001_0001, false_values, self),
            0b0010 => self_from_const_op!(_mm256_blend_ps, 0b0010_0010, false_values, self),
            0b0011 => self_from_const_op!(_mm256_blend_ps, 0b0011_0011, false_values, self),
            0b0100 => self_from_const_op!(_mm256_blend_ps, 0b0100_0100, false_values, self),
            0b0101 => self_from_const_op!(_mm256_blend_ps, 0b0101_0101, false_values, self),
            0b0110 => self_from_const_op!(_mm256_blend_ps, 0b0110_0110, false_values, self),
            0b0111 => self_from_const_op!(_mm256_blend_ps, 0b0111_0111, false_values, self),
            0b1000 => self_from_const_op!(_mm256_blend_ps, 0b1000_1000, false_values, self),
            0b1001 => self_from_const_op!(_mm256_blend_ps, 0b1001_1001, false_values, self),
            0b1010 => self_from_const_op!(_mm256_blend_ps, 0b1010_1010, false_values, self),
            0b1011 => self_from_const_op!(_mm256_blend_ps, 0b1011_1011, false_values, self),
            0b1100 => self_from_const_op!(_mm256_blend_ps, 0b1100_1100, false_values, self),
            0b1101 => self_from_const_op!(_mm256_blend_ps, 0b1101_1101, false_values, self),
            0b1110 => self_from_const_op!(_mm256_blend_ps, 0b1110_1110, false_values, self),
            0b1111 => self,
            _ => unreachable!(),
        }
    }
    #[inline(always)]
    unsafe fn blend_16<const N: i32>(self, false_values: Self) -> Self {
        self_from_const_op!(_mm256_blend_epi16, N, false_values, self)
    }
}

impl SimdInterleave for Avx2Reg {
    map_intrinsic!(interleave_lower_block_64, _mm256_unpacklo_epi64, self, rhs: Self);
    map_intrinsic!(interleave_lower_block_32, _mm256_unpacklo_epi32, self, rhs: Self);
    map_intrinsic!(interleave_lower_block_16, _mm256_unpacklo_epi16, self, rhs: Self);
    map_intrinsic!(interleave_lower_block_8, _mm256_unpacklo_epi8, self, rhs: Self);
    map_intrinsic!(interleave_upper_block_64, _mm256_unpackhi_epi64, self, rhs: Self);
    map_intrinsic!(interleave_upper_block_32, _mm256_unpackhi_epi32, self, rhs: Self);
    map_intrinsic!(interleave_upper_block_16, _mm256_unpackhi_epi16, self, rhs: Self);
    map_intrinsic!(interleave_upper_block_8, _mm256_unpackhi_epi8, self, rhs: Self);

    #[inline(always)]
    unsafe fn interleave_lower_128(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_128::<0>(rhs) }
    }
    #[inline(always)]
    unsafe fn interleave_lower_64(self, rhs: Self) -> Self {
        unsafe {
            let lo = self_from_op!(_mm256_unpacklo_epi64, self, rhs);
            let hi = self_from_op!(_mm256_unpackhi_epi64, self, rhs);
            lo.shuffle2_128::<0>(hi)
        }
    }
    #[inline(always)]
    unsafe fn interleave_lower_32(self, rhs: Self) -> Self {
        unsafe {
            let lo = self_from_op!(_mm256_unpacklo_epi32, self, rhs);
            let hi = self_from_op!(_mm256_unpackhi_epi32, self, rhs);
            lo.shuffle2_128::<0>(hi)
        }
    }
    #[inline(always)]
    unsafe fn interleave_lower_16(self, rhs: Self) -> Self {
        unsafe {
            let lo = self_from_op!(_mm256_unpacklo_epi16, self, rhs);
            let hi = self_from_op!(_mm256_unpackhi_epi16, self, rhs);
            lo.shuffle2_128::<0>(hi)
        }
    }
    #[inline(always)]
    unsafe fn interleave_lower_8(self, rhs: Self) -> Self {
        unsafe {
            let lo = self_from_op!(_mm256_unpacklo_epi8, self, rhs);
            let hi = self_from_op!(_mm256_unpackhi_epi8, self, rhs);
            lo.shuffle2_128::<0>(hi)
        }
    }
    #[inline(always)]
    unsafe fn interleave_upper_128(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_128::<0b11>(rhs) }
    }
    #[inline(always)]
    unsafe fn interleave_upper_64(self, rhs: Self) -> Self {
        unsafe {
            let lo = self_from_op!(_mm256_unpacklo_epi64, self, rhs);
            let hi = self_from_op!(_mm256_unpackhi_epi64, self, rhs);
            lo.shuffle2_128::<0b11>(hi)
        }
    }
    #[inline(always)]
    unsafe fn interleave_upper_32(self, rhs: Self) -> Self {
        unsafe {
            let lo = self_from_op!(_mm256_unpacklo_epi32, self, rhs);
            let hi = self_from_op!(_mm256_unpackhi_epi32, self, rhs);
            lo.shuffle2_128::<0b11>(hi)
        }
    }
    #[inline(always)]
    unsafe fn interleave_upper_16(self, rhs: Self) -> Self {
        unsafe {
            let lo = self_from_op!(_mm256_unpacklo_epi16, self, rhs);
            let hi = self_from_op!(_mm256_unpackhi_epi16, self, rhs);
            lo.shuffle2_128::<0b11>(hi)
        }
    }
    #[inline(always)]
    unsafe fn interleave_upper_8(self, rhs: Self) -> Self {
        unsafe {
            let lo = self_from_op!(_mm256_unpacklo_epi8, self, rhs);
            let hi = self_from_op!(_mm256_unpackhi_epi8, self, rhs);
            lo.shuffle2_128::<0b11>(hi)
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
        unsafe { Self::splat_16(0x00FF).vblend_8(self, rhs.slli_16::<8>()) }
    }
    #[inline(always)]
    unsafe fn interleave_odd_128(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_128::<0b11>(rhs) }
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
    unsafe fn deinterleave_even_128(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_128::<0>(rhs) }
    }

    #[inline(always)]
    unsafe fn deinterleave_even_64(self, rhs: Self) -> Self {
        let shuf = unsafe { self.shuffle2_64::<0>(rhs) };
        self_from_const_op!(_mm256_permute4x64_epi64, 0xD8, shuf)
    }

    #[inline(always)]
    unsafe fn deinterleave_even_32(self, rhs: Self) -> Self {
        let shuf = unsafe { self.shuffle2_32::<0x88>(rhs) };
        self_from_const_op!(_mm256_permute4x64_epi64, 0xD8, shuf)
    }

    #[inline(always)]
    unsafe fn deinterleave_even_16(self, rhs: Self) -> Self {
        unsafe {
            let mask = Self::splat_32(0x0000FFFF);
            let low = self.and(mask);
            let high = rhs.and(mask);
            let packed = self_from_op!(_mm256_packus_epi32, low, high);
            self_from_const_op!(_mm256_permute4x64_epi64, 0xD8, packed)
        }
    }

    #[inline(always)]
    unsafe fn deinterleave_even_8(self, rhs: Self) -> Self {
        unsafe {
            let mask = Self::splat_16(0x00FF);
            let low = self.and(mask);
            let high = rhs.and(mask);
            let packed = self_from_op!(_mm256_packus_epi16, low, high);
            self_from_const_op!(_mm256_permute4x64_epi64, 0xD8, packed)
        }
    }

    #[inline(always)]
    unsafe fn deinterleave_odd_128(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_128::<0b11>(rhs) }
    }

    #[inline(always)]
    unsafe fn deinterleave_odd_64(self, rhs: Self) -> Self {
        let shuf = unsafe { self.shuffle2_64::<0b11>(rhs) };
        self_from_const_op!(_mm256_permute4x64_epi64, 0xD8, shuf)
    }

    #[inline(always)]
    unsafe fn deinterleave_odd_32(self, rhs: Self) -> Self {
        let shuf = unsafe { self.shuffle2_32::<0xDD>(rhs) };
        self_from_const_op!(_mm256_permute4x64_epi64, 0xD8, shuf)
    }

    #[inline(always)]
    unsafe fn deinterleave_odd_16(self, rhs: Self) -> Self {
        unsafe {
            let low = self.srli_32::<16>();
            let high = rhs.srli_32::<16>();
            let packed = self_from_op!(_mm256_packus_epi32, low, high);
            self_from_const_op!(_mm256_permute4x64_epi64, 0xD8, packed)
        }
    }

    #[inline(always)]
    unsafe fn deinterleave_odd_8(self, rhs: Self) -> Self {
        unsafe {
            let low = self.srli_16::<8>();
            let high = rhs.srli_16::<8>();
            let packed = self_from_op!(_mm256_packus_epi16, low, high);
            self_from_const_op!(_mm256_permute4x64_epi64, 0xD8, packed)
        }
    }
}

impl SimdMulAdd for Avx2Reg {
    #[inline(always)]
    unsafe fn mul_add_f64(self, mult: Self, add: Self) -> Self {
        self_from_op!(_mm256_fmadd_pd, self, mult, add)
    }
    #[inline(always)]
    unsafe fn mul_sub_f64(self, mult: Self, sub: Self) -> Self {
        self_from_op!(_mm256_fmsub_pd, self, mult, sub)
    }
    #[inline(always)]
    unsafe fn negated_mul_add_f64(self, mult: Self, add: Self) -> Self {
        self_from_op!(_mm256_fnmadd_pd, self, mult, add)
    }
    #[inline(always)]
    unsafe fn negated_mul_sub_f64(self, mult: Self, sub: Self) -> Self {
        self_from_op!(_mm256_fnmsub_pd, self, mult, sub)
    }
    #[inline(always)]
    unsafe fn mul_add_f32(self, mult: Self, add: Self) -> Self {
        self_from_op!(_mm256_fmadd_ps, self, mult, add)
    }
    #[inline(always)]
    unsafe fn mul_sub_f32(self, mult: Self, sub: Self) -> Self {
        self_from_op!(_mm256_fmsub_ps, self, mult, sub)
    }
    #[inline(always)]
    unsafe fn negated_mul_add_f32(self, mult: Self, add: Self) -> Self {
        self_from_op!(_mm256_fnmadd_ps, self, mult, add)
    }
    #[inline(always)]
    unsafe fn negated_mul_sub_f32(self, mult: Self, sub: Self) -> Self {
        self_from_op!(_mm256_fnmsub_ps, self, mult, sub)
    }
}

impl SimdRound for Avx2Reg {
    #[inline(always)]
    unsafe fn round_f64(self) -> Self {
        self_from_const_op!(
            _mm256_round_pd,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            self
        )
    }
    #[inline(always)]
    unsafe fn round_f32(self) -> Self {
        self_from_const_op!(
            _mm256_round_ps,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            self
        )
    }
    #[inline(always)]
    unsafe fn floor_f64(self) -> Self {
        self_from_const_op!(
            _mm256_round_pd,
            _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC,
            self
        )
    }
    #[inline(always)]
    unsafe fn floor_f32(self) -> Self {
        self_from_const_op!(
            _mm256_round_ps,
            _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC,
            self
        )
    }
    #[inline(always)]
    unsafe fn ceil_f64(self) -> Self {
        self_from_const_op!(
            _mm256_round_pd,
            _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC,
            self
        )
    }
    #[inline(always)]
    unsafe fn ceil_f32(self) -> Self {
        self_from_const_op!(
            _mm256_round_ps,
            _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC,
            self
        )
    }
}

macro_rules! map_cmp {
    ($func:ident, $intrinsic:ident, $flag: ident) => {
        #[inline(always)]
        unsafe fn $func(self, rhs: Self) -> Self {
            self_from_const_op!($intrinsic, $flag, self, rhs)
        }
    };
}

impl SimdPartialOrd for Avx2Reg {
    type MaskType = Self;
    map_cmp!(cmp_f64_eq, _mm256_cmp_pd, _CMP_EQ_OQ);
    map_cmp!(cmp_f64_lt, _mm256_cmp_pd, _CMP_LT_OQ);
    map_cmp!(cmp_f64_le, _mm256_cmp_pd, _CMP_LE_OQ);
    map_cmp!(cmp_f64_gt, _mm256_cmp_pd, _CMP_GT_OQ);
    map_cmp!(cmp_f64_ge, _mm256_cmp_pd, _CMP_GE_OQ);
    map_cmp!(cmp_f64_ne, _mm256_cmp_pd, _CMP_NEQ_OQ);
    map_cmp!(cmp_f32_eq, _mm256_cmp_ps, _CMP_EQ_OQ);
    map_cmp!(cmp_f32_lt, _mm256_cmp_ps, _CMP_LT_OQ);
    map_cmp!(cmp_f32_le, _mm256_cmp_ps, _CMP_LE_OQ);
    map_cmp!(cmp_f32_gt, _mm256_cmp_ps, _CMP_GT_OQ);
    map_cmp!(cmp_f32_ge, _mm256_cmp_ps, _CMP_GE_OQ);
    map_cmp!(cmp_f32_ne, _mm256_cmp_ps, _CMP_NEQ_OQ);
    map_intrinsic!(cmp_i64_eq, _mm256_cmpeq_epi64, self, rhs: Self);
    map_intrinsic!(cmp_i64_gt, _mm256_cmpgt_epi64, self, rhs: Self);
    map_intrinsic!(cmp_i32_eq, _mm256_cmpeq_epi32, self, rhs: Self);
    map_intrinsic!(cmp_i32_gt, _mm256_cmpgt_epi32, self, rhs: Self);
    map_intrinsic!(cmp_i16_eq, _mm256_cmpeq_epi16, self, rhs: Self);
    map_intrinsic!(cmp_i16_gt, _mm256_cmpgt_epi16, self, rhs: Self);
    map_intrinsic!(cmp_i8_eq, _mm256_cmpeq_epi8, self, rhs: Self);
    map_intrinsic!(cmp_i8_gt, _mm256_cmpgt_epi8, self, rhs: Self);
    map_intrinsic!(max_f64, _mm256_max_pd, self, rhs: Self);
    map_intrinsic!(min_f64, _mm256_min_pd, self, rhs: Self);
    map_intrinsic!(max_f32, _mm256_max_ps, self, rhs: Self);
    map_intrinsic!(min_f32, _mm256_min_ps, self, rhs: Self);
    map_intrinsic!(max_i32, _mm256_max_epi32, self, rhs: Self);
    map_intrinsic!(min_i32, _mm256_min_epi32, self, rhs: Self);
    map_intrinsic!(max_i16, _mm256_max_epi16, self, rhs: Self);
    map_intrinsic!(min_i16, _mm256_min_epi16, self, rhs: Self);
    map_intrinsic!(max_i8, _mm256_max_epi8, self, rhs: Self);
    map_intrinsic!(min_i8, _mm256_min_epi8, self, rhs: Self);
    map_intrinsic!(max_u32, _mm256_max_epu32, self, rhs: Self);
    map_intrinsic!(min_u32, _mm256_min_epu32, self, rhs: Self);
    map_intrinsic!(max_u16, _mm256_max_epu16, self, rhs: Self);
    map_intrinsic!(min_u16, _mm256_min_epu16, self, rhs: Self);
    map_intrinsic!(max_u8, _mm256_max_epu8, self, rhs: Self);
    map_intrinsic!(min_u8, _mm256_min_epu8, self, rhs: Self);

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
impl SimdSplat for Avx2Reg {
    #[inline(always)]
    unsafe fn splat_64<T>(val: T) -> Self {
        self_from_op!(_mm256_set1_epi64x, val)
    }
    #[inline(always)]
    unsafe fn splat_32<T>(val: T) -> Self {
        self_from_op!(_mm256_set1_epi32, val)
    }
    #[inline(always)]
    unsafe fn splat_16<T>(val: T) -> Self {
        self_from_op!(_mm256_set1_epi16, val)
    }
    #[inline(always)]
    unsafe fn splat_8<T>(val: T) -> Self {
        self_from_op!(_mm256_set1_epi8, val)
    }
}

impl SimdGather for Avx2Reg {
    #[inline(always)]
    unsafe fn gather_32_from_32<T, const B: i32>(self, ptr: *const T) -> Self {
        self_from_const_op!(_mm256_i32gather_epi32, B, ptr, self)
    }
    // #[inline(always)] unsafe fn gather_64_from_32<T, const B: i32>(self, ptr: *const T) -> Self { self_from_const_op!(_mm256_i32gather_epi64, B, ptr, self) }
    // #[inline(always)] unsafe fn gather_32_from_64<T, const B: i32>(self, ptr: *const T) -> Self { self_from_const_op!(_mm256_i64gather_epi32, B, ptr, self) }
    #[inline(always)]
    unsafe fn gather_64_from_64<T, const B: i32>(self, ptr: *const T) -> Self {
        self_from_const_op!(_mm256_i64gather_epi64, B, ptr, self)
    }
}

impl SimdSqrt for Avx2Reg {
    #[inline(always)]
    unsafe fn sqrt_f64(self) -> Self {
        self_from_op!(_mm256_sqrt_pd, self)
    }
    #[inline(always)]
    unsafe fn sqrt_f32(self) -> Self {
        self_from_op!(_mm256_sqrt_ps, self)
    }
    #[inline(always)]
    unsafe fn rsqrt_f32(self) -> Self {
        self_from_op!(_mm256_rsqrt_ps, self)
    }
}

impl SimdAllBits for Avx2Reg {
    #[inline(always)]
    unsafe fn all_zero(self) -> bool {
        execute_intrinsic!(_mm256_testz_si256, self, self) != 0
    }
}

impl SimdNegate for Avx2Reg {
    #[inline(always)]
    unsafe fn negate_f64(self) -> Self {
        unsafe { Self::splat_64(-0.0f64).xor(self) }
    }
    #[inline(always)]
    unsafe fn negate_f32(self) -> Self {
        unsafe { Self::splat_32(-0.0f64).xor(self) }
    }
}

impl SimdBlockShift for Avx2Reg {
    #[inline(always)]
    unsafe fn block_left_byte_shift<const N: i32>(self) -> Self {
        self_from_const_op!(_mm256_bslli_epi128, N, self)
    }
    #[inline(always)]
    unsafe fn block_right_byte_shift<const N: i32>(self) -> Self {
        self_from_const_op!(_mm256_bsrli_epi128, N, self)
    }
}

impl SimdMaskBitConversion for Avx2Reg {
    #[inline(always)]
    unsafe fn to_bits_64(self) -> u64 {
        execute_intrinsic!(_mm256_movemask_pd, self) as u64
    }
    #[inline(always)]
    unsafe fn to_bits_32(self) -> u64 {
        execute_intrinsic!(_mm256_movemask_ps, self) as u64
    }
    #[inline(always)]
    unsafe fn to_bits_8(self) -> u64 {
        execute_intrinsic!(_mm256_movemask_epi8, self) as u64
    }
    #[inline(always)]
    unsafe fn from_bits_64(bitmask: u64) -> Self {
        let mask = self_from_op!(_mm256_set_epi64x, 1, 2, 4, 8);
        let bits = self_from_op!(_mm256_set1_epi64x, bitmask as i64);
        self_from_op!(_mm256_cmpgt_epi64, bits.and(mask), Self::zero())
    }
    #[inline(always)]
    unsafe fn from_bits_32(bitmask: u64) -> Self {
        let mask = self_from_op!(_mm256_set_epi32, 1, 2, 4, 8, 16, 32, 64, 128);
        let bits = self_from_op!(_mm256_set1_epi32, bitmask as u32);
        self_from_op!(_mm256_cmpgt_epi32, bits.and(mask), Self::zero())
    }
    #[inline(always)]
    unsafe fn from_bits_16(bitmask: u64) -> Self {
        #[rustfmt::skip]
        let mask = self_from_op!(_mm256_set_epi16, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, -32768);
        let bits = self_from_op!(_mm256_set1_epi16, bitmask as i16);
        unsafe { self_from_op!(_mm256_cmpeq_epi16, bits.and(mask), Self::zero()).not() }
    }
    #[inline(always)]
    unsafe fn from_bits_8(bitmask: u64) -> Self {
        let b1 = bitmask as i8;
        let b2 = (bitmask >> 8) as i8;
        let b3 = (bitmask >> 16) as i8;
        let b4 = (bitmask >> 24) as i8;
        #[rustfmt::skip]
        let mask = self_from_op!(_mm256_set_epi8,
            1, 2, 4, 8, 16, 32, 64, -128,
            1, 2, 4, 8, 16, 32, 64, -128,
            1, 2, 4, 8, 16, 32, 64, -128,
            1, 2, 4, 8, 16, 32, 64, -128
        );
        #[rustfmt::skip]
        let bits = self_from_op!(_mm256_set_epi8,
            b1, b1, b1, b1, b1, b1, b1, b1,
            b2, b2, b2, b2, b2, b2, b2, b2,
            b3, b3, b3, b3, b3, b3, b3, b3,
            b4, b4, b4, b4, b4, b4, b4, b4
        );
        unsafe { self_from_op!(_mm256_cmpeq_epi8, bits.and(mask), Self::zero()).not() }
    }
}

impl SimdLaneShift for Avx2Reg {
    #[inline(always)]
    unsafe fn left_lane_shift_32(self, n: u32) -> Self {
        unsafe {
            let a = self.0;
            let z = _mm256_setzero_si256();
            let t = _mm256_permute2x128_si256::<0x81>(a, z);
            Self(match n {
                0 => a,
                1 => _mm256_alignr_epi8::<4>(t, a),
                2 => _mm256_alignr_epi8::<8>(t, a),
                3 => _mm256_alignr_epi8::<12>(t, a),
                4 => t,
                5 => _mm256_srli_si256::<4>(t),
                6 => _mm256_srli_si256::<8>(t),
                7 => _mm256_srli_si256::<12>(t),
                _ => z,
            })
        }
    }

    #[inline(always)]
    unsafe fn right_lane_shift_32(self, n: u32) -> Self {
        unsafe {
            let a = self.0;
            let z = _mm256_setzero_si256();
            let t = _mm256_permute2x128_si256::<0x08>(a, a);
            Self(match n {
                0 => a,
                1 => _mm256_alignr_epi8::<12>(a, t),
                2 => _mm256_alignr_epi8::<8>(a, t),
                3 => _mm256_alignr_epi8::<4>(a, t),
                4 => t,
                5 => _mm256_slli_si256::<4>(t),
                6 => _mm256_slli_si256::<8>(t),
                7 => _mm256_slli_si256::<12>(t),
                _ => z,
            })
        }
    }
}
