use std::arch::aarch64::*;
use std::mem::{transmute, transmute_copy};

use crate::architectures::interface::*;
use crate::architectures::macros::*;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct NeonReg(pub float32x4_t);
impl SimdArch for NeonReg {}
impl MaskArch for NeonReg {}

impl SimdAdd for NeonReg {
    #[inline(always)]
    unsafe fn f64_add(self, rhs: Self) -> Self {
        self_from_op!(vaddq_f64, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_add(self, rhs: Self) -> Self {
        self_from_op!(vaddq_f32, self, rhs)
    }
    #[inline(always)]
    unsafe fn i64_add(self, rhs: Self) -> Self {
        self_from_op!(vaddq_s64, self, rhs)
    }
    #[inline(always)]
    unsafe fn i32_add(self, rhs: Self) -> Self {
        self_from_op!(vaddq_s32, self, rhs)
    }
    #[inline(always)]
    unsafe fn i16_add(self, rhs: Self) -> Self {
        self_from_op!(vaddq_s16, self, rhs)
    }
    #[inline(always)]
    unsafe fn i8_add(self, rhs: Self) -> Self {
        self_from_op!(vaddq_s8, self, rhs)
    }
}

impl SimdSub for NeonReg {
    #[inline(always)]
    unsafe fn f64_sub(self, rhs: Self) -> Self {
        self_from_op!(vsubq_f64, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_sub(self, rhs: Self) -> Self {
        self_from_op!(vsubq_f32, self, rhs)
    }
    #[inline(always)]
    unsafe fn i64_sub(self, rhs: Self) -> Self {
        self_from_op!(vsubq_s64, self, rhs)
    }
    #[inline(always)]
    unsafe fn i32_sub(self, rhs: Self) -> Self {
        self_from_op!(vsubq_s32, self, rhs)
    }
    #[inline(always)]
    unsafe fn i16_sub(self, rhs: Self) -> Self {
        self_from_op!(vsubq_s16, self, rhs)
    }
    #[inline(always)]
    unsafe fn i8_sub(self, rhs: Self) -> Self {
        self_from_op!(vsubq_s8, self, rhs)
    }
}

impl SimdMul for NeonReg {
    #[inline(always)]
    unsafe fn f64_mul(self, rhs: Self) -> Self {
        self_from_op!(vmulq_f64, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_mul(self, rhs: Self) -> Self {
        self_from_op!(vmulq_f32, self, rhs)
    }
    #[inline(always)]
    unsafe fn i32_mul(self, rhs: Self) -> Self {
        self_from_op!(vmulq_s32, self, rhs)
    }
    #[inline(always)]
    unsafe fn i16_mul(self, rhs: Self) -> Self {
        self_from_op!(vmulq_s16, self, rhs)
    }
}

impl SimdDiv for NeonReg {
    #[inline(always)]
    unsafe fn f64_div(self, rhs: Self) -> Self {
        self_from_op!(vdivq_f64, self, rhs)
    }
    #[inline(always)]
    unsafe fn f32_div(self, rhs: Self) -> Self {
        self_from_op!(vdivq_f32, self, rhs)
    }
}

impl SimdBitwise for NeonReg {
    #[inline(always)]
    unsafe fn and(self, rhs: Self) -> Self {
        self_from_op!(vandq_u32, self, rhs)
    }
    #[inline(always)]
    unsafe fn or(self, rhs: Self) -> Self {
        self_from_op!(vorrq_u32, self, rhs)
    }
    #[inline(always)]
    unsafe fn xor(self, rhs: Self) -> Self {
        self_from_op!(veorq_u32, self, rhs)
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        unsafe { Self(self.xor(Self::splat_32(!0u32)).0) }
    }
    #[inline(always)]
    unsafe fn and_not(self, rhs: Self) -> Self {
        self_from_op!(vbicq_u32, self, rhs)
    }
}

impl SimdShift for NeonReg {
    #[inline(always)]
    unsafe fn sll_64(self, rhs: u32) -> Self {
        unsafe { self.sllv_64(Self::splat_64(rhs as u64)) }
    }
    #[inline(always)]
    unsafe fn srl_64(self, rhs: u32) -> Self {
        unsafe { self.srlv_64(Self::splat_64(rhs as u64)) }
    }
    #[inline(always)]
    unsafe fn sra_64(self, rhs: u32) -> Self {
        unsafe { self.srav_64(Self::splat_64(rhs as u64)) }
    }
    #[inline(always)]
    unsafe fn sll_32(self, rhs: u32) -> Self {
        unsafe { self.sllv_32(Self::splat_32(rhs)) }
    }
    #[inline(always)]
    unsafe fn srl_32(self, rhs: u32) -> Self {
        unsafe { self.srlv_32(Self::splat_32(rhs)) }
    }
    #[inline(always)]
    unsafe fn sra_32(self, rhs: u32) -> Self {
        unsafe { self.srav_32(Self::splat_32(rhs)) }
    }
    #[inline(always)]
    unsafe fn sll_16(self, rhs: u32) -> Self {
        unsafe { self.sllv_16(Self::splat_16(rhs as u16)) }
    }
    #[inline(always)]
    unsafe fn srl_16(self, rhs: u32) -> Self {
        unsafe { self.srlv_16(Self::splat_16(rhs as u16)) }
    }
    #[inline(always)]
    unsafe fn sra_16(self, rhs: u32) -> Self {
        unsafe { self.srav_16(Self::splat_16(rhs as u16)) }
    }

    map_intrinsic!(slli_64, vshlq_n_u64, const N: i32, self);
    map_intrinsic!(srli_64, vshrq_n_u64, const N: i32, self);
    map_intrinsic!(srai_64, vshrq_n_s64, const N: i32, self);
    map_intrinsic!(slli_32, vshlq_n_u32, const N: i32, self);
    map_intrinsic!(srli_32, vshrq_n_u32, const N: i32, self);
    map_intrinsic!(srai_32, vshrq_n_s32, const N: i32, self);
    map_intrinsic!(slli_16, vshlq_n_u16, const N: i32, self);
    map_intrinsic!(srli_16, vshrq_n_u16, const N: i32, self);
    map_intrinsic!(srai_16, vshrq_n_s16, const N: i32, self);

    #[inline(always)]
    unsafe fn sllv_64(self, rhs: Self) -> Self {
        self_from_op!(vshlq_s64, self, rhs)
    }
    #[inline(always)]
    unsafe fn srlv_64(self, rhs: Self) -> Self {
        self_from_op!(vshlq_u64, self, self_from_op!(vnegq_s64, rhs))
    }
    #[inline(always)]
    unsafe fn srav_64(self, rhs: Self) -> Self {
        self_from_op!(vshlq_s64, self, self_from_op!(vnegq_s64, rhs))
    }
    #[inline(always)]
    unsafe fn sllv_32(self, rhs: Self) -> Self {
        self_from_op!(vshlq_s32, self, rhs)
    }
    #[inline(always)]
    unsafe fn srlv_32(self, rhs: Self) -> Self {
        self_from_op!(vshlq_u32, self, self_from_op!(vnegq_s32, rhs))
    }
    #[inline(always)]
    unsafe fn srav_32(self, rhs: Self) -> Self {
        self_from_op!(vshlq_s32, self, self_from_op!(vnegq_s32, rhs))
    }
    #[inline(always)]
    unsafe fn sllv_16(self, rhs: Self) -> Self {
        self_from_op!(vshlq_s16, self, rhs)
    }
    #[inline(always)]
    unsafe fn srlv_16(self, rhs: Self) -> Self {
        self_from_op!(vshlq_u16, self, self_from_op!(vnegq_s16, rhs))
    }
    #[inline(always)]
    unsafe fn srav_16(self, rhs: Self) -> Self {
        self_from_op!(vshlq_s16, self, self_from_op!(vnegq_s16, rhs))
    }
}

impl SimdLoad for NeonReg {
    type MaskType = Self;
    #[inline(always)]
    unsafe fn load_aligned<T>(ptr: *const T) -> Self {
        self_from_op!(vld1q_u32, ptr)
    }
    #[inline(always)]
    unsafe fn load_unaligned<T>(ptr: *const T) -> Self {
        self_from_op!(vld1q_u32, ptr)
    }
    #[inline(always)]
    unsafe fn masked_load_64<T>(ptr: *const T, mask: Self::MaskType) -> Self {
        unsafe {
            Self(transmute(vandq_u64(
                transmute(vld1q_u64(ptr as *const u64)),
                transmute(mask.0),
            )))
        }
    }
    #[inline(always)]
    unsafe fn masked_load_32<T>(ptr: *const T, mask: Self::MaskType) -> Self {
        unsafe {
            Self(transmute(vandq_u32(
                transmute(vld1q_u32(ptr as *const u32)),
                transmute(mask.0),
            )))
        }
    }
}

impl SimdStore for NeonReg {
    type MaskType = Self;
    #[inline(always)]
    unsafe fn store_aligned<T>(self, ptr: *mut T) {
        execute_intrinsic!(vst1q_u32, ptr, self);
    }
    #[inline(always)]
    unsafe fn store_unaligned<T>(self, ptr: *mut T) {
        execute_intrinsic!(vst1q_u32, ptr, self);
    }
    #[inline(always)]
    unsafe fn masked_store_64<T>(self, ptr: *mut T, mask: Self::MaskType) {
        unsafe {
            let current = vld1q_u64(ptr as *const u64);
            let masked = vbslq_u64(transmute(mask.0), transmute(self.0), current);
            vst1q_u64(ptr as *mut u64, masked);
        }
    }

    #[inline(always)]
    unsafe fn masked_store_32<T>(self, ptr: *mut T, mask: Self::MaskType) {
        unsafe {
            let current = vld1q_u32(ptr as *const u32);
            let masked = vbslq_u32(transmute(mask.0), transmute(self.0), current);
            vst1q_u32(ptr as *mut u32, masked);
        }
    }
}

impl SimdZero for NeonReg {
    #[inline(always)]
    unsafe fn zero() -> Self {
        unsafe { Self(transmute(vdupq_n_u32(0))) }
    }
}

impl SimdFloatCasts for NeonReg {
    #[inline(always)]
    unsafe fn float_to_int_trunc(self) -> Self {
        self_from_op!(vcvtq_s32_f32, self)
    }
    #[inline(always)]
    unsafe fn float_to_int_round(self) -> Self {
        self_from_op!(vcvtnq_s32_f32, self)
    }
}

impl SimdIntCasts for NeonReg {
    #[inline(always)]
    unsafe fn int_to_float(self) -> Self {
        self_from_op!(vcvtq_f32_s32, self)
    }
}

impl SimdPermute for NeonReg {
    #[inline(always)]
    unsafe fn vpermute_64(self, rhs: Self) -> Self {
        unsafe {
            let lo = self_from_const_op!(vdupq_laneq_u64, 0, self);
            let hi = self_from_const_op!(vdupq_laneq_u64, 1, self);
            let mask = self_from_op!(vtstq_u64, rhs, Self::splat_64(1u64));
            mask.vblend_64(lo, hi)
        }
    }
    #[inline(always)]
    unsafe fn vpermute_32(self, rhs: Self) -> Self {
        unsafe {
            let mult = Self::splat_32(0x04040404u32);
            let add = Self::splat_32(0x03020100u32);
            let byte_indices = self_from_op!(vmlaq_u32, add, rhs, mult);
            self_from_op!(vqtbl1q_u8, self, byte_indices)
        }
    }
    #[inline(always)]
    unsafe fn vshuffle_16(self, rhs: Self) -> Self {
        unsafe {
            let mult = Self::splat_16(0x0202u16);
            let add = Self::splat_32(0x0100u32);
            let byte_indices = self_from_op!(vmlaq_u16, add, rhs, mult);
            self_from_op!(vqtbl1q_u8, self, byte_indices)
        }
    }
    #[inline(always)]
    unsafe fn vshuffle_8(self, rhs: Self) -> Self {
        self_from_op!(vqtbl1q_u8, self, rhs)
    }

    #[inline(always)]
    unsafe fn permute_128<const N: i32>(self) -> Self {
        self
    }

    #[inline(always)]
    unsafe fn shuffle_64<const N: i32>(self) -> Self {
        match N & 0b11 {
            0b00 => self_from_const_op!(vdupq_laneq_u64, 0, self),
            0b01 => self_from_const_op!(vextq_u64, 1, self, self),
            0b10 => self,
            0b11 => self_from_const_op!(vdupq_laneq_u64, 1, self),
            _ => unreachable!(),
        }
    }
    #[inline(always)]
    unsafe fn shuffle_32<const N: i32>(self) -> Self {
        const fn gen_indices(n: u32) -> [u8; 16] {
            let mut indices = [0; 16];

            let mut i = 0;
            while i < 4 {
                let sel = ((n >> (i * 2)) & 3) as u8;
                indices[i * 4] = sel * 4;
                indices[i * 4 + 1] = sel * 4 + 1;
                indices[i * 4 + 2] = sel * 4 + 2;
                indices[i * 4 + 3] = sel * 4 + 3;
                i += 1;
            }

            indices
        }

        match N & 0xFF {
            0b11_10_01_00 => self,
            0b00_11_10_01 => self_from_const_op!(vextq_u32, 1, self, self),
            0b01_00_11_10 => self_from_const_op!(vextq_u32, 2, self, self),
            0b10_01_00_11 => self_from_const_op!(vextq_u32, 3, self, self),
            0b00_00_00_00 => self_from_const_op!(vdupq_laneq_u32, 0, self),
            0b01_01_01_01 => self_from_const_op!(vdupq_laneq_u32, 1, self),
            0b10_10_10_10 => self_from_const_op!(vdupq_laneq_u32, 2, self),
            0b11_11_11_11 => self_from_const_op!(vdupq_laneq_u32, 3, self),
            0b01_00_01_00 => self_from_const_op!(vdupq_laneq_u64, 0, self),
            0b11_10_11_10 => self_from_const_op!(vdupq_laneq_u64, 1, self),
            0b10_10_00_00 => self_from_op!(vtrn1q_u32, self, self),
            0b11_11_01_01 => self_from_op!(vtrn2q_u32, self, self),
            0b01_01_00_00 => self_from_op!(vzip1q_u32, self, self),
            0b11_11_10_10 => self_from_op!(vzip2q_u32, self, self),
            0b10_00_10_00 => self_from_op!(vuzp1q_u32, self, self),
            0b11_01_11_01 => self_from_op!(vuzp2q_u32, self, self),
            0b10_11_00_01 => self_from_op!(vrev64q_u32, self),
            0b11_10_00_00 => self_from_const2_op!(vcopyq_laneq_u32, 1, 0, self, self),
            0b11_00_01_00 => self_from_const2_op!(vcopyq_laneq_u32, 2, 0, self, self),
            0b00_10_01_00 => self_from_const2_op!(vcopyq_laneq_u32, 3, 0, self, self),
            0b11_10_01_01 => self_from_const2_op!(vcopyq_laneq_u32, 0, 1, self, self),
            0b11_01_01_00 => self_from_const2_op!(vcopyq_laneq_u32, 2, 1, self, self),
            0b01_10_01_00 => self_from_const2_op!(vcopyq_laneq_u32, 3, 1, self, self),
            0b11_10_01_10 => self_from_const2_op!(vcopyq_laneq_u32, 0, 2, self, self),
            0b11_10_10_00 => self_from_const2_op!(vcopyq_laneq_u32, 1, 2, self, self),
            0b10_10_01_00 => self_from_const2_op!(vcopyq_laneq_u32, 3, 2, self, self),
            0b11_10_01_11 => self_from_const2_op!(vcopyq_laneq_u32, 0, 3, self, self),
            0b11_10_11_00 => self_from_const2_op!(vcopyq_laneq_u32, 1, 3, self, self),
            0b11_11_01_00 => self_from_const2_op!(vcopyq_laneq_u32, 2, 3, self, self),
            _ => {
                let indices = const { gen_indices(N as u32 & 0xFF) };
                unsafe {
                    let indices = Self::load_unaligned(indices.as_ptr());
                    self.vshuffle_8(indices)
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn shuffle_16<const N: i32>(self) -> Self {
        const fn gen_indices(n: u32) -> [u8; 16] {
            let mut indices = [0; 16];

            let mut cur = 0;
            while cur < 2 {
                let mut i = 0;
                while i < 4 {
                    let sel = ((n >> (i * 2)) & 3) as usize;
                    let src = (sel * 2 + cur * 8) as u8;
                    let dst = i * 2 + cur * 8;
                    indices[dst] = src;
                    indices[dst + 1] = src + 1;
                    i += 1;
                }
                cur += 1;
            }

            indices
        }

        match N & 0xFF {
            0b11_10_01_00 => self,
            0b10_11_00_01 => self_from_op!(vrev32q_u16, self),
            0b00_01_10_11 => self_from_op!(vrev64q_u16, self),
            0b01_00_11_10 => self_from_op!(vrev64q_u32, self),
            0b10_10_00_00 => self_from_op!(vtrn1q_u16, self, self),
            0b11_11_01_01 => self_from_op!(vtrn2q_u16, self, self),
            0b01_00_01_00 => self_from_op!(vtrn1q_u32, self, self),
            0b11_10_11_10 => self_from_op!(vtrn2q_u32, self, self),
            _ => {
                let indices = const { gen_indices(N as u32 & 0xFF) };
                unsafe {
                    let indices = Self::load_unaligned(indices.as_ptr());
                    self.vshuffle_8(indices)
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn permute2_half<const N: i32>(self, rhs: Self) -> Self {
        unsafe { self.shuffle2_64::<N>(rhs) }
    }

    #[inline(always)]
    unsafe fn shuffle2_128<const N: i32>(self, _rhs: Self) -> Self {
        self
    }
    #[inline(always)]
    unsafe fn shuffle2_64<const N: i32>(self, rhs: Self) -> Self {
        match N & 0b11 {
            0b00 => self_from_op!(vzip1q_u64, self, rhs),
            0b01 => self_from_const_op!(vextq_u64, 1, self, rhs),
            0b10 => self_from_const2_op!(vcopyq_laneq_u64, 1, 1, self, rhs),
            0b11 => self_from_op!(vzip2q_u64, self, rhs),
            _ => unreachable!(),
        }
    }
    #[inline(always)]
    unsafe fn shuffle2_32<const N: i32>(self, rhs: Self) -> Self {
        const fn gen_indices(n: u32) -> [u8; 16] {
            let mut indices = [0; 16];
            let mut i = 0;
            while i < 4 {
                let reg = (i & 2) >> 1;
                let sel = ((n >> (i * 2)) & 3) as usize;
                let src = sel * 4 + reg * 16;
                let mut j = 0;
                while j < 4 {
                    indices[i * 4 + j] = (src + j) as u8;
                    j += 1;
                }
                i += 1;
            }
            indices
        }

        match N & 0xFF {
            0b11_10_01_00 => self_from_const2_op!(vcopyq_laneq_u64, 1, 1, self, rhs),
            0b01_00_11_10 => self_from_const_op!(vextq_u32, 2, self, rhs),
            0b01_00_01_00 => self_from_op!(vtrn1q_u64, self, rhs),
            0b11_10_11_10 => self_from_op!(vtrn2q_u64, self, rhs),
            0b10_00_10_00 => self_from_op!(vuzp1q_u32, self, rhs),
            0b11_01_11_01 => self_from_op!(vuzp2q_u32, self, rhs),
            _ => {
                let indices = const { gen_indices(N as u32 & 0xFF) };
                unsafe {
                    let indices = Self::load_unaligned(indices.as_ptr());
                    self_from_op!(vqtbl2q_u8, (self, rhs), indices)
                }
            }
        }
    }
}

impl SimdVariableBlend for NeonReg {
    type VecType = Self;
    #[inline(always)]
    unsafe fn vblend_64(self, true_values: Self, false_values: Self) -> Self {
        self_from_op!(vbslq_f64, self, true_values, false_values)
    }
    #[inline(always)]
    unsafe fn vblend_32(self, true_values: Self, false_values: Self) -> Self {
        self_from_op!(vbslq_f32, self, true_values, false_values)
    }
    #[inline(always)]
    unsafe fn vblend_16(self, true_values: Self, false_values: Self) -> Self {
        self_from_op!(vbslq_u16, self, true_values, false_values)
    }
    #[inline(always)]
    unsafe fn vblend_8(self, true_values: Self, false_values: Self) -> Self {
        self_from_op!(vbslq_u8, self, true_values, false_values)
    }
}

impl SimdInterleave for NeonReg {
    #[inline(always)]
    unsafe fn interleave_lower_128(self, _rhs: Self) -> Self {
        self
    }
    map_intrinsic!(interleave_lower_64, vzip1q_u64, self, rhs: Self);
    map_intrinsic!(interleave_lower_32, vzip1q_u32, self, rhs: Self);
    map_intrinsic!(interleave_lower_16, vzip1q_u16, self, rhs: Self);
    map_intrinsic!(interleave_lower_8, vzip1q_u8, self, rhs: Self);

    #[inline(always)]
    unsafe fn interleave_upper_128(self, _rhs: Self) -> Self {
        self
    }
    map_intrinsic!(interleave_upper_64, vzip2q_u64, self, rhs: Self);
    map_intrinsic!(interleave_upper_32, vzip2q_u32, self, rhs: Self);
    map_intrinsic!(interleave_upper_16, vzip2q_u16, self, rhs: Self);
    map_intrinsic!(interleave_upper_8, vzip2q_u8, self, rhs: Self);

    #[inline(always)]
    unsafe fn interleave_even_128(self, _rhs: Self) -> Self {
        self
    }
    map_intrinsic!(interleave_even_64, vtrn1q_u64, self, rhs: Self);
    map_intrinsic!(interleave_even_32, vtrn1q_u32, self, rhs: Self);
    map_intrinsic!(interleave_even_16, vtrn1q_u16, self, rhs: Self);
    map_intrinsic!(interleave_even_8, vtrn1q_u8, self, rhs: Self);

    #[inline(always)]
    unsafe fn interleave_odd_128(self, _rhs: Self) -> Self {
        self
    }
    map_intrinsic!(interleave_odd_64, vtrn2q_u64, self, rhs: Self);
    map_intrinsic!(interleave_odd_32, vtrn2q_u32, self, rhs: Self);
    map_intrinsic!(interleave_odd_16, vtrn2q_u16, self, rhs: Self);
    map_intrinsic!(interleave_odd_8, vtrn2q_u8, self, rhs: Self);

    #[inline(always)]
    unsafe fn deinterleave_even_128(self, _rhs: Self) -> Self {
        self
    }
    map_intrinsic!(deinterleave_even_64, vuzp1q_u64, self, rhs: Self);
    map_intrinsic!(deinterleave_even_32, vuzp1q_u32, self, rhs: Self);
    map_intrinsic!(deinterleave_even_16, vuzp1q_u16, self, rhs: Self);
    map_intrinsic!(deinterleave_even_8, vuzp1q_u8, self, rhs: Self);

    #[inline(always)]
    unsafe fn deinterleave_odd_128(self, _rhs: Self) -> Self {
        self
    }
    map_intrinsic!(deinterleave_odd_64, vuzp2q_u64, self, rhs: Self);
    map_intrinsic!(deinterleave_odd_32, vuzp2q_u32, self, rhs: Self);
    map_intrinsic!(deinterleave_odd_16, vuzp2q_u16, self, rhs: Self);
    map_intrinsic!(deinterleave_odd_8, vuzp2q_u8, self, rhs: Self);
}

impl SimdImmediateBlend for NeonReg {
    #[inline(always)]
    unsafe fn blend_64<const N: i32>(self, false_values: Self) -> Self {
        const {
            assert!(N < 4, "N must be less than 4");
        }

        const MASKS: [[u64; 2]; 4] = [
            [0, 0],
            [0xFFFFFFFFFFFFFFFF, 0],
            [0, 0xFFFFFFFFFFFFFFFF],
            [0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF],
        ];

        let mask = self_from_op!(vld1q_u64, MASKS[N as usize].as_ptr());
        unsafe { mask.vblend_64(self, false_values) }
    }

    #[inline(always)]
    unsafe fn blend_32<const N: i32>(self, false_values: Self) -> Self {
        const {
            assert!(N < 16, "N must be less than 16");
        }

        const MASKS: [[u32; 4]; 16] = [
            [0, 0, 0, 0],
            [0xFFFFFFFF, 0, 0, 0],
            [0, 0xFFFFFFFF, 0, 0],
            [0xFFFFFFFF, 0xFFFFFFFF, 0, 0],
            [0, 0, 0xFFFFFFFF, 0],
            [0xFFFFFFFF, 0, 0xFFFFFFFF, 0],
            [0, 0xFFFFFFFF, 0xFFFFFFFF, 0],
            [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0],
            [0, 0, 0, 0xFFFFFFFF],
            [0xFFFFFFFF, 0, 0, 0xFFFFFFFF],
            [0, 0xFFFFFFFF, 0, 0xFFFFFFFF],
            [0xFFFFFFFF, 0xFFFFFFFF, 0, 0xFFFFFFFF],
            [0, 0, 0xFFFFFFFF, 0xFFFFFFFF],
            [0xFFFFFFFF, 0, 0xFFFFFFFF, 0xFFFFFFFF],
            [0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
            [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
        ];

        let mask = self_from_op!(vld1q_u32, MASKS[N as usize].as_ptr());
        unsafe { mask.vblend_32(self, false_values) }
    }

    #[inline(always)]
    unsafe fn blend_16<const N: i32>(self, false_values: Self) -> Self {
        const {
            assert!(N < 256, "N must be less than 256");
        }

        let mask = unsafe { Self::from_bits_16(N as u64) };
        self_from_op!(vbslq_u16, mask, self, false_values)
    }
}

impl SimdMulAdd for NeonReg {
    #[inline(always)]
    unsafe fn mul_add_f64(self, mult: Self, add: Self) -> Self {
        self_from_op!(vfmaq_f64, add, self, mult)
    }
    #[inline(always)]
    unsafe fn mul_sub_f64(self, mult: Self, sub: Self) -> Self {
        unsafe { self_from_op!(vfmsq_f64, sub, self, mult).negate_f64() }
    }
    #[inline(always)]
    unsafe fn negated_mul_add_f64(self, mult: Self, add: Self) -> Self {
        self_from_op!(vfmsq_f64, add, self, mult)
    }
    #[inline(always)]
    unsafe fn negated_mul_sub_f64(self, mult: Self, sub: Self) -> Self {
        unsafe { self_from_op!(vfmaq_f64, sub, self, mult).negate_f64() }
    }
    #[inline(always)]
    unsafe fn mul_add_f32(self, mult: Self, add: Self) -> Self {
        self_from_op!(vfmaq_f32, add, self, mult)
    }
    #[inline(always)]
    unsafe fn mul_sub_f32(self, mult: Self, sub: Self) -> Self {
        unsafe { self_from_op!(vfmsq_f32, sub, self, mult).negate_f32() }
    }
    #[inline(always)]
    unsafe fn negated_mul_add_f32(self, mult: Self, add: Self) -> Self {
        self_from_op!(vfmsq_f32, add, self, mult)
    }
    #[inline(always)]
    unsafe fn negated_mul_sub_f32(self, mult: Self, sub: Self) -> Self {
        unsafe { self_from_op!(vfmaq_f32, sub, self, mult).negate_f32() }
    }
}

impl SimdRound for NeonReg {
    #[inline(always)]
    unsafe fn round_f64(self) -> Self {
        self_from_op!(vrndnq_f64, self)
    }
    #[inline(always)]
    unsafe fn round_f32(self) -> Self {
        self_from_op!(vrndnq_f32, self)
    }
    #[inline(always)]
    unsafe fn floor_f64(self) -> Self {
        self_from_op!(vrndmq_f64, self)
    }
    #[inline(always)]
    unsafe fn floor_f32(self) -> Self {
        self_from_op!(vrndmq_f32, self)
    }
    #[inline(always)]
    unsafe fn ceil_f64(self) -> Self {
        self_from_op!(vrndpq_f64, self)
    }
    #[inline(always)]
    unsafe fn ceil_f32(self) -> Self {
        self_from_op!(vrndpq_f32, self)
    }
}

impl SimdPartialOrd for NeonReg {
    type MaskType = Self;
    map_intrinsic!(cmp_f64_eq, vceqq_f64, self, rhs: Self);
    map_intrinsic!(cmp_f64_lt, vcltq_f64, self, rhs: Self);
    map_intrinsic!(cmp_f64_le, vcleq_f64, self, rhs: Self);
    map_intrinsic!(cmp_f64_gt, vcgtq_f64, self, rhs: Self);
    map_intrinsic!(cmp_f64_ge, vcgeq_f64, self, rhs: Self);
    map_intrinsic!(cmp_f32_eq, vceqq_f32, self, rhs: Self);
    map_intrinsic!(cmp_f32_lt, vcltq_f32, self, rhs: Self);
    map_intrinsic!(cmp_f32_le, vcleq_f32, self, rhs: Self);
    map_intrinsic!(cmp_f32_gt, vcgtq_f32, self, rhs: Self);
    map_intrinsic!(cmp_f32_ge, vcgeq_f32, self, rhs: Self);
    map_intrinsic!(cmp_i64_eq, vceqq_s64, self, rhs: Self);
    map_intrinsic!(cmp_i64_gt, vcgtq_s64, self, rhs: Self);
    map_intrinsic!(cmp_i32_eq, vceqq_s32, self, rhs: Self);
    map_intrinsic!(cmp_i32_gt, vcgtq_s32, self, rhs: Self);
    map_intrinsic!(cmp_i16_eq, vceqq_s16, self, rhs: Self);
    map_intrinsic!(cmp_i16_gt, vcgtq_s16, self, rhs: Self);
    map_intrinsic!(cmp_i8_eq, vceqq_s8, self, rhs: Self);
    map_intrinsic!(cmp_i8_gt, vcgtq_s8, self, rhs: Self);
    map_intrinsic!(cmp_u64_gt, vcgtq_u64, self, rhs: Self);
    map_intrinsic!(cmp_u32_gt, vcgtq_u32, self, rhs: Self);
    map_intrinsic!(cmp_u16_gt, vcgtq_u16, self, rhs: Self);
    map_intrinsic!(cmp_u8_gt, vcgtq_u8, self, rhs: Self);
    map_intrinsic!(max_f64, vmaxq_f64, self, rhs: Self);
    map_intrinsic!(min_f64, vminq_f64, self, rhs: Self);
    map_intrinsic!(max_f32, vmaxq_f32, self, rhs: Self);
    map_intrinsic!(min_f32, vminq_f32, self, rhs: Self);
    map_intrinsic!(max_i32, vmaxq_s32, self, rhs: Self);
    map_intrinsic!(min_i32, vminq_s32, self, rhs: Self);
    map_intrinsic!(max_i16, vmaxq_s16, self, rhs: Self);
    map_intrinsic!(min_i16, vminq_s16, self, rhs: Self);
    map_intrinsic!(max_i8, vmaxq_s8, self, rhs: Self);
    map_intrinsic!(min_i8, vminq_s8, self, rhs: Self);
    map_intrinsic!(max_u32, vmaxq_u32, self, rhs: Self);
    map_intrinsic!(min_u32, vminq_u32, self, rhs: Self);
    map_intrinsic!(max_u16, vmaxq_u16, self, rhs: Self);
    map_intrinsic!(min_u16, vminq_u16, self, rhs: Self);
    map_intrinsic!(max_u8, vmaxq_u8, self, rhs: Self);
    map_intrinsic!(min_u8, vminq_u8, self, rhs: Self);

    #[inline(always)]
    unsafe fn cmp_f64_ne(self, rhs: Self) -> Self {
        unsafe { Self(self.cmp_f64_eq(rhs).not().0) }
    }
    #[inline(always)]
    unsafe fn cmp_f32_ne(self, rhs: Self) -> Self {
        unsafe { Self(self.cmp_f32_eq(rhs).not().0) }
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

impl SimdSplat for NeonReg {
    #[inline(always)]
    unsafe fn splat_64<T>(val: T) -> Self {
        self_from_op!(vdupq_n_s64, val)
    }
    #[inline(always)]
    unsafe fn splat_32<T>(val: T) -> Self {
        self_from_op!(vdupq_n_s32, val)
    }
    #[inline(always)]
    unsafe fn splat_16<T>(val: T) -> Self {
        self_from_op!(vdupq_n_s16, val)
    }
    #[inline(always)]
    unsafe fn splat_8<T>(val: T) -> Self {
        self_from_op!(vdupq_n_s8, val)
    }
}

impl SimdGather for NeonReg {
    unsafe fn gather_32_from_32<T, const B: i32>(self, ptr: *const T) -> Self {
        unsafe {
            let ptr_32 = ptr as *const u32;
            let mut temp = [0u32; 4];
            self.store_unaligned(temp.as_mut_ptr());
            for i in 0..4 {
                temp[i] = *ptr_32.add(temp[i] as usize);
            }
            Self::load_unaligned(temp.as_ptr())
        }
    }

    #[inline(always)]
    unsafe fn gather_64_from_64<T, const B: i32>(self, ptr: *const T) -> Self {
        unsafe {
            let ptr_64 = ptr as *const u64;
            let mut temp = [0u64; 2];
            self.store_unaligned(temp.as_mut_ptr());
            for i in 0..2 {
                temp[i] = *ptr_64.add(temp[i] as usize);
            }
            Self::load_unaligned(temp.as_ptr())
        }
    }
}

impl SimdSqrt for NeonReg {
    #[inline(always)]
    unsafe fn sqrt_f64(self) -> Self {
        self_from_op!(vsqrtq_f64, self)
    }
    #[inline(always)]
    unsafe fn sqrt_f32(self) -> Self {
        self_from_op!(vsqrtq_f32, self)
    }
    #[inline(always)]
    unsafe fn rsqrt_f32(self) -> Self {
        self_from_op!(vrsqrteq_f32, self)
    }
}

impl SimdAllBits for NeonReg {
    #[inline(always)]
    unsafe fn all_zero(self) -> bool {
        unsafe { vmaxvq_u8(transmute_copy(&self.0)) == 0 }
    }
}

impl SimdNegate for NeonReg {
    #[inline(always)]
    unsafe fn negate_f64(self) -> Self {
        self_from_op!(vnegq_f64, self)
    }
    #[inline(always)]
    unsafe fn negate_f32(self) -> Self {
        self_from_op!(vnegq_f32, self)
    }
}

impl SimdBlockShift for NeonReg {
    // Stabilize const generic expr plssssss.
    #[inline(always)]
    unsafe fn block_left_byte_shift<const N: i32>(self) -> Self {
        match N {
            0 => self,
            1 => self_from_const_op!(vextq_u8, 15, Self::zero(), self),
            2 => self_from_const_op!(vextq_u8, 14, Self::zero(), self),
            3 => self_from_const_op!(vextq_u8, 13, Self::zero(), self),
            4 => self_from_const_op!(vextq_u8, 12, Self::zero(), self),
            5 => self_from_const_op!(vextq_u8, 11, Self::zero(), self),
            6 => self_from_const_op!(vextq_u8, 10, Self::zero(), self),
            7 => self_from_const_op!(vextq_u8, 9, Self::zero(), self),
            8 => self_from_const_op!(vextq_u8, 8, Self::zero(), self),
            9 => self_from_const_op!(vextq_u8, 7, Self::zero(), self),
            10 => self_from_const_op!(vextq_u8, 6, Self::zero(), self),
            11 => self_from_const_op!(vextq_u8, 5, Self::zero(), self),
            12 => self_from_const_op!(vextq_u8, 4, Self::zero(), self),
            13 => self_from_const_op!(vextq_u8, 3, Self::zero(), self),
            14 => self_from_const_op!(vextq_u8, 2, Self::zero(), self),
            15 => self_from_const_op!(vextq_u8, 1, Self::zero(), self),
            16 => unsafe { Self::zero() },
            _ => unreachable!(),
        }
    }
    #[inline(always)]
    unsafe fn block_right_byte_shift<const N: i32>(self) -> Self {
        self_from_const_op!(vextq_u8, N, self, Self::zero())
    }
}

// TODO: Add to_bits for other bit sizes for NEON.
impl SimdMaskBitConversion for NeonReg {
    #[inline(always)]
    unsafe fn to_bits_64(self) -> u64 {
        0u64
    }
    #[inline(always)]
    unsafe fn to_bits_32(self) -> u64 {
        unsafe {
            let single_bits = vshrq_n_u32::<31>(transmute(self.0));
            let iota = vld1q_u32([0, 1, 2, 3].as_ptr());
            let shifted = vshlq_u32(single_bits, transmute(iota));
            vaddvq_u32(shifted).into()
        }
    }
    #[inline(always)]
    unsafe fn to_bits_8(self) -> u64 {
        0u64
    }

    #[inline(always)]
    unsafe fn from_bits_64(bitmask: u64) -> Self {
        let mask: Self = unsafe { Self(transmute([1u64, 2u64])) };
        let bits = self_from_op!(vdupq_n_u64, bitmask);
        self_from_op!(vtstq_u64, bits, mask)
    }

    #[inline(always)]
    unsafe fn from_bits_32(bitmask: u64) -> Self {
        let mask: Self = unsafe { Self(transmute([1u32, 2u32, 4u32, 8u32])) };
        let bits = self_from_op!(vdupq_n_u32, bitmask as u32);
        self_from_op!(vtstq_u32, bits, mask)
    }

    #[inline(always)]
    unsafe fn from_bits_16(bitmask: u64) -> Self {
        let mask: Self = unsafe {
            Self(transmute([
                1u16, 2u16, 4u16, 8u16, 16u16, 32u16, 64u16, 128u16,
            ]))
        };
        let bits = self_from_op!(vdupq_n_u16, bitmask as u16);
        self_from_op!(vtstq_u16, bits, mask)
    }

    #[inline(always)]
    unsafe fn from_bits_8(bitmask: u64) -> Self {
        let mask: Self = unsafe {
            Self(transmute([
                1u8, 2u8, 4u8, 8u8, 16u8, 32u8, 64u8, 128u8, 1u8, 2u8, 4u8, 8u8, 16u8, 32u8, 64u8,
                128u8,
            ]))
        };
        let b0 = bitmask as u8;
        let b1 = (bitmask >> 8) as u8;
        let bits: Self = unsafe {
            Self(transmute([
                b0, b0, b0, b0, b0, b0, b0, b0, b1, b1, b1, b1, b1, b1, b1, b1,
            ]))
        };
        self_from_op!(vtstq_u8, bits, mask)
    }
}

impl SimdLaneShift for NeonReg {
    #[inline(always)]
    unsafe fn left_lane_shift_32(self, n: u32) -> Self {
        match n {
            0 => self,
            1 => self_from_const_op!(vextq_u8, 4, self, Self::zero()),
            2 => self_from_const_op!(vextq_u8, 8, self, Self::zero()),
            3 => self_from_const_op!(vextq_u8, 12, self, Self::zero()),
            _ => unsafe { Self::zero() },
        }
    }
    #[inline(always)]
    unsafe fn right_lane_shift_32(self, n: u32) -> Self {
        match n {
            0 => self,
            1 => self_from_const_op!(vextq_u8, 12, Self::zero(), self),
            2 => self_from_const_op!(vextq_u8, 8, Self::zero(), self),
            3 => self_from_const_op!(vextq_u8, 4, Self::zero(), self),
            _ => unsafe { Self::zero() },
        }
    }
}
