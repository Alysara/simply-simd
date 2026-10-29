#![allow(clippy::missing_safety_doc)]
use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use crate::{Architecture, SimdElement, register::Simd, simd_array::SimdToArray};

pub trait Arch: Clone + Copy + Default {
    const SIMD_WIDTH: usize;
    const NUM_SIMD_REG: usize;
    const ARCHITECTURE: Architecture;

    type Block2<T: SimdElement>: Deref<Target = [Simd<T, Self>]> + DerefMut + Default;

    type Block4<T: SimdElement>: Deref<Target = [Simd<T, Self>]> + DerefMut + Default;

    type Vec: SimdArch
        + Copy
        + Clone
        + SimdLoad<MaskType = Self::Mask>
        + SimdStore<MaskType = Self::Mask>
        + SimdPartialOrd<MaskType = Self::Mask>;
    type Mask: MaskArch + Copy + Clone + SimdVariableBlend<VecType = Self::Vec>;
    type ScalarArch: Arch;

    type Array64<T: Debug + Copy>: Debug + Copy + SimdToArray<T>;
    type Array32<T: Debug + Copy>: Debug + Copy + SimdToArray<T>;
    type Array16<T: Debug + Copy>: Debug + Copy + SimdToArray<T>;
    type Array8<T: Debug + Copy>: Debug + Copy + SimdToArray<T>;
}

pub trait SimdArch:
    Copy
    + Clone
    + SimdAdd
    + SimdSub
    + SimdMul
    + SimdDiv
    + SimdBitwise
    + SimdShift
    + SimdLoad
    + SimdStore
    + SimdZero
    + SimdFloatCasts
    + SimdIntCasts
    + SimdPermute
    + SimdInterleave
    + SimdMulAdd
    + SimdRound
    + SimdPartialOrd
    + SimdSplat
    + SimdGather
    + SimdSqrt
    + SimdNegate
    + SimdBlockShift
    + SimdImmediateBlend
    + SimdLaneShift
{
}

pub trait MaskArch:
    Copy + Clone + SimdBitwise + SimdAllBits + SimdVariableBlend + SimdMaskBitConversion
{
}

#[repr(transparent)]
pub struct Block<V, const N: usize>([V; N]);

impl<V: Default, const N: usize> Default for Block<V, N> {
    #[inline(always)]
    fn default() -> Self {
        Block(core::array::from_fn(|_| V::default()))
    }
}

impl<V, const N: usize> Deref for Block<V, N> {
    type Target = [V];
    #[inline(always)]
    fn deref(&self) -> &[V] {
        &self.0
    }
}

impl<V, const N: usize> DerefMut for Block<V, N> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [V] {
        &mut self.0
    }
}

// pub trait BlockExt<V> {
//     /// # Safety
//     /// `index` must be in bounds.
//     unsafe fn get_unchecked(&self, index: usize) -> &V;
//     /// # Safety
//     /// `index` must be in bounds.
//     unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut V;
// }
//
// impl<V, B> BlockExt<V> for B
// where
//     B: AsRef<[V]> + AsMut<[V]>,
// {
//     #[inline(always)]
//     unsafe fn get_unchecked(&self, index: usize) -> &V {
//         unsafe { self.as_ref().get_unchecked(index) }
//     }
//
//     #[inline(always)]
//     unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut V {
//         unsafe { self.as_mut().get_unchecked_mut(index) }
//     }
// }

// === Arithmetic ===
pub trait SimdAdd {
    unsafe fn f64_add(self, rhs: Self) -> Self;
    unsafe fn f32_add(self, rhs: Self) -> Self;
    unsafe fn i64_add(self, rhs: Self) -> Self;
    unsafe fn i32_add(self, rhs: Self) -> Self;
    unsafe fn i16_add(self, rhs: Self) -> Self;
    unsafe fn i8_add(self, rhs: Self) -> Self;
}

pub trait SimdSub {
    unsafe fn f64_sub(self, rhs: Self) -> Self;
    unsafe fn f32_sub(self, rhs: Self) -> Self;
    unsafe fn i64_sub(self, rhs: Self) -> Self;
    unsafe fn i32_sub(self, rhs: Self) -> Self;
    unsafe fn i16_sub(self, rhs: Self) -> Self;
    unsafe fn i8_sub(self, rhs: Self) -> Self;
}

pub trait SimdMul {
    unsafe fn f64_mul(self, rhs: Self) -> Self;
    unsafe fn f32_mul(self, rhs: Self) -> Self;
    unsafe fn i32_mul(self, rhs: Self) -> Self;
    unsafe fn i16_mul(self, rhs: Self) -> Self;
}

pub trait SimdDiv {
    unsafe fn f64_div(self, rhs: Self) -> Self;
    unsafe fn f32_div(self, rhs: Self) -> Self;
}

pub trait SimdBitwise {
    unsafe fn and(self, rhs: Self) -> Self;
    unsafe fn or(self, rhs: Self) -> Self;
    unsafe fn xor(self, rhs: Self) -> Self;
    unsafe fn not(self) -> Self;
    unsafe fn and_not(self, rhs: Self) -> Self;
}

pub trait SimdShift {
    unsafe fn sll_64(self, rhs: u32) -> Self;
    unsafe fn srl_64(self, rhs: u32) -> Self;
    unsafe fn sra_64(self, rhs: u32) -> Self;
    unsafe fn sll_32(self, rhs: u32) -> Self;
    unsafe fn srl_32(self, rhs: u32) -> Self;
    unsafe fn sra_32(self, rhs: u32) -> Self;
    unsafe fn sll_16(self, rhs: u32) -> Self;
    unsafe fn srl_16(self, rhs: u32) -> Self;
    unsafe fn sra_16(self, rhs: u32) -> Self;
    unsafe fn slli_64<const N: i32>(self) -> Self;
    unsafe fn srli_64<const N: i32>(self) -> Self;
    unsafe fn srai_64<const N: i32>(self) -> Self;
    unsafe fn slli_32<const N: i32>(self) -> Self;
    unsafe fn srli_32<const N: i32>(self) -> Self;
    unsafe fn srai_32<const N: i32>(self) -> Self;
    unsafe fn slli_16<const N: i32>(self) -> Self;
    unsafe fn srli_16<const N: i32>(self) -> Self;
    unsafe fn srai_16<const N: i32>(self) -> Self;
    unsafe fn sllv_64(self, shift: Self) -> Self;
    unsafe fn srlv_64(self, shift: Self) -> Self;
    unsafe fn srav_64(self, shift: Self) -> Self;
    unsafe fn sllv_32(self, shift: Self) -> Self;
    unsafe fn srlv_32(self, shift: Self) -> Self;
    unsafe fn srav_32(self, shift: Self) -> Self;
    unsafe fn sllv_16(self, shift: Self) -> Self;
    unsafe fn srlv_16(self, shift: Self) -> Self;
    unsafe fn srav_16(self, shift: Self) -> Self;
}

pub trait SimdLoad {
    type MaskType;
    unsafe fn load_aligned<T>(ptr: *const T) -> Self;
    unsafe fn load_unaligned<T>(ptr: *const T) -> Self;
    unsafe fn masked_load_64<T>(ptr: *const T, mask: Self::MaskType) -> Self;
    unsafe fn masked_load_32<T>(ptr: *const T, mask: Self::MaskType) -> Self;
    // TODO: There are byte and short mask loaders as well.
}

pub trait SimdStore {
    type MaskType;
    unsafe fn store_aligned<T>(self, ptr: *mut T);
    unsafe fn store_unaligned<T>(self, ptr: *mut T);
    unsafe fn masked_store_64<T>(self, ptr: *mut T, mask: Self::MaskType);
    unsafe fn masked_store_32<T>(self, ptr: *mut T, mask: Self::MaskType);
    // TODO: There are byte and short mask storers as well.
}

// pub trait SimdExtract {
//     unsafe fn extract_64<T, const N: i32>(self) -> T;
//     unsafe fn extract_32<T, const N: i32>(self) -> T;
//     unsafe fn extract_16<T, const N: i32>(self) -> T;
//     unsafe fn extract_8<T, const N: i32>(self) -> T;
// }

// pub trait SimdInsert {
//     unsafe fn insert_64<T, const N: i32>(self, val: T) -> Self;
//     unsafe fn insert_32<T, const N: i32>(self, val: T) -> Self;
//     unsafe fn insert_16<T, const N: i32>(self, val: T) -> Self;
//     unsafe fn insert_8<T, const N: i32>(self, val: T) -> Self;
// }

pub trait SimdZero {
    unsafe fn zero() -> Self;
}

pub trait SimdFloatCasts {
    unsafe fn float_to_int_trunc(self) -> Self;
    unsafe fn float_to_int_round(self) -> Self;
}

pub trait SimdIntCasts {
    unsafe fn int_to_float(self) -> Self;
}

pub trait SimdPermute {
    unsafe fn vpermute_64(self, rhs: Self) -> Self;
    unsafe fn vpermute_32(self, rhs: Self) -> Self;
    unsafe fn vshuffle_16(self, rhs: Self) -> Self;
    unsafe fn vshuffle_8(self, rhs: Self) -> Self;
    unsafe fn permute_128<const N: i32>(self) -> Self;
    unsafe fn shuffle_64<const N: i32>(self) -> Self;
    unsafe fn shuffle_32<const N: i32>(self) -> Self;
    unsafe fn shuffle_16<const N: i32>(self) -> Self;
    unsafe fn permute2_half<const N: i32>(self, rhs: Self) -> Self;
    unsafe fn shuffle2_128<const N: i32>(self, rhs: Self) -> Self;
    unsafe fn shuffle2_64<const N: i32>(self, rhs: Self) -> Self;
    unsafe fn shuffle2_32<const N: i32>(self, rhs: Self) -> Self;
}

pub trait SimdVariableBlend {
    type VecType;
    unsafe fn vblend_64(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> Self::VecType;
    unsafe fn vblend_32(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> Self::VecType;
    unsafe fn vblend_16(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> Self::VecType;
    unsafe fn vblend_8(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> Self::VecType;
}

pub trait SimdImmediateBlend {
    unsafe fn blend_64<const N: i32>(self, false_values: Self) -> Self;
    unsafe fn blend_32<const N: i32>(self, false_values: Self) -> Self;
    unsafe fn blend_16<const N: i32>(self, false_values: Self) -> Self;
}

pub trait SimdInterleave {
    unsafe fn interleave_lower_128(self, rhs: Self) -> Self;
    unsafe fn interleave_lower_64(self, rhs: Self) -> Self;
    unsafe fn interleave_lower_32(self, rhs: Self) -> Self;
    unsafe fn interleave_lower_16(self, rhs: Self) -> Self;
    unsafe fn interleave_lower_8(self, rhs: Self) -> Self;
    unsafe fn interleave_upper_128(self, rhs: Self) -> Self;
    unsafe fn interleave_upper_64(self, rhs: Self) -> Self;
    unsafe fn interleave_upper_32(self, rhs: Self) -> Self;
    unsafe fn interleave_upper_16(self, rhs: Self) -> Self;
    unsafe fn interleave_upper_8(self, rhs: Self) -> Self;
    unsafe fn interleave_lower_block_64(self, rhs: Self) -> Self;
    unsafe fn interleave_lower_block_32(self, rhs: Self) -> Self;
    unsafe fn interleave_lower_block_16(self, rhs: Self) -> Self;
    unsafe fn interleave_lower_block_8(self, rhs: Self) -> Self;
    unsafe fn interleave_upper_block_64(self, rhs: Self) -> Self;
    unsafe fn interleave_upper_block_32(self, rhs: Self) -> Self;
    unsafe fn interleave_upper_block_16(self, rhs: Self) -> Self;
    unsafe fn interleave_upper_block_8(self, rhs: Self) -> Self;
    unsafe fn interleave_even_128(self, rhs: Self) -> Self;
    unsafe fn interleave_even_64(self, rhs: Self) -> Self;
    unsafe fn interleave_even_32(self, rhs: Self) -> Self;
    unsafe fn interleave_even_16(self, rhs: Self) -> Self;
    unsafe fn interleave_even_8(self, rhs: Self) -> Self;
    unsafe fn interleave_odd_128(self, rhs: Self) -> Self;
    unsafe fn interleave_odd_64(self, rhs: Self) -> Self;
    unsafe fn interleave_odd_32(self, rhs: Self) -> Self;
    unsafe fn interleave_odd_16(self, rhs: Self) -> Self;
    unsafe fn interleave_odd_8(self, rhs: Self) -> Self;
    unsafe fn deinterleave_even_128(self, rhs: Self) -> Self;
    unsafe fn deinterleave_even_64(self, rhs: Self) -> Self;
    unsafe fn deinterleave_even_32(self, rhs: Self) -> Self;
    unsafe fn deinterleave_even_16(self, rhs: Self) -> Self;
    unsafe fn deinterleave_even_8(self, rhs: Self) -> Self;
    unsafe fn deinterleave_odd_128(self, rhs: Self) -> Self;
    unsafe fn deinterleave_odd_64(self, rhs: Self) -> Self;
    unsafe fn deinterleave_odd_32(self, rhs: Self) -> Self;
    unsafe fn deinterleave_odd_16(self, rhs: Self) -> Self;
    unsafe fn deinterleave_odd_8(self, rhs: Self) -> Self;
}

pub trait SimdMulAdd {
    unsafe fn mul_add_f64(self, mult: Self, add: Self) -> Self;
    unsafe fn mul_sub_f64(self, mult: Self, sub: Self) -> Self;
    unsafe fn negated_mul_add_f64(self, mult: Self, add: Self) -> Self;
    unsafe fn negated_mul_sub_f64(self, mult: Self, sub: Self) -> Self;
    unsafe fn mul_add_f32(self, mult: Self, add: Self) -> Self;
    unsafe fn mul_sub_f32(self, mult: Self, sub: Self) -> Self;
    unsafe fn negated_mul_add_f32(self, mult: Self, add: Self) -> Self;
    unsafe fn negated_mul_sub_f32(self, mult: Self, sub: Self) -> Self;
}

pub trait SimdRound {
    unsafe fn round_f64(self) -> Self;
    unsafe fn round_f32(self) -> Self;
    unsafe fn floor_f64(self) -> Self;
    unsafe fn floor_f32(self) -> Self;
    unsafe fn ceil_f64(self) -> Self;
    unsafe fn ceil_f32(self) -> Self;
}

pub trait SimdPartialOrd {
    type MaskType;
    unsafe fn cmp_f64_eq(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_f64_lt(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_f64_le(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_f64_gt(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_f64_ge(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_f64_ne(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_f32_eq(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_f32_lt(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_f32_le(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_f32_gt(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_f32_ge(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_f32_ne(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_i64_eq(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_i64_gt(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_u64_gt(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_i32_eq(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_i32_gt(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_u32_gt(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_i16_eq(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_i16_gt(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_u16_gt(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_i8_eq(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_i8_gt(self, rhs: Self) -> Self::MaskType;
    unsafe fn cmp_u8_gt(self, rhs: Self) -> Self::MaskType;
    unsafe fn max_f64(self, rhs: Self) -> Self;
    unsafe fn min_f64(self, rhs: Self) -> Self;
    unsafe fn max_f32(self, rhs: Self) -> Self;
    unsafe fn min_f32(self, rhs: Self) -> Self;
    unsafe fn max_i64(self, rhs: Self) -> Self;
    unsafe fn min_i64(self, rhs: Self) -> Self;
    unsafe fn max_i32(self, rhs: Self) -> Self;
    unsafe fn min_i32(self, rhs: Self) -> Self;
    unsafe fn max_i16(self, rhs: Self) -> Self;
    unsafe fn min_i16(self, rhs: Self) -> Self;
    unsafe fn max_i8(self, rhs: Self) -> Self;
    unsafe fn min_i8(self, rhs: Self) -> Self;
    unsafe fn max_u64(self, rhs: Self) -> Self;
    unsafe fn min_u64(self, rhs: Self) -> Self;
    unsafe fn max_u32(self, rhs: Self) -> Self;
    unsafe fn min_u32(self, rhs: Self) -> Self;
    unsafe fn max_u16(self, rhs: Self) -> Self;
    unsafe fn min_u16(self, rhs: Self) -> Self;
    unsafe fn max_u8(self, rhs: Self) -> Self;
    unsafe fn min_u8(self, rhs: Self) -> Self;
}

pub trait SimdSplat {
    unsafe fn splat_64<T>(val: T) -> Self;
    unsafe fn splat_32<T>(val: T) -> Self;
    unsafe fn splat_16<T>(val: T) -> Self;
    unsafe fn splat_8<T>(val: T) -> Self;
}

pub trait SimdGather {
    unsafe fn gather_32_from_32<T, const B: i32>(self, ptr: *const T) -> Self;
    // unsafe fn gather_64_from_32<T, const B: i32>(ptr: *const T, indicies: Self) -> Self;
    // unsafe fn gather_32_from_64<T, const B: i32>(ptr: *const T, indicies: Self) -> Self;
    unsafe fn gather_64_from_64<T, const B: i32>(self, ptr: *const T) -> Self;
}

pub trait SimdSqrt {
    unsafe fn sqrt_f64(self) -> Self;
    unsafe fn sqrt_f32(self) -> Self;
    // unsafe fn rsqrt_f64(self) -> Self;
    unsafe fn rsqrt_f32(self) -> Self;
}

pub trait SimdAllBits {
    unsafe fn all_zero(self) -> bool;
}

pub trait SimdNegate {
    unsafe fn negate_f64(self) -> Self;
    unsafe fn negate_f32(self) -> Self;
}

pub trait SimdBlockShift {
    unsafe fn block_right_byte_shift<const N: i32>(self) -> Self;
    unsafe fn block_left_byte_shift<const N: i32>(self) -> Self;
}

// TODO: Add to_bits_16 and from_bits.
pub trait SimdMaskBitConversion {
    unsafe fn to_bits_64(self) -> u64;
    unsafe fn to_bits_32(self) -> u64;
    // unsafe fn to_bits_16(self) -> u64;
    unsafe fn to_bits_8(self) -> u64;
    unsafe fn from_bits_64(bitmask: u64) -> Self;
    unsafe fn from_bits_32(bitmask: u64) -> Self;
    unsafe fn from_bits_16(bitmask: u64) -> Self;
    unsafe fn from_bits_8(bitmask: u64) -> Self;
}

pub trait SimdLaneShift {
    unsafe fn right_lane_shift_32(self, n: u32) -> Self;
    unsafe fn left_lane_shift_32(self, n: u32) -> Self;
}
