use std::mem::transmute_copy;

use crate::architectures::interface::*;

#[derive(Copy, Clone)]
#[repr(align(8))]
pub struct ScalarReg<const N: usize>(pub [u8; N]);
#[derive(Copy, Clone)]
pub struct ScalarMask<const N: usize>(pub [bool; N]);

impl<const N: usize> SimdArch for ScalarReg<N> {}
impl<const N: usize> MaskArch for ScalarMask<N> {}

macro_rules! scalar_token_op {
    ($type:ty, $op:tt, $self:ident, $rhs:ident, $size:expr) => {
        unsafe {
            let mut new = ScalarReg::<$size>([0; $size]);

            let self_ptr: *const $type = $self.0.as_ptr() as *const $type;
            let rhs_ptr: *const $type = $rhs.0.as_ptr() as *const $type;
            let new_ptr: *mut $type = new.0.as_mut_ptr() as *mut $type;

            for i in 0..($size / size_of::<$type>()) {
                *new_ptr.add(i) = *self_ptr.add(i) $op *rhs_ptr.add(i);
            }

            new
        }
    }
}

macro_rules! scalar_token_op_usize_rhs {
    ($type:ty, $op:tt, $self:ident, $rhs:ident, $size:expr) => {
        unsafe {
            let mut new = ScalarReg::<$size>([0; $size]);

            let self_ptr: *const $type = $self.0.as_ptr() as *const $type;
            let rhs_ptr: *const $type = $rhs.0.as_ptr() as *const $type;
            let new_ptr: *mut $type = new.0.as_mut_ptr() as *mut $type;

            for i in 0..($size / size_of::<$type>()) {
                *new_ptr.add(i) = *self_ptr.add(i) $op (*rhs_ptr.add(i) as usize);
            }

            new
        }
    }
}

macro_rules! scalar_func_op {
    ($type:ty, $op:ident, $self:ident, $rhs:ident, $size:expr) => {
        unsafe {
            let mut new = ScalarReg::<$size>([0; $size]);

            let self_ptr: *const $type = $self.0.as_ptr() as *const $type;
            let rhs_ptr: *const $type = $rhs.0.as_ptr() as *const $type;
            let new_ptr: *mut $type = new.0.as_mut_ptr() as *mut $type;

            for i in 0..($size / size_of::<$type>()) {
                *new_ptr.add(i) = (*self_ptr.add(i)).$op(*rhs_ptr.add(i));
            }

            new
        }
    };
}

macro_rules! map_scalar_op {
    ($func:ident, $type:ty, $op:ident, $size:expr) => {
        #[inline(always)]
        unsafe fn $func(self, rhs: Self) -> Self {
            scalar_func_op!($type, $op, self, rhs, $size)
        }
    };
}

macro_rules! scalar_self_op {
    ($type:ty, $op:ident, $self:ident, $size:expr) => {
        unsafe {
            let mut new = ScalarReg::<$size>([0; $size]);

            let self_ptr: *const $type = $self.0.as_ptr() as *const $type;
            let new_ptr: *mut $type = new.0.as_mut_ptr() as *mut $type;

            for i in 0..($size / size_of::<$type>()) {
                *new_ptr.add(i) = (*self_ptr.add(i)).$op();
            }
            new
        }
    };
}

macro_rules! scalar_fma_expr_op {
    ($type:ty, $self:ident, $mult:ident, $add:ident, $size:expr, |$a:ident, $b:ident, $c:ident| $op:expr) => {
        unsafe {
            let mut new = ScalarReg::<$size>([0; $size]);

            let a_ptr: *const $type = $self.0.as_ptr() as *const $type;
            let b_ptr: *const $type = $mult.0.as_ptr() as *const $type;
            let c_ptr: *const $type = $add.0.as_ptr() as *const $type;
            let new_ptr: *mut $type = new.0.as_mut_ptr() as *mut $type;

            for i in 0..($size / size_of::<$type>()) {
                let $a = *a_ptr.add(i);
                let $b = *b_ptr.add(i);
                let $c = *c_ptr.add(i);
                *new_ptr.add(i) = $op;
            }
            new
        }
    };
}

macro_rules! scalar_splat {
    {$type:ty, $self:ident, $val:expr, $size:expr} => {
        unsafe {
            let mut new = ScalarReg::<$size>([0; $size]);
            let new_ptr = new.0.as_mut_ptr() as *mut $type;

            for i in 0..(N / size_of::<$type>()) {
                *new_ptr.add(i) = $val;
            }

            new
        }
    }
}

macro_rules! scalar_shift {
    ($ty:ty, $shift:tt, $self:ident, $rhs:ident, $size:expr) => {
        unsafe {
            let mut new = ScalarReg::<$size>([0; $size]);
            let new_ptr = new.0.as_mut_ptr() as *mut $ty;
            let old_ptr = $self.0.as_ptr() as *const $ty;

            for i in 0..(N / size_of::<$ty>()) {
                *new_ptr.add(i) = *old_ptr.add(i) $shift $rhs as usize;
            }

            new
        }
    }
}

impl<const N: usize> SimdAdd for ScalarReg<N> {
    #[inline(always)]
    unsafe fn f64_add(self, rhs: Self) -> Self {
        scalar_token_op!(f64, +, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn f32_add(self, rhs: Self) -> Self {
        scalar_token_op!(f32, +, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn i64_add(self, rhs: Self) -> Self {
        scalar_func_op!(i64, wrapping_add, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn i32_add(self, rhs: Self) -> Self {
        scalar_func_op!(i32, wrapping_add, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn i16_add(self, rhs: Self) -> Self {
        scalar_func_op!(i16, wrapping_add, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn i8_add(self, rhs: Self) -> Self {
        scalar_func_op!(i8, wrapping_add, self, rhs, N)
    }
}

impl<const N: usize> SimdSub for ScalarReg<N> {
    #[inline(always)]
    unsafe fn f64_sub(self, rhs: Self) -> Self {
        scalar_token_op!(f64, -, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn f32_sub(self, rhs: Self) -> Self {
        scalar_token_op!(f32, -, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn i64_sub(self, rhs: Self) -> Self {
        scalar_func_op!(i64, wrapping_sub, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn i32_sub(self, rhs: Self) -> Self {
        scalar_func_op!(i32, wrapping_sub, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn i16_sub(self, rhs: Self) -> Self {
        scalar_func_op!(i16, wrapping_sub, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn i8_sub(self, rhs: Self) -> Self {
        scalar_func_op!(i8, wrapping_sub, self, rhs, N)
    }
}

impl<const N: usize> SimdMul for ScalarReg<N> {
    #[inline(always)]
    unsafe fn f64_mul(self, rhs: Self) -> Self {
        scalar_token_op!(f64, *, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn f32_mul(self, rhs: Self) -> Self {
        scalar_token_op!(f32, *, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn i32_mul(self, rhs: Self) -> Self {
        scalar_func_op!(i32, wrapping_mul, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn i16_mul(self, rhs: Self) -> Self {
        scalar_func_op!(i16, wrapping_mul, self, rhs, N)
    }
}

impl<const N: usize> SimdDiv for ScalarReg<N> {
    #[inline(always)]
    unsafe fn f64_div(self, rhs: Self) -> Self {
        scalar_token_op!(f64, /, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn f32_div(self, rhs: Self) -> Self {
        scalar_token_op!(f32, /, self, rhs, N)
    }
}

impl<const N: usize> SimdBitwise for ScalarReg<N> {
    #[inline(always)]
    unsafe fn and(self, rhs: Self) -> Self {
        scalar_token_op!(u64, &, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn or(self, rhs: Self) -> Self {
        scalar_token_op!(u64, |, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn xor(self, rhs: Self) -> Self {
        scalar_token_op!(u64, ^, self, rhs, N)
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        unsafe { self.xor(Self([255; N])) }
    }
    #[inline(always)]
    unsafe fn and_not(self, rhs: Self) -> Self {
        unsafe { self.and(rhs.not()) }
    }
}

macro_rules! map_scalar_shift {
    ($func:ident, $ty:ty, $shift:tt) => {
        #[inline(always)]
        unsafe fn $func(self, rhs: u32) -> Self {
            scalar_shift!($ty, $shift, self, rhs, N)
        }
    };
}

macro_rules! map_scalar_ishift {
    ($func:ident, $ty:ty, $shift:tt) => {
        #[inline(always)]
        unsafe fn $func<const M: i32>(self) -> Self {
            scalar_shift!($ty, $shift, self, M, N)
        }
    };
}

macro_rules! map_scalar_vshift {
    ($func:ident, $ty:ty, $shift:tt) => {
        #[inline(always)]
        unsafe fn $func(self, rhs: Self) -> Self {
            scalar_token_op_usize_rhs!($ty, $shift, self, rhs, N)
        }
    };
}

impl<const N: usize> SimdShift for ScalarReg<N> {
    map_scalar_shift!(sll_64, u64, <<);
    map_scalar_shift!(srl_64, u64, >>);
    map_scalar_shift!(sra_64, i64, >>);
    map_scalar_shift!(sll_32, u32, <<);
    map_scalar_shift!(srl_32, u32, >>);
    map_scalar_shift!(sra_32, i32, >>);
    map_scalar_shift!(sll_16, u16, <<);
    map_scalar_shift!(srl_16, u16, >>);
    map_scalar_shift!(sra_16, i16, >>);
    map_scalar_ishift!(slli_64, u64, <<);
    map_scalar_ishift!(srli_64, u64, >>);
    map_scalar_ishift!(srai_64, i64, >>);
    map_scalar_ishift!(slli_32, u32, <<);
    map_scalar_ishift!(srli_32, u32, >>);
    map_scalar_ishift!(srai_32, i32, >>);
    map_scalar_ishift!(slli_16, u16, <<);
    map_scalar_ishift!(srli_16, u16, >>);
    map_scalar_ishift!(srai_16, i16, >>);
    map_scalar_vshift!(sllv_64, u64, <<);
    map_scalar_vshift!(srlv_64, u64, >>);
    map_scalar_vshift!(srav_64, i64, >>);
    map_scalar_vshift!(sllv_32, u32, <<);
    map_scalar_vshift!(srlv_32, u32, >>);
    map_scalar_vshift!(srav_32, i32, >>);
    map_scalar_vshift!(sllv_16, u16, <<);
    map_scalar_vshift!(srlv_16, u16, >>);
    map_scalar_vshift!(srav_16, i16, >>);
}

impl<const N: usize> SimdLoad for ScalarReg<N> {
    type MaskType = ScalarMask<N>;
    #[inline(always)]
    unsafe fn load_aligned<T>(ptr: *const T) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            std::ptr::copy_nonoverlapping(ptr as *const u8, new.0.as_mut_ptr(), N);
            new
        }
    }
    #[inline(always)]
    unsafe fn load_unaligned<T>(ptr: *const T) -> Self {
        unsafe { Self::load_aligned::<T>(ptr) }
    }
    #[inline(always)]
    unsafe fn masked_load_64<T>(ptr: *const T, mask: Self::MaskType) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u64;
            let data_ptr = ptr as *const u64;
            for i in 0..(N / 8) {
                *new_ptr.add(i) = if mask.0[i] { *data_ptr.add(i) } else { 0 };
            }
            new
        }
    }
    #[inline(always)]
    unsafe fn masked_load_32<T>(ptr: *const T, mask: Self::MaskType) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u32;
            let data_ptr = ptr as *const u32;
            for i in 0..(N / 4) {
                *new_ptr.add(i) = if mask.0[i] { *data_ptr.add(i) } else { 0 };
            }
            new
        }
    }
}

impl<const N: usize> SimdStore for ScalarReg<N> {
    type MaskType = ScalarMask<N>;
    #[inline(always)]
    unsafe fn store_aligned<T>(self, ptr: *mut T) {
        unsafe {
            std::ptr::copy_nonoverlapping(self.0.as_ptr(), ptr as *mut u8, N);
        }
    }
    #[inline(always)]
    unsafe fn store_unaligned<T>(self, ptr: *mut T) {
        unsafe { self.store_aligned(ptr) };
    }
    #[inline(always)]
    unsafe fn masked_store_64<T>(self, ptr: *mut T, mask: Self::MaskType) {
        unsafe {
            let self_ptr = self.0.as_ptr() as *const u64;
            let store_ptr = ptr as *mut u64;
            for i in 0..(N / 8) {
                if mask.0[i] {
                    *store_ptr.add(i) = *self_ptr.add(i);
                }
            }
        }
    }
    #[inline(always)]
    unsafe fn masked_store_32<T>(self, ptr: *mut T, mask: Self::MaskType) {
        unsafe {
            let self_ptr = self.0.as_ptr() as *const u32;
            let store_ptr = ptr as *mut u32;
            for i in 0..(N / 4) {
                if mask.0[i] {
                    *store_ptr.add(i) = *self_ptr.add(i);
                }
            }
        }
    }
}

impl<const N: usize> SimdZero for ScalarReg<N> {
    #[inline(always)]
    unsafe fn zero() -> Self {
        Self([0; N])
    }
}

impl<const N: usize> SimdFloatCasts for ScalarReg<N> {
    #[inline(always)]
    unsafe fn float_to_int_trunc(self) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let float_ptr = self.0.as_ptr() as *const f32;
            let int_ptr = new.0.as_mut_ptr() as *mut i32;
            for i in 0..(N / 4) {
                *int_ptr.add(i) = (*float_ptr.add(i)) as i32;
            }
            new
        }
    }
    #[inline(always)]
    unsafe fn float_to_int_round(self) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let float_ptr = self.0.as_ptr() as *const f32;
            let int_ptr = new.0.as_mut_ptr() as *mut i32;
            for i in 0..(N / 4) {
                *int_ptr.add(i) = (*float_ptr.add(i)).round_ties_even() as i32;
            }
            new
        }
    }
}

impl<const N: usize> SimdIntCasts for ScalarReg<N> {
    #[inline(always)]
    unsafe fn int_to_float(self) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let int_ptr = self.0.as_ptr() as *const i32;
            let float_ptr = new.0.as_mut_ptr() as *mut f32;
            for i in 0..(N / 4) {
                *float_ptr.add(i) = (*int_ptr.add(i)) as f32;
            }
            new
        }
    }
}

impl<const N: usize> SimdPermute for ScalarReg<N> {
    unsafe fn vpermute_64(self, rhs: Self) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u64;
            let self_ptr = self.0.as_ptr() as *const u64;
            let indices_ptr = rhs.0.as_ptr() as *const u64;
            for i in 0..(N / 8) {
                let index = (*indices_ptr.add(i) as usize) & (N / 8 - 1);
                *new_ptr.add(i) = *self_ptr.add(index);
            }
            new
        }
    }
    unsafe fn vpermute_32(self, rhs: Self) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u32;
            let self_ptr = self.0.as_ptr() as *const u32;
            let indices_ptr = rhs.0.as_ptr() as *const u32;
            for i in 0..(N / 4) {
                let index = (*indices_ptr.add(i) as usize) & (N / 4 - 1);
                *new_ptr.add(i) = *self_ptr.add(index);
            }
            new
        }
    }
    unsafe fn vshuffle_16(self, rhs: Self) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u16;
            let self_ptr = self.0.as_ptr() as *const u16;
            let indices_ptr = rhs.0.as_ptr() as *const u16;
            for i in 0..(N / 2) {
                let lane_base = i & !7; // which 16-byte lane we're in (0 or 8)
                let index = (*indices_ptr.add(i) as usize & 7) + lane_base;
                *new_ptr.add(i) = *self_ptr.add(index);
            }
            new
        }
    }
    unsafe fn vshuffle_8(self, rhs: Self) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr();
            let self_ptr = self.0.as_ptr();
            let indices_ptr = rhs.0.as_ptr();
            for i in 0..N {
                let lane_base = i & !15; // which 16-byte lane we're in (0 or 16)
                let index = (*indices_ptr.add(i) as usize & 15) + lane_base;
                *new_ptr.add(i) = *self_ptr.add(index);
            }
            new
        }
    }
    unsafe fn permute_128<const M: i32>(self) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let lanes = N / 16;
            let src = self.0.as_ptr();
            let dst = new.0.as_mut_ptr();

            for lane in 0..lanes {
                let sel = ((M >> (2 * lane)) & (lanes as i32 - 1)) as usize;
                core::ptr::copy_nonoverlapping(src.add(sel * 16), dst.add(lane * 16), 16);
            }
            new
        }
    }
    unsafe fn shuffle_64<const M: i32>(self) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u64;
            let self_ptr = self.0.as_ptr() as *const u64;

            let sel1 = (M & 1) as usize;
            let sel2 = ((M >> 1) & 1) as usize;

            for i in (0..(N / 8)).step_by(2) {
                *new_ptr.add(i) = *self_ptr.add(i + sel1);
                *new_ptr.add(i + 1) = *self_ptr.add(i + sel2);
            }

            new
        }
    }
    unsafe fn shuffle_32<const M: i32>(self) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u32;
            let self_ptr = self.0.as_ptr() as *const u32;

            for i in 0..(N / 4) {
                let base = i & !3;
                let sel = (M >> ((i & 3) * 2)) as usize & 3;
                *new_ptr.add(i) = *self_ptr.add(base + sel);
            }

            new
        }
    }
    unsafe fn shuffle_16<const M: i32>(self) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u16;
            let self_ptr = self.0.as_ptr() as *const u16;

            for i in 0..(N / 2) {
                let base = i & !3;
                let sel = (M >> ((i & 3) * 2)) as usize & 3;
                *new_ptr.add(i) = *self_ptr.add(base + sel);
            }

            new
        }
    }
    unsafe fn permute2_half<const M: i32>(self, rhs: Self) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u64;
            let self_ptr = self.0.as_ptr() as *const u64;
            let rhs_ptr = rhs.0.as_ptr() as *const u64;

            for n in 0..2 {
                let src = if n & 1 == 0 { self_ptr } else { rhs_ptr };
                let base = n * (N / 16);
                let sel = ((M >> n) & 1) as usize * (N / 16);
                for i in 0..(N / 16) {
                    *new_ptr.add(base + i) = *src.add(sel + i);
                }
            }

            new
        }
    }
    unsafe fn shuffle2_128<const M: i32>(self, rhs: Self) -> Self {
        if N == 16 {
            return self;
        }

        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u64;
            let self_ptr = self.0.as_ptr() as *const u64;
            let rhs_ptr = rhs.0.as_ptr() as *const u64;

            for i in 0..(N / 16) {
                let idx = ((i & !1) + ((M >> (i & 1)) as usize & 1)) * 2;
                let src = if i & 1 == 0 { self_ptr } else { rhs_ptr };
                *new_ptr.add(i * 2) = *src.add(idx);
                *new_ptr.add(i * 2 + 1) = *src.add(idx + 1);
            }

            new
        }
    }
    unsafe fn shuffle2_64<const M: i32>(self, rhs: Self) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u64;
            let self_ptr = self.0.as_ptr() as *const u64;
            let rhs_ptr = rhs.0.as_ptr() as *const u64;

            for i in 0..(N / 8) {
                let idx = (i & !1) + ((M >> (i & 1)) as usize & 1);
                let src = if i & 1 == 0 { self_ptr } else { rhs_ptr };
                *new_ptr.add(i) = *src.add(idx);
            }

            new
        }
    }
    unsafe fn shuffle2_32<const M: i32>(self, rhs: Self) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u32;
            let self_ptr = self.0.as_ptr() as *const u32;
            let rhs_ptr = rhs.0.as_ptr() as *const u32;

            for i in 0..(N / 4) {
                let idx = (i & !3) + ((M >> ((i & 3) * 2)) as usize & 3);
                let src = if i & 2 == 0 { self_ptr } else { rhs_ptr };
                *new_ptr.add(i) = *src.add(idx);
            }

            new
        }
    }
}

impl<const N: usize> SimdVariableBlend for ScalarMask<N> {
    type VecType = ScalarReg<N>;
    #[inline(always)]
    unsafe fn vblend_64(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> ScalarReg<N> {
        unsafe {
            let mut new = ScalarReg::<N>([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u64;
            let false_ptr = false_values.0.as_ptr() as *const u64;
            let true_ptr = true_values.0.as_ptr() as *const u64;
            for i in 0..(N / 8) {
                *new_ptr.add(i) = if self.0[i] {
                    *true_ptr.add(i)
                } else {
                    *false_ptr.add(i)
                };
            }
            new
        }
    }
    #[inline(always)]
    unsafe fn vblend_32(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> ScalarReg<N> {
        unsafe {
            let mut new = ScalarReg::<N>([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u32;
            let false_ptr = false_values.0.as_ptr() as *const u32;
            let true_ptr = true_values.0.as_ptr() as *const u32;
            for i in 0..(N / 4) {
                *new_ptr.add(i) = if self.0[i] {
                    *true_ptr.add(i)
                } else {
                    *false_ptr.add(i)
                };
            }
            new
        }
    }
    #[inline(always)]
    unsafe fn vblend_16(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> ScalarReg<N> {
        unsafe {
            let mut new = ScalarReg::<N>([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u16;
            let false_ptr = false_values.0.as_ptr() as *const u16;
            let true_ptr = true_values.0.as_ptr() as *const u16;
            for i in 0..(N / 2) {
                *new_ptr.add(i) = match self.0[i] {
                    true => *true_ptr.add(i),
                    false => *false_ptr.add(i),
                };
            }
            new
        }
    }
    #[inline(always)]
    unsafe fn vblend_8(
        self,
        true_values: Self::VecType,
        false_values: Self::VecType,
    ) -> ScalarReg<N> {
        unsafe {
            let mut new = ScalarReg::<N>([0; N]);
            let new_ptr = new.0.as_mut_ptr();
            let false_ptr = false_values.0.as_ptr();
            let true_ptr = true_values.0.as_ptr();
            for i in 0..N {
                *new_ptr.add(i) = if self.0[i] {
                    *true_ptr.add(i)
                } else {
                    *false_ptr.add(i)
                };
            }
            new
        }
    }
}

impl<const M: usize> SimdImmediateBlend for ScalarReg<M> {
    #[inline(always)]
    unsafe fn blend_64<const N: i32>(self, false_values: Self) -> Self {
        unsafe {
            let n = (N as u64 & 0b11) * 0x5555555555555555;
            let mut new = ScalarReg::<M>([0; M]);
            let new_ptr = new.0.as_mut_ptr() as *mut u64;
            let false_ptr = false_values.0.as_ptr() as *const u64;
            let true_ptr = self.0.as_ptr() as *const u64;
            for i in 0..(M >> 3) {
                let cond = ((n >> i) & 1) == 1;
                *new_ptr.add(i) = if cond {
                    *true_ptr.add(i)
                } else {
                    *false_ptr.add(i)
                };
            }
            new
        }
    }
    #[inline(always)]
    unsafe fn blend_32<const N: i32>(self, false_values: Self) -> Self {
        unsafe {
            let n = (N as u64 & 0xF) * 0x1111111111111111;
            let mut new = ScalarReg::<M>([0; M]);
            let new_ptr = new.0.as_mut_ptr() as *mut u32;
            let false_ptr = false_values.0.as_ptr() as *const u32;
            let true_ptr = self.0.as_ptr() as *const u32;
            for i in 0..(M >> 2) {
                let cond = ((n >> i) & 1) == 1;
                *new_ptr.add(i) = if cond {
                    *true_ptr.add(i)
                } else {
                    *false_ptr.add(i)
                };
            }
            new
        }
    }

    #[inline(always)]
    unsafe fn blend_16<const N: i32>(self, false_values: Self) -> Self {
        unsafe {
            let n = (N as u64 & 0xFF) * 0x0101010101010101;
            let mut new = ScalarReg::<M>([0; M]);
            let new_ptr = new.0.as_mut_ptr() as *mut u16;
            let false_ptr = false_values.0.as_ptr() as *const u16;
            let true_ptr = self.0.as_ptr() as *const u16;
            for block in 0..(M >> 4) {
                for i in 0..8 {
                    let idx = block * 8 + i;
                    let cond = ((n >> i) & 1) == 1;
                    *new_ptr.add(idx) = if cond {
                        *true_ptr.add(idx)
                    } else {
                        *false_ptr.add(idx)
                    };
                }
            }
            new
        }
    }
}

macro_rules! map_interleave_hilo {
    ($func:ident, $ty:ty, $size:expr, $offset:expr) => {
        #[inline(always)]
        unsafe fn $func(self, rhs: Self) -> Self {
            unsafe {
                let mut new = Self([0; N]);
                let new_ptr = new.0.as_mut_ptr() as *mut $ty;
                let self_ptr = self.0.as_ptr() as *const $ty;
                let rhs_ptr = rhs.0.as_ptr() as *const $ty;

                let loops = $size / (size_of::<$ty>() * 2);
                let offset = if $offset { loops } else { 0 };
                for i in 0..loops {
                    *new_ptr.add(i * 2) = *self_ptr.add(i + offset);
                    *new_ptr.add(i * 2 + 1) = *rhs_ptr.add(i + offset);
                }

                new
            }
        }
    };
}

macro_rules! map_interleave_hilo_block {
    ($func:ident, $ty:ty, $size:expr, $offset:expr) => {
        #[inline(always)]
        unsafe fn $func(self, rhs: Self) -> Self {
            unsafe {
                let mut new = Self([0; N]);
                let new_ptr = new.0.as_mut_ptr() as *mut $ty;
                let self_ptr = self.0.as_ptr() as *const $ty;
                let rhs_ptr = rhs.0.as_ptr() as *const $ty;

                for b in 0..(N / 16) {
                    let width = 16 / size_of::<$ty>();
                    let loops = width / 2;
                    let offset = if $offset { loops } else { 0 };
                    let base = b * width;
                    for i in 0..width {
                        *new_ptr.add(base + i * 2) = *self_ptr.add(base + i + offset);
                        *new_ptr.add(base + i * 2 + 1) = *rhs_ptr.add(base + i + offset);
                    }
                }

                new
            }
        }
    };
}

macro_rules! map_interleave_even {
    ($func:ident, $ty:ty, $size:expr) => {
        #[inline(always)]
        unsafe fn $func(self, rhs: Self) -> Self {
            unsafe {
                let mut new = Self([0; N]);
                let new_ptr = new.0.as_mut_ptr() as *mut $ty;
                let self_ptr = self.0.as_ptr() as *const $ty;
                let rhs_ptr = rhs.0.as_ptr() as *const $ty;

                let loops = $size / (size_of::<$ty>() * 2);
                for i in 0..loops {
                    *new_ptr.add(i * 2) = *self_ptr.add(i * 2);
                    *new_ptr.add(i * 2 + 1) = *rhs_ptr.add(i * 2);
                }

                new
            }
        }
    };
}

macro_rules! map_interleave_odd {
    ($func:ident, $ty:ty, $size:expr) => {
        #[inline(always)]
        unsafe fn $func(self, rhs: Self) -> Self {
            unsafe {
                let mut new = Self([0; N]);
                let new_ptr = new.0.as_mut_ptr() as *mut $ty;
                let self_ptr = self.0.as_ptr() as *const $ty;
                let rhs_ptr = rhs.0.as_ptr() as *const $ty;

                let loops = $size / (size_of::<$ty>() * 2);
                for i in 0..loops {
                    *new_ptr.add(i * 2) = *self_ptr.add(i * 2 + 1);
                    *new_ptr.add(i * 2 + 1) = *rhs_ptr.add(i * 2 + 1);
                }

                new
            }
        }
    };
}

macro_rules! map_deinterleave_even {
    ($func:ident, $ty:ty, $size:expr) => {
        #[inline(always)]
        unsafe fn $func(self, rhs: Self) -> Self {
            unsafe {
                let mut new = Self([0; N]);
                let new_ptr = new.0.as_mut_ptr() as *mut $ty;
                let self_ptr = self.0.as_ptr() as *const $ty;
                let rhs_ptr = rhs.0.as_ptr() as *const $ty;

                let loops = $size / (size_of::<$ty>() * 2);
                for i in 0..loops {
                    *new_ptr.add(i) = *self_ptr.add(i * 2);
                }
                for i in 0..loops {
                    *new_ptr.add(i + loops) = *rhs_ptr.add(i * 2);
                }

                new
            }
        }
    };
}

macro_rules! map_deinterleave_odd {
    ($func:ident, $ty:ty, $size:expr) => {
        #[inline(always)]
        unsafe fn $func(self, rhs: Self) -> Self {
            unsafe {
                let mut new = Self([0; N]);
                let new_ptr = new.0.as_mut_ptr() as *mut $ty;
                let self_ptr = self.0.as_ptr() as *const $ty;
                let rhs_ptr = rhs.0.as_ptr() as *const $ty;

                let loops = $size / (size_of::<$ty>() * 2);
                for i in 0..loops {
                    *new_ptr.add(i) = *self_ptr.add(i * 2 + 1);
                }
                for i in 0..loops {
                    *new_ptr.add(i + loops) = *rhs_ptr.add(i * 2 + 1);
                }

                new
            }
        }
    };
}

impl<const N: usize> SimdInterleave for ScalarReg<N> {
    map_interleave_hilo!(interleave_lower_64, u64, N, false);
    map_interleave_hilo!(interleave_lower_32, u32, N, false);
    map_interleave_hilo!(interleave_lower_16, u16, N, false);
    map_interleave_hilo!(interleave_lower_8, u8, N, false);
    map_interleave_hilo!(interleave_upper_64, u64, N, true);
    map_interleave_hilo!(interleave_upper_32, u32, N, true);
    map_interleave_hilo!(interleave_upper_16, u16, N, true);
    map_interleave_hilo!(interleave_upper_8, u8, N, true);
    map_interleave_hilo_block!(interleave_lower_block_64, u64, N, false);
    map_interleave_hilo_block!(interleave_lower_block_32, u32, N, false);
    map_interleave_hilo_block!(interleave_lower_block_16, u16, N, false);
    map_interleave_hilo_block!(interleave_lower_block_8, u8, N, false);
    map_interleave_hilo_block!(interleave_upper_block_64, u64, N, true);
    map_interleave_hilo_block!(interleave_upper_block_32, u32, N, true);
    map_interleave_hilo_block!(interleave_upper_block_16, u16, N, true);
    map_interleave_hilo_block!(interleave_upper_block_8, u8, N, true);

    #[inline(always)]
    unsafe fn interleave_lower_128(self, rhs: Self) -> Self {
        if N == 16 {
            return self;
        }

        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u64;
            let self_ptr = self.0.as_ptr() as *const u64;
            let rhs_ptr = rhs.0.as_ptr() as *const u64;

            for i in 0..N / 32 {
                *new_ptr.add(i * 4) = *self_ptr.add(i * 2);
                *new_ptr.add(i * 4 + 1) = *self_ptr.add(i * 2 + 1);
                *new_ptr.add(i * 4 + 2) = *rhs_ptr.add(i * 2);
                *new_ptr.add(i * 4 + 3) = *rhs_ptr.add(i * 2 + 1);
            }

            new
        }
    }

    unsafe fn interleave_upper_128(self, rhs: Self) -> Self {
        if N == 16 {
            return self;
        }

        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u64;
            let self_ptr = self.0.as_ptr() as *const u64;
            let rhs_ptr = rhs.0.as_ptr() as *const u64;

            for i in 0..N / 32 {
                let offset = i * 2 + N / 16;
                *new_ptr.add(i * 4) = *self_ptr.add(offset);
                *new_ptr.add(i * 4 + 1) = *self_ptr.add(offset + 1);
                *new_ptr.add(i * 4 + 2) = *rhs_ptr.add(offset);
                *new_ptr.add(i * 4 + 3) = *rhs_ptr.add(offset + 1);
            }

            new
        }
    }

    unsafe fn interleave_even_128(self, rhs: Self) -> Self {
        if N == 16 {
            return self;
        }

        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u64;
            let self_ptr = self.0.as_ptr() as *const u64;
            let rhs_ptr = rhs.0.as_ptr() as *const u64;

            for i in 0..N / 32 {
                *new_ptr.add(i * 4) = *self_ptr.add(i * 4);
                *new_ptr.add(i * 4 + 1) = *self_ptr.add(i * 4 + 1);
                *new_ptr.add(i * 4 + 2) = *rhs_ptr.add(i * 4);
                *new_ptr.add(i * 4 + 3) = *rhs_ptr.add(i * 4 + 1);
            }

            new
        }
    }

    map_interleave_even!(interleave_even_64, u64, N);
    map_interleave_even!(interleave_even_32, u32, N);
    map_interleave_even!(interleave_even_16, u16, N);
    map_interleave_even!(interleave_even_8, u8, N);

    unsafe fn interleave_odd_128(self, rhs: Self) -> Self {
        if N == 16 {
            return self;
        }

        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u64;
            let self_ptr = self.0.as_ptr() as *const u64;
            let rhs_ptr = rhs.0.as_ptr() as *const u64;

            for i in 0..N / 32 {
                *new_ptr.add(i * 4) = *self_ptr.add(i * 4 + 2);
                *new_ptr.add(i * 4 + 1) = *self_ptr.add(i * 4 + 3);
                *new_ptr.add(i * 4 + 2) = *rhs_ptr.add(i * 4 + 2);
                *new_ptr.add(i * 4 + 3) = *rhs_ptr.add(i * 4 + 3);
            }

            new
        }
    }
    map_interleave_odd!(interleave_odd_64, u64, N);
    map_interleave_odd!(interleave_odd_32, u32, N);
    map_interleave_odd!(interleave_odd_16, u16, N);
    map_interleave_odd!(interleave_odd_8, u8, N);

    unsafe fn deinterleave_even_128(self, rhs: Self) -> Self {
        if N == 16 {
            return self;
        }

        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u64;
            let self_ptr = self.0.as_ptr() as *const u64;
            let rhs_ptr = rhs.0.as_ptr() as *const u64;

            for i in 0..N / 32 {
                *new_ptr.add(i * 2) = *self_ptr.add(i * 4);
                *new_ptr.add(i * 2 + 1) = *self_ptr.add(i * 4 + 1);
            }
            for i in 0..N / 32 {
                let offset = i + N / 32;
                *new_ptr.add(offset * 2) = *rhs_ptr.add(i * 4);
                *new_ptr.add(offset * 2 + 1) = *rhs_ptr.add(i * 4 + 1);
            }

            new
        }
    }

    map_deinterleave_even!(deinterleave_even_64, u64, N);
    map_deinterleave_even!(deinterleave_even_32, u32, N);
    map_deinterleave_even!(deinterleave_even_16, u16, N);
    map_deinterleave_even!(deinterleave_even_8, u8, N);

    unsafe fn deinterleave_odd_128(self, rhs: Self) -> Self {
        if N == 16 {
            return self;
        }

        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u64;
            let self_ptr = self.0.as_ptr() as *const u64;
            let rhs_ptr = rhs.0.as_ptr() as *const u64;

            for i in 0..N / 32 {
                *new_ptr.add(i * 2) = *self_ptr.add(i * 4 + 2);
                *new_ptr.add(i * 2 + 1) = *self_ptr.add(i * 4 + 3);
            }
            for i in 0..N / 32 {
                let offset = i + N / 32;
                *new_ptr.add(offset * 2) = *rhs_ptr.add(i * 4 + 2);
                *new_ptr.add(offset * 2 + 1) = *rhs_ptr.add(i * 4 + 3);
            }

            new
        }
    }
    map_deinterleave_odd!(deinterleave_odd_64, u64, N);
    map_deinterleave_odd!(deinterleave_odd_32, u32, N);
    map_deinterleave_odd!(deinterleave_odd_16, u16, N);
    map_deinterleave_odd!(deinterleave_odd_8, u8, N);
}

impl<const N: usize> SimdMulAdd for ScalarReg<N> {
    #[inline(always)]
    unsafe fn mul_add_f64(self, mult: Self, add: Self) -> Self {
        scalar_fma_expr_op!(f64, self, mult, add, N, |a, b, c| f64::mul_add(a, b, c))
    }
    #[inline(always)]
    unsafe fn mul_sub_f64(self, mult: Self, sub: Self) -> Self {
        scalar_fma_expr_op!(f64, self, mult, sub, N, |a, b, c| f64::mul_add(a, b, -c))
    }
    #[inline(always)]
    unsafe fn negated_mul_add_f64(self, mult: Self, add: Self) -> Self {
        scalar_fma_expr_op!(f64, self, mult, add, N, |a, b, c| f64::mul_add(-a, b, c))
    }
    #[inline(always)]
    unsafe fn negated_mul_sub_f64(self, mult: Self, sub: Self) -> Self {
        scalar_fma_expr_op!(f64, self, mult, sub, N, |a, b, c| f64::mul_add(-a, b, -c))
    }
    #[inline(always)]
    unsafe fn mul_add_f32(self, mult: Self, add: Self) -> Self {
        scalar_fma_expr_op!(f32, self, mult, add, N, |a, b, c| f32::mul_add(a, b, c))
    }
    #[inline(always)]
    unsafe fn mul_sub_f32(self, mult: Self, sub: Self) -> Self {
        scalar_fma_expr_op!(f32, self, mult, sub, N, |a, b, c| f32::mul_add(a, b, -c))
    }
    #[inline(always)]
    unsafe fn negated_mul_add_f32(self, mult: Self, add: Self) -> Self {
        scalar_fma_expr_op!(f32, self, mult, add, N, |a, b, c| f32::mul_add(-a, b, c))
    }
    #[inline(always)]
    unsafe fn negated_mul_sub_f32(self, mult: Self, sub: Self) -> Self {
        scalar_fma_expr_op!(f32, self, mult, sub, N, |a, b, c| f32::mul_add(-a, b, -c))
    }
}

impl<const N: usize> SimdRound for ScalarReg<N> {
    #[inline(always)]
    unsafe fn round_f64(self) -> Self {
        scalar_self_op!(f64, round_ties_even, self, N)
    }
    #[inline(always)]
    unsafe fn round_f32(self) -> Self {
        scalar_self_op!(f32, round_ties_even, self, N)
    }
    #[inline(always)]
    unsafe fn floor_f64(self) -> Self {
        scalar_self_op!(f64, floor, self, N)
    }
    #[inline(always)]
    unsafe fn floor_f32(self) -> Self {
        scalar_self_op!(f32, floor, self, N)
    }
    #[inline(always)]
    unsafe fn ceil_f64(self) -> Self {
        scalar_self_op!(f64, ceil, self, N)
    }
    #[inline(always)]
    unsafe fn ceil_f32(self) -> Self {
        scalar_self_op!(f32, ceil, self, N)
    }
}

macro_rules! map_scalar_cmp {
    {$func:ident, $type:ty, $op:tt, $size:expr} => {
        #[inline(always)]
        unsafe fn $func(self, rhs: Self) -> Self::MaskType {
            unsafe {
                let mut result = ScalarMask::<$size>([false; $size]);

                let self_ptr = self.0.as_ptr() as *const $type;
                let rhs_ptr = rhs.0.as_ptr() as *const $type;

                for i in 0..($size / size_of::<$type>()) {
                    result.0[i] = *self_ptr.add(i) $op *rhs_ptr.add(i);
                }

                result
            }
        }
    }
}

impl<const N: usize> SimdPartialOrd for ScalarReg<N> {
    type MaskType = ScalarMask<N>;
    map_scalar_cmp!(cmp_f64_eq, f64, ==, N);
    map_scalar_cmp!(cmp_f64_lt, f64, <, N);
    map_scalar_cmp!(cmp_f64_le, f64, <=, N);
    map_scalar_cmp!(cmp_f64_gt, f64, >, N);
    map_scalar_cmp!(cmp_f64_ge, f64, >=, N);
    map_scalar_cmp!(cmp_f64_ne, f64, !=, N);
    map_scalar_cmp!(cmp_f32_eq, f32, ==, N);
    map_scalar_cmp!(cmp_f32_lt, f32, <, N);
    map_scalar_cmp!(cmp_f32_le, f32, <=, N);
    map_scalar_cmp!(cmp_f32_gt, f32, >, N);
    map_scalar_cmp!(cmp_f32_ge, f32, >=, N);
    map_scalar_cmp!(cmp_f32_ne, f32, !=, N);
    map_scalar_cmp!(cmp_i64_eq, i64, ==, N);
    map_scalar_cmp!(cmp_i64_gt, i64, >, N);
    map_scalar_cmp!(cmp_i32_eq, i32, ==, N);
    map_scalar_cmp!(cmp_i32_gt, i32, >, N);
    map_scalar_cmp!(cmp_i16_eq, i16, ==, N);
    map_scalar_cmp!(cmp_i16_gt, i16, >, N);
    map_scalar_cmp!(cmp_i8_eq, i8, ==, N);
    map_scalar_cmp!(cmp_i8_gt, i8, >, N);
    map_scalar_cmp!(cmp_u64_gt, u64, >, N);
    map_scalar_cmp!(cmp_u32_gt, u32, >, N);
    map_scalar_cmp!(cmp_u16_gt, u16, >, N);
    map_scalar_cmp!(cmp_u8_gt, u8, >, N);
    map_scalar_op!(max_f64, f64, max, N);
    map_scalar_op!(min_f64, f64, min, N);
    map_scalar_op!(max_f32, f32, max, N);
    map_scalar_op!(min_f32, f32, min, N);
    map_scalar_op!(max_i64, i64, max, N);
    map_scalar_op!(min_i64, i64, min, N);
    map_scalar_op!(max_i32, i32, max, N);
    map_scalar_op!(min_i32, i32, min, N);
    map_scalar_op!(max_i16, i16, max, N);
    map_scalar_op!(min_i16, i16, min, N);
    map_scalar_op!(max_i8, i8, max, N);
    map_scalar_op!(min_i8, i8, min, N);
    map_scalar_op!(max_u64, u64, max, N);
    map_scalar_op!(min_u64, u64, min, N);
    map_scalar_op!(max_u32, u32, max, N);
    map_scalar_op!(min_u32, u32, min, N);
    map_scalar_op!(max_u16, u16, max, N);
    map_scalar_op!(min_u16, u16, min, N);
    map_scalar_op!(max_u8, u8, max, N);
    map_scalar_op!(min_u8, u8, min, N);
}

// TODO: Make a custom trait for handling this transmutation into i*.
impl<const N: usize> SimdSplat for ScalarReg<N> {
    #[inline(always)]
    unsafe fn splat_64<T>(val: T) -> Self {
        scalar_splat!(u64, self, transmute_copy(&val), N)
    }
    #[inline(always)]
    unsafe fn splat_32<T>(val: T) -> Self {
        scalar_splat!(u32, self, transmute_copy(&val), N)
    }
    #[inline(always)]
    unsafe fn splat_16<T>(val: T) -> Self {
        scalar_splat!(u16, self, transmute_copy(&val), N)
    }
    #[inline(always)]
    unsafe fn splat_8<T>(val: T) -> Self {
        scalar_splat!(u8, self, transmute_copy(&val), N)
    }
}

impl<const N: usize> SimdGather for ScalarReg<N> {
    #[inline(always)]
    unsafe fn gather_32_from_32<T, const B: i32>(self, ptr: *const T) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u32;
            let indices_ptr = self.0.as_ptr() as *const u32;
            let data_ptr = ptr as *const u32;
            for i in 0..(N / 4) {
                let index = *indices_ptr.add(i) as usize;
                *new_ptr.add(i) = *data_ptr.add(index);
            }
            new
        }
    }
    #[inline(always)]
    unsafe fn gather_64_from_64<T, const B: i32>(self, ptr: *const T) -> Self {
        unsafe {
            let mut new = Self([0; N]);
            let new_ptr = new.0.as_mut_ptr() as *mut u64;
            let indices_ptr = self.0.as_ptr() as *const u64;
            let data_ptr = ptr as *const u64;
            for i in 0..(N / 8) {
                let index = *indices_ptr.add(i) as usize;
                *new_ptr.add(i) = *data_ptr.add(index);
            }
            new
        }
    }
}

impl<const N: usize> SimdSqrt for ScalarReg<N> {
    #[inline(always)]
    unsafe fn sqrt_f64(self) -> Self {
        scalar_self_op!(f64, sqrt, self, N)
    }
    #[inline(always)]
    unsafe fn sqrt_f32(self) -> Self {
        scalar_self_op!(f32, sqrt, self, N)
    }
    #[inline(always)]
    unsafe fn rsqrt_f32(self) -> Self {
        unsafe { Self::splat_32(1.0f32).f32_div(self.sqrt_f32()) }
    }
}

impl<const N: usize> SimdAllBits for ScalarMask<N> {
    #[inline(always)]
    unsafe fn all_zero(self) -> bool {
        !self.0.iter().any(|&x| x)
    }
}

impl<const N: usize> SimdBitwise for ScalarMask<N> {
    #[inline(always)]
    unsafe fn and(self, rhs: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] & rhs.0[i]))
    }
    #[inline(always)]
    unsafe fn or(self, rhs: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] | rhs.0[i]))
    }
    #[inline(always)]
    unsafe fn xor(self, rhs: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] ^ rhs.0[i]))
    }
    #[inline(always)]
    unsafe fn not(self) -> Self {
        Self(std::array::from_fn(|i| !self.0[i]))
    }
    #[inline(always)]
    unsafe fn and_not(self, rhs: Self) -> Self {
        unsafe { self.and(rhs.not()) }
    }
}

impl<const N: usize> SimdNegate for ScalarReg<N> {
    #[inline(always)]
    unsafe fn negate_f64(self) -> Self {
        unsafe { Self::splat_64(-0.0f64).xor(self) }
    }
    #[inline(always)]
    unsafe fn negate_f32(self) -> Self {
        unsafe { Self::splat_32(-0.0f64).xor(self) }
    }
}

impl<const N: usize> SimdBlockShift for ScalarReg<N> {
    #[inline(always)]
    unsafe fn block_left_byte_shift<const M: i32>(self) -> Self {
        let mut new = unsafe { Self::splat_8(0) };
        for block_start in (0..N).step_by(16) {
            let block_end = block_start + 16;
            for i in (block_start + M as usize)..block_end {
                new.0[i] = self.0[i - M as usize];
            }
        }
        new
    }
    #[inline(always)]
    unsafe fn block_right_byte_shift<const M: i32>(self) -> Self {
        let mut new = unsafe { Self::splat_8(0) };
        for block_start in (0..N).step_by(16) {
            let block_end = block_start + 16;
            for i in block_start..(block_end - M as usize) {
                new.0[i] = self.0[i + M as usize];
            }
        }
        new
    }
}

impl<const N: usize> SimdMaskBitConversion for ScalarMask<N> {
    #[inline(always)]
    unsafe fn to_bits_64(self) -> u64 {
        let mut bits = 0u64;
        for i in 0..(N >> 3) {
            bits ^= (self.0[i] as u64) << i
        }
        bits
    }
    #[inline(always)]
    unsafe fn to_bits_32(self) -> u64 {
        let mut bits = 0u64;
        for i in 0..(N >> 2) {
            bits ^= (self.0[i] as u64) << i
        }
        bits
    }
    #[inline(always)]
    unsafe fn to_bits_8(self) -> u64 {
        let mut bits = 0u64;
        for i in 0..N {
            bits ^= (self.0[i] as u64) << i
        }
        bits
    }
    #[inline(always)]
    unsafe fn from_bits_64(bitmask: u64) -> Self {
        let mut new_mask = Self([false; N]);
        for i in 0..N {
            new_mask.0[i] = ((bitmask >> i) & 1) == 1;
        }
        new_mask
    }
    unsafe fn from_bits_32(bitmask: u64) -> Self {
        let mut new_mask = Self([false; N]);
        for i in 0..N {
            new_mask.0[i] = ((bitmask >> i) & 1) == 1;
        }
        new_mask
    }
    unsafe fn from_bits_16(bitmask: u64) -> Self {
        let mut new_mask = Self([false; N]);
        for i in 0..N {
            new_mask.0[i] = ((bitmask >> i) & 1) == 1;
        }
        new_mask
    }
    unsafe fn from_bits_8(bitmask: u64) -> Self {
        let mut new_mask = Self([false; N]);
        for i in 0..N {
            new_mask.0[i] = ((bitmask >> i) & 1) == 1;
        }
        new_mask
    }
}

impl<const M: usize> SimdLaneShift for ScalarReg<M> {
    #[inline(always)]
    unsafe fn left_lane_shift_32(self, n: u32) -> Self {
        let mut new = unsafe { Self::zero() };
        if n as usize * 4 >= M {
            new
        } else {
            let bytes = (n * 4) as usize;
            for i in 0..(M - bytes) {
                new.0[i] = self.0[i + bytes];
            }
            new
        }
    }

    #[inline(always)]
    unsafe fn right_lane_shift_32(self, n: u32) -> Self {
        let mut new = unsafe { Self::zero() };
        if n as usize * 4 >= M {
            new
        } else {
            let bytes = (n * 4) as usize;
            for i in bytes..M {
                new.0[i] = self.0[i - bytes];
            }
            new
        }
    }
}
