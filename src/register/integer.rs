use std::ops::*;

use crate::architectures::interface::*;
use crate::register::Simd;
use crate::simd_types::*;

impl<T: SimdInteger, A: Arch> BitAnd for Simd<T, A> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self::new(A::Vec::and(self.data, rhs.data)) }
    }
}

impl<T: SimdInteger, A: Arch> BitOr for Simd<T, A> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self::new(A::Vec::or(self.data, rhs.data)) }
    }
}

impl<T: SimdInteger, A: Arch> BitXor for Simd<T, A> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe { Self::new(A::Vec::xor(self.data, rhs.data)) }
    }
}

impl<T: SimdInteger, A: Arch> Simd<T, A> {
    #[inline(always)]
    pub fn andnot(self, rhs: Self) -> Self {
        unsafe { Self::new(A::Vec::and_not(self.data, rhs.data)) }
    }
}

impl<T: SimdInteger, A: Arch> Not for Simd<T, A> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self::new(A::Vec::not(self.data)) }
    }
}

// === Shifts ===
impl<T: SimdIntegerNotByte, A: Arch> Shl<Simd<<T as SimdInteger>::Unsigned, A>> for Simd<T, A> {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Simd<<T as SimdInteger>::Unsigned, A>) -> Self {
        Self::new(unsafe {
            match T::BIT_SIZE {
                BitSize::Size64 => self.data.sllv_64(rhs.data),
                BitSize::Size32 => self.data.sllv_32(rhs.data),
                BitSize::Size16 => self.data.sllv_16(rhs.data),
                _ => unreachable!(),
            }
        })
    }
}

impl<T: SimdIntegerNotByte, A: Arch> Shr<Simd<<T as SimdInteger>::Unsigned, A>> for Simd<T, A> {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Simd<<T as SimdInteger>::Unsigned, A>) -> Self {
        unsafe {
            Self::new(match T::TYPE {
                SimdType::U64 => self.data.srlv_64(rhs.data),
                SimdType::U32 => self.data.srlv_32(rhs.data),
                SimdType::U16 => self.data.srlv_16(rhs.data),
                SimdType::I64 => self.data.srav_64(rhs.data),
                SimdType::I32 => self.data.srav_32(rhs.data),
                SimdType::I16 => self.data.srav_16(rhs.data),
                _ => unreachable!(),
            })
        }
    }
}

// === Scalar shifts ===

impl<T: SimdIntegerNotByte, A: Arch> Shl<usize> for Simd<T, A> {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: usize) -> Self {
        unsafe {
            Self::new(match T::BIT_SIZE {
                BitSize::Size64 => self.data.sll_64(rhs as u32),
                BitSize::Size32 => self.data.sll_32(rhs as u32),
                BitSize::Size16 => self.data.sll_16(rhs as u32),
                _ => unreachable!(),
            })
        }
    }
}

impl<T: SimdIntegerNotByte, A: Arch> Shr<usize> for Simd<T, A> {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: usize) -> Self {
        unsafe {
            Self::new(match T::TYPE {
                SimdType::U64 => self.data.srl_64(rhs as u32),
                SimdType::U32 => self.data.srl_32(rhs as u32),
                SimdType::U16 => self.data.srl_16(rhs as u32),
                SimdType::I64 => self.data.sra_64(rhs as u32),
                SimdType::I32 => self.data.sra_32(rhs as u32),
                SimdType::I16 => self.data.sra_16(rhs as u32),
                _ => unreachable!(),
            })
        }
    }
}

// === Immediate shifts ===

impl<T: SimdIntegerNotByte, A: Arch> Simd<T, A> {
    #[inline(always)]
    pub fn shift_left<const N: i32>(self) -> Self {
        unsafe {
            Self::new(match T::BIT_SIZE {
                BitSize::Size64 => self.data.slli_64::<N>(),
                BitSize::Size32 => self.data.slli_32::<N>(),
                BitSize::Size16 => self.data.slli_16::<N>(),
                _ => unreachable!(),
            })
        }
    }

    #[inline(always)]
    pub fn shift_right<const N: i32>(self) -> Self {
        unsafe {
            Self::new(match T::TYPE {
                SimdType::U64 => self.data.srli_64::<N>(),
                SimdType::U32 => self.data.srli_32::<N>(),
                SimdType::U16 => self.data.srli_16::<N>(),
                SimdType::I64 => self.data.srai_64::<N>(),
                SimdType::I32 => self.data.srai_32::<N>(),
                SimdType::I16 => self.data.srai_16::<N>(),
                _ => unreachable!(),
            })
        }
    }
}

// === Arbitrary shifts ===

impl<T: SimdElement, A: Arch> Simd<T, A> {
    /// Shift left logical on 64-bit lanes. Fills in
    /// incoming bits with zeroes.
    #[inline(always)]
    pub fn shift_ll_64<const N: i32>(self) -> Self {
        unsafe { Self::new(self.data.slli_64::<N>()) }
    }
    /// Shift left logical on 32-bit lanes. Fills in
    /// incoming bits with zeroes.
    #[inline(always)]
    pub fn shift_ll_32<const N: i32>(self) -> Self {
        unsafe { Self::new(self.data.slli_32::<N>()) }
    }
    /// Shift left logical on 16-bit lanes. Fills in
    /// incoming bits with zeroes.
    #[inline(always)]
    pub fn shift_ll_16<const N: i32>(self) -> Self {
        unsafe { Self::new(self.data.slli_16::<N>()) }
    }
    /// Shift right logical on 64-bit lanes. Fills in
    /// incoming bits with zeroes.
    #[inline(always)]
    pub fn shift_rl_64<const N: i32>(self) -> Self {
        unsafe { Self::new(self.data.srli_64::<N>()) }
    }
    /// Shift right logical on 32-bit lanes. Fills in
    /// incoming bits with zeroes.
    #[inline(always)]
    pub fn shift_rl_32<const N: i32>(self) -> Self {
        unsafe { Self::new(self.data.srli_32::<N>()) }
    }
    /// Shift right logical on 16-bit lanes. Fills in
    /// incoming bits with zeroes.
    #[inline(always)]
    pub fn shift_rl_16<const N: i32>(self) -> Self {
        unsafe { Self::new(self.data.srli_16::<N>()) }
    }
    /// Shift right arithmetic on 64-bit lanes. Fills in
    /// incoming bits with ones.
    #[inline(always)]
    pub fn shift_ra_64<const N: i32>(self) -> Self {
        unsafe { Self::new(self.data.srai_64::<N>()) }
    }
    /// Shift right arithmetic on 32-bit lanes. Fills in
    /// incoming bits with ones.
    #[inline(always)]
    pub fn shift_ra_32<const N: i32>(self) -> Self {
        unsafe { Self::new(self.data.srai_32::<N>()) }
    }
    /// Shift right arithmetic on 16-bit lanes. Fills in
    /// incoming bits with ones.
    #[inline(always)]
    pub fn shift_ra_16<const N: i32>(self) -> Self {
        unsafe { Self::new(self.data.srai_16::<N>()) }
    }
}

// === Addition ===

impl<T: SimdMulType, A: Arch> Mul for Simd<T, A> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            Self::new(match T::TYPE {
                SimdType::F64 => self.data.f64_mul(rhs.data),
                SimdType::F32 => self.data.f32_mul(rhs.data),
                SimdType::I32 => self.data.i32_mul(rhs.data),
                SimdType::I16 => self.data.i16_mul(rhs.data),
                SimdType::U32 => self.data.i32_mul(rhs.data),
                SimdType::U16 => self.data.i16_mul(rhs.data),
                _ => unreachable!(),
            })
        }
    }
}

impl<T: SimdFloat, A: Arch> Div for Simd<T, A> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe {
            Self::new(match T::TYPE {
                SimdType::F64 => self.data.f64_div(rhs.data),
                SimdType::F32 => self.data.f32_div(rhs.data),
                _ => unreachable!(),
            })
        }
    }
}

// === Casts ===

impl<T: SimdInteger + HasSigned, A: Arch> Simd<T, A> {
    #[inline(always)]
    pub fn cast_signed(self) -> Simd<<T as SimdInteger>::Signed, A> {
        Simd::new(self.data)
    }
}

impl<T: SimdInteger + HasUnsigned, A: Arch> Simd<T, A> {
    #[inline(always)]
    pub fn cast_unsigned(self) -> Simd<<T as SimdInteger>::Unsigned, A> {
        Simd::new(self.data)
    }
}

impl<T: SimdInteger + HasFloat, A: Arch> Simd<T, A> {
    #[inline(always)]
    pub fn cast_float(self) -> Simd<<T as HasFloat>::Float, A> {
        unsafe { Simd::new(self.data.int_to_float()) }
    }
}
