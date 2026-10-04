use std::marker::PhantomData;
use std::ops::*;

use crate::architectures::interface::{Arch, SimdAllBitsImpl, *};
use crate::mask::Mask;
use crate::register::Simd;
use crate::simd_types::*;

impl<T: SimdElement, A: Arch> Mask<T, A> {
    #[inline(always)]
    pub(crate) fn new(data: A::Mask) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn raw_cast<S: SimdElement>(self) -> Mask<S, A> {
        Mask::new(self.data)
    }


    #[inline(always)]
    pub fn all_false(self) -> bool {
        unsafe { self.data.all_zero() }
    }

    // TODO: Support other bit_sizes
    #[inline(always)]
    pub fn first_n_true(n: u32) -> Mask<T, A> {
        let iota = Simd::iota(0u32);
        let n_vec = Simd::splat(n);
        n_vec.simd_gt(iota).raw_cast()
    }

    #[inline(always)]
    pub fn first_n_false(n: u32) -> Mask<T, A> {
        let iota = Simd::iota(1u32);
        let n_vec = Simd::splat(n);
        iota.simd_gt(n_vec).raw_cast()
    }
}

impl<T: SimdElement, A: Arch> BitAnd for Mask<T, A> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.and(rhs.data)) }
    }
}

impl<T: SimdElement, A: Arch> BitOr for Mask<T, A> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.or(rhs.data)) }
    }
}

impl<T: SimdElement, A: Arch> BitXor for Mask<T, A> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.xor(rhs.data)) }
    }
}

impl<T: SimdElement, A: Arch> Not for Mask<T, A> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe { Self::new(self.data.not()) }
    }
}

impl<T: SimdElement, A: Arch> Mask<T, A> {
    #[inline(always)]
    pub fn andnot(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.and_not(rhs.data)) }
    }

    // TODO: Add 16 bit select.
    pub fn select(self, true_values: Simd<T, A>, false_values: Simd<T, A>) -> Simd<T, A> {
        unsafe {
            match T::BIT_SIZE {
                BitSize::Size64 => {
                    Simd::new(self.data.vblend_64(true_values.data, false_values.data))
                }
                BitSize::Size32 => {
                    Simd::new(self.data.vblend_32(true_values.data, false_values.data))
                }
                BitSize::Size8 => {
                    Simd::new(self.data.vblend_8(true_values.data, false_values.data))
                }
                _ => panic!("Select for 16 bit types not implemented yet!"),
            }
        }
    }
}

impl<T: SimdElement, A: Arch> Mask<T, A> {
    pub fn to_bits(self) -> u64 {
        unsafe {
            match T::BIT_SIZE {
                BitSize::Size64 => self.data.to_bits_64(),
                BitSize::Size32 => self.data.to_bits_32(),
                BitSize::Size8 => self.data.to_bits_8(),
                _ => unreachable!(), // TODO: Add to_bits_16.
            }
        }
    }

    pub fn from_bits(bitmask: u64) -> Self {
        unsafe {
            match T::BIT_SIZE {
                BitSize::Size64 => Self::new(A::Mask::from_bits_64(bitmask)),
                BitSize::Size32 => Self::new(A::Mask::from_bits_32(bitmask)),
                BitSize::Size16 => Self::new(A::Mask::from_bits_16(bitmask)),
                BitSize::Size8 => Self::new(A::Mask::from_bits_8(bitmask)),
            }
        }
    }
}
