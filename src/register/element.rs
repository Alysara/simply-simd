use std::fmt;
use std::marker::PhantomData;
use std::ops::*;

use num_traits::NumCast;

use crate::SimdElement;
use crate::architectures::interface::*;
use crate::mask::Mask;
use crate::register::Simd;
use crate::simd_array::SimdToArray;
use crate::simd_types::{B32, B64, BitSize, SimdType, SimdWideType};

impl<T: SimdElement, A: Arch> Simd<T, A> {
    #[inline(always)]
    pub(crate) fn new(data: A::Vec) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn zero() -> Self {
        unsafe { Self::new(A::Vec::zero()) }
    }

    #[inline(always)]
    pub fn from_aligned_slice(slice: &[T]) -> Self {
        let ptr = slice.as_ptr();
        assert!(ptr.align_offset(Self::SIMD_WIDTH) == 0);
        assert!(slice.len() >= Self::LANES);
        unsafe { Self::new(A::Vec::load_aligned(ptr)) }
    }

    /// # Safety
    /// Does not check if the slice goes out of bounds.
    #[inline(always)]
    pub unsafe fn from_aligned_slice_unchecked(slice: &[T]) -> Self {
        let ptr = slice.as_ptr();
        debug_assert!(ptr.align_offset(Self::SIMD_WIDTH) == 0);
        debug_assert!(slice.len() >= Self::LANES);
        unsafe { Self::new(A::Vec::load_aligned(ptr)) }
    }

    #[inline(always)]
    pub fn from_slice(slice: &[T]) -> Self {
        if slice.len() >= Self::LANES {
            unsafe { Self::new(A::Vec::load_unaligned(slice.as_ptr())) }
        } else {
            let mut array = Self::zero().to_array();
            for (arr, val) in array.iter_mut().zip(slice.iter()) {
                *arr = *val;
            }
            unsafe { Self::from_slice_unchecked(array.as_slice()) }
        }
    }

    /// # Safety
    /// Requires allocated memory to be behind (left) of the slice.
    /// Bounds are not checked.
    /// Length of the slice must be less than or equal to the number of lanes.
    #[inline(always)]
    pub unsafe fn from_slice_partial(slice: &[T]) -> Self {
        debug_assert!(slice.len() <= Self::LANES);
        unsafe {
            let offset = Self::LANES - slice.len();
            let raw_ptr = slice.as_ptr().sub(offset);
            let simd = Self::new(A::Vec::load_unaligned(raw_ptr));
            simd.left_lane_shift(offset as u32)
        }
    }

    /// # Safety
    /// Does not check if the slice goes out of bounds.
    #[inline(always)]
    pub unsafe fn from_slice_unchecked(slice: &[T]) -> Self {
        debug_assert!(slice.len() >= Self::LANES);
        unsafe { Self::new(A::Vec::load_unaligned(slice.as_ptr())) }
    }

    #[inline(always)]
    pub fn copy_to_aligned_slice(self, slice: &mut [T]) {
        let ptr = slice.as_mut_ptr();
        assert!(ptr.align_offset(Self::SIMD_WIDTH) == 0);
        assert!(slice.len() >= Self::LANES);
        unsafe { self.data.store_aligned(ptr) };
    }

    /// # Safety
    /// Does not check if the slice goes out of bounds.
    #[inline(always)]
    pub unsafe fn copy_to_aligned_slice_unchecked(self, slice: &mut [T]) {
        let ptr = slice.as_mut_ptr();
        debug_assert!(ptr.align_offset(Self::SIMD_WIDTH) == 0);
        debug_assert!(slice.len() >= Self::LANES);
        unsafe { self.data.store_aligned(ptr) };
    }

    #[inline(always)]
    pub fn copy_to_slice(self, slice: &mut [T]) {
        if slice.len() >= Self::LANES {
            let ptr = slice.as_mut_ptr();
            unsafe { self.data.store_unaligned(ptr) };
        } else {
            // Scalar/tail case.
            let array = self.to_array();
            slice
                .iter_mut()
                .zip(array.iter())
                .for_each(|(src, new)| *src = *new);
        }
    }

    /// # Safety
    /// Does not check if the slice goes out of bounds.
    #[inline(always)]
    pub unsafe fn copy_to_slice_unchecked(self, slice: &mut [T]) {
        let ptr = slice.as_mut_ptr();
        debug_assert!(slice.len() >= Self::LANES);
        unsafe { self.data.store_unaligned(ptr) };
    }

    /// Converts the Simd register into an array.
    ///
    /// # Example
    ///
    /// TODO
    /// use simply_simd::
    #[inline(always)]
    pub fn to_array(self) -> T::Array<A> {
        let mut array = T::Array::<A>::from_fn(|_| T::from(0).unwrap());
        self.copy_to_slice(array.as_mut_slice());
        array
    }

    #[inline(always)]
    pub fn iota(offset: T) -> Self {
        let iota_array =
            T::Array::<A>::from_fn(|i| <T as NumCast>::from(i).unwrap().safe_add(offset));
        Self::from_slice(iota_array.as_slice())
    }
}

impl<T: SimdElement, A: Arch> fmt::Debug for Simd<T, A> {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let buf = self.to_array();
        write!(f, "{:?}", buf)
    }
}

// === Assign operations ===
impl<T: SimdElement, A: Arch> AddAssign for Simd<T, A>
where
    Self: Add<Output = Self> + Copy,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: SimdElement, A: Arch> SubAssign for Simd<T, A>
where
    Self: Sub<Output = Self> + Copy,
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: SimdElement, A: Arch> MulAssign for Simd<T, A>
where
    Self: Mul<Output = Self> + Copy,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: SimdElement, A: Arch> DivAssign for Simd<T, A>
where
    Self: Div<Output = Self> + Copy,
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: SimdElement, A: Arch> RemAssign for Simd<T, A>
where
    Self: Rem<Output = Self> + Copy,
{
    #[inline(always)]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl<T: SimdElement, A: Arch> BitAndAssign for Simd<T, A>
where
    Self: BitAnd<Output = Self> + Copy,
{
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl<T: SimdElement, A: Arch> BitOrAssign for Simd<T, A>
where
    Self: BitOr<Output = Self> + Copy,
{
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

impl<T: SimdElement, A: Arch> BitXorAssign for Simd<T, A>
where
    Self: BitXor<Output = Self> + Copy,
{
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}

impl<T: SimdElement, A: Arch> Neg for Simd<T, A> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::zero() - self
    }
}

impl<T: SimdElement, A: Arch> Simd<T, A> {
    #[inline(always)]
    pub fn raw_cast<S: SimdElement>(self) -> Simd<S, A> {
        Simd::new(self.data)
    }
}

impl<T: SimdElement, A: Arch> Default for Simd<T, A> {
    #[inline(always)]
    fn default() -> Self {
        Self::splat(<T as NumCast>::from(T::default()).unwrap())
    }
}

impl<T: SimdElement, A: Arch> Add for Simd<T, A> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            Self::new(match T::TYPE {
                SimdType::F64 => self.data.f64_add(rhs.data),
                SimdType::F32 => self.data.f32_add(rhs.data),
                SimdType::I64 => self.data.i64_add(rhs.data),
                SimdType::I32 => self.data.i32_add(rhs.data),
                SimdType::I16 => self.data.i16_add(rhs.data),
                SimdType::I8 => self.data.i8_add(rhs.data),
                SimdType::U64 => self.data.i64_add(rhs.data),
                SimdType::U32 => self.data.i32_add(rhs.data),
                SimdType::U16 => self.data.i16_add(rhs.data),
                SimdType::U8 => self.data.i8_add(rhs.data),
            })
        }
    }
}

impl<T: SimdElement, A: Arch> Sub for Simd<T, A> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            Self::new(match T::TYPE {
                SimdType::F64 => self.data.f64_sub(rhs.data),
                SimdType::F32 => self.data.f32_sub(rhs.data),
                SimdType::I64 => self.data.i64_sub(rhs.data),
                SimdType::I32 => self.data.i32_sub(rhs.data),
                SimdType::I16 => self.data.i16_sub(rhs.data),
                SimdType::I8 => self.data.i8_sub(rhs.data),
                SimdType::U64 => self.data.i64_sub(rhs.data),
                SimdType::U32 => self.data.i32_sub(rhs.data),
                SimdType::U16 => self.data.i16_sub(rhs.data),
                SimdType::U8 => self.data.i8_sub(rhs.data),
            })
        }
    }
}

impl<T: SimdElement, A: Arch> Simd<T, A> {
    /// Clamps the values in a register between two bounds, inclusive.
    ///
    /// # Parameters:
    /// - `lower_bound`: Minimum value after clamping
    /// - `upper_bound`: Maximum value after clamping
    #[inline(always)]
    pub fn clamp(self, lower_bound: Self, upper_bound: Self) -> Self {
        self.min(upper_bound).max(lower_bound)
    }

    /// Shifts bytes N to the left within 128-bit blocks.
    #[inline(always)]
    pub fn block_left_byte_shift<const N: i32>(self) -> Self {
        unsafe { Self::new(self.data.block_left_byte_shift::<N>()) }
    }

    /// Shifts bytes N to the right within 128-bit blocks.
    #[inline(always)]
    pub fn block_right_byte_shift<const N: i32>(self) -> Self {
        unsafe { Self::new(self.data.block_right_byte_shift::<N>()) }
    }

    /// Blends two registers using an immediate mask.
    #[inline(always)]
    pub fn blend<const N: i32>(self, false_values: Self) -> Self {
        unsafe {
            match T::BIT_SIZE {
                BitSize::Size64 => Self::new(self.data.blend_64::<N>(false_values.data)),
                BitSize::Size32 => Self::new(self.data.blend_32::<N>(false_values.data)),
                BitSize::Size16 => Self::new(self.data.blend_16::<N>(false_values.data)),
                _ => unreachable!(), // TODO: Add 8 for immediate blend.
            }
        }
    }

    /// Blends 64-bit lanes of two registers using an immediate mask.
    #[inline(always)]
    pub fn blend_64<const N: i32>(self, false_values: Self) -> Self {
        unsafe { Self::new(self.data.blend_64::<N>(false_values.data)) }
    }
    /// Blends 32-bit lanes of two registers using an immediate mask.
    #[inline(always)]
    pub fn blend_32<const N: i32>(self, false_values: Self) -> Self {
        unsafe { Self::new(self.data.blend_32::<N>(false_values.data)) }
    }
    /// Blends 16-bit lanes of two registers using an immediate mask.
    #[inline(always)]
    pub fn blend_16<const N: i32>(self, false_values: Self) -> Self {
        unsafe { Self::new(self.data.blend_16::<N>(false_values.data)) }
    }
}

impl<T: SimdElement, A: Arch> Simd<T, A> {
    /// Broadcasts a value across the entire register.
    #[inline(always)]
    pub fn splat(val: T) -> Self {
        unsafe {
            Self::new(match T::BIT_SIZE {
                BitSize::Size64 => A::Vec::splat_64(val),
                BitSize::Size32 => A::Vec::splat_32(val),
                BitSize::Size16 => A::Vec::splat_16(val),
                BitSize::Size8 => A::Vec::splat_8(val),
            })
        }
    }

    /// Loads a register according to a mask.
    #[inline(always)]
    pub fn masked_load(slice: &[T], mask: Mask<T, A>) -> Self {
        unsafe {
            Self::new(match T::BIT_SIZE {
                BitSize::Size64 => A::Vec::masked_load_64(slice.as_ptr(), mask.data),
                BitSize::Size32 => A::Vec::masked_load_32(slice.as_ptr(), mask.data),
                _ => unreachable!(),
            })
        }
    }

    /// Loads only the first `amount` elements into the register.
    pub fn partial_load(slice: &[T], amount: usize) -> Self {
        assert!(
            slice.len() >= Self::LANES,
            "Attempted to do a partial load, but index is out of bounds!"
        );
        let amount_vec = Self::splat(<T as NumCast>::from(amount).unwrap());
        let mask = Self::iota(<T as NumCast>::from(0).unwrap()).simd_lt(amount_vec);
        Self::masked_load(slice, mask)
    }

    /// Loads only the first `amount` elements into the register.
    ///
    /// # Safety
    /// - Slice must be greater than or equal to `ArchSimd::<T>::LANES`
    pub unsafe fn partial_load_unchecked(slice: &[T], amount: usize) -> Self {
        debug_assert!(
            slice.len() >= Self::LANES,
            "Index is out of bounds in unsafe code!"
        );
        let amount_vec = Self::splat(<T as NumCast>::from(amount).unwrap());
        let mask = Self::iota(<T as NumCast>::from(0).unwrap()).simd_lt(amount_vec);
        Self::masked_load(slice, mask)
    }

    /// Stores the register using a given mask.
    #[inline(always)]
    pub fn masked_store(self, slice: &mut [T], mask: Mask<T, A>) {
        unsafe {
            match T::BIT_SIZE {
                BitSize::Size64 => {
                    A::Vec::masked_store_64(self.data, slice.as_mut_ptr(), mask.data)
                }
                BitSize::Size32 => {
                    A::Vec::masked_store_32(self.data, slice.as_mut_ptr(), mask.data)
                }
                _ => unreachable!(), // TODO: ADD SUPPORT AOR OTHER SIZES!!!
            }
        }
    }

    /// Stores only the first `amount` elements into the register.
    pub fn partial_store(self, slice: &mut [T], amount: usize) {
        match T::BIT_SIZE {
            BitSize::Size64 => {
                let iota = Simd::iota(0u64);
                let n_vec = Simd::splat(amount as u64);
                let mask = n_vec.simd_gt(iota);
                Self::masked_store(self, slice, mask.raw_cast());
            }
            BitSize::Size32 => {
                let iota = Simd::iota(0u32);
                let n_vec = Simd::splat(amount as u32);
                let mask = n_vec.simd_gt(iota);
                Self::masked_store(self, slice, mask.raw_cast());
            }
            _ => unreachable!(),
        }
    }

    #[inline(always)]
    pub fn simd_eq(self, rhs: Self) -> Mask<T, A> {
        unsafe {
            Mask::new(match T::TYPE {
                SimdType::F64 => self.data.cmp_f64_eq(rhs.data),
                SimdType::F32 => self.data.cmp_f32_eq(rhs.data),
                SimdType::U64 => self.data.cmp_i64_eq(rhs.data),
                SimdType::U32 => self.data.cmp_i32_eq(rhs.data),
                SimdType::U16 => self.data.cmp_i16_eq(rhs.data),
                SimdType::U8 => self.data.cmp_i8_eq(rhs.data),
                SimdType::I64 => self.data.cmp_i64_eq(rhs.data),
                SimdType::I32 => self.data.cmp_i32_eq(rhs.data),
                SimdType::I16 => self.data.cmp_i16_eq(rhs.data),
                SimdType::I8 => self.data.cmp_i8_eq(rhs.data),
                // _ => unreachable!() // TODO: Add integer types .
            })
        }
    }

    #[inline(always)]
    pub fn simd_ne(self, rhs: Self) -> Mask<T, A> {
        unsafe {
            Mask::new(match T::TYPE {
                SimdType::F64 => self.data.cmp_f64_ne(rhs.data),
                SimdType::F32 => self.data.cmp_f32_ne(rhs.data),
                _ => self.simd_eq(rhs).data,
            })
        }
    }

    #[inline(always)]
    pub fn simd_gt(self, rhs: Self) -> Mask<T, A> {
        unsafe {
            Mask::new(match T::TYPE {
                SimdType::F64 => self.data.cmp_f64_gt(rhs.data),
                SimdType::F32 => self.data.cmp_f32_gt(rhs.data),
                SimdType::U64 => self.data.cmp_u64_gt(rhs.data),
                SimdType::U32 => self.data.cmp_u32_gt(rhs.data),
                SimdType::U16 => self.data.cmp_u16_gt(rhs.data),
                SimdType::U8 => self.data.cmp_u8_gt(rhs.data),
                SimdType::I64 => self.data.cmp_i64_gt(rhs.data),
                SimdType::I32 => self.data.cmp_i32_gt(rhs.data),
                SimdType::I16 => self.data.cmp_i16_gt(rhs.data),
                SimdType::I8 => self.data.cmp_i8_gt(rhs.data),
            })
        }
    }

    // TODO: Aind better way to handle comparisons.
    #[inline(always)]
    pub fn simd_ge(self, rhs: Self) -> Mask<T, A> {
        unsafe {
            Mask::new(match T::TYPE {
                SimdType::F64 => self.data.cmp_f64_ge(rhs.data),
                SimdType::F32 => self.data.cmp_f32_ge(rhs.data),
                SimdType::U64 => self
                    .data
                    .cmp_u64_gt(rhs.data)
                    .or(self.data.cmp_i64_eq(rhs.data)),
                SimdType::U32 => self
                    .data
                    .cmp_u32_gt(rhs.data)
                    .or(self.data.cmp_i32_eq(rhs.data)),
                SimdType::U16 => self
                    .data
                    .cmp_u16_gt(rhs.data)
                    .or(self.data.cmp_i16_eq(rhs.data)),
                SimdType::U8 => self
                    .data
                    .cmp_u8_gt(rhs.data)
                    .or(self.data.cmp_i8_eq(rhs.data)),
                SimdType::I64 => self
                    .data
                    .cmp_i64_gt(rhs.data)
                    .or(self.data.cmp_i64_eq(rhs.data)),
                SimdType::I32 => self
                    .data
                    .cmp_i32_gt(rhs.data)
                    .or(self.data.cmp_i32_eq(rhs.data)),
                SimdType::I16 => self
                    .data
                    .cmp_i16_gt(rhs.data)
                    .or(self.data.cmp_i16_eq(rhs.data)),
                SimdType::I8 => self
                    .data
                    .cmp_i8_gt(rhs.data)
                    .or(self.data.cmp_i8_eq(rhs.data)),
            })
        }
    }

    #[inline(always)]
    pub fn simd_lt(self, rhs: Self) -> Mask<T, A> {
        unsafe {
            Mask::new(match T::TYPE {
                SimdType::F64 => self.data.cmp_f64_lt(rhs.data),
                SimdType::F32 => self.data.cmp_f32_lt(rhs.data),
                _ => panic!("Less than for integers not implemented!"), // TODO: Add integer types .
            })
        }
    }

    #[inline(always)]
    pub fn simd_le(self, rhs: Self) -> Mask<T, A> {
        unsafe {
            Mask::new(match T::TYPE {
                SimdType::F64 => self.data.cmp_f64_le(rhs.data),
                SimdType::F32 => self.data.cmp_f32_le(rhs.data),
                _ => panic!("Less than or equal not implemented for integers!"), // TODO: Add integer types .
            })
        }
    }

    pub fn max(self, rhs: Self) -> Self {
        unsafe {
            Self::new(match T::TYPE {
                SimdType::F64 => self.data.max_f64(rhs.data),
                SimdType::F32 => self.data.max_f32(rhs.data),
                SimdType::I64 => self.data.max_i64(rhs.data),
                SimdType::I32 => self.data.max_i32(rhs.data),
                SimdType::I16 => self.data.max_i16(rhs.data),
                SimdType::I8 => self.data.max_i8(rhs.data),
                SimdType::U64 => self.data.max_u64(rhs.data),
                SimdType::U32 => self.data.max_u32(rhs.data),
                SimdType::U16 => self.data.max_u16(rhs.data),
                SimdType::U8 => self.data.max_u8(rhs.data),
            })
        }
    }

    pub fn min(self, rhs: Self) -> Self {
        unsafe {
            Self::new(match T::TYPE {
                SimdType::F64 => self.data.min_f64(rhs.data),
                SimdType::F32 => self.data.min_f32(rhs.data),
                SimdType::I64 => self.data.min_i64(rhs.data),
                SimdType::I32 => self.data.min_i32(rhs.data),
                SimdType::I16 => self.data.min_i16(rhs.data),
                SimdType::I8 => self.data.min_i8(rhs.data),
                SimdType::U64 => self.data.min_u64(rhs.data),
                SimdType::U32 => self.data.min_u32(rhs.data),
                SimdType::U16 => self.data.min_u16(rhs.data),
                SimdType::U8 => self.data.min_u8(rhs.data),
            })
        }
    }
}

impl<T: SimdWideType, A: Arch> Simd<T, A> {
    #[inline(always)]
    pub fn permute(self, indices: Simd<T::UType, A>) -> Self {
        unsafe {
            match T::BIT_SIZE {
                BitSize::Size64 => Self::new(self.data.vpermute_64(indices.data)),
                BitSize::Size32 => Self::new(self.data.vpermute_32(indices.data)),
                _ => unreachable!(),
            }
        }
    }
}

impl<T: SimdElement, A: Arch> Simd<T, A> {
    /// Permutes all of the 64-bit chunks in a register according
    /// to the indices in another register.
    #[inline(always)]
    pub fn permute_64(self, indices: Simd<u64, A>) -> Self {
        unsafe { Self::new(self.data.vpermute_64(indices.data)) }
    }

    /// Permutes all of the 32-bit chunks in a register according
    /// to the indices in another register.
    #[inline(always)]
    pub fn permute_32(self, indices: Simd<u32, A>) -> Self {
        unsafe { Self::new(self.data.vpermute_32(indices.data)) }
    }

    /// Shuffles the 16-bit chunks for each 128-bit lane in a register
    /// according to the indices in another register. Each index is
    /// relative to its 128-bit chunk lanes cannot cross the 128-bit boundary.
    #[inline(always)]
    pub fn shuffle_16(self, indices: Simd<u16, A>) -> Self {
        unsafe { Self::new(self.data.vshuffle_16(indices.data)) }
    }

    /// Shuffles the 8-bit chunks for each 128-bit lane in a register
    /// according to the indices in another register. Each index is
    /// relative to its 128-bit chunk lanes cannot cross the 128-bit boundary.
    #[inline(always)]
    pub fn shuffle_8(self, indices: Simd<u8, A>) -> Self {
        unsafe { Self::new(self.data.vshuffle_8(indices.data)) }
    }

    /// Shuffles 128 bit chunks inside every 256-bit lane repeating
    /// using a 2 bit immediate. Each bit selects a 128-bit chunk from
    /// the current 256-bit lane, repeating for every 256 bit chunk.
    /// If the register is only 128 bits, then the register is returned
    /// unmodified.
    #[inline(always)]
    pub fn permute_imm_128<const N: i32>(self) -> Self {
        unsafe { Self::new(self.data.permute_128::<N>()) }
    }

    /// Shuffles 64 bit chunks inside every 128-bit lane repeating
    /// using a 2 bit immediate. Each bit selects a 64-bit chunk from
    /// the current 128-bit lane, repeating for every 128 bit chunk.
    #[inline(always)]
    pub fn shuffle_imm_64<const N: i32>(self) -> Self {
        unsafe { Self::new(self.data.shuffle_64::<N>()) }
    }

    /// Shuffles 32 bit chunks inside every 128-bit lane repeating
    /// using an 8 bit immediate. Each 2-bit segment selects a 32-bit chunk
    /// from the current 128-bit lane, repeating for every 128 bit lane.
    #[inline(always)]
    pub fn shuffle_imm_32<const N: i32>(self) -> Self {
        unsafe { Self::new(self.data.shuffle_32::<N>()) }
    }

    /// Shuffles 16 bit chunks inside every 64-bit lane repeating
    /// using an 8 bit immediate. Each 2-bit segment selects a 16-bit chunk
    /// from the current 64-bit lane, repeating for every 64 bit lane.
    #[inline(always)]
    pub fn shuffle_imm_16<const N: i32>(self) -> Self {
        unsafe { Self::new(self.data.shuffle_16::<N>()) }
    }

    /// Combines two input registers by selecting the upper or lower
    /// half from each as decided by a 2 bit immediate.
    #[inline(always)]
    pub fn permute2_imm_half<const N: i32>(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.permute2_half::<N>(rhs.data)) }
    }

    /// Shuffles 128 bit chunks inside every 256-bit lane repeating
    /// using a 2 bit immediate. Each bit selects a 128-bit chunk from
    /// the current 256-bit lane, repeating for every 256 bit chunk.
    /// The first bit picks from `self`, while the second bit picks from `rhs`.
    /// If the register is only 128 bits, then the register is returned
    /// unmodified.
    #[inline(always)]
    pub fn shuffle2_imm_128<const N: i32>(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.shuffle2_128::<N>(rhs.data)) }
    }

    /// Shuffles 64 bit chunks inside every 128-bit lane repeating
    /// using a 2 bit immediate. Each bit selects a 64-bit chunk from
    /// the current 128-bit lane, repeating for every 128 bit chunk.
    /// The first bit picks from `self`, while the second bit picks from `rhs`.
    #[inline(always)]
    pub fn shuffle2_imm_64<const N: i32>(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.shuffle2_64::<N>(rhs.data)) }
    }

    /// Shuffles 32 bit chunks inside every 128-bit lane repeating
    /// using an 8 bit immediate. Each 2-bit segment selects a 32-bit chunk
    /// from the current 128-bit lane, repeating for every 128 bit lane.
    /// The first 4-bits (first two selections) select from `self`, while
    /// the second 4-bits select from `rhs`.
    #[inline(always)]
    pub fn shuffle2_imm_32<const N: i32>(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.shuffle2_32::<N>(rhs.data)) }
    }

    /// Takes the lower halves of two registers and interleaves them
    /// in 128-bit chunks. If the register is only 128-bits, it is
    /// returned unmodified.
    #[inline(always)]
    pub fn interleave_lower_128(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_lower_128(rhs.data)) }
    }

    /// Takes the lower halves of two registers and interleaves them
    /// in 64-bit chunks.
    #[inline(always)]
    pub fn interleave_lower_64(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_lower_64(rhs.data)) }
    }

    /// Takes the lower halves of two registers and interleaves them
    /// in 32-bit chunks.
    #[inline(always)]
    pub fn interleave_lower_32(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_lower_32(rhs.data)) }
    }

    /// Takes the lower halves of two registers and interleaves them
    /// in 16-bit chunks.
    #[inline(always)]
    pub fn interleave_lower_16(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_lower_16(rhs.data)) }
    }

    /// Takes the lower halves of two registers and interleaves them
    /// in 8-bit chunks.
    #[inline(always)]
    pub fn interleave_lower_8(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_lower_8(rhs.data)) }
    }

    /// Takes the upper halves of two registers and interleaves them
    /// in 128-bit chunks. If the register is only 128-bits, it is
    /// returned unmodified.
    #[inline(always)]
    pub fn interleave_upper_128(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_upper_128(rhs.data)) }
    }

    /// Takes the upper halves of two registers and interleaves them
    /// in 64-bit chunks.
    #[inline(always)]
    pub fn interleave_upper_64(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_upper_64(rhs.data)) }
    }

    /// Takes the upper halves of two registers and interleaves them
    /// in 32-bit chunks.
    #[inline(always)]
    pub fn interleave_upper_32(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_upper_32(rhs.data)) }
    }

    /// Takes the upper halves of two registers and interleaves them
    /// in 16-bit chunks.
    #[inline(always)]
    pub fn interleave_upper_16(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_upper_16(rhs.data)) }
    }

    /// Takes the upper halves of two registers and interleaves them
    /// in 8-bit chunks.
    #[inline(always)]
    pub fn interleave_upper_8(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_upper_8(rhs.data)) }
    }

    /// Interleaves the even 128-bit chunks of two registers.
    /// If the register is only 128 bits, then it is returned unmodified.
    #[inline(always)]
    pub fn interleave_even_128(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_even_128(rhs.data)) }
    }

    /// Interleaves the even 64-bit chunks of two registers.
    #[inline(always)]
    pub fn interleave_even_64(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_even_64(rhs.data)) }
    }

    /// Interleaves the even 32-bit chunks of two registers.
    #[inline(always)]
    pub fn interleave_even_32(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_even_32(rhs.data)) }
    }

    /// Interleaves the even 16-bit chunks of two registers.
    #[inline(always)]
    pub fn interleave_even_16(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_even_16(rhs.data)) }
    }

    /// Interleaves the even 8-bit chunks of two registers.
    #[inline(always)]
    pub fn interleave_even_8(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_even_8(rhs.data)) }
    }

    /// Interleaves the odd 128-bit chunks of two registers.
    /// If the register is only 128 bits, then it is returned unmodified.
    #[inline(always)]
    pub fn interleave_odd_128(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_odd_128(rhs.data)) }
    }

    /// Interleaves the odd 64-bit chunks of two registers.
    #[inline(always)]
    pub fn interleave_odd_64(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_odd_64(rhs.data)) }
    }

    /// Interleaves the odd 32-bit chunks of two registers.
    #[inline(always)]
    pub fn interleave_odd_32(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_odd_32(rhs.data)) }
    }

    /// Interleaves the odd 16-bit chunks of two registers.
    #[inline(always)]
    pub fn interleave_odd_16(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_odd_16(rhs.data)) }
    }

    /// Interleaves the odd 8-bit chunks of two registers.
    #[inline(always)]
    pub fn interleave_odd_8(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.interleave_odd_8(rhs.data)) }
    }

    /// Concatenates two registers together and packs all of the even
    /// 128-bit chunks into one register. If the register is only 128 bits,
    /// then it is returned unmodified.
    #[inline(always)]
    pub fn deinterleave_even_128(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.deinterleave_even_128(rhs.data)) }
    }

    /// Concatenates two registers together and packs all of the even
    /// 64-bit chunks into one register.
    #[inline(always)]
    pub fn deinterleave_even_64(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.deinterleave_even_64(rhs.data)) }
    }

    /// Concatenates two registers together and packs all of the even
    /// 32-bit chunks into one register.
    #[inline(always)]
    pub fn deinterleave_even_32(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.deinterleave_even_32(rhs.data)) }
    }

    /// Concatenates two registers together and packs all of the even
    /// 16-bit chunks into one register.
    #[inline(always)]
    pub fn deinterleave_even_16(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.deinterleave_even_16(rhs.data)) }
    }

    /// Concatenates two registers together and packs all of the even
    /// 8-bit chunks into one register.
    #[inline(always)]
    pub fn deinterleave_even_8(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.deinterleave_even_8(rhs.data)) }
    }

    /// Concatenates two registers together and packs all of the odd
    /// 128-bit chunks into one register. If the register is only 128 bits,
    /// then it is returned unmodified.
    #[inline(always)]
    pub fn deinterleave_odd_128(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.deinterleave_odd_128(rhs.data)) }
    }

    /// Concatenates two registers together and packs all of the odd
    /// 64-bit chunks into one register.
    #[inline(always)]
    pub fn deinterleave_odd_64(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.deinterleave_odd_64(rhs.data)) }
    }

    /// Concatenates two registers together and packs all of the odd
    /// 32-bit chunks into one register.
    #[inline(always)]
    pub fn deinterleave_odd_32(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.deinterleave_odd_32(rhs.data)) }
    }

    /// Concatenates two registers together and packs all of the odd
    /// 16-bit chunks into one register.
    #[inline(always)]
    pub fn deinterleave_odd_16(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.deinterleave_odd_16(rhs.data)) }
    }

    /// Concatenates two registers together and packs all of the odd
    /// 8-bit chunks into one register.
    #[inline(always)]
    pub fn deinterleave_odd_8(self, rhs: Self) -> Self {
        unsafe { Self::new(self.data.deinterleave_odd_8(rhs.data)) }
    }

    /// Concatenates the lower half of the first register with
    /// the lower half of the second register.
    #[inline(always)]
    pub fn concat_lo_lo(self, rhs: Self) -> Self {
        self.shuffle2_imm_128::<0b00>(rhs)
    }

    /// Concatenates the lower half of the first register with
    /// the upper half of the second register.
    #[inline(always)]
    pub fn concat_lo_hi(self, rhs: Self) -> Self {
        self.shuffle2_imm_128::<0b10>(rhs)
    }

    /// Concatenates the upper half of the first register with
    /// the lower half of the second register.
    #[inline(always)]
    pub fn concat_hi_lo(self, rhs: Self) -> Self {
        self.shuffle2_imm_128::<0b01>(rhs)
    }

    /// Concatenates the upper half of the first register with
    /// the upper half of the second register.
    #[inline(always)]
    pub fn concat_hi_hi(self, rhs: Self) -> Self {
        self.shuffle2_imm_128::<0b11>(rhs)
    }
}

impl<T: SimdElement, A: Arch> Simd<T, A> {
    #[inline(always)]
    pub fn shuffle_8_pattern_32(self, indices: [u8; 4]) -> Self {
        let pattern = u32::from_ne_bytes(indices);
        let pattern_vec = Simd::<u32, A>::splat(pattern).raw_cast();
        self.shuffle_8(pattern_vec)
    }
}

// TODO: Super early version gather.
impl<A: Arch> Simd<u32, A> {
    pub fn gather<S: SimdElement + SimdElement<BitWidthType = B32>, const N: usize>(
        self,
        slice: &[S; N],
    ) -> Simd<S, A> {
        if N <= Self::LANES {
            let data = Simd::<S, A>::from_slice(&slice[..]);
            data.permute_32(self)
        } else {
            unsafe { Simd::new(self.data.gather_32_from_32::<S, 4>(slice.as_ptr())) }
        }
    }
}

impl<A: Arch> Simd<u64, A> {
    pub fn gather<S: SimdElement + SimdElement<BitWidthType = B64>, const N: usize>(
        self,
        slice: &[S; N],
    ) -> Simd<S, A> {
        unsafe { Simd::new(self.data.gather_64_from_64::<S, 8>(slice.as_ptr())) }
    }
}

// TODO: Add other types of lane shifts.
impl<T: SimdElement, A: Arch> Simd<T, A> {
    pub fn left_lane_shift(self, n: u32) -> Self {
        match T::BIT_SIZE {
            BitSize::Size32 => unsafe { Self::new(self.data.left_lane_shift_32(n)) },
            _ => unreachable!(),
        }
    }
    pub fn right_lane_shift(self, n: u32) -> Self {
        match T::BIT_SIZE {
            BitSize::Size32 => unsafe { Self::new(self.data.right_lane_shift_32(n)) },
            _ => unreachable!(),
        }
    }
}
