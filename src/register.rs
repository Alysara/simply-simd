use std::marker::PhantomData;

use crate::architectures::interface::*;
use crate::{StaticArch, simd_types::*};

pub mod element;
pub mod integer;
pub mod float;
pub mod iters;
pub mod math;

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Simd<T: SimdElement, A: Arch = StaticArch> {
    pub(crate) data: A::Vec,
    pub(crate) _marker: PhantomData<T>,
}

impl<T: SimdElement, A: Arch> Simd<T, A> {
    pub const SIMD_WIDTH: usize = A::SIMD_WIDTH;
    pub const LANE_SIZE: usize = std::mem::size_of::<T>();
    pub const LANES: usize = A::SIMD_WIDTH / Self::LANE_SIZE;
}
