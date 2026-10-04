use std::marker::PhantomData;

use crate::{StaticArch, architectures::interface::Arch};

pub mod element;

#[derive(Clone, Copy)]
pub struct Mask<T, A: Arch = StaticArch> {
    pub(crate) data: A::Mask,
    pub(crate) _marker: PhantomData<T>,
}
