#![doc = include_str!("../README.md")]

//! No-boilerplate portable SIMD library for raw registers with runtime feature detection

pub mod architectures;
mod dispatch;
pub mod mask;
pub mod register;
mod simd_array;
pub mod simd_types;
mod static_simd;

pub use architectures::interface::Arch;
pub use dispatch::*;
pub use mask::Mask;
pub use register::Simd;
pub use register::iters::SimdSliceIterExt;
pub use simd_array::SimdToArray;
pub use simd_types::{SimdElement, SimdFloat, SimdInteger};
pub use simply_simd_macros::{dispatch_simd, enable_targets};
pub use static_simd::*;
// pub use dispatch::{Architecture, detect_architecture, dispatch};
