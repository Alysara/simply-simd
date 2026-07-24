#![doc = include_str!("../README.md")]

//! No-boilerplate portable SIMD rust library for raw registers with runtime feature detection

pub mod architectures;
pub mod simd_array;
mod dispatch;
pub mod mask;
pub mod register;
mod static_simd;
pub mod simd_types;
mod aliases;

pub use architectures::interface::Arch;
pub use dispatch::*;
pub use mask::Mask;
pub use simply_simd_macros::{dispatch_simd, enable_targets};
pub use register::Simd;
pub use register::iters::SimdSliceIterExt;
pub use static_simd::*;
pub use simd_types::{SimdElement, SimdFloat, SimdInteger};
// pub use dispatch::{Architecture, detect_architecture, dispatch};
