#[macro_export]
macro_rules! execute_intrinsic {
    ($intrinsic:ident, $($arg:expr),*) => {
        unsafe { $intrinsic($(transmute_copy(&$arg)),*) }
    };
}

// #[macro_export]
// macro_rules! execute_intrinsic {
//     ($intrinsic:ident, $($arg:expr),*) => {
//         unsafe { $intrinsic($(transmute($arg)),*) }
//     };
// }

#[macro_export]
macro_rules! execute_const_intrinsic {
    ($intrinsic:ident, $const:expr, $($arg:expr),*) => {
        unsafe { $intrinsic::<{$const}>($(transmute($arg)),*) }
    };
}

// #[macro_export]
// macro_rules! self_from_op_copy {
//     ($intrinsic:ident, $($arg:expr),*) => {
//         unsafe { Self(transmute(execute_intrinsic_copy!($intrinsic, $($arg),*))) }
//     }
// }

#[macro_export]
macro_rules! self_from_op {
    ($intrinsic:ident, $($arg:expr),*) => {
        unsafe { Self(transmute_copy(&$intrinsic($(transmute_copy(&$arg)),*))) }
    }
}

#[macro_export]
macro_rules! self_from_const_op {
    ($intrinsic:ident, $const:expr, $($arg:expr),*) => {
        unsafe { Self(transmute($intrinsic::<{$const}>($(transmute($arg)),*))) }
    }
}

#[macro_export]
macro_rules! self_from_const2_op {
    ($intrinsic:ident, $const1:expr, $const2:expr, $($arg:expr),*) => {
        unsafe { Self(transmute($intrinsic::<{$const1}, {$const2}>($(transmute($arg)),*))) }
    }
}

#[macro_export]
macro_rules! map_intrinsic {
    ($func:ident, $intrinsic:ident, self $(, $arg:ident : $ty:ty)* $(,)?) => {
        #[inline(always)]
        unsafe fn $func(self $(, $arg: $ty)*) -> Self {
            unsafe { self_from_op!($intrinsic, self $(, $arg)*) }
        }
    };

    ($func:ident, $intrinsic:ident $(, $arg:ident : $ty:ty)* $(,)?) => {
        #[inline(always)]
        unsafe fn $func($(, $arg: $ty)*) -> Self {
            unsafe { self_from_op!($intrinsic, $(, $arg)*) }
        }
    };

    ($func:ident, $intrinsic:ident, const $const_arg:ident : $const_ty:ty, self $(, $arg:ident : $ty:ty)* $(,)?) => {
        #[inline(always)]
        unsafe fn $func<const $const_arg: $const_ty>(self $(, $arg: $ty)*) -> Self {
            unsafe { self_from_const_op!($intrinsic, $const_arg, self $(, $arg)*) }
        }
    };

    ($func:ident, $intrinsic:ident, const $const_arg:ident : $const_ty:ty, self, self) => {
        #[inline(always)]
        unsafe fn $func<const $const_arg: $const_ty>(self) -> Self {
            unsafe { self_from_const_op!($intrinsic, $const_arg, self, self) }
        }
    };
}

#[macro_export]
macro_rules! map_x86_shift {
    ($func:ident, $intrinsic:ident) => {
        #[inline(always)] unsafe fn $func(self, rhs: u32) -> Self {
            let shift = unsafe { _mm_cvtsi32_si128(rhs as i32) };
            self_from_op!($intrinsic, self, shift)
        }
    }
}

// Compiler folds into immediate shift, no need for match table.
#[macro_export]
macro_rules! map_x86_conversion_imm_shift {
    ($func:ident, $intrinsic:ident) => {
        #[inline(always)]
        unsafe fn $func<const N: i32>(self) -> Self {
            let shift = unsafe { _mm_cvtsi32_si128(N) };
            self_from_op!($intrinsic, self, shift)
        }
    }
}


pub use crate::execute_intrinsic;
// pub use crate::execute_intrinsic_copy;
pub use crate::execute_const_intrinsic;
pub use crate::self_from_op;
// pub use crate::self_from_op_copy;
pub use crate::self_from_const_op;
pub use crate::self_from_const2_op;
pub use crate::map_intrinsic;
pub use crate::map_x86_shift;
pub use crate::map_x86_conversion_imm_shift;
