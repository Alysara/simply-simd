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

pub use crate::execute_intrinsic;
// pub use crate::execute_intrinsic_copy;
pub use crate::execute_const_intrinsic;
pub use crate::self_from_op;
// pub use crate::self_from_op_copy;
pub use crate::self_from_const_op;
