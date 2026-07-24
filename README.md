No-boilerplate portable SIMD rust library for raw registers with runtime feature detection 

# Usage

## Static Dispatch

`StaticSimd` gives you easy access to simd using the feature set provided by
your compiler flags. Using the flag `target-cpu=native` will ensure you always
use the best simd architecutre.

```rust
use simply_simd::StaticSimd;

let a = StaticSimd::splat(1.0f32);
let b = StaticSimd::splat(1.5f32);

let c = a + b;

println!("Result: {:?}", c);
```

## Dynamic Dispatch

For dynamic runtime feature detection, `dispatch_simd` will automatically
compile your function for each of the supported targets.

```rust
use simply_simd::{Simd, dispatch_simd};

fn main() {
    // Call like normal.
    simd_work();
}

#[dispatch_simd(A)]
fn simd_work() {
    let a = Simd::<f32, A>::splat(1.5);
    let b = Simd::<f32, A>::iota(1.0);

    let c = b.mul_add(b, a);
}
```

Since the dispatch requires branching at runtime,
it is best to do this outside of hot loops.

Rust will not emit SIMD instructions inline if it does not have
the associated feature flags. Functions with generic type `A: Arch`
must either inline or use the `enable_targets` macro. If neither
of these are done, Rust may not inline the SIMD instructions and
performance will significantly degrade, up to 10x+ times slower.

```rust
use simply_simd::{Arch, dispatch_simd, enable_targets};

// Dispatch simd here to avoid repeated dispatching
// in every iteration.
#[dispatch_simd(A)]
pub fn simd_entry() {
    for _ in 0..1024 {
        simd_work_1::<A>();
        simd_work_2::<A>();
        broken_simd_work::<A>();
    }
}

// Avoid using #[dispatch_simd(A)] again here.
// Instead, use #[enable_targets(A)] with <A: Arch>.
#[enable_targets(A)]
pub fn simd_work_1<A: Arch>() {
    // ...
}

// #[inline(always)] can be used as well.
#[inline(always)]
pub fn simd_work_2<A: Arch>() {
    // ...
}

// No inline or enable_targets,
// compiler may not optimize or inline 
// SIMD instructions.
pub fn broken_simd_work<A: Arch>() {
    // ...
}

```

`dispatch_simd` and `enable_targets` can both be used on impl blocks as well. This is necessary
when `A: Arch` is generic across a struct. It will also apply to every function that contains
a generic `A: Arch` (or other identifier you specify). This method must be used for trait implementations:

```rust
use simply_simd::{Arch, enable_targets};

trait SimdTask {
    fn simd_work<A: Arch>();
}
struct SimdWorker {}

#[enable_targets(A)]
impl SimdTask for SimdWorker {
    fn simd_work<A: Arch>() {}
}
```

If `#[dispatch_simd(A)]` is applied to an associated function using generic
parameters from its impl block, the macro will not have enough information to tell it's an
associated function and requires an additional flag: `#[dispatch_simd(A, associated)]`.
For the majority of cases, this flag can be omitted.

Unfortunately, these restrictions make it impossible to ensure other functions that you do not own
use the dispatched target. This limitation applies to all SIMD libraries.

## Variable Lanes

Unlike std::simd, it does not use a fixed width and instead operates
in the largest supported lane count.

<!-- ```rust -->
<!-- use simple_simd::StaticSimd; -->
<!---->
<!-- for  -->
<!-- ``` -->
<!---->
