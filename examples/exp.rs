use std::hint::black_box;

use simply_simd::StaticSimd;

fn main() {
    let input = black_box(3.3_f64);

    let values = StaticSimd::splat(input);
    let result = unsafe {values.exp()};

    let values_array = black_box(values.to_array());
    values_array.iter().for_each(|x| {
        black_box(x.exp());
    });

    println!("input:      {:?}\nexp result: {:?}", values, result);

} 
