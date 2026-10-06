use std::hint::black_box;

use simply_simd::StaticSimd;

fn main() {
    let input1 = black_box(3.3_f32);
    let input2 = black_box(2.3_f32);

    let values1 = StaticSimd::splat(input1);
    let values2 = StaticSimd::splat(input2);
    // let result = unsafe {values1.log2()};
    let result = unsafe {values1.pow_unchecked(values2)};


    let values_array = black_box(values1.to_array());
    values_array.iter().for_each(|x| {
        black_box(x.exp());
    });

    // println!("input:      {:?}\nexp result: {:?}", values, result);
    println!("input1:     {:?}\ninput2:     {:?}\nexp result: {:?}", values1, values2, result);

} 
