mod matrix_vector;
mod neural_network;

use matrix_vector::{Matrix, Vector};

fn main() {
    let matrix: Matrix = Matrix{
        value: vec![vec![2.,1.,1.];2],
        hight: 2,
        width: 3,
    };
    let vector: Vector = Vector{
        value: vec![0.,0.,2.],
        length: 3,
    };
    println!("Matrix is: {:?}, Vector is: {:?}", matrix, vector);

    println!("Product: {:?}", &(&matrix * &vector) + &vector);
}
