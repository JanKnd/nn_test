mod matrix_vector;
mod neural_network;
mod training_data;

use matrix_vector::{Matrix, Vector};
use crate::neural_network::{NeuralNetwork, tanh};
use crate::training_data::TrainingData;

fn main() {
    /*
    let matrix: Matrix = Matrix{
        value: vec![vec![2.,1.,1.];2],
        hight: 2,
        width: 3,
    };
    let vector: Vector = Vector{
        value: vec![0.,0.,2.],
        length: 3,
    };
    println!("Matrix is: {:?}, Vector is: {:?}", &matrix, &vector);

    println!("Product: {:?}", &matrix * &vector);
     */

    let nn = NeuralNetwork::new_random(784, 10, 200, 50);
    let a = TrainingData::new();
    let outputs = nn.run(a.inputs[0].clone());
    println!("output: {:?}",outputs);
}
