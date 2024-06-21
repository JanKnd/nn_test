
#[macro_use]
extern crate savefile_derive;
mod matrix_vector;
mod neural_network;
mod training_data;

use savefile::load_file;
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
    let mut nn:NeuralNetwork = load_file("neural_network_1.bin", 0).unwrap();
    //let mut nn = NeuralNetwork::new_random(784, 10, 5, 20);
    let a:TrainingData = load_file("Training_data.bin", 0).unwrap();

    println!("len: {:?}", a.inputs.len());
    let mut prev_msn:f64 = 1000.;
    let mut learn_rate:f64 = -0.001;
    for _a in 0..1000000 {
        let mut meanmsn = 0.;
        for i in 0..a.inputs.len() {
            let res = nn.bp_single(a.inputs[i].clone(), a.outputs[i].clone(), learn_rate);
            nn = res.0;
            meanmsn += res.1;
        }
        meanmsn = meanmsn / a.inputs.len() as f64;
        if meanmsn > prev_msn {
            println!("STEP TO LARGEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE");
            learn_rate = learn_rate / 2.;
            continue
        }
        prev_msn = meanmsn;
        println!("epoch {:?} : {:?}", _a, meanmsn);

        //let outputs = nn.run(a.inputs[0].clone(), a.outputs[0].clone());
        //let a = Matrix::new_random(2, 3,0.,10.);
        //println!("{:?}",);
            nn.save();
    }
}
