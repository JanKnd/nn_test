use std::ffi::NulError;
use std::process::Output;
use rand::Rng;
use crate::matrix_vector::{Matrix, Vector};



pub fn tanh(x: &f64) -> f64{
    f64::tanh(*x)
}

pub fn tanh_derivative(x: &f64) -> f64{
    1. - tanh(x).powi(2)
}


pub struct NeuralNetwork{
    pub num_inputs: usize,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Vector>,
    pub num_outputs: usize,
}

impl NeuralNetwork{
    pub fn new_random(num_inputs: usize, num_outputs: usize, hidden_layers: usize, hidden_layers_neuron_count: usize) -> NeuralNetwork{
        let mut rng = rand::thread_rng();

        //need to implement bahavior for hiddenlayers = 0


        let mut weights= vec![];
        weights.push( Matrix::new_random(hidden_layers_neuron_count, num_inputs, -1.,1.));
            for i in 0..hidden_layers-1{
                weights.push(Matrix::new_random(hidden_layers_neuron_count, hidden_layers_neuron_count, -1.,1.,));
            }
            weights.push(Matrix::new_random(num_outputs,hidden_layers_neuron_count, -1.,1.));


        let mut biases = vec![];
            for i in 0..hidden_layers{
                biases.push(Vector::new_random(hidden_layers_neuron_count, -1.,1.));
            }
            biases.push(Vector::new_random(num_outputs, -1.,1.));

        NeuralNetwork{
            num_inputs,
            weights,
            biases,
            num_outputs,
        }
    }

    pub fn run(&self, input: Vector, label: Vector) -> f64{
        let mut result: Vector = input;
        for i in 0..self.biases.len(){
            result = (&(&self.weights[i] * &result) + &self.biases[i]).squish_vector();

            println!("i: {:?}, res: {:?}",i,result);
        }
        result= result.squish_vector();
        result.mse(label)
    }

}