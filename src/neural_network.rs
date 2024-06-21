
use std::ffi::NulError;
use std::process::Output;
use rand::Rng;
use crate::matrix_vector::{Matrix, Vector};

extern crate savefile;
use savefile::prelude::*;
use savefile_derive::Savefile;




pub fn tanh(x: &f64) -> f64{
    f64::tanh(*x)
}

pub fn tanh_derivative(x: &f64) -> f64{
    1. - tanh(x).powi(2)
}


#[derive(Debug)]
#[derive(Savefile)]
pub struct NeuralNetwork{
    pub num_inputs: usize,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Vector>,
    pub num_outputs: usize,
}

impl NeuralNetwork{
    pub fn new_random(num_inputs: usize, num_outputs: usize, hidden_layers: usize, hidden_layers_neuron_count: usize) -> NeuralNetwork{
        let mut weights= vec![];
        weights.push( Matrix::new_random(hidden_layers_neuron_count, num_inputs, -1.,1.));
            for _ in 0..hidden_layers-1{
                weights.push(Matrix::new_random(hidden_layers_neuron_count, hidden_layers_neuron_count, -1.,1.,));
            }
            weights.push(Matrix::new_random(num_outputs,hidden_layers_neuron_count, -1.,1.));


        let mut biases = vec![];
            for _ in 0..hidden_layers{
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

            //println!("i: {:?}, res: {:?}",i,result);
        }
        result= result.squish_vector();
        result.mse(label)
    }

    pub fn bp_single(mut self, input: Vector, label: Vector, learn_rate: f64) -> (Self, f64){
        let learning_rate: f64 = learn_rate;
        let mut result: Vector = input;
        let mut xs: Vec<Vector> = vec![];
        let mut ys: Vec<Vector> = vec![];
        let mut ys_squished: Vec<Vector> = vec![];
        for i in 0..self.biases.len(){
            xs.push(result.clone());
            result = (&(&self.weights[i] * &result) + &self.biases[i]);
            ys.push(result.clone());
            result = result.squish_vector();
            ys_squished.push(result.clone());
        }
        result= result.squish_vector();
        let error = result.mse(label.clone());
        //println!("{:?}", error);

        let grad_output = &(&result + &(&label * &-1.)) * &(2. / result.length as f64);
        let grad_input_for_activation = ys[ys.len()-1].clone().grad_tanh();
        let mut grad_y: Vector = grad_output.mul_element_wise(&grad_input_for_activation);
        let mut grad_x: Vector;
        for i in (0..self.biases.len()).rev(){
            grad_x = &self.weights[i].clone().transpose() * &grad_y;
            let grad_weights = &grad_y * &xs[i];
            let grad_biases = grad_y.clone();

            self.biases[i] = &self.biases[i].clone() + &(&grad_biases * &learning_rate);
            self.weights[i] = &self.weights[i].clone() + &(&grad_weights * &learning_rate );

            if i == 0{
                break;
            } else {
                grad_y = grad_x.mul_element_wise(&ys[i-1].clone().grad_tanh());
            }
        }
        (self, error)
    }

    pub fn save(&self){
        save_file("neural_network_1.bin", 0, self).unwrap()
    }

}


