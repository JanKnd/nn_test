use std::ffi::NulError;
use std::process::Output;
use rand::Rng;
use crate::matrix_vector::{Matrix, Vector};



pub fn tanh(x: f64) -> f64{
    (( 2. * x ).exp() - 1.) / (( 2. * x ).exp() + 1.)
}

pub fn tanh_derivative(x: f64) -> f64{
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
        weights.push(Matrix {
            value: vec![vec![rng.gen_range(0_f64..1_f64); num_inputs]; hidden_layers_neuron_count],
            hight: hidden_layers_neuron_count,
            width: num_inputs,
        });
            for i in 0..hidden_layers-1{
                weights.push(Matrix{
                    value: vec![vec![rng.gen_range(0_f64..1_f64); hidden_layers_neuron_count]; hidden_layers_neuron_count],
                    hight: hidden_layers_neuron_count,
                    width: hidden_layers_neuron_count,
                });
            }
            weights.push(Matrix{
                value: vec![vec![rng.gen_range(0_f64..1_f64); num_inputs]; hidden_layers_neuron_count],
                hight: num_outputs,
                width: hidden_layers_neuron_count,
            });


        let mut biases = vec![];
            for i in 0..hidden_layers{
                biases.push(Vector{
                    value: vec![rng.gen_range(0_f64..1_f64);hidden_layers_neuron_count],
                    length: hidden_layers_neuron_count,
                });
            }
            biases.push(Vector{
                value: vec![rng.gen_range(0_f64..1_f64); num_outputs],
                length: num_outputs,
            });

        NeuralNetwork{
            num_inputs,
            weights,
            biases,
            num_outputs,
        }
    }


}