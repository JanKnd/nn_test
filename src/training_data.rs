use std::error::Error;
use std::process::Output;
use csv::{ReaderBuilder, StringRecord};
use datetime;
use crate::matrix_vector::Vector;

#[derive(Debug)]
pub struct TrainingData {
    pub inputs: Vec<Vector>,
    pub outputs: Vec<Vector>,
}


impl TrainingData {
    pub fn new() -> TrainingData {
        let mut rdr = ReaderBuilder::new().from_path("archive/mnist_train.csv");

        let mut result_inputs: Vec<Vector> = vec![];
        let mut result_outputs: Vec<Vector> = vec![];

        for result in rdr.unwrap().records() {
            let record = result.unwrap();
            let mut whole_row: Vec<f64> = vec![];
            for i in 0..record.len() {
                whole_row.push(record[i].parse().unwrap());
            }
            let mut raw_outputs: Vec<f64> = vec![0.;10];
            raw_outputs[whole_row[0] as usize] = 1.;

            result_inputs.push(Vector{
                value: whole_row[1..].to_owned(),
                length: whole_row.len()-1,
            });
            result_outputs.push( Vector{
                value: raw_outputs,
                length: 10,
            })
        }
        TrainingData{
            inputs: result_inputs,
            outputs: result_outputs,
        }
    }
}
