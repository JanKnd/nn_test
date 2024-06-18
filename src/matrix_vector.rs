use std::fmt::Error;
use std::ops::{Add, Mul};
use std::os::windows::ffi::EncodeWide;
use rand::distributions::Open01;
use rand::prelude::*;
use log::error;


#[derive(Debug,Clone)]
pub struct Matrix{
    pub value: Vec<Vec<f64>>,
    pub hight: usize,
    pub width: usize,
}

impl Matrix {
    pub fn new_zeros(hight: usize, width: usize) -> Matrix{
        Matrix{
            value: vec![vec![0.; width];hight],
            hight,
            width,
        }
    }


    pub fn new_random(hight: usize, width: usize, lower_bound: f64, upper_bound: f64) -> Matrix{

        let mut rng = rand::thread_rng();
        Matrix{
            value: vec![vec![rng.gen_range(lower_bound..upper_bound); width];hight],
            hight,
            width,
        }
    }
}
impl Mul<&Vector> for &Matrix{
    type Output = Vector;
    fn mul(self, rhs: &Vector) -> Self::Output {
        if self.width != rhs.length {
            panic!("Cannot multiply matrix of width n with vector of length m.");
        }
        let mut result = vec![];

        for row in 0..self.hight {
            let mut row_entry:f64 = 0.;
            for column in 0..self.width {
                row_entry += self.value[row][column] * rhs.value[column];
            }
            result.push(row_entry);
        }

        Vector {
            value: result,
            length: self.hight, }

    }
}


#[derive(Debug, Clone)]
pub struct Vector{
    pub value: Vec<f64>,
    pub length: usize,
}

impl Vector {
    pub fn new_zeros(length: usize, width: usize) -> Vector {
        Vector {
            value: vec![0.; length],
            length,
        }
    }
    pub fn new_random(length: usize, lower_bound: f64, upper_bound: f64) -> Vector {

        let mut rng = rand::thread_rng();
        Vector {
            value: vec![rng.gen_range(lower_bound..upper_bound); length],
            length,
        }
    }

}

impl Add for &Vector {
    type Output = Vector;
    fn add(self, rhs: &Vector) -> Self::Output {
        if self.length != rhs.length {
            panic!("Vectors must be of the same length to add.")
        }

        let mut result = vec![];
        for i in 0..self.length {
            result.push(self.value[i] + rhs.value[i])
        }
        Vector {
            value: result,
            length: self.length,
        }
    }
}

