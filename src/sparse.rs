//! Sparse matrix basics: COO format, sparse-dense multiplication.

use crate::Mat;
use crate::VecN;

/// A sparse matrix in COOrdinate format.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SparseCOO {
    pub nrows: usize,
    pub ncols: usize,
    pub rows: std::vec::Vec<usize>,
    pub cols: std::vec::Vec<usize>,
    pub vals: std::vec::Vec<f64>,
}

impl SparseCOO {
    /// Create a new empty sparse matrix.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        SparseCOO {
            nrows,
            ncols,
            rows: std::vec::Vec::new(),
            cols: std::vec::Vec::new(),
            vals: std::vec::Vec::new(),
        }
    }

    /// Add an entry.
    pub fn add_entry(&mut self, i: usize, j: usize, v: f64) {
        assert!(i < self.nrows && j < self.ncols);
        self.rows.push(i);
        self.cols.push(j);
        self.vals.push(v);
    }

    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    /// Multiply sparse matrix by dense matrix: self * dense.
    pub fn mul_dense(&self, b: &Mat) -> Mat {
        let _k = b.nrows();
        let p = b.ncols();
        assert_eq!(b.nrows(), self.ncols, "Dimension mismatch in sparse-dense multiply");

        let mut result = Mat::zeros(self.nrows, p);
        for idx in 0..self.vals.len() {
            let i = self.rows[idx];
            let j = self.cols[idx];
            let v = self.vals[idx];
            for col in 0..p {
                result[(i, col)] += v * b[(j, col)];
            }
        }
        result
    }

    /// Multiply sparse matrix by dense vector: self * v.
    pub fn mul_vec(&self, v: &VecN) -> VecN {
        assert_eq!(v.len(), self.ncols, "Dimension mismatch in sparse-vector multiply");
        let mut result = nalgebra::DVector::zeros(self.nrows);
        for idx in 0..self.vals.len() {
            let i = self.rows[idx];
            let j = self.cols[idx];
            result[i] += self.vals[idx] * v[j];
        }
        result
    }

    /// Convert to dense matrix.
    pub fn to_dense(&self) -> Mat {
        let mut m = Mat::zeros(self.nrows, self.ncols);
        for idx in 0..self.vals.len() {
            m[(self.rows[idx], self.cols[idx])] += self.vals[idx];
        }
        m
    }

    /// Transpose.
    pub fn transpose(&self) -> SparseCOO {
        SparseCOO {
            nrows: self.ncols,
            ncols: self.nrows,
            rows: self.cols.clone(),
            cols: self.rows.clone(),
            vals: self.vals.clone(),
        }
    }
}
