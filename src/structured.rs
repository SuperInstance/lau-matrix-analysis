//! Structured matrices: Toeplitz, circulant, Vandermonde — fast algorithms.

use crate::Mat;

/// Create a Toeplitz matrix from first column (c) and first row (r).
/// c[0] must equal r[0] (the diagonal element).
pub fn toeplitz(col: &[f64], row: &[f64]) -> Mat {
    let n = col.len();
    let m = row.len();
    let mut mat = Mat::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            if j >= i {
                mat[(i, j)] = row[j - i];
            } else {
                mat[(i, j)] = col[i - j];
            }
        }
    }
    mat
}

/// Create a symmetric Toeplitz matrix from its first column.
pub fn toeplitz_symmetric(col: &[f64]) -> Mat {
    toeplitz(col, col)
}

/// Create a circulant matrix from its first row.
pub fn circulant(first_row: &[f64]) -> Mat {
    let n = first_row.len();
    let mut mat = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = (j + n - i) % n;
            mat[(i, j)] = first_row[idx];
        }
    }
    mat
}

/// Fast circulant-vector multiplication via DFT-like approach (O(n^2) fallback).
/// For production, would use FFT — here we use the circulant structure directly.
pub fn circulant_mul(first_row: &[f64], v: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
    let n = first_row.len();
    assert_eq!(v.len(), n);
    let mut result = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            let idx = (j + n - i) % n;
            result[i] += first_row[idx] * v[j];
        }
    }
    nalgebra::DVector::from_vec(result)
}

/// Create a Vandermonde matrix from a vector of nodes.
/// V_{i,j} = x_i^{j} for i = 0..n-1, j = 0..m-1.
pub fn vandermonde(x: &[f64], m: usize) -> Mat {
    let n = x.len();
    let mut mat = Mat::zeros(n, m);
    for i in 0..n {
        let mut pow = 1.0;
        for j in 0..m {
            mat[(i, j)] = pow;
            pow *= x[i];
        }
    }
    mat
}

/// Vandermonde determinant: product of (x_i - x_j) for i > j.
/// Returns None if nodes are not distinct.
pub fn vandermonde_det(x: &[f64]) -> Option<f64> {
    let n = x.len();
    let mut det = 1.0;
    for i in 1..n {
        for j in 0..i {
            let diff = x[i] - x[j];
            if diff.abs() < 1e-15 {
                return None; // Not distinct
            }
            det *= diff;
        }
    }
    Some(det)
}

/// Create a Hankel matrix: constant anti-diagonals, specified by first column and last row.
pub fn hankel(col: &[f64], row: &[f64]) -> Mat {
    let n = col.len();
    let m = row.len();
    let mut mat = Mat::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            let idx = i + j;
            if idx < n {
                mat[(i, j)] = col[idx];
            } else {
                mat[(i, j)] = row[idx - n + 1];
            }
        }
    }
    mat
}
