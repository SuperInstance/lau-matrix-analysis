//! Matrix norms: operator, Frobenius, nuclear, condition number.

use crate::Mat;
use crate::decompositions::svd;

/// Frobenius norm: sqrt(sum of all |a_ij|^2).
pub fn frobenius_norm(a: &Mat) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Operator norm (spectral norm = largest singular value).
pub fn operator_norm(a: &Mat) -> f64 {
    if a.nrows() == 0 || a.ncols() == 0 {
        return 0.0;
    }
    let ata = a.transpose() * a;
    // Power iteration for largest eigenvalue of A^T A
    let n = ata.ncols();
    let mut v = nalgebra::DVector::from_element(n, 1.0 / (n as f64).sqrt());

    for _ in 0..300 {
        let w = &ata * &v;
        let norm = w.norm();
        if norm < 1e-30 {
            return 0.0;
        }
        v = w / norm;
    }

    let lambda = v.dot(&(&ata * &v));
    lambda.abs().sqrt()
}

/// Nuclear norm (sum of singular values).
pub fn nuclear_norm(a: &Mat) -> f64 {
    match svd(a) {
        Some(s) => s.sigma.iter().sum(),
        None => 0.0,
    }
}

/// 1-norm (max column absolute sum).
pub fn norm_1(a: &Mat) -> f64 {
    let mut max_sum = 0.0;
    for j in 0..a.ncols() {
        let col_sum: f64 = a.column(j).iter().map(|x| x.abs()).sum();
        if col_sum > max_sum {
            max_sum = col_sum;
        }
    }
    max_sum
}

/// Infinity-norm (max row absolute sum).
pub fn norm_inf(a: &Mat) -> f64 {
    let mut max_sum = 0.0;
    for i in 0..a.nrows() {
        let row_sum: f64 = a.row(i).iter().map(|x| x.abs()).sum();
        if row_sum > max_sum {
            max_sum = row_sum;
        }
    }
    max_sum
}

/// Condition number (using the operator/spectral norm).
pub fn condition_number(a: &Mat) -> f64 {
    let op = operator_norm(a);
    if op < 1e-15 {
        return f64::INFINITY;
    }
    // Use SVD for more accurate condition number
    match svd(a) {
        Some(s) => {
            let max_sv = s.sigma.iter().cloned().fold(0.0f64, f64::max);
            let min_sv = s.sigma.iter().cloned().fold(f64::INFINITY, f64::min);
            if min_sv < 1e-15 {
                f64::INFINITY
            } else {
                max_sv / min_sv
            }
        }
        None => f64::INFINITY,
    }
}

/// Trace of a square matrix.
pub fn trace(a: &Mat) -> f64 {
    (0..a.nrows().min(a.ncols())).map(|i| a[(i, i)]).sum()
}

/// Rank estimation via SVD (count singular values above tolerance).
pub fn rank(a: &Mat, tol: f64) -> usize {
    match svd(a) {
        Some(s) => s.sigma.iter().filter(|&&v| v > tol).count(),
        None => 0,
    }
}
