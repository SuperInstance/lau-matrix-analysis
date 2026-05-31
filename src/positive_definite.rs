//! Positive definite matrices: verification, nearest PSD, Cholesky.

use crate::Mat;
use crate::decompositions::eigen;

/// Check if a symmetric matrix is positive definite via Cholesky.
pub fn is_positive_definite(a: &Mat) -> bool {
    if a.nrows() != a.ncols() {
        return false;
    }
    crate::decompositions::cholesky(a).is_some()
}

/// Check if a symmetric matrix is positive semi-definite.
pub fn is_positive_semidefinite(a: &Mat, tol: f64) -> bool {
    if a.nrows() != a.ncols() {
        return false;
    }
    let eig = match eigen(a) {
        Some(e) => e,
        None => return false,
    };
    eig.eigenvalues.iter().all(|&v| v >= -tol)
}

/// Compute the nearest positive semi-definite matrix to A (Higham's algorithm).
pub fn nearest_psd(a: &Mat) -> Option<Mat> {
    let n = a.nrows();
    if n != a.ncols() {
        return None;
    }

    let at = a.transpose();
    let sym_a = (a + &at) * 0.5;

    let mut x = sym_a;

    for _ in 0..100 {
        let eig = eigen(&x)?;
        let mut d = Mat::zeros(n, n);
        for (i, &val) in eig.eigenvalues.iter().enumerate() {
            d[(i, i)] = val.max(0.0);
        }
        let ev_clone = eig.eigenvectors.clone();
        let v_inv = ev_clone.try_inverse()?;
        let x_new = eig.eigenvectors * d * v_inv;
        
        let diff = &x_new - &x;
        let change = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
        x = x_new;
        if change < 1e-10 {
            break;
        }
    }

    Some(x)
}

/// Result of positive-definiteness verification.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PDVerifyResult {
    pub is_symmetric: bool,
    pub min_eigenvalue: f64,
    pub max_eigenvalue: f64,
    pub is_pd: bool,
    pub is_psd: bool,
}

/// Verify positive definiteness and return eigenvalue range.
pub fn pd_verify(a: &Mat) -> Option<PDVerifyResult> {
    let n = a.nrows();
    if n != a.ncols() || n == 0 {
        return None;
    }

    let sym_diff = a - &a.transpose();
    let sym_err = sym_diff.iter().map(|v| v * v).sum::<f64>().sqrt();

    let eig = eigen(a)?;
    let min_eig = eig.eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_eig = eig.eigenvalues.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Some(PDVerifyResult {
        is_symmetric: sym_err < 1e-10,
        min_eigenvalue: min_eig,
        max_eigenvalue: max_eig,
        is_pd: min_eig > 0.0,
        is_psd: min_eig >= -1e-10,
    })
}
