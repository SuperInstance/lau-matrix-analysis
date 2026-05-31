//! Perturbation theory: eigenvalue sensitivity, condition number of eigenproblem.

use crate::Mat;
use crate::decompositions::eigen;
use crate::norms::operator_norm;

/// Eigenvalue condition number: for each eigenvalue λ_i, compute the condition number
/// κ_i = 1 / |y_i^H x_i| where x_i, y_i are right and left eigenvectors.
pub fn eigenvalue_condition_numbers(a: &Mat) -> std::vec::Vec<f64> {
    let n = a.nrows();
    let eig = match eigen(a) {
        Some(e) => e,
        None => return std::vec![f64::NAN; n],
    };

    let eig_left = match eigen(&a.transpose()) {
        Some(e) => e,
        None => return std::vec![f64::NAN; n],
    };

    let mut cond_nums = std::vec::Vec::with_capacity(n);
    for i in 0..n {
        let xi = eig.eigenvectors.column(i);
        let yi = eig_left.eigenvectors.column(i);
        let overlap = yi.dot(&xi).abs();
        if overlap < 1e-15 {
            cond_nums.push(f64::INFINITY);
        } else {
            cond_nums.push(1.0 / overlap);
        }
    }
    cond_nums
}

/// Bauer-Fike theorem bound.
pub fn bauer_fike_bound(a: &Mat, e: &Mat) -> Option<f64> {
    let eig = eigen(a)?;
    let ev_clone = eig.eigenvectors.clone();
    let v_inv = ev_clone.try_inverse()?;
    let cond_v = operator_norm(&eig.eigenvectors) * operator_norm(&v_inv);
    let e_norm = operator_norm(e);
    Some(cond_v * e_norm)
}

/// Weyl's inequality bound for symmetric matrices.
pub fn weyl_bound(e: &Mat) -> f64 {
    operator_norm(e)
}

/// Eigenvalue sensitivity: approximate derivative dλ/dA_{ij} via perturbation.
pub fn eigenvalue_sensitivity(a: &Mat, eigenvalue_idx: usize, eps: f64) -> Option<Mat> {
    let n = a.nrows();
    let eig_original = eigen(a)?;
    if eigenvalue_idx >= eig_original.eigenvalues.len() {
        return None;
    }
    let lambda_0 = eig_original.eigenvalues[eigenvalue_idx];

    let mut sensitivity = Mat::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            let mut a_pert = a.clone();
            a_pert[(i, j)] += eps;
            let eig_pert = eigen(&a_pert)?;
            let mut min_dist = f64::INFINITY;
            for k in 0..eig_pert.eigenvalues.len() {
                let dist = (eig_pert.eigenvalues[k] - lambda_0).abs();
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            sensitivity[(i, j)] = min_dist / eps;
        }
    }

    Some(sensitivity)
}
