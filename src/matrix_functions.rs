//! Matrix functions: exponential via Padé, logarithm, square root.

use crate::Mat;
use crate::decompositions::eigen;

/// Matrix exponential via eigendecomposition: exp(A) = V * exp(D) * V^{-1}.
pub fn matrix_exp(a: &Mat) -> Option<Mat> {
    let n = a.nrows();
    if n != a.ncols() || n == 0 {
        return None;
    }

    let eig = eigen(a)?;
    let d = eig.eigenvalues.map(|v| v.exp());
    let mut d_mat = Mat::zeros(n, n);
    for (i, val) in d.iter().enumerate() {
        d_mat[(i, i)] = *val;
    }
    let v_inv = eig.eigenvectors.clone().try_inverse()?;
    Some(eig.eigenvectors * d_mat * v_inv)
}

/// Matrix exponential via Padé approximation with scaling and squaring.
/// Uses the [1/1] Padé approximant: exp(A) ≈ (I - A/2)^{-1}(I + A/2)
pub fn matrix_exp_pade(a: &Mat) -> Option<Mat> {
    let n = a.nrows();
    if n != a.ncols() || n == 0 {
        return None;
    }

    let norm = a.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    // Choose s such that ||A/2^s|| < 0.5
    let mut s = 0usize;
    let mut scaled_norm = norm;
    while scaled_norm > 0.5 {
        scaled_norm /= 2.0;
        s += 1;
    }
    let scale = 1.0 / (1u64 << s) as f64;
    let scaled_a = a.scale(scale);

    let i_n = Mat::identity(n, n);
    let half_a = scaled_a.scale(0.5);
    let numer = &i_n + &half_a;
    let denom = (&i_n - half_a).try_inverse()?;
    let mut result = denom * numer;

    // Repeated squaring: exp(A) = (exp(A/2^s))^{2^s}
    for _ in 0..s {
        result = result.clone() * &result;
    }

    Some(result)
}

/// Matrix logarithm: if A = V D V^{-1}, then log(A) = V log(D) V^{-1}.
pub fn matrix_log(a: &Mat) -> Option<Mat> {
    let n = a.nrows();
    if n != a.ncols() || n == 0 {
        return None;
    }

    let eig = eigen(a)?;

    for &v in eig.eigenvalues.iter() {
        if v <= 0.0 {
            return None;
        }
    }

    let d = eig.eigenvalues.map(|v| v.ln());
    let mut d_mat = Mat::zeros(n, n);
    for (i, val) in d.iter().enumerate() {
        d_mat[(i, i)] = *val;
    }
    let v_inv = eig.eigenvectors.clone().try_inverse()?;
    Some(eig.eigenvectors * d_mat * v_inv)
}

/// Matrix square root: if A = V D V^{-1}, then sqrt(A) = V sqrt(D) V^{-1}.
pub fn matrix_sqrt(a: &Mat) -> Option<Mat> {
    let n = a.nrows();
    if n != a.ncols() || n == 0 {
        return None;
    }

    let eig = eigen(a)?;

    for &v in eig.eigenvalues.iter() {
        if v < -1e-10 {
            return None;
        }
    }

    let d = eig.eigenvalues.map(|v| v.max(0.0).sqrt());
    let mut d_mat = Mat::zeros(n, n);
    for (i, val) in d.iter().enumerate() {
        d_mat[(i, i)] = *val;
    }
    let v_inv = eig.eigenvectors.clone().try_inverse()?;
    Some(eig.eigenvectors * d_mat * v_inv)
}

/// Matrix power: A^t for real t.
pub fn matrix_pow(a: &Mat, t: f64) -> Option<Mat> {
    let n = a.nrows();
    if n != a.ncols() || n == 0 {
        return None;
    }

    let eig = eigen(a)?;

    for &v in eig.eigenvalues.iter() {
        if v <= 0.0 && t != t.trunc() {
            return None;
        }
    }

    let d = eig.eigenvalues.map(|v| v.powf(t));
    let mut d_mat = Mat::zeros(n, n);
    for (i, val) in d.iter().enumerate() {
        d_mat[(i, i)] = *val;
    }
    let v_inv = eig.eigenvectors.clone().try_inverse()?;
    Some(eig.eigenvectors * d_mat * v_inv)
}
