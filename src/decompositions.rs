//! Matrix decompositions: LU, QR, Cholesky, SVD, eigendecomposition.

use crate::{Mat, VecN};

/// LU decomposition result with PA = LU.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LUResult {
    /// Permutation matrix P such that PA = LU.
    pub p: Mat,
    /// Lower triangular matrix L with unit diagonal.
    pub l: Mat,
    /// Upper triangular matrix U.
    pub u: Mat,
}

/// LU decomposition with partial pivoting. Returns P, L, U such that PA = LU.
pub fn lu(a: &Mat) -> Option<LUResult> {
    let n = a.nrows();
    if n != a.ncols() || n == 0 {
        return None;
    }

    let mut u = a.clone();
    let mut p = Mat::identity(n, n);
    let mut l = Mat::zeros(n, n);

    for k in 0..n {
        // Find pivot
        let mut max_val = u[(k, k)].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = u[(i, k)].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }

        if max_val < 1e-15 {
            l[(k, k)] = 1.0;
            continue;
        }

        // Swap rows in u and p
        if max_row != k {
            u.swap_rows(k, max_row);
            p.swap_rows(k, max_row);
            if k > 0 {
                for j in 0..k {
                    let tmp = l[(k, j)];
                    l[(k, j)] = l[(max_row, j)];
                    l[(max_row, j)] = tmp;
                }
            }
        }

        l[(k, k)] = 1.0;

        for i in (k + 1)..n {
            let factor = u[(i, k)] / u[(k, k)];
            l[(i, k)] = factor;
            for j in k..n {
                u[(i, j)] -= factor * u[(k, j)];
            }
        }
    }

    Some(LUResult { p, l, u })
}

/// QR decomposition result: A = QR.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QRResult {
    /// Orthogonal matrix Q.
    pub q: Mat,
    /// Upper triangular matrix R.
    pub r: Mat,
}

/// QR decomposition via modified Gram-Schmidt.
pub fn qr(a: &Mat) -> Option<QRResult> {
    let m = a.nrows();
    let n = a.ncols();
    if m < n {
        return None;
    }

    let mut q = Mat::zeros(m, n);
    let mut r = Mat::zeros(n, n);

    let mut v: std::vec::Vec<VecN> = std::vec::Vec::with_capacity(n);
    for j in 0..n {
        v.push(a.column(j).clone_owned());
    }

    for j in 0..n {
        r[(j, j)] = v[j].norm();
        if r[(j, j)] < 1e-15 {
            q.set_column(j, &nalgebra::DVector::zeros(m));
        } else {
            let qj = &v[j] / r[(j, j)];
            q.set_column(j, &qj);
        }

        for k_idx in (j + 1)..n {
            let dot = q.column(j).dot(&v[k_idx]);
            r[(j, k_idx)] = dot;
            v[k_idx] -= dot * q.column(j);
        }
    }

    Some(QRResult { q, r })
}

/// Cholesky decomposition result: A = L * L^T.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CholeskyResult {
    /// Lower triangular matrix L such that A = L * L^T.
    pub l: Mat,
}

/// Cholesky decomposition. Returns None if matrix is not positive definite.
pub fn cholesky(a: &Mat) -> Option<CholeskyResult> {
    let n = a.nrows();
    if n != a.ncols() || n == 0 {
        return None;
    }

    let mut l = Mat::zeros(n, n);

    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[(j, k)] * l[(j, k)];
        }
        let diag = a[(j, j)] - sum;
        if diag <= 0.0 {
            return None;
        }
        l[(j, j)] = diag.sqrt();

        for i in (j + 1)..n {
            let mut sum2 = 0.0;
            for k in 0..j {
                sum2 += l[(i, k)] * l[(j, k)];
            }
            l[(i, j)] = (a[(i, j)] - sum2) / l[(j, j)];
        }
    }

    Some(CholeskyResult { l })
}

/// SVD result: A = U * Σ * V^T.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SVDResult {
    /// Left singular vectors.
    pub u: Mat,
    /// Singular values (diagonal).
    pub sigma: VecN,
    /// Right singular vectors.
    pub v: Mat,
}

/// Compute SVD via eigendecomposition of A^T A with deflation.
pub fn svd(a: &Mat) -> Option<SVDResult> {
    let m = a.nrows();
    let n = a.ncols();
    if m == 0 || n == 0 {
        return None;
    }

    let ata = a.transpose() * a;

    let k = n.min(m);
    let mut singular_values = std::vec::Vec::with_capacity(k);
    let mut v_columns = std::vec::Vec::with_capacity(k);
    let mut residual = ata.clone();

    for _ in 0..k {
        let mut v = VecN::from_fn(n, |i, _| {
            1.0 / (n as f64).sqrt() + (i as f64) * 0.001
        });

        for _ in 0..200 {
            let w = &residual * &v;
            let norm = w.norm();
            if norm < 1e-30 {
                break;
            }
            v = w / norm;
        }

        let sigma_sq = v.dot(&(&residual * &v));
        if sigma_sq < 1e-30 {
            break;
        }
        let sigma = sigma_sq.sqrt();
        singular_values.push(sigma);
        v_columns.push(v.clone());

        let outer = &v * v.transpose().scale(sigma_sq);
        residual -= Mat::from(outer);
    }

    let r = singular_values.len();
    let sigma_vec = VecN::from_vec(singular_values);
    let mut v_mat = Mat::zeros(n, r);
    for (j, col) in v_columns.iter().enumerate() {
        v_mat.set_column(j, col);
    }

    let mut u_mat = Mat::zeros(m, r);
    for j in 0..r {
        let av = a * v_mat.column(j);
        u_mat.set_column(j, &(av / sigma_vec[j]));
    }

    Some(SVDResult {
        u: u_mat,
        sigma: sigma_vec,
        v: v_mat,
    })
}

/// Eigendecomposition result: A = V * D * V^{-1}.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EigenResult {
    /// Eigenvalues.
    pub eigenvalues: VecN,
    /// Eigenvectors as columns.
    pub eigenvectors: Mat,
}

/// Eigendecomposition via QR algorithm (simple shiftless for symmetric matrices).
pub fn eigen(a: &Mat) -> Option<EigenResult> {
    let n = a.nrows();
    if n != a.ncols() || n == 0 {
        return None;
    }

    let symmetric = *a == a.transpose();
    if symmetric {
        let eig = a.clone().try_symmetric_eigen(1e-10, 200)?;
        Some(EigenResult {
            eigenvalues: eig.eigenvalues,
            eigenvectors: eig.eigenvectors,
        })
    } else {
        let mut t = a.clone();
        let mut q_accum = Mat::identity(n, n);

        for _ in 0..500 {
            let qr_res = t.clone().qr();
            let q = qr_res.q();
            let r = qr_res.r();
            t = &r * &q;
            q_accum = &q_accum * &q;
        }

        let eigenvalues = t.diagonal();
        Some(EigenResult {
            eigenvalues: eigenvalues.clone_owned(),
            eigenvectors: q_accum,
        })
    }
}

/// Solve linear system Ax = b via LU decomposition.
pub fn solve(a: &Mat, b: &VecN) -> Option<VecN> {
    let lu_res = lu(a)?;
    let pb = &lu_res.p * b;

    let n = a.nrows();
    // Forward substitution: Ly = Pb
    let mut y = VecN::zeros(n);
    for i in 0..n {
        let mut sum = pb[i];
        for j in 0..i {
            sum -= lu_res.l[(i, j)] * y[j];
        }
        y[i] = sum / lu_res.l[(i, i)];
    }

    // Back substitution: Ux = y
    let mut x = VecN::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= lu_res.u[(i, j)] * x[j];
        }
        if lu_res.u[(i, i)].abs() < 1e-15 {
            return None;
        }
        x[i] = sum / lu_res.u[(i, i)];
    }

    Some(x)
}
