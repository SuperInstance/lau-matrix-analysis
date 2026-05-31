//! Kronecker products and properties.

use crate::Mat;

/// Kronecker product of two matrices: A ⊗ B.
/// If A is m×n and B is p×q, then A ⊗ B is mp×nq.
pub fn kron(a: &Mat, b: &Mat) -> Mat {
    let m = a.nrows();
    let n = a.ncols();
    let p = b.nrows();
    let q = b.ncols();

    let mut result = Mat::zeros(m * p, n * q);

    for i in 0..m {
        for j in 0..n {
            let aij = a[(i, j)];
            for k in 0..p {
                for l in 0..q {
                    result[(i * p + k, j * q + l)] = aij * b[(k, l)];
                }
            }
        }
    }

    result
}

/// Kronecker product properties:
/// - (A ⊗ B)(C ⊗ D) = AC ⊗ BD (when dimensions compatible)
/// - (A ⊗ B)^T = A^T ⊗ B^T
/// - (A ⊗ B)^{-1} = A^{-1} ⊗ B^{-1} (when A, B invertible)
/// - det(A ⊗ B) = det(A)^p * det(B)^m for A m×m, B p×p

/// Verify (A ⊗ B)^T = A^T ⊗ B^T.
pub fn verify_kron_transpose(a: &Mat, b: &Mat) -> bool {
    let kron_ab = kron(a, b);
    let kron_ab_t = kron_ab.transpose();
    let kron_at_bt = kron(&a.transpose(), &b.transpose());
    let diff = &kron_ab_t - &kron_at_bt;
    diff.iter().all(|v| v.abs() < 1e-10)
}

/// Compute (A ⊗ B)(C ⊗ D) and compare with AC ⊗ BD.
pub fn verify_kron_mixed_product(a: &Mat, b: &Mat, c: &Mat, d: &Mat) -> bool {
    let ac = a * c;
    let bd = b * d;
    let kron_ac_bd = kron(&ac, &bd);

    let kron_ab = kron(a, b);
    let kron_cd = kron(c, d);
    let product = &kron_ab * &kron_cd;

    let diff = product - kron_ac_bd;
    diff.iter().all(|v| v.abs() < 1e-8)
}

/// Kronecker sum: A ⊕ B = A ⊗ I_p + I_m ⊗ B (for A m×m, B p×p).
pub fn kron_sum(a: &Mat, b: &Mat) -> Option<Mat> {
    let m = a.nrows();
    let p = b.nrows();
    if m != a.ncols() || p != b.ncols() {
        return None;
    }

    let im = Mat::identity(m, m);
    let ip = Mat::identity(p, p);

    Some(kron(a, &ip) + kron(&im, b))
}
