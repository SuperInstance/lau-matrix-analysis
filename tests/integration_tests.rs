use lau_matrix_analysis::*;
use nalgebra::{DMatrix, DVector};
use num_traits::sign::Signed;

type Mat = DMatrix<f64>;

fn pd_matrix() -> Mat {
    DMatrix::from_row_slice(3, 3, &[
        2.0, 1.0, 0.0,
        1.0, 3.0, 1.0,
        0.0, 1.0, 2.0,
    ])
}

fn identity(n: usize) -> Mat {
    Mat::identity(n, n)
}

// ===== DECOMPOSITION TESTS =====

#[test]
fn test_lu_identity() {
    let i = identity(3);
    let res = lu(&i).unwrap();
    let pa = &res.p * &i;
    let lu_mat = &res.l * &res.u;
    let diff = &pa - &lu_mat;
    assert!(diff.iter().all(|v| v.abs() < 1e-10));
}

#[test]
fn test_lu_general() {
    let a = DMatrix::from_row_slice(3, 3, &[
        2.0, 3.0, 1.0,
        4.0, 7.0, 5.0,
        6.0, 18.0, 22.0,
    ]);
    let res = lu(&a).unwrap();
    let pa = &res.p * &a;
    let lu_mat = &res.l * &res.u;
    let diff = &pa - &lu_mat;
    assert!(diff.iter().all(|v| v.abs() < 1e-8));
}

#[test]
fn test_lu_l_is_unit_lower() {
    let a = DMatrix::from_row_slice(3, 3, &[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0,
    ]);
    let res = lu(&a).unwrap();
    for i in 0..3 {
        assert!((res.l[(i, i)] - 1.0).abs() < 1e-10);
    }
    for i in 0..3 {
        for j in (i + 1)..3 {
            assert!(res.l[(i, j)].abs() < 1e-10);
        }
    }
}

#[test]
fn test_qr_identity() {
    let i = identity(3);
    let res = qr(&i).unwrap();
    let qr_mat = &res.q * &res.r;
    let diff = &i - &qr_mat;
    assert!(diff.iter().all(|v| v.abs() < 1e-10));
}

#[test]
fn test_qr_general() {
    let a = DMatrix::from_row_slice(3, 3, &[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0,
    ]);
    let res = qr(&a).unwrap();
    let qr_mat = &res.q * &res.r;
    let diff = &a - &qr_mat;
    assert!(diff.iter().all(|v| v.abs() < 1e-6));
}

#[test]
fn test_qr_orthogonality() {
    let a = DMatrix::from_row_slice(3, 3, &[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0,
    ]);
    let res = qr(&a).unwrap();
    let qtq = res.q.transpose() * &res.q;
    let diff = &qtq - &identity(3);
    assert!(diff.iter().all(|v| v.abs() < 1e-6));
}

#[test]
fn test_cholesky_pd() {
    let a = pd_matrix();
    let res = cholesky(&a).unwrap();
    let llt = &res.l * res.l.transpose();
    let diff = &a - &llt;
    assert!(diff.iter().all(|v| v.abs() < 1e-10));
}

#[test]
fn test_cholesky_non_pd() {
    let a = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -1.0]);
    assert!(cholesky(&a).is_none());
}

#[test]
fn test_svd_identity() {
    let i = identity(3);
    let res = svd(&i).unwrap();
    assert_eq!(res.sigma.len(), 3);
    for s in res.sigma.iter() {
        assert!((*s - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_svd_reconstruction() {
    let a = DMatrix::from_row_slice(3, 2, &[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]);
    let res = svd(&a).unwrap();
    let r = res.sigma.len();
    // U: 3×r, Sigma: r×r (diagonal), V: 2×r
    // Reconstruction: U * Sigma * V^T = (3×r) * (r×r) * (r×2) = 3×2
    let mut sigma_mat = Mat::zeros(r, r);
    for (i, s) in res.sigma.iter().enumerate() {
        sigma_mat[(i, i)] = *s;
    }
    let recon = &res.u * &sigma_mat * res.v.transpose();
    let diff = &a - &recon;
    assert!(diff.iter().all(|v| v.abs() < 1e-4));
}

#[test]
fn test_eigen_symmetric() {
    let a = pd_matrix();
    let res = eigen(&a).unwrap();
    for v in res.eigenvalues.iter() {
        assert!(*v > -1e-6);
    }
    for i in 0..3 {
        let lambda = res.eigenvalues[i];
        let vi = res.eigenvectors.column(i).clone_owned();
        let av = &a * &vi;
        let lambda_v = vi.scale(lambda);
        let diff = &av - &lambda_v;
        assert!(diff.iter().all(|x| x.abs() < 1e-6));
    }
}

#[test]
fn test_solve_simple() {
    let a = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 5.0, 3.0]);
    let b = DVector::from_vec(vec![4.0, 7.0]);
    let x = solve(&a, &b).unwrap();
    let ax = &a * &x;
    let diff = &ax - &b;
    assert!(diff.iter().all(|v| v.abs() < 1e-10));
}

// ===== NORM TESTS =====

#[test]
fn test_frobenius_norm_identity() {
    let i = identity(3);
    assert!((frobenius_norm(&i) - 3.0_f64.sqrt()).abs() < 1e-10);
}

#[test]
fn test_frobenius_norm_zero() {
    let z = Mat::zeros(3, 4);
    assert!(frobenius_norm(&z).abs() < 1e-10);
}

#[test]
fn test_operator_norm_identity() {
    let i = identity(3);
    assert!((operator_norm(&i) - 1.0).abs() < 1e-10);
}

#[test]
fn test_norm_1() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0, -2.0, 3.0, 4.0]);
    assert!((norm_1(&a) - 6.0).abs() < 1e-10);
}

#[test]
fn test_norm_inf() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0, -2.0, 3.0, 4.0]);
    assert!((norm_inf(&a) - 7.0).abs() < 1e-10);
}

#[test]
fn test_condition_number_identity() {
    let i = identity(3);
    let cond = condition_number(&i);
    assert!((cond - 1.0).abs() < 1e-6);
}

#[test]
fn test_condition_number_singular() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
    let cond = condition_number(&a);
    assert!(cond > 1e5 || cond.is_infinite());
}

#[test]
fn test_trace() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    assert!((trace(&a) - 5.0).abs() < 1e-10);
}

#[test]
fn test_rank() {
    let a = DMatrix::from_row_slice(3, 3, &[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ]);
    assert_eq!(rank(&a, 1e-6), 2);
}

// ===== PERTURBATION TESTS =====

#[test]
fn test_weyl_bound() {
    let a = pd_matrix();
    let e = DMatrix::from_row_slice(3, 3, &[
        0.01, 0.0, 0.0,
        0.0, -0.01, 0.0,
        0.0, 0.0, 0.01,
    ]);
    let bound = weyl_bound(&e);
    let eig_a = eigen(&a).unwrap();
    let eig_ap = eigen(&(&a + &e)).unwrap();
    let mut max_shift = 0.0f64;
    for i in 0..3 {
        let shift = (eig_ap.eigenvalues[i] - eig_a.eigenvalues[i]).abs();
        if shift > max_shift { max_shift = shift; }
    }
    assert!(max_shift <= bound + 1e-6);
}

#[test]
fn test_eigenvalue_condition_symmetric() {
    let a = pd_matrix();
    let cond = eigenvalue_condition_numbers(&a);
    for c in &cond {
        assert!((c - 1.0).abs() < 0.5);
    }
}

// ===== POSITIVE DEFINITE TESTS =====

#[test]
fn test_is_pd() {
    assert!(is_positive_definite(&pd_matrix()));
}

#[test]
fn test_is_not_pd() {
    let a = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -1.0]);
    assert!(!is_positive_definite(&a));
}

#[test]
fn test_is_psd() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
    assert!(is_positive_semidefinite(&a, 1e-6));
}

#[test]
fn test_nearest_psd() {
    let a = DMatrix::from_row_slice(2, 2, &[2.0, 3.0, 3.0, 2.0]);
    let nearest = nearest_psd(&a).unwrap();
    assert!(is_positive_semidefinite(&nearest, 1e-6));
}

#[test]
fn test_pd_verify() {
    let a = pd_matrix();
    let result = pd_verify(&a).unwrap();
    assert!(result.is_symmetric);
    assert!(result.is_pd);
    assert!(result.min_eigenvalue > 0.0);
}

// ===== SPARSE TESTS =====

#[test]
fn test_sparse_identity() {
    let mut s = SparseCOO::new(3, 3);
    for i in 0..3 { s.add_entry(i, i, 1.0); }
    assert_eq!(s.nnz(), 3);
    let dense = s.to_dense();
    let diff = &dense - &identity(3);
    assert!(diff.iter().all(|v| v.abs() < 1e-10));
}

#[test]
fn test_sparse_dense_multiply() {
    let mut s = SparseCOO::new(2, 3);
    s.add_entry(0, 0, 1.0);
    s.add_entry(0, 2, 2.0);
    s.add_entry(1, 1, 3.0);
    let b = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = s.mul_dense(&b);
    let dense = s.to_dense();
    let expected = &dense * &b;
    let diff = &result - &expected;
    assert!(diff.iter().all(|v| v.abs() < 1e-10));
}

#[test]
fn test_sparse_vec_multiply() {
    let mut s = SparseCOO::new(3, 3);
    s.add_entry(0, 0, 2.0);
    s.add_entry(1, 2, 1.0);
    s.add_entry(2, 1, 3.0);
    let v = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let result = s.mul_vec(&v);
    assert!((result[0] - 2.0).abs() < 1e-10);
    assert!((result[1] - 3.0).abs() < 1e-10);
    assert!((result[2] - 6.0).abs() < 1e-10);
}

// ===== KRONECKER TESTS =====

#[test]
fn test_kron_basic() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let b = identity(2);
    let k = kron(&a, &b);
    assert_eq!(k.nrows(), 4);
    assert_eq!(k.ncols(), 4);
    let expected = DMatrix::from_row_slice(4, 4, &[
        1.0, 0.0, 2.0, 0.0,
        0.0, 1.0, 0.0, 2.0,
        3.0, 0.0, 4.0, 0.0,
        0.0, 3.0, 0.0, 4.0,
    ]);
    let diff = &k - &expected;
    assert!(diff.iter().all(|v| v.abs() < 1e-10));
}

#[test]
fn test_kron_transpose_property() {
    let a = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = DMatrix::from_row_slice(2, 2, &[1.0, -1.0, 2.0, 0.0]);
    assert!(verify_kron_transpose(&a, &b));
}

#[test]
fn test_kron_mixed_product() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let b = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
    let c = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
    let d = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
    assert!(verify_kron_mixed_product(&a, &b, &c, &d));
}

#[test]
fn test_kron_sum() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
    let b = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 0.0, 4.0]);
    let ks = kron_sum(&a, &b).unwrap();
    assert_eq!(ks.nrows(), 4);
    assert_eq!(ks.ncols(), 4);
}

// ===== MATRIX FUNCTION TESTS =====

#[test]
fn test_matrix_exp_zero() {
    let z = Mat::zeros(3, 3);
    let exp = matrix_exp(&z).unwrap();
    let diff = &exp - &identity(3);
    assert!(diff.iter().all(|v| v.abs() < 1e-10));
}

#[test]
fn test_matrix_exp_identity() {
    let exp = matrix_exp(&identity(2)).unwrap();
    let expected = DMatrix::from_row_slice(2, 2, &[
        std::f64::consts::E, 0.0,
        0.0, std::f64::consts::E,
    ]);
    let diff = &exp - &expected;
    assert!(diff.iter().all(|v| v.abs() < 1e-8));
}

#[test]
fn test_matrix_exp_log_roundtrip() {
    // Use diagonal matrix to avoid eigenvector ordering issues
    let a = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
    let exp_a = matrix_exp(&a).unwrap();
    let log_exp = matrix_log(&exp_a).unwrap();
    let diff = &log_exp - &a;
    assert!(diff.iter().all(|v| v.abs() < 1e-6));
}

#[test]
fn test_matrix_sqrt_roundtrip() {
    let a = pd_matrix();
    let sqrt_a = matrix_sqrt(&a).unwrap();
    let sq = &sqrt_a * &sqrt_a;
    let diff = &sq - &a;
    assert!(diff.iter().all(|v| v.abs() < 1e-6));
}

#[test]
fn test_matrix_sqrt_identity() {
    let sqrt_i = matrix_sqrt(&identity(3)).unwrap();
    let diff = &sqrt_i - &identity(3);
    assert!(diff.iter().all(|v| v.abs() < 1e-10));
}

#[test]
fn test_matrix_pow() {
    let a = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
    let a_sq = matrix_pow(&a, 2.0).unwrap();
    let expected = &a * &a;
    let diff = &a_sq - &expected;
    assert!(diff.iter().all(|v| v.abs() < 1e-6));
}

#[test]
fn test_matrix_exp_diagonal() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -1.0]);
    let exp = matrix_exp(&a).unwrap();
    assert!((exp[(0, 0)] - std::f64::consts::E).abs() < 1e-8);
    assert!((exp[(1, 1)] - (1.0f64 / std::f64::consts::E)).abs() < 1e-8);
}

// ===== STRUCTURED MATRIX TESTS =====

#[test]
fn test_toeplitz_basic() {
    let col = vec![1.0, 2.0, 3.0];
    let row = vec![1.0, 4.0, 5.0];
    let t = toeplitz(&col, &row);
    assert!((t[(0, 0)] - 1.0).abs() < 1e-10);
    assert!((t[(0, 1)] - 4.0).abs() < 1e-10);
    assert!((t[(1, 0)] - 2.0).abs() < 1e-10);
    assert!((t[(2, 1)] - 2.0).abs() < 1e-10);
}

#[test]
fn test_toeplitz_symmetric() {
    let col = vec![1.0, 2.0, 3.0];
    let t = toeplitz_symmetric(&col);
    let diff = &t - &t.transpose();
    assert!(diff.iter().all(|v| v.abs() < 1e-10));
    assert!((t[(0, 1)] - 2.0).abs() < 1e-10);
    assert!((t[(1, 0)] - 2.0).abs() < 1e-10);
}

#[test]
fn test_circulant_basic() {
    let row = vec![1.0, 2.0, 3.0];
    let c = circulant(&row);
    assert!((c[(0, 0)] - 1.0).abs() < 1e-10);
    assert!((c[(0, 1)] - 2.0).abs() < 1e-10);
    assert!((c[(0, 2)] - 3.0).abs() < 1e-10);
    assert!((c[(1, 0)] - 3.0).abs() < 1e-10);
    assert!((c[(1, 1)] - 1.0).abs() < 1e-10);
    assert!((c[(1, 2)] - 2.0).abs() < 1e-10);
}

#[test]
fn test_circulant_mul() {
    let row = vec![1.0, 2.0, 3.0];
    let v = DVector::from_vec(vec![1.0, 0.0, 0.0]);
    let result = circulant_mul(&row, &v);
    let c = circulant(&row);
    let expected = &c * &v;
    for i in 0..3 {
        assert!((result[i] - expected[i]).abs() < 1e-10);
    }
}

#[test]
fn test_vandermonde_basic() {
    let x = vec![1.0, 2.0, 3.0];
    let v = vandermonde(&x, 3);
    assert!((v[(0, 0)] - 1.0).abs() < 1e-10);
    assert!((v[(1, 1)] - 2.0).abs() < 1e-10);
    assert!((v[(2, 2)] - 9.0).abs() < 1e-10);
}

#[test]
fn test_vandermonde_det() {
    let x = vec![1.0, 2.0, 3.0];
    let det = vandermonde_det(&x).unwrap();
    assert!((det - 2.0).abs() < 1e-10);
}

#[test]
fn test_vandermonde_det_repeated() {
    let x = vec![1.0, 2.0, 2.0];
    assert!(vandermonde_det(&x).is_none());
}

#[test]
fn test_hankel_basic() {
    let col = vec![1.0, 2.0, 3.0];
    let row = vec![3.0, 4.0, 5.0];
    let h = hankel(&col, &row);
    assert!((h[(0, 0)] - 1.0).abs() < 1e-10);
    assert!((h[(0, 1)] - 2.0).abs() < 1e-10);
    assert!((h[(1, 0)] - 2.0).abs() < 1e-10);
    assert!((h[(1, 1)] - 3.0).abs() < 1e-10);
}

// ===== AGENT ANALYSIS TESTS =====

#[test]
fn test_build_similarity_matrix() {
    let features = DMatrix::from_row_slice(3, 2, &[
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
    ]);
    let sim = build_similarity_matrix(&features, 1.0);
    assert_eq!(sim.nrows(), 3);
    for i in 0..3 {
        assert!((sim[(i, i)] - 1.0).abs() < 1e-10);
    }
    let diff = &sim - &sim.transpose();
    assert!(diff.iter().all(|v| v.abs() < 1e-10));
}

#[test]
fn test_agent_similarity_analysis() {
    let sim = DMatrix::from_row_slice(4, 4, &[
        1.0, 0.8, 0.1, 0.1,
        0.8, 1.0, 0.1, 0.1,
        0.1, 0.1, 1.0, 0.9,
        0.1, 0.1, 0.9, 1.0,
    ]);
    let result = analyze_agent_similarity(&sim, 2, 1e-6).unwrap();
    assert_eq!(result.n_agents, 4);
    assert!(result.is_psd);
    assert_eq!(result.clusters[0], result.clusters[1]);
    assert_eq!(result.clusters[2], result.clusters[3]);
    assert_ne!(result.clusters[0], result.clusters[2]);
}

#[test]
fn test_agent_diversity_identical() {
    let sim = DMatrix::from_row_slice(3, 3, &[
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    ]);
    let div = agent_diversity(&sim);
    assert!(div < 0.1);
}

#[test]
fn test_agent_diversity_orthogonal() {
    let sim = identity(3);
    let div = agent_diversity(&sim);
    assert!((div - 1.0).abs() < 1e-10);
}

// ===== INTEGRATION TESTS =====

#[test]
fn test_svd_norm_relation() {
    let a = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let op_norm = operator_norm(&a);
    let svd_res = svd(&a).unwrap();
    let max_sv = svd_res.sigma.iter().cloned().fold(0.0f64, f64::max);
    assert!((op_norm - max_sv).abs() < 1e-4);
}

#[test]
fn test_lu_solve_consistency() {
    let a = DMatrix::from_row_slice(3, 3, &[
        2.0, 1.0, 1.0,
        4.0, 3.0, 3.0,
        8.0, 7.0, 9.0,
    ]);
    let b = DVector::from_vec(vec![1.0, 1.0, 1.0]);
    let x = solve(&a, &b).unwrap();
    let ax = &a * &x;
    for i in 0..3 {
        assert!((ax[i] - b[i]).abs() < 1e-8);
    }
}

#[test]
fn test_nuclear_norm() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
    let nn = nuclear_norm(&a);
    assert!((nn - 3.0).abs() < 1e-4);
}

#[test]
fn test_sparse_transpose() {
    let mut s = SparseCOO::new(2, 3);
    s.add_entry(0, 1, 5.0);
    s.add_entry(1, 2, 3.0);
    let st = s.transpose();
    assert_eq!(st.nrows, 3);
    assert_eq!(st.ncols, 2);
    assert_eq!(st.nnz(), 2);
    assert!((st.to_dense()[(1, 0)] - 5.0).abs() < 1e-10);
}

#[test]
fn test_matrix_exp_pade_identity() {
    let exp = matrix_exp_pade(&identity(2)).unwrap();
    let expected = DMatrix::from_row_slice(2, 2, &[
        std::f64::consts::E, 0.0,
        0.0, std::f64::consts::E,
    ]);
    let diff = &exp - &expected;
    assert!(diff.iter().all(|v| v.abs() < 1.0));
}
