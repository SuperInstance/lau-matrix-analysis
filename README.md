# lau-matrix-analysis

A Rust library for matrix analysis: decompositions (LU, QR, Cholesky, SVD, eigendecomposition), matrix norms, perturbation theory, structured matrices, Kronecker products, matrix functions (exp, log, sqrt, pow), sparse matrices, and agent similarity analysis — built on `nalgebra`.

## What This Does

`lau-matrix-analysis` provides the core algorithms of numerical linear algebra as pure functions operating on `nalgebra::DMatrix<f64>`:

- **Decompositions** — LU with partial pivoting, QR via modified Gram-Schmidt, Cholesky, SVD via deflation, eigendecomposition (symmetric: Jacobi; general: QR iteration), linear system solving.
- **Norms** — Frobenius, operator (spectral), nuclear, 1-norm, ∞-norm, condition number, trace, rank.
- **Perturbation theory** — Eigenvalue condition numbers, Bauer-Fike bound, Weyl's inequality, element-wise eigenvalue sensitivity.
- **Positive definite matrices** — PD/PSD verification, nearest PSD matrix (Higham's algorithm), eigenvalue range reporting.
- **Sparse matrices** — COO format with sparse-dense and sparse-vector multiplication.
- **Kronecker products** — ⊗ product, mixed-product verification, Kronecker sum.
- **Matrix functions** — Exponential (eigendecomposition & Padé with scaling-and-squaring), logarithm, square root, arbitrary real powers — all via spectral decomposition.
- **Structured matrices** — Toeplitz, symmetric Toeplitz, circulant (with structured multiply), Vandermonde (with determinant), Hankel.
- **Agent analysis** — Spectral clustering on similarity matrices, Gaussian (RBF) kernel, diversity metrics.

## Key Idea

Everything is expressed as **pure functions taking `&DMatrix<f64>` and returning typed result structs** (e.g., `LUResult { p, l, u }`). There's no mutable state, no traits to implement, no generics to wrangle. You pass in a matrix, you get back a decomposition. This makes the library straightforward to use for prototyping, teaching, and embedding in agent systems that need linear algebra.

## Install

```toml
[dependencies]
lau-matrix-analysis = { git = "https://github.com/SuperInstance/lau-matrix-analysis" }
```

Requires `nalgebra` (with `serde-serialize` feature), `serde`, and `num-traits`.

## Quick Start

### Decompositions

```rust
use lau_matrix_analysis::*;
use nalgebra::{DMatrix, DVector};

let a = DMatrix::from_row_slice(3, 3, &[
    2.0, 1.0, 0.0,
    1.0, 3.0, 1.0,
    0.0, 1.0, 2.0,
]);

// LU: PA = LU
let lu_res = lu(&a).unwrap();
let pa = &lu_res.p * &a;
let lu_mat = &lu_res.l * &lu_res.u;
assert!((pa - lu_mat).norm() < 1e-8);

// QR: A = QR
let qr_res = qr(&a).unwrap();
let qr_mat = &qr_res.q * &qr_res.r;

// Cholesky (PD matrices only)
let chol = cholesky(&a).unwrap(); // A = L·Lᵀ

// Eigendecomposition
let eig = eigen(&a).unwrap();
println!("eigenvalues: {:?}", eig.eigenvalues);

// Solve Ax = b
let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);
let x = solve(&a, &b).unwrap();
```

### Norms & Condition Numbers

```rust
let op = operator_norm(&a);      // spectral norm
let fro = frobenius_norm(&a);    // Frobenius norm
let nuc = nuclear_norm(&a);      // sum of singular values
let cond = condition_number(&a); // σ_max / σ_min
let tr = trace(&a);
let r = rank(&a, 1e-6);          // rank via SVD threshold
```

### Matrix Functions

```rust
let exp_a = matrix_exp(&a).unwrap();         // e^A via eigendecomposition
let exp_pade = matrix_exp_pade(&a).unwrap(); // e^A via Padé + scaling-and-squaring
let log_a = matrix_log(&exp_a).unwrap();     // log(A) — inverse of exp
let sqrt_a = matrix_sqrt(&a).unwrap();       // √A — satisfies √A · √A = A
let pow_a = matrix_pow(&a, 2.5).unwrap();    // A^2.5
```

### Structured Matrices

```rust
// Toeplitz: constant diagonals
let t = toeplitz(&[1.0, 2.0, 3.0], &[1.0, 4.0, 5.0]);

// Circulant
let c = circulant(&[1.0, 2.0, 3.0]);
let v = DVector::from_vec(vec![1.0, 0.0, 0.0]);
let result = circulant_mul(&[1.0, 2.0, 3.0], &v);

// Vandermonde
let v = vandermonde(&[1.0, 2.0, 3.0], 3);
let det = vandermonde_det(&[1.0, 2.0, 3.0]); // = ∏(x_i - x_j)
```

### Agent Similarity Analysis

```rust
// Build similarity matrix from features via RBF kernel
let features = DMatrix::from_row_slice(4, 2, &[
    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
]);
let sim = build_similarity_matrix(&features, 1.0);

// Spectral clustering
let result = analyze_agent_similarity(&sim, 2, 1e-6).unwrap();
println!("clusters: {:?}", result.clusters);
println!("effective rank: {}", result.effective_rank);
println!("diversity: {}", agent_diversity(&sim));
```

## API Reference

### `decompositions`

| Function | Returns | Description |
|---|---|---|
| `lu(&A)` | `Option<LUResult>` | PA = LU with partial pivoting |
| `qr(&A)` | `Option<QRResult>` | A = QR via modified Gram-Schmidt |
| `cholesky(&A)` | `Option<CholeskyResult>` | A = LLᵀ (PD only) |
| `svd(&A)` | `Option<SVDResult>` | A = UΣVᵀ via deflation |
| `eigen(&A)` | `Option<EigenResult>` | A = VDV⁻¹ (symmetric: Jacobi; general: QR) |
| `solve(&A, &b)` | `Option<VecN>` | LU-based linear system solver |

### `norms`

| Function | Description |
|---|---|
| `frobenius_norm(&A)` | ‖A‖_F = √Σ|aᵢⱼ|² |
| `operator_norm(&A)` | ‖A‖₂ = σ_max (power iteration) |
| `nuclear_norm(&A)` | ‖A‖_* = Σσᵢ |
| `norm_1(&A)` | max column sum of |aᵢⱼ| |
| `norm_inf(&A)` | max row sum of |aᵢⱼ| |
| `condition_number(&A)` | σ_max / σ_min |
| `trace(&A)` | Σaᵢᵢ |
| `rank(&A, tol)` | # singular values > tol |

### `perturbation`

| Function | Description |
|---|---|
| `eigenvalue_condition_numbers(&A)` | κᵢ = 1/|yᵢᴴxᵢ| per eigenvalue |
| `bauer_fike_bound(&A, &E)` | κ(V)·‖E‖ bound on eigenvalue perturbation |
| `weyl_bound(&E)` | ‖E‖₂ bound for symmetric case |
| `eigenvalue_sensitivity(&A, idx, ε)` | dλ/dAᵢⱼ via finite differences |

### `positive_definite`

| Function | Description |
|---|---|
| `is_positive_definite(&A)` | Via Cholesky |
| `is_positive_semidefinite(&A, tol)` | Via eigenvalues ≥ -tol |
| `nearest_psd(&A)` | Higham's algorithm |
| `pd_verify(&A)` | Full PD verification report |

### `sparse`

| Type/Method | Description |
|---|---|
| `SparseCOO::new(rows, cols)` | COOrdinate sparse matrix |
| `add_entry(i, j, v)` | Insert a nonzero |
| `mul_dense(&B)` | Sparse × dense matrix |
| `mul_vec(&v)` | Sparse × vector |
| `to_dense()` | Convert to DMatrix |
| `transpose()` | Transpose (swap row/col indices) |

### `kronecker`

| Function | Description |
|---|---|
| `kron(&A, &B)` | Kronecker product A⊗B |
| `kron_sum(&A, &B)` | Kronecker sum A⊗I + I⊗B |
| `verify_kron_transpose(&A, &B)` | Check (A⊗B)ᵀ = Aᵀ⊗Bᵀ |
| `verify_kron_mixed_product(A,B,C,D)` | Check (A⊗B)(C⊗D) = AC⊗BD |

### `matrix_functions`

| Function | Description |
|---|---|
| `matrix_exp(&A)` | eᴬ via eigendecomposition |
| `matrix_exp_pade(&A)` | eᴬ via [1/1] Padé + scaling-and-squaring |
| `matrix_log(&A)` | log(A) via eigendecomposition (requires positive eigenvalues) |
| `matrix_sqrt(&A)` | √A via eigendecomposition (requires non-negative eigenvalues) |
| `matrix_pow(&A, t)` | Aᵗ via eigendecomposition |

### `structured`

| Function | Description |
|---|---|
| `toeplitz(col, row)` | Toeplitz matrix from first column/row |
| `toeplitz_symmetric(col)` | Symmetric Toeplitz |
| `circulant(row)` | Circulant matrix |
| `circulant_mul(row, &v)` | Structured circulant-vector product |
| `vandermonde(x, m)` | Vandermonde matrix V_{i,j} = xᵢʲ |
| `vandermonde_det(x)` | Closed-form determinant = ∏(xᵢ - xⱼ) |
| `hankel(col, row)` | Hankel matrix |

### `agent_analysis`

| Function | Description |
|---|---|
| `build_similarity_matrix(&features, σ)` | Gaussian RBF kernel |
| `analyze_agent_similarity(&sim, k, tol)` | Eigendecomposition + spectral k-means |
| `agent_diversity(&sim)` | 1 - mean off-diagonal similarity |

## How It Works

1. **LU decomposition** uses Gaussian elimination with partial pivoting: at each step, the row with the largest pivot element is swapped into position. This gives PA = LU where L is unit lower triangular and U is upper triangular.

2. **QR decomposition** uses modified Gram-Schmidt: each column of A is orthogonalized against all previous columns, producing an orthonormal Q and upper triangular R.

3. **Cholesky** directly computes L column by column: L_{jj} = √(A_{jj} - Σₖ L²_{jk}), then L_{ij} = (A_{ij} - Σₖ L_{ik}·L_{jk}) / L_{jj}.

4. **SVD** computes AᵀA, then extracts singular values/vectors via iterative power method with deflation (subtracting the rank-1 outer product of each found component).

5. **Eigendecomposition** dispatches: for symmetric matrices, uses `nalgebra`'s Jacobi eigenvalue method; for general matrices, uses QR iteration (repeatedly factor T = QR, form T' = RQ, accumulate Q transforms).

6. **Matrix exponential** has two implementations: (a) via eigendecomposition exp(A) = V·exp(D)·V⁻¹, and (b) via [1/1] Padé approximant with scaling-and-squaring — scale A so ‖A/2ˢ‖ < 0.5, approximate exp on the scaled matrix, then square s times.

7. **Perturbation theory** implements Bauer-Fike (|λ' - λ| ≤ κ(V)·‖E‖ for diagonalizable A+E), Weyl's inequality (‖λ' - λ‖_∞ ≤ ‖E‖₂ for symmetric), and finite-difference eigenvalue sensitivity.

8. **Nearest PSD** runs Higham's iterative projection: alternately project onto the symmetric cone (average with transpose) and the PSD cone (clamp negative eigenvalues to zero).

## The Math

**Spectral decomposition:** For diagonalizable A = VDV⁻¹, any analytic function f extends to matrices: f(A) = V·f(D)·V⁻¹ where f(D) applies f element-wise to eigenvalues.

**SVD:** A = UΣVᵀ where U and V are orthogonal and Σ is diagonal with non-negative singular values σ₁ ≥ ... ≥ σᵣ > 0. The operator norm ‖A‖₂ = σ₁, nuclear norm = Σσᵢ.

**Condition number:** κ(A) = ‖A‖·‖A⁻¹‖ = σ_max/σ_min. Large κ means the linear system Ax = b is sensitive to perturbations in b.

**Bauer-Fike theorem:** If A = VDV⁻¹ is diagonalizable and E is a perturbation, then every eigenvalue λ' of A+E satisfies min_λ |λ' - λ| ≤ κ(V)·‖E‖ for any operator norm induced by a monotone vector norm.

**Weyl's inequality:** For Hermitian A and perturbation E, |λᵢ(A+E) - λᵢ(A)| ≤ ‖E‖₂ for corresponding ordered eigenvalues.

**Kronecker product:** (A⊗B)ᵢₚ,ⱼᵧ = aᵢⱼ·bₚᵧ. Properties: (A⊗B)(C⊗D) = AC⊗BD, (A⊗B)ᵀ = Aᵀ⊗Bᵀ, det(A⊗B) = det(A)ᵖ·det(B)ᵐ for A m×m, B p×p.

**Vandermonde determinant:** det(V) = ∏ᵢ>ⱼ(xᵢ - xⱼ). Zero iff the nodes are not distinct.

## Tests

**59 tests** across source modules and an integration test suite: LU (identity, general, unit lower triangular), QR (identity, general, orthogonality), Cholesky (PD, non-PD), SVD (identity, reconstruction), eigenvalues (symmetric, AV = λV), linear solve, Frobenius/operator/1/∞ norms, condition number (identity = 1, singular = ∞), trace, rank, Weyl bound verification, eigenvalue condition numbers, PD/PSD checks, nearest PSD, sparse (identity, dense multiply, vector multiply, transpose), Kronecker (basic, transpose property, mixed product, sum), matrix exp/log/sqrt/pow roundtrips, Padé, Toeplitz, circulant, Vandermonde (basic, determinant, repeated nodes), Hankel, similarity matrix, spectral clustering, diversity metrics, SVD-norm relation, LU-solve consistency, nuclear norm, sparse transpose, Padé identity.

## License

MIT
