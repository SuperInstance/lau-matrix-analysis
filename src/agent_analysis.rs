//! Agent correlation analysis: eigenvalue methods for agent similarity matrices.

use crate::{Mat, VecN};
use crate::decompositions::eigen;
use crate::norms::condition_number;

/// Agent similarity analysis result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentSimilarityResult {
    /// Number of agents.
    pub n_agents: usize,
    /// Eigenvalues of the similarity matrix.
    pub eigenvalues: VecN,
    /// Effective rank (number of significant eigenvalues).
    pub effective_rank: usize,
    /// Condition number of the similarity matrix.
    pub condition_number: f64,
    /// Whether the matrix is positive semi-definite.
    pub is_psd: bool,
    /// Agent clusters via spectral method.
    pub clusters: std::vec::Vec<usize>,
    /// Top-k principal components.
    pub top_components: std::vec::Vec<VecN>,
}

/// Analyze agent similarity matrix using eigenvalue methods.
pub fn analyze_agent_similarity(sim: &Mat, n_clusters: usize, tol: f64) -> Option<AgentSimilarityResult> {
    let n = sim.nrows();
    if n != sim.ncols() || n == 0 {
        return None;
    }

    let eig = eigen(sim)?;
    let eigenvalues = eig.eigenvalues.clone_owned();

    let effective_rank = eigenvalues.iter().filter(|&&v| v.abs() > tol).count();
    let cond = condition_number(sim);
    let is_psd = eigenvalues.iter().all(|&v| v >= -tol);

    let mut indices: std::vec::Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

    let k = n_clusters.min(n);
    let top_k: std::vec::Vec<usize> = indices.iter().take(k).copied().collect();

    let mut top_components = std::vec::Vec::with_capacity(k);
    for &idx in &top_k {
        top_components.push(eig.eigenvectors.column(idx).clone_owned());
    }

    let clusters = spectral_cluster(&eig.eigenvectors, &top_k, n_clusters, 50);

    Some(AgentSimilarityResult {
        n_agents: n,
        eigenvalues,
        effective_rank,
        condition_number: cond,
        is_psd,
        clusters,
        top_components,
    })
}

fn spectral_cluster(v: &Mat, top_k: &[usize], n_clusters: usize, max_iter: usize) -> std::vec::Vec<usize> {
    let n = v.nrows();
    let k = top_k.len().min(n_clusters);
    if k == 0 {
        return std::vec![0; n];
    }

    let mut features = Mat::zeros(n, k);
    for (j, &idx) in top_k.iter().enumerate() {
        for i in 0..n {
            features[(i, j)] = v[(i, idx)];
        }
    }

    let mut centroids = Mat::zeros(k, k);
    for (i, row_idx) in (0..n).take(k).enumerate() {
        for j in 0..k {
            centroids[(i, j)] = features[(row_idx, j)];
        }
    }

    let mut assignments = std::vec![0usize; n];

    for _ in 0..max_iter {
        let mut changed = false;
        for i in 0..n {
            let mut best_dist = f64::INFINITY;
            let mut best_c = 0;
            for c in 0..k {
                let mut dist = 0.0;
                for j in 0..k {
                    let d = features[(i, j)] - centroids[(c, j)];
                    dist += d * d;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        let mut counts = std::vec![0usize; k];
        centroids.fill(0.0);
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for j in 0..k {
                centroids[(c, j)] += features[(i, j)];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..k {
                    centroids[(c, j)] /= counts[c] as f64;
                }
            }
        }
    }

    assignments
}

/// Build a similarity matrix from feature vectors using Gaussian (RBF) kernel.
pub fn build_similarity_matrix(features: &Mat, sigma: f64) -> Mat {
    let n = features.nrows();
    let mut sim = Mat::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            let mut dist_sq = 0.0;
            for k in 0..features.ncols() {
                let d = features[(i, k)] - features[(j, k)];
                dist_sq += d * d;
            }
            sim[(i, j)] = (-dist_sq / (2.0 * sigma * sigma)).exp();
        }
    }

    sim
}

/// Compute agent diversity metric from similarity matrix.
pub fn agent_diversity(sim: &Mat) -> f64 {
    let n = sim.nrows();
    if n <= 1 {
        return 1.0;
    }

    let total: f64 = sim.iter().sum();
    let diagonal: f64 = (0..n).map(|i| sim[(i, i)]).sum();
    let off_diag = total - diagonal;
    let max_off_diag = (n * n - n) as f64;

    1.0 - off_diag / max_off_diag
}
