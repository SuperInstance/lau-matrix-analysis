//! # lau-matrix-analysis
//!
//! Matrix analysis: decompositions, perturbation theory, structured matrices, and matrix functions.

pub mod decompositions;
pub mod norms;
pub mod perturbation;
pub mod positive_definite;
pub mod sparse;
pub mod kronecker;
pub mod matrix_functions;
pub mod structured;
pub mod agent_analysis;

pub use decompositions::*;
pub use norms::*;
pub use perturbation::*;
pub use positive_definite::*;
pub use sparse::*;
pub use kronecker::*;
pub use matrix_functions::*;
pub use structured::*;
pub use agent_analysis::*;

use nalgebra::DMatrix;
use nalgebra::DVector;

/// Type alias for real-valued dynamic matrices.
pub type Mat = DMatrix<f64>;
/// Type alias for real-valued dynamic vectors.
pub type VecN = DVector<f64>;
