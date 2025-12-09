## Geometric Gait Analysis Project
This project analyzes gait data (human motion sequences) using Riemannian geometry on Symmetric Positive Definite (SPD) matrices. It processes skeletal pose data to compute stability and variability metrics, comparing a geometric approach against a standard Euclidean baseline.

## Directory Structure
data/: Input data files in JSON format (e.g., run_fast_100%.json, run_slow_100%.json).
features/: Core mathematical logic.
spd_geom.py: Functions for Riemannian geometry (SPD matrices, geodesics, Fréchet variance).
embedder.py: Dimensionality reduction (UMAP) for visualization.
metrics.py: Calculation of baseline Euclidean metrics.
feature_maker.py: Transformation of raw coordinates into feature vectors.
gait/: Logic for gait segmentation (step detection).
step_detector.py: Detects individual steps based on ankle vertical markers.
pre/: Preprocessing tools.
preprocessor.py: Normalization routines (centering on pelvis, scaling by leg length).
io_pkg/: Input/Output utilities for loading pose data.
main.py: The entry point that runs the analysis pipeline.

## Analysis Pipeline
The primary workflow is defined in main.py. The pipeline executes the following steps:
Data Loading: Reads factorial JSON pose data.
Preprocessing:
Centers the skeleton on the pelvis.
Scales dimensions based on leg length to ensure subject invariance.
Joint Selection: Filters specific joints for analysis (currently configured for the Right Leg: Hip, Knee, Ankle).
Step Detection: Identifies individual steps by analyzing the vertical movement (Y-axis) of the ankle.
Metric Calculation: For every detected step, the system calculates two sets of metrics:
Euclidean Baseline: Path length (Smoothness), Mean Velocity, and Variance calculated on raw Cartesian coordinates.
Riemannian (SPD):
Converts step features into an SPD matrix (covariance representation).
Computes Smoothness (Geodesic length of the sequence on the manifold).
Computes Average Velocity on the manifold.
Computes Fréchet Variance across multiple steps.
Comparison: Outputs a comparison of stability metrics between the geometric and Euclidean approaches.
## Usage

To run the analysis on a specific dataset, execute the main.py script.
The script currently defaults to processing data/run_fast_100%.json. You can modify the run_spd_pipeline call in main.py to target different files (e.g., mid or slow runs)
## Key Configuration

Adjustable parameters within : `main.py`
- : List of joints to include in the analysis. `JOINTS_RIGHT`
- : Boolean flag.
    - `True`: Computes Smoothness based on a sequence of SPD matrices per step.
    - `False`: Collapses the entire step into a single SPD matrix.

`USE_SEQ`
