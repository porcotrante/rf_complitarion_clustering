//! Data structures and loading logic for Random Forest data.
//! 
//! This module does not have performance critical code. Safety checks are allowed!

use serde::Deserialize;
use std::error::Error;
use std::fs;
use std::path::Path;

use crate::{NUM_CLASSES, NUM_FEATURES};

/// Data structure representing a Random Forest loaded from a JSON file.
///
/// This structure mirrors the common representation used by scikit-learn,
/// storing tree structures as parallel arrays. Includes pre-baked threshold bits.
#[derive(Deserialize, Debug, Clone)]
pub struct RandomForestData {
    /// The total number of trees in the forest.
    pub n_total_trees: usize,
    /// `features[tree_idx][node_idx]` stores the feature index used for splitting
    /// at that node, or -2 if it's a leaf node. Must be < NUM_FEATURES.
    pub features: Vec<Vec<i64>>,
    /// `thresholds[tree_idx][node_idx]` stores the threshold value for the split
    /// at that node. Leaf nodes have a sentinel value of -2.0.
    pub thresholds: Vec<Vec<f64>>,
    /// Pre-baked `thresholds` as `u64` bits for efficient hashing and comparison during transform.
    #[serde(skip)] // Skip deserialization, populate manually after loading
    pub thresholds_bits: Vec<Vec<u64>>,
    /// `children_left[tree_idx][node_idx]` stores the index of the left child node
    /// (taken if `feature <= threshold`). Leaf nodes have -1.
    pub children_left: Vec<Vec<i64>>,
    /// `children_right[tree_idx][node_idx]` stores the index of the right child node
    /// (taken if `feature > threshold`). Leaf nodes have -1.
    pub children_right: Vec<Vec<i64>>,
    /// `values[tree_idx][node_idx][class_idx]` stores the class distribution (counts or probabilities)
    /// at that node. For leaves, this determines the prediction.
    pub values: Vec<Vec<Vec<f64>>>,
    /// Precomputed majority class for each leaf. `usize::MAX` for non-leaves.
    #[serde(skip)] // Skip deserialization, populate manually after loading
    pub leaf_classes: Vec<Vec<usize>>,
}

/// Loads Random Forest data from a JSON file.
///
/// The JSON file is expected to contain the fields defined in `RandomForestData`.
/// Performs basic validation and pre-bakes threshold bits and leaf classes.
///
/// # Arguments
/// * `filename` - The path to the JSON file containing the Random Forest data.
///
/// # Returns
/// A `Result` containing the loaded `RandomForestData` or a `Box<dyn Error>`.
pub fn load_rf_data<P: AsRef<Path>>(filename: P) -> Result<RandomForestData, Box<dyn Error>> {
    let filename_ref = filename.as_ref();

    let file_content = fs::read_to_string(filename_ref)
        .map_err(|e| format!("Failed to read file {:?}: {}", filename_ref, e))?;

    let mut rf_data: RandomForestData = serde_json::from_str(&file_content)
        .map_err(|e| format!("Failed to parse JSON from {:?}: {}", filename_ref, e))?;

    // --- Validation (on raw data) ---
    if rf_data.n_total_trees != rf_data.features.len()
        || rf_data.n_total_trees != rf_data.thresholds.len()
        || rf_data.n_total_trees != rf_data.children_left.len()
        || rf_data.n_total_trees != rf_data.children_right.len()
        || rf_data.n_total_trees != rf_data.values.len()
    {
        return Err(format!(
            "Inconsistent number of trees ({}) found in raw data arrays in file {:?}.",
            rf_data.n_total_trees, filename_ref
        )
        .into());
    }

    // --- Validate Feature Indices ---
    for (tree_idx, features_in_tree) in rf_data.features.iter().enumerate() {
        for (node_idx, &feature_idx_signed) in features_in_tree.iter().enumerate() {
            if feature_idx_signed >= 0 {
                let feature_idx = feature_idx_signed as usize;
                if feature_idx >= NUM_FEATURES {
                    return Err(format!(
                        "Feature index {} at tree {}, node {} is out of bounds (>= NUM_FEATURES = {}).",
                        feature_idx, tree_idx, node_idx, NUM_FEATURES
                    ).into());
                }
            } else if feature_idx_signed != -2 {
                 eprintln!("Warning: Unexpected negative feature index {} at tree {}, node {}", feature_idx_signed, tree_idx, node_idx);
            }
        }
    }

    // --- Pre-bake Threshold Bits and Leaf Classes ---
    rf_data.thresholds_bits = Vec::with_capacity(rf_data.n_total_trees);
    rf_data.leaf_classes = Vec::with_capacity(rf_data.n_total_trees);

    for i in 0..rf_data.n_total_trees {
        // Pre-bake bits
        let thresholds_bits: Vec<u64> = rf_data.thresholds[i].iter().map(|&t| t.to_bits()).collect();
        rf_data.thresholds_bits.push(thresholds_bits);

        // Pre-compute leaf classes
        let leaf_classes = compute_leaf_classes(&rf_data.features[i], &rf_data.values[i]);
        rf_data.leaf_classes.push(leaf_classes);
    }

    Ok(rf_data)
}

/// Helper to compute leaf classes for a single tree's nodes.
pub(crate) fn compute_leaf_classes(features: &[i64], values: &[Vec<f64>]) -> Vec<usize> {
    let num_nodes = features.len();
    let mut tree_leaf_classes = Vec::with_capacity(num_nodes);
    for node_idx in 0..num_nodes {
        if features[node_idx] == -2 { // Is leaf
             if node_idx >= values.len() || values[node_idx].len() != NUM_CLASSES {
                 eprintln!("Warning: Inconsistent values array for leaf node {} during leaf class computation.", node_idx);
                 tree_leaf_classes.push(0); // Default to class 0 on error
                 continue;
             }
             // Use _FAST_get_majority_class logic (adapted for f64 values)
             let mut max_val = f64::NEG_INFINITY;
             let mut leaf_class = 0;
             for (cls_idx, &val) in values[node_idx].iter().enumerate() {
                 if val > max_val {
                     max_val = val;
                     leaf_class = cls_idx;
                 }
                 // Tie-breaking: prefer lower index (already handled by iteration order)
             }
            tree_leaf_classes.push(leaf_class);
        } else {
            tree_leaf_classes.push(usize::MAX); // Sentinel for non-leaves
        }
    }
    tree_leaf_classes
}
