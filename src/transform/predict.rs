//! Provides functions for making class predictions using either the original 
//! Random Forest structure or the compiled, single Decision Tree representation.

use crate::{tree::{Arena, NodeId, FEATURE_LEAF_SENTINEL}, utils, NUM_CLASSES};
use super::data::RandomForestData;

/// Predicts the class for a single input instance by traversing the compiled 
/// decision tree stored in the `Arena`.
///
/// This function simulates the standard decision path traversal: starting from the 
/// `root_id`, it follows the true/false branches based on feature comparisons 
/// against thresholds until a leaf node is encountered. The class associated 
/// with that leaf node is returned as the prediction.
///
/// # Arguments
/// * `arena` - The Arena containing the nodes of the compiled decision tree.
/// * `root_id` - The NodeId of the root node of the compiled tree in the `arena`.
/// * `features` - A slice containing the input feature values for the instance.
///
/// # Returns
/// The predicted class index (usize).
pub fn predict_with_merged_tree(arena: &Arena, root_id: NodeId, features: &[f64]) -> usize {
    let mut current_node_id = root_id;

    loop {
        // Check if the current node is a leaf
        if arena.is_leaf(current_node_id) {
            // Reached a leaf, return its class
            return arena.get_leaf_class(current_node_id);
        }
        
        // Internal node: evaluate the split condition
        let feature_idx = arena.get_feature_usize(current_node_id);
        let threshold = arena.get_threshold(current_node_id); // Get f64 threshold for comparison
        let feature_value = features[feature_idx];

        // Determine the next node based on the feature value
        let next_node_id = if feature_value <= threshold {
            arena.get_true_id(current_node_id) // Follow the 'true' branch (feature <= threshold)
        } else {
            arena.get_false_id(current_node_id) // Follow the 'false' branch (feature > threshold)
        };

        current_node_id = next_node_id;
    }
}

/// Predicts the class for a single input instance by simulating the prediction 
/// process of the *original* Random Forest ensemble.
///
/// This function iterates through each individual tree within the `rf_data`, 
/// traverses it based on the input `features`, and records the prediction (class) 
/// from the reached leaf. Finally, it performs a majority vote over the predictions 
/// from all trees to determine the ensemble's final predicted class.
/// 
/// This serves as a way to get predictions directly from the source RF model, 
/// often used for comparison or verification against the compiled tree's predictions 
/// (`predict_with_merged_tree`). It includes robust checks against invalid indices 
/// or data issues within the `rf_data` structure.
///
/// # Arguments
/// * `rf_data` - A structure holding the data for the original Random Forest, 
///   containing separate arrays/vectors for node features, thresholds, 
///   children, and leaf classes for each tree.
/// * `features` - A slice representing the input feature values for the instance.
///
/// # Returns
/// A tuple containing three elements:
///   1. `usize`: The final predicted class index determined by the majority vote.
///   2. `Vec<u32>`: A vector (size `NUM_CLASSES`) containing the raw vote counts for each class 
///      accumulated across all trees in the ensemble.
///   3. `Vec<usize>`: A vector containing the class prediction made by each individual 
///      tree in the forest. If an error occurred during a specific tree's traversal 
///      (e.g., invalid node index), `usize::MAX` is used as a sentinel value for 
///      that tree's prediction.
pub fn predict_with_original_rf(rf_data: &RandomForestData, features: &[f64]) -> (usize, Vec<u32>, Vec<usize>) {
    let mut votes = vec![0u32; NUM_CLASSES];
    // Store the prediction from each individual tree
    let mut individual_preds = Vec::with_capacity(rf_data.n_total_trees); 

    // Iterate through each tree in the original forest structure
    for tree_idx in 0..rf_data.n_total_trees {
        let mut current_node_idx: i64 = 0;
        let mut prediction_for_this_tree = usize::MAX; // Default to sentinel in case of error

        // Traverse the current tree from root (index 0) to a leaf
        loop {
            // Safeguard against invalid indices (potential data issue or traversal error)
            if current_node_idx < 0 || current_node_idx as usize >= rf_data.features[tree_idx].len() {
                 eprintln!("[WARN] Invalid node index {} encountered during traversal in original tree {}. Stopping traversal for this tree.", current_node_idx, tree_idx);
                 break; 
            }
            let current_node_usize = current_node_idx as usize;

            let feature_idx_raw = rf_data.features[tree_idx][current_node_usize];

            // Check if it's a leaf node (using the sentinel value)
            if feature_idx_raw == FEATURE_LEAF_SENTINEL {
                 // Bounds check before accessing leaf class data
                 if current_node_usize >= rf_data.leaf_classes[tree_idx].len() {
                     eprintln!("[WARN] Node index {} out of bounds for leaf_classes in original tree {}. Stopping traversal.", current_node_usize, tree_idx);
                     break;
                 }
                let leaf_class = rf_data.leaf_classes[tree_idx][current_node_usize];
                prediction_for_this_tree = leaf_class; 

                // Add vote if the class is valid
                if leaf_class < NUM_CLASSES {
                    votes[leaf_class] += 1;
                } else {
                    // Handle unexpected invalid leaf class value
                    eprintln!("[WARN] Invalid leaf class {} found in original tree {} node {}", leaf_class, tree_idx, current_node_idx);
                    prediction_for_this_tree = usize::MAX; // Mark prediction as invalid
                }
                break; // Reached leaf, exit loop for this tree
            }

            // Internal node: get split info and decide next step
            let feature_idx = feature_idx_raw as usize;

            // Perform bounds checks before accessing feature/threshold/children data
            if feature_idx >= features.len() {
                 eprintln!("[WARN] Feature index {} out of bounds for input features (len={}) in original tree {} node {}. Stopping traversal.", feature_idx, features.len(), tree_idx, current_node_idx);
                 break;
            }
            if current_node_usize >= rf_data.thresholds[tree_idx].len() {
                 eprintln!("[WARN] Node index {} out of bounds for thresholds in original tree {}. Stopping traversal.", current_node_usize, tree_idx);
                 break;
            }
             if current_node_usize >= rf_data.children_left[tree_idx].len() || current_node_usize >= rf_data.children_right[tree_idx].len() {
                 eprintln!("[WARN] Node index {} out of bounds for children arrays in original tree {}. Stopping traversal.", current_node_usize, tree_idx);
                 break;
             }


            let threshold = rf_data.thresholds[tree_idx][current_node_usize]; 
            let feature_value = features[feature_idx];

            // Determine the next node index based on the split condition
            let next_node_idx: i64 = if feature_value <= threshold {
                rf_data.children_left[tree_idx][current_node_usize]
            } else {
                rf_data.children_right[tree_idx][current_node_usize]
            };
            current_node_idx = next_node_idx;
        }
        // Record the prediction (or sentinel) for the tree just traversed
        individual_preds.push(prediction_for_this_tree);
    }

    // Determine the final ensemble prediction via majority vote on collected votes
    let final_prediction = utils::_FAST_get_majority_class(&votes);

    // Return the final prediction, the raw vote counts, and the list of individual tree predictions
    (final_prediction, votes, individual_preds)
}