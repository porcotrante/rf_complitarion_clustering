//! Functionality for exporting the merged decision tree (stored in an `Arena`)
//! into a JSON format compatible with scikit-learn's internal tree representation.
//! This allows the merged tree to be loaded and used within Python environments.
//! 
//! Note: this functionality is not being used currently, because the benchmarking is done in Rust itself.
//! It might be useful if the merged tree needs to be exported to Python.

use crate::tree::{Arena, NodeId};
use serde::Serialize;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::collections::HashMap;
use std::cmp;

/// Structure mirroring scikit-learn's internal `_tree.Tree` state for export.
///
/// This structure holds the flattened arrays that represent the tree structure,
/// allowing it to be easily serialized to JSON and reconstructed in Python.
/// Field names are chosen to align with scikit-learn where possible.
#[derive(Serialize, Debug)]
struct SklearnTreeData {
    /// Total number of nodes in the exported tree.
    node_count: usize,
    /// Number of classes the original model was trained for.
    n_classes: usize,
    /// Maximum depth reached in the exported tree structure.
    max_depth: usize,
    /// `features[sklearn_node_id]` = feature index for split, or -2 for leaves.
    features: Vec<i64>,
    /// `thresholds[sklearn_node_id]` = threshold value for split, or -2.0 for leaves.
    thresholds: Vec<f64>,
    /// `children_left[sklearn_node_id]` = index of left child, or -1 for leaves.
    children_left: Vec<i64>,
    /// `children_right[sklearn_node_id]` = index of right child, or -1 for leaves.
    children_right: Vec<i64>,
    /// `values[sklearn_node_id][class_idx]` = class distribution at the node.
    /// For leaves, typically one class has a value of 1.0, others 0.0.
    /// For internal nodes, this might be zeroed or represent aggregated values (here, zeroed).
    values: Vec<Vec<f64>>,
    /// `impurity[sklearn_node_id]` = impurity measure (e.g., Gini). Placeholder (0.0).
    impurity: Vec<f64>,
    /// `n_node_samples[sklearn_node_id]` = number of samples reaching the node. Placeholder (1).
    n_node_samples: Vec<i64>,
}

/// Recursive helper function to traverse the merged tree in the `Arena`
/// and build the flattened scikit-learn compatible arrays.
///
/// Performs a depth-first traversal. It uses a `node_map` to handle the
/// potential DAG structure created by hash-consing, ensuring each unique
/// node in the `Arena` is processed and assigned a sequential scikit-learn
/// node index exactly once.
///
/// # Arguments
/// * `arena` - The arena containing the merged tree nodes.
/// * `node_id` - The `NodeId` (in the arena) of the current node to process.
/// * `features`, `thresholds`, ... - Mutable references to the vectors being populated.
/// * `node_map` - Maps `Arena` `NodeId` to the sequential `sklearn_node_id` assigned.
/// * `sklearn_node_id_counter` - Counter for assigning sequential sklearn node IDs.
/// * `current_depth` - The depth of the current node in the traversal.
/// * `max_depth` - Mutable reference to track the maximum depth reached.
/// * `n_classes` - The number of classes.
///
/// # Returns
/// The sequential scikit-learn node index assigned to the processed `node_id`.
fn build_sklearn_arrays_recursive(
    arena: &Arena,
    node_id: NodeId,
    features: &mut Vec<i64>,
    thresholds: &mut Vec<f64>,
    children_left: &mut Vec<i64>,
    children_right: &mut Vec<i64>,
    values: &mut Vec<Vec<f64>>,
    impurity: &mut Vec<f64>,
    n_node_samples: &mut Vec<i64>,
    node_map: &mut HashMap<NodeId, usize>,
    sklearn_node_id_counter: &mut usize,
    current_depth: usize,
    max_depth: &mut usize,
    n_classes: usize,
) -> usize {
    if let Some(&sklearn_id) = node_map.get(&node_id) {
        return sklearn_id;
    }

    let current_sklearn_id = *sklearn_node_id_counter;
    *sklearn_node_id_counter += 1;
    node_map.insert(node_id, current_sklearn_id);

    features.push(-99);
    thresholds.push(-99.0);
    children_left.push(-99);
    children_right.push(-99);
    impurity.push(-99.0);
    n_node_samples.push(-99);
    values.push(vec![-99.0; n_classes]);

    if current_depth > *max_depth {
        *max_depth = current_depth;
    }

    // Use Arena SoA accessors
    if arena.is_leaf(node_id) {
        let predicted_class = arena.get_leaf_class(node_id);

        features[current_sklearn_id] = -2;
        thresholds[current_sklearn_id] = -2.0;
        children_left[current_sklearn_id] = -1;
        children_right[current_sklearn_id] = -1;
        impurity[current_sklearn_id] = 0.0;
        n_node_samples[current_sklearn_id] = 1;

        let mut leaf_value = vec![0.0; n_classes];
        if predicted_class < n_classes {
            leaf_value[predicted_class] = 1.0;
        } else {
            eprintln!(
                "Warning: Predicted class {} out of bounds {} during export.",
                predicted_class, n_classes
            );
        }
        values[current_sklearn_id] = leaf_value;

        current_sklearn_id
    } else { // Internal node
        let feature = arena.get_feature_usize(node_id);
        let threshold = arena.get_threshold(node_id); // Use precomputed f64
        let true_id = arena.get_true_id(node_id);
        let false_id = arena.get_false_id(node_id);

        let left_sklearn_id = build_sklearn_arrays_recursive(
            arena,
            true_id,
            features,
            thresholds,
            children_left,
            children_right,
            values,
            impurity,
            n_node_samples,
            node_map,
            sklearn_node_id_counter,
            current_depth + 1,
            max_depth,
            n_classes,
        );
        let right_sklearn_id = build_sklearn_arrays_recursive(
            arena,
            false_id,
            features,
            thresholds,
            children_left,
            children_right,
            values,
            impurity,
            n_node_samples,
            node_map,
            sklearn_node_id_counter,
            current_depth + 1,
            max_depth,
            n_classes,
        );

        features[current_sklearn_id] = feature as i64;
        thresholds[current_sklearn_id] = threshold;
        children_left[current_sklearn_id] = left_sklearn_id as i64;
        children_right[current_sklearn_id] = right_sklearn_id as i64;
        impurity[current_sklearn_id] = 0.0;
        n_node_samples[current_sklearn_id] = 1;
        values[current_sklearn_id] = vec![0.0; n_classes];

        current_sklearn_id
    }
}

/// Exports the merged decision tree (stored in an `Arena`) to a JSON file
/// in a format compatible with scikit-learn's internal tree structure.
///
/// This function orchestrates the traversal and serialization process.
///
/// # Arguments
/// * `arena` - The `Arena` containing the nodes of the merged tree.
/// * `root_id` - The `NodeId` of the root node of the merged tree in the `arena`.
/// * `n_classes` - The number of classes the original model was trained for.
/// * `filename` - The path to the output JSON file.
///
/// # Returns
/// A `Result` indicating success or containing an error if traversal, serialization, or file writing fails.
pub fn export_merged_tree_to_json<P: AsRef<Path>>(
    arena: &Arena,
    root_id: NodeId,
    n_classes: usize,
    filename: P,
) -> Result<(), Box<dyn Error>> {
    let filename_ref = filename.as_ref();

    let mut features = Vec::new();
    let mut thresholds = Vec::new();
    let mut children_left = Vec::new();
    let mut children_right = Vec::new();
    let mut values = Vec::new();
    let mut impurity = Vec::new();
    let mut n_node_samples = Vec::new();

    let mut node_map = HashMap::new();
    let mut sklearn_node_id_counter = 0;
    let mut max_depth = 0;

    build_sklearn_arrays_recursive(
        arena,
        root_id,
        &mut features,
        &mut thresholds,
        &mut children_left,
        &mut children_right,
        &mut values,
        &mut impurity,
        &mut n_node_samples,
        &mut node_map,
        &mut sklearn_node_id_counter,
        0,
        &mut max_depth,
        n_classes,
    );

    let node_count = sklearn_node_id_counter;

    features.truncate(node_count);
    thresholds.truncate(node_count);
    children_left.truncate(node_count);
    children_right.truncate(node_count);
    values.truncate(node_count);
    impurity.truncate(node_count);
    n_node_samples.truncate(node_count);

    let export_data = SklearnTreeData {
        node_count,
        n_classes,
        max_depth,
        features,
        thresholds,
        children_left,
        children_right,
        values,
        impurity,
        n_node_samples,
    };

    println!(
        "Tree traversal complete. Node count: {}, Max depth: {}",
        node_count, max_depth
    );

    let json_string = serde_json::to_string_pretty(&export_data)?;

    fs::write(filename_ref, json_string)
        .map_err(|e| format!("Failed to write JSON to {:?}: {}", filename_ref, e))?;

    Ok(())
}

/// Recursive helper to calculate the height of a tree in the Arena.
fn calculate_height_recursive(arena: &Arena, node_id: NodeId, visited: &mut HashMap<NodeId, usize>) -> usize {
    if let Some(&height) = visited.get(&node_id) {
        return height;
    }

    // Use Arena SoA accessors
    let height = if arena.is_leaf(node_id) {
        0 // Height of a leaf is 0
    } else { // Internal node
        let true_id = arena.get_true_id(node_id);
        let false_id = arena.get_false_id(node_id);
        let left_height = calculate_height_recursive(arena, true_id, visited);
        let right_height = calculate_height_recursive(arena, false_id, visited);
        1 + cmp::max(left_height, right_height) // Height is 1 + max height of children
    };

    visited.insert(node_id, height);
    height
}

/// Calculates the height (maximum depth) of the merged tree stored in the Arena.
/// Handles DAG structures correctly by using a visited map.
pub fn calculate_tree_height(arena: &Arena, root_id: NodeId) -> usize {
    let mut visited = HashMap::new();
    calculate_height_recursive(arena, root_id, &mut visited)
}

pub fn get_tree_acc(arena: &Arena, root_id: NodeId, seed: u16) -> f64 {
    let filename = format!("src/rf/test_data_{}.csv", seed);

    let file = File::open(&filename)
        .expect(&format!("Erro ao abrir arquivo {}", filename));

    let reader = BufReader::new(file);

    // Aqui você pode acumular estatísticas depois
    let mut total = 0usize;
    let mut correct = 0usize;

    for line in reader.lines() {
        let line = line.expect("Erro ao ler linha");

        // Split por vírgula
        let values: Vec<f64> = line
            .split(',')
            .map(|v| v.parse::<f64>().expect("Erro ao converter valor"))
            .collect();

        // Assumindo:
        // - últimas coluna = y
        // - resto = features
        let (x, y_true) = values.split_at(values.len() - 1);
        let y_true = y_true[0];

        // 🔴 Aqui entra sua lógica de inferência da árvore
        // Exemplo fictício:
        let y_pred = predict_with_merged_tree(arena, root_id, x) as f64;

        if y_pred == y_true {
            correct += 1;
        }
        total += 1;
    }

    if total == 0 {
        0.0
    } else {
        (correct as f64 / total as f64) * 100.0
    }
}

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