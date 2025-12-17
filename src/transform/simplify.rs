//! Contains functions for simplifying decision tree structures by applying path 
//! condition analysis. This involves pruning branches that become unreachable 
//! given a set of feature bounds (`BoundsMap`). Simplification can occur either 
//! during the initial conversion of trees from an external format (like arrays) 
//! into the `Arena` structure, or as a refinement step within the partitioning 
//! strategy where tighter bounds specific to a partition are applied.

use crate::tree::{Arena, NodeId, ConditionStatus};
use super::common::{BoundsMap, check_condition_bounds};
use std::collections::HashMap;

/// Converts a single decision tree, originally represented by flat arrays (e.g., 
/// from scikit-learn), into the `Arena` format while simultaneously simplifying it. 
/// 
/// The simplification process uses the provided `bounds_map` to prune branches. 
/// If a node's split condition is always true or always false given the bounds, 
/// the traversal only proceeds down the necessary child branch. The resulting 
/// simplified tree structure is built within the provided `arena` using its 
/// interning mechanism (hash-consing) for node deduplication.
///
/// This function is typically used for the initial processing of input trees 
/// before merging or partitioning begins. The initial `bounds_map` might represent 
/// unrestricted bounds `(-inf, +inf)` for all features.
///
/// # Arguments
/// * `arena` - The mutable `Arena` where the simplified tree nodes will be interned.
/// * `orig_features`, `orig_thresholds`, ... - Slices representing the structure 
///   of the original tree in array format.
/// * `bounds_map` - The feature bounds constraints to apply during simplification.
/// * `count`, `kept_count` - Counters for detailed statistics (conditional compilation).
///
/// # Returns
/// The `NodeId` of the root of the simplified and interned tree within the `arena`.
pub(crate) fn simplify_single_tree(
    arena: &mut Arena,
    orig_features: &[i64], // Assumes sklearn-like format where feature index is stored
    orig_thresholds: &[f64], // Thresholds for comparison
    orig_thresholds_bits: &[u64], // Thresholds as bits for stable interning keys
    orig_children_left: &[i64], // Indices of left children in the original arrays
    orig_children_right: &[i64], // Indices of right children in the original arrays
    orig_leaf_classes: &[usize], // Class labels for leaf nodes
    bounds_map: &mut BoundsMap, // Feature bounds for pruning
    #[cfg(feature = "detailed-stats")]
    count: &mut usize, // Total nodes processed in original tree (conditional compilation)
    #[cfg(feature = "detailed-stats")]
    kept_count: &mut usize, // Nodes kept after simplification (conditional compilation)
) -> NodeId {

    // Delegate the recursive work to simplify_recursive_arena
    let root_id = simplify_recursive_arena(
        0, // Start traversal from the root node (index 0) of the original array tree
        bounds_map,
        orig_features,
        orig_thresholds,
        orig_thresholds_bits,
        orig_children_left,
        orig_children_right,
        orig_leaf_classes,
        arena,
        #[cfg(feature = "detailed-stats")]
        count,
        #[cfg(feature = "detailed-stats")]
        kept_count,
    );

    // Return the NodeId corresponding to the simplified tree's root in the Arena
    root_id 
}

/// Recursive helper function for `simplify_single_tree`. 
/// Traverses the original tree structure defined by arrays (`orig_*`) and builds 
/// the simplified equivalent within the `arena` by interning nodes.
///
/// Applies path condition pruning based on `bounds_map`.
fn simplify_recursive_arena(
    orig_node_idx: usize, // Current node index in the original array representation
    bounds_map: &mut BoundsMap, // Current feature bounds (modified during recursion)
    // --- Original tree data (read-only) ---
    orig_features: &[i64],
    orig_thresholds: &[f64],
    orig_thresholds_bits: &[u64],
    orig_children_left: &[i64],
    orig_children_right: &[i64],
    orig_leaf_classes: &[usize],
    // --- Output ---
    arena: &mut Arena, // Arena to intern simplified nodes into
    // --- Stats ---
    #[cfg(feature = "detailed-stats")]
    num_pruned_nodes: &mut usize,
    #[cfg(feature = "detailed-stats")]
    num_kept_nodes: &mut usize,
) -> NodeId {

    // Base Case: Current node in original tree is a leaf.
    // Assumes sklearn's sentinel value (-2) for leaves in the feature array.
    if orig_features[orig_node_idx] == -2 { 
        let predicted_class = orig_leaf_classes[orig_node_idx];
        // Intern the leaf into the arena and return its NodeId.
        let node_id = arena.intern_leaf(predicted_class); 
        return node_id;
    }

    // Recursive Step: Current node is an internal node.
    let feature = orig_features[orig_node_idx] as usize;
    let threshold = orig_thresholds[orig_node_idx]; // f64 for checking bounds
    let threshold_bits = orig_thresholds_bits[orig_node_idx]; // u64 for interning key
    let orig_left_child_idx = orig_children_left[orig_node_idx] as usize;
    let orig_right_child_idx = orig_children_right[orig_node_idx] as usize;

    // Check if the split condition is determined by the current bounds.
    let condition_status = check_condition_bounds(bounds_map, feature, threshold);

    let result_node_id = match condition_status {
        ConditionStatus::AlwaysTrue => {
            // Path condition guarantees feature <= threshold. Prune the right branch.
            #[cfg(feature = "detailed-stats")] { *num_pruned_nodes += 1; }
            // Recursively simplify only the left child.
            simplify_recursive_arena(
                orig_left_child_idx, bounds_map, orig_features, orig_thresholds, 
                orig_thresholds_bits, orig_children_left, orig_children_right, 
                orig_leaf_classes, arena, 
                #[cfg(feature = "detailed-stats")] num_pruned_nodes, 
                #[cfg(feature = "detailed-stats")] num_kept_nodes
            )
        }
        ConditionStatus::AlwaysFalse => {
            // Path condition guarantees feature > threshold. Prune the left branch.
            #[cfg(feature = "detailed-stats")] { *num_pruned_nodes += 1; }
            // Recursively simplify only the right child.
            simplify_recursive_arena(
                orig_right_child_idx, bounds_map, orig_features, orig_thresholds, 
                orig_thresholds_bits, orig_children_left, orig_children_right, 
                orig_leaf_classes, arena, 
                #[cfg(feature = "detailed-stats")] num_pruned_nodes, 
                #[cfg(feature = "detailed-stats")] num_kept_nodes
            )
        }
        ConditionStatus::Undetermined => {
            // Path condition does not determine the outcome; both branches are potentially reachable.
            #[cfg(feature = "detailed-stats")] { *num_kept_nodes += 1; }
            
            // Store the current bounds for the split feature before modifying.
            let (original_lower, original_upper) = bounds_map[feature];

            // --- Process True Branch (feature <= threshold) ---
            // Temporarily tighten the upper bound for the left child recursion.
            bounds_map[feature].1 = original_upper.min(threshold); 
            let simplified_true_id = simplify_recursive_arena(
                orig_left_child_idx, bounds_map, orig_features, orig_thresholds, 
                orig_thresholds_bits, orig_children_left, orig_children_right, 
                orig_leaf_classes, arena, 
                #[cfg(feature = "detailed-stats")] num_pruned_nodes, 
                #[cfg(feature = "detailed-stats")] num_kept_nodes
            );
            // Restore the original upper bound for backtracking.
            bounds_map[feature].1 = original_upper; 

            // --- Process False Branch (feature > threshold) ---
            // Temporarily tighten the lower bound for the right child recursion.
            bounds_map[feature].0 = original_lower.max(threshold); 
            let simplified_false_id = simplify_recursive_arena(
                orig_right_child_idx, bounds_map, orig_features, orig_thresholds, 
                orig_thresholds_bits, orig_children_left, orig_children_right, 
                orig_leaf_classes, arena, 
                #[cfg(feature = "detailed-stats")] num_pruned_nodes, 
                #[cfg(feature = "detailed-stats")] num_kept_nodes
            );
            // Restore the original lower bound for backtracking.
            bounds_map[feature].0 = original_lower; 

            // Intern the resulting internal node into the arena. 
            // The arena's interning handles the case where simplified_true_id == simplified_false_id, 
            // effectively simplifying the node away if both branches lead to the same result.
            arena.intern_internal(feature, threshold_bits, simplified_true_id, simplified_false_id)
        }
    };

    result_node_id // Return the NodeId of the simplified (sub)tree root in the arena
}


// --- Functions for Re-simplifying from an existing Arena ---
// These are used specifically within the partitioning approach (k > 0).

/// Takes a set of trees already represented in a `base_arena` (potentially after some 
/// initial global simplification) and applies *further* simplification based on 
/// tighter `partition_bounds` specific to a particular partition/subproblem. 
/// The results are interned into a `local_arena`.
///
/// This corresponds to Step 3 ("Simplify Subproblems") in the partitioning algorithm 
/// described in Section 4.3 of the paper. It refines the tree structures for the 
/// specific constraints of the current partition before the merging step (`MergeForest`) 
/// is applied within that partition.
///
/// # Arguments
/// * `base_arena` - The arena containing the nodes of the trees to be re-simplified.
/// * `base_root_ids` - Slice of root NodeIds in `base_arena` for the trees to process.
/// * `partition_bounds` - The *tighter* bounds specific to the current partition.
/// * `local_arena` - The arena where the re-simplified nodes will be interned.
/// * `num_pruned_nodes`, `num_kept_nodes` - Counters for partition-specific simplification stats.
///
/// # Returns
/// A `Vec<NodeId>` containing the root IDs of the re-simplified trees within the `local_arena`.
pub(crate) fn resimplify_from_arena(
    base_arena: &Arena,
    base_root_ids: &[NodeId],
    partition_bounds: &mut BoundsMap, // Bounds specific to this partition
    local_arena: &mut Arena, // Target arena for this partition's simplified trees
    #[cfg(feature = "detailed-stats")]
    num_pruned_nodes: &mut usize,
    #[cfg(feature = "detailed-stats")]
    num_kept_nodes: &mut usize,
) -> Vec<NodeId> {
    base_root_ids
        .iter()
        .map(|&base_root_id| {
            // Use a memoization cache *per tree* being re-simplified to handle shared subtrees 
            // within that original tree structure efficiently. Maps base_arena NodeId -> local_arena NodeId.
            let mut memoization_cache = HashMap::new(); 
            
            resimplify_recursive_arena(
                base_arena,
                base_root_id,
                partition_bounds,
                local_arena,
                &mut memoization_cache,
                #[cfg(feature = "detailed-stats")]
                num_pruned_nodes,
                #[cfg(feature = "detailed-stats")] 
                num_kept_nodes
            )
        })
        .collect() // Collect the new root IDs in the local_arena
}

/// Recursive helper function for `resimplify_from_arena`.
/// Traverses a tree structure within `base_arena`, applies pruning based on the 
/// tighter `partition_bounds`, and interns the resulting simplified structure 
/// into `local_arena`. Uses a `memoization_cache` to avoid redundant processing 
/// of shared subtrees during this re-simplification traversal.
fn resimplify_recursive_arena(
    base_arena: &Arena, // Source arena (read-only)
    base_node_id: NodeId, // Current node ID in base_arena
    partition_bounds: &mut BoundsMap, // Partition-specific bounds (modified during recursion)
    local_arena: &mut Arena, // Destination arena for simplified nodes
    memoization_cache: &mut HashMap<NodeId, NodeId>, // Memoization: base_id -> local_id
    #[cfg(feature = "detailed-stats")]
    num_pruned_nodes: &mut usize,
    #[cfg(feature = "detailed-stats")]
    num_kept_nodes: &mut usize,
) -> NodeId {
    // Check cache: If this base_node_id has already been processed for this partition, return its local_id.
    if let Some(&local_id) = memoization_cache.get(&base_node_id) {
        return local_id;
    }

    // Base Case: Node is a leaf in the base_arena.
    if base_arena.is_leaf(base_node_id) {
        let predicted_class = base_arena.get_leaf_class(base_node_id);
        // Intern the leaf directly into the local_arena.
        let local_id = local_arena.intern_leaf(predicted_class);
        // Cache the mapping before returning.
        memoization_cache.insert(base_node_id, local_id);
        return local_id;
    }

    // Recursive Step: Node is an internal node in base_arena.
    let feature = base_arena.get_feature_usize(base_node_id);
    let threshold = base_arena.get_threshold(base_node_id); // f64 for checking bounds
    let threshold_bits = base_arena.get_threshold_bits(base_node_id); // u64 for interning key
    let base_left_child_id = base_arena.get_true_id(base_node_id);
    let base_right_child_id = base_arena.get_false_id(base_node_id);

    // Check the split condition against the *partition-specific* bounds.
    let condition_status = check_condition_bounds(partition_bounds, feature, threshold);

    let result_local_id = match condition_status {
        ConditionStatus::AlwaysTrue => {
            // Partition bounds guarantee feature <= threshold. Prune right branch.
            #[cfg(feature = "detailed-stats")] { *num_pruned_nodes += 1; }
            // Recursively simplify only the left child from base_arena into local_arena.
            resimplify_recursive_arena(
                base_arena, base_left_child_id, partition_bounds, local_arena, 
                memoization_cache, 
                #[cfg(feature = "detailed-stats")] num_pruned_nodes, 
                #[cfg(feature = "detailed-stats")] num_kept_nodes
            )
        }
        ConditionStatus::AlwaysFalse => {
            // Partition bounds guarantee feature > threshold. Prune left branch.
            #[cfg(feature = "detailed-stats")] { *num_pruned_nodes += 1; }
            // Recursively simplify only the right child from base_arena into local_arena.
            resimplify_recursive_arena(
                base_arena, base_right_child_id, partition_bounds, local_arena, 
                memoization_cache, 
                #[cfg(feature = "detailed-stats")] num_pruned_nodes, 
                #[cfg(feature = "detailed-stats")] num_kept_nodes
            )
        }
        ConditionStatus::Undetermined => {
            // Partition bounds do not determine the outcome; need to explore both branches.
            #[cfg(feature = "detailed-stats")] { *num_kept_nodes += 1; }

            // Store current partition bounds for this feature before modifying.
            let (original_lower_bound, original_upper_bound) = partition_bounds[feature];

            // --- Process True Branch (feature <= threshold) ---
            partition_bounds[feature].1 = original_upper_bound.min(threshold); // Tighten upper bound
            let simplified_true_id = resimplify_recursive_arena( // Result is NodeId in local_arena
                base_arena, base_left_child_id, partition_bounds, local_arena, 
                memoization_cache, 
                #[cfg(feature = "detailed-stats")] num_pruned_nodes, 
                #[cfg(feature = "detailed-stats")] num_kept_nodes
            );
            partition_bounds[feature].1 = original_upper_bound; // Restore upper bound

            // --- Process False Branch (feature > threshold) ---
            partition_bounds[feature].0 = original_lower_bound.max(threshold); // Tighten lower bound
            let simplified_false_id = resimplify_recursive_arena( // Result is NodeId in local_arena
                base_arena, base_right_child_id, partition_bounds, local_arena, 
                memoization_cache, 
                #[cfg(feature = "detailed-stats")] num_pruned_nodes, 
                #[cfg(feature = "detailed-stats")] num_kept_nodes
            );
            partition_bounds[feature].0 = original_lower_bound; // Restore lower bound

            // Intern the resulting node in the local_arena using the *local* IDs of its children.
            local_arena.intern_internal(feature, threshold_bits, simplified_true_id, simplified_false_id)
        }
    };

    // Cache the mapping from base_node_id to the resulting local_id before returning.
    memoization_cache.insert(base_node_id, result_local_id);
    result_local_id
}