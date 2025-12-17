//! Implements the different early stopping strategies (ES, AbsES, HEUR, ORD) 
//! used during the RF-to-DT compilation process, as described in the paper [17].
//! This includes the core abstract interpretation logic required for AbsES, HEUR, 
//! and ORD strategies to estimate guaranteed outcomes and prune the search space.

use crate::tree::{NodeId, ConditionStatus, FEATURE_LEAF_SENTINEL};
use crate::utils::_FAST_get_top_two_votes_with_safe;
use super::common::{BoundsMap, check_condition_bounds};
use super::builder::ArenaProvider;
use crate::NUM_CLASSES;

/// Enumerates the available strategies for deciding when to stop the recursive 
/// tree merging process early. These strategies balance computational cost 
/// against the potential for earlier pruning. See paper [17] for details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyStoppingStrategy {
    /// **Standard Early Stopping (ES):** The simplest strategy. Checks if the current 
    /// majority class has enough votes to win even if all remaining trees vote 
    /// for the second-best class. Typically applied only after processing half the trees.
    Standard,
    /// **Abstract Early Stopping (AbsES):** Uses abstract interpretation (`abstract_interpret_dt`) 
    /// to determine the *guaranteed* class prediction for each remaining tree under 
    /// the current path's constraints. Sums these guaranteed votes with current votes 
    /// to check for a guaranteed overall winner. More powerful but computationally 
    /// more expensive than Standard ES.
    Abstract,
    /// **Heuristic Early Stopping (HEUR):** A hybrid approach. Uses Standard ES after 
    /// processing half the trees. Before that, it uses a heuristic (e.g., vote margin) 
    /// to decide whether to invoke the more expensive AbsES check. Aims for a practical 
    /// trade-off between speed and pruning power.
    HEUR,
    /// **Ordered Early Stopping (ORD):** Extends HEUR by dynamically ordering the 
    /// processing of remaining trees. It prioritizes trees estimated to be simpler 
    /// (fewer reachable leaves under current constraints, via `count_reachable_leaves`) 
    /// to potentially enable earlier pruning. Uses the same HEUR logic for the 
    /// early stopping check itself.
    ORD,
}

/// Performs Abstract Interpretation on a single decision tree node to determine 
/// the *guaranteed* class prediction given the current path constraints.
/// This is essentially Algorithm 2 from the paper [17].
///
/// If the path constraints (`bounds_map`) force the traversal down a single path 
/// to a specific leaf, that leaf's class is returned. If the constraints allow 
/// reaching leaves of *different* classes, it returns `NUM_CLASSES` (acting as 
/// an "unsure" or "any class" sentinel value) because no single class is guaranteed.
///
/// **Performance Critical:** Optimized for speed, no safety checks.
///
/// # Arguments
/// * `provider` - Provides access to the node data in the arena.
/// * `node_id` - The ID of the current node in the tree being interpreted.
/// * `bounds_map` - The current feature bounds imposed by the path taken in the merged tree.
///
/// # Returns
/// The guaranteed class label (0 to K-1) if one exists, otherwise `NUM_CLASSES` (K).
pub(crate) fn abstract_interpret_dt<'a, P: ArenaProvider<'a>>(
    provider: &P,
    node_id: NodeId,
    bounds_map: &BoundsMap,
) -> usize {
    // if leaf { return class or NUM_CLASSES }
    // if internal { check bounds -> recurse true, recurse false, or recurse both & compare }

    let feature_idx_signed = provider.get_feature_raw(node_id);

    if feature_idx_signed == FEATURE_LEAF_SENTINEL { // Leaf node
        let leaf_class = provider.get_leaf_class(node_id);
        if leaf_class >= NUM_CLASSES { 
            return NUM_CLASSES; // Returns "unsure"
        }
        return leaf_class;
    }

    // Internal node
    let feature = provider.get_feature_usize(node_id);
    let threshold = provider.get_threshold(node_id);
    let left_child_id = provider.get_true_id(node_id);
    let right_child_id = provider.get_false_id(node_id);

    let condition_status = check_condition_bounds(bounds_map, feature, threshold);

    match condition_status {
        ConditionStatus::AlwaysTrue => {
            abstract_interpret_dt(provider, left_child_id, bounds_map)
        }
        ConditionStatus::AlwaysFalse => {
            abstract_interpret_dt(provider, right_child_id, bounds_map)
        }
        ConditionStatus::Undetermined => {
            let true_class = abstract_interpret_dt(provider, left_child_id, bounds_map);
            let false_class = abstract_interpret_dt(provider, right_child_id, bounds_map);

            if true_class == false_class {
                true_class
            } else {
                NUM_CLASSES // "unsure"
            }
        }
    }
}

/// Checks if an early stop is possible using the Abstract Early Stopping logic (AbsES).
/// This implements Algorithm 3 from the paper [17].
///
/// It iterates through the remaining trees (from `tree_index + 1` onwards), performs 
/// abstract interpretation (`abstract_interpret_dt`) on each using the current `bounds_map`, 
/// and aggregates the guaranteed votes (`safe_votes`). It then compares the potential 
/// maximum score of the current winner (current votes + safe votes for that class) 
/// against the maximum potential score of the second-best class (current votes + 
/// safe votes + all "unsure" votes).
///
/// **Performance Critical:** Optimized for speed, no safety checks.
///
/// # Arguments
/// * `tree_index` - The index of the tree just processed.
/// * `current_votes` - Votes accumulated along the path so far.
/// * `bounds_map` - Current path constraints.
/// * `root_ids` - Slice containing root IDs of all trees in the sequence.
/// * `provider` - Mutable provider to intern the leaf node if stopping early.
///
/// # Returns
/// `Some(NodeId)` of the interned leaf node representing the guaranteed winning class 
/// if early stopping is possible, otherwise `None`.
pub(crate) fn abstract_early_stopping<'a, P: ArenaProvider<'a>>( // Generic over provider
    tree_index: usize,
    current_votes: &[u32],
    bounds_map: &BoundsMap,
    root_ids: &[NodeId],
    provider: &mut P,
) -> Option<NodeId> {

    // Iterate tree_index+1..n_total_trees
    //   call abstract_interpret_dt for each root_ids[j]
    // Aggregate results into safe_votes
    // Calculate free_votes (unsure)
    // Check if winner is guaranteed using _FAST_get_top_two_votes_with_safe
    // If guaranteed, return Some(provider.intern_leaf(winning_class)), else None

    let n_total_trees = root_ids.len();

    let results = (tree_index + 1..n_total_trees)
        .into_iter() // Could use into_par_iter() for parallel processing
        .map(|j| {
            let root_id = root_ids[j];
            abstract_interpret_dt(provider, root_id, bounds_map)
        });

    // Last index (NUM_CLASSES) is the "unsure" class
    let mut safe_votes = vec![0u32; NUM_CLASSES + 1];
    for result_class in results {
        safe_votes[result_class] += 1;
    }
    let free_votes = safe_votes[NUM_CLASSES]; // "unsure" votes
    // Don't worry, _FAST_get_top_two_votes_with_safe won't care about this last index in safe_votes
    let (idx, max_total, second_max_total) = _FAST_get_top_two_votes_with_safe(current_votes, &safe_votes);

    let second_max_potential = second_max_total + free_votes;
    if max_total > second_max_potential || (max_total == second_max_potential && idx == 0) {
        Some(provider.intern_leaf(idx))
    } else {
        None
    }
}

/// Recursively counts the number of leaves reachable within a subtree given the 
/// current path constraints (`bounds`). This is used by the ORD strategy to estimate 
/// the "simplicity" of a remaining tree under the current path - fewer reachable 
/// leaves suggests the tree is more constrained and might lead to faster pruning 
/// if processed next.
///
/// It traverses the tree, pruning branches that are inconsistent with the `bounds`.
///
/// **Performance Critical:** Optimized for speed, no safety checks.
///
/// # Arguments
/// * `provider` - Provides access to node data.
/// * `node_id` - The root of the subtree to evaluate.
/// * `bounds` - The current feature bounds (mutable temporarily during recursion).
///
/// # Returns
/// The total count of reachable leaves in the subtree.
pub(crate) fn count_reachable_leaves<'a, P: ArenaProvider<'a>>( // Generic over provider
    provider: &P, // Accept provider by reference
    node_id: NodeId,
    bounds: &mut BoundsMap, // Bounds are modified temporarily within the recursion
) -> usize {
    let feature_idx_signed = provider.get_feature_raw(node_id);

    // Base Case: Leaf Node
    if feature_idx_signed == FEATURE_LEAF_SENTINEL {
        return 1; // This leaf is reachable
    }

    // Internal Node
    let feature_idx = provider.get_feature_usize(node_id);
    let threshold_f64 = provider.get_threshold(node_id);
    let left_child_id = provider.get_true_id(node_id);
    let right_child_id = provider.get_false_id(node_id);

    match check_condition_bounds(bounds, feature_idx, threshold_f64) {
        ConditionStatus::AlwaysTrue => {
            count_reachable_leaves(provider, left_child_id, bounds)
        }
        ConditionStatus::AlwaysFalse => {
            count_reachable_leaves(provider, right_child_id, bounds)
        }
        ConditionStatus::Undetermined => {
            // Explore both branches, modifying bounds

            // Store original bounds for restoration
            let original_upper_bound = bounds[feature_idx].1;
            let original_lower_bound = bounds[feature_idx].0;

            // Recurse left: Tighten upper bound
            bounds[feature_idx].1 = original_upper_bound.min(threshold_f64);
            let left_count = count_reachable_leaves(provider, left_child_id, bounds);
            bounds[feature_idx].1 = original_upper_bound; // Restore

            // Recurse right: Tighten lower bound
            bounds[feature_idx].0 = original_lower_bound.max(threshold_f64);
            let right_count = count_reachable_leaves(provider, right_child_id, bounds);
            bounds[feature_idx].0 = original_lower_bound; // Restore

            left_count + right_count
        }
    }
}

/// Checks if an early stop is possible using Abstract Early Stopping logic, specifically 
/// adapted for the ORD strategy.
///
/// This is nearly identical to `abstract_early_stopping`, but instead of iterating 
/// over a fixed range `tree_index + 1..n_total_trees`, it iterates over the 
/// `remaining_indices` provided by the ORD strategy's dynamic selection process.
///
/// **Performance Critical:** Optimized for speed, no safety checks.
///
/// # Arguments
/// * `current_votes` - Votes accumulated along the path so far.
/// * `bounds_map` - Current path constraints.
/// * `root_ids` - Slice containing root IDs of all trees in the sequence.
/// * `remaining_indices` - Slice of indices into `root_ids` representing the trees yet to be processed.
/// * `provider` - Mutable provider to intern the leaf node if stopping early.
///
/// # Returns
/// `Some(NodeId)` of the interned leaf node representing the guaranteed winning class 
/// if early stopping is possible, otherwise `None`.
pub(crate) fn abstract_early_stopping_ord<'a, P: ArenaProvider<'a>>( // Generic over provider
    current_votes: &[u32],
    bounds_map: &BoundsMap,
    root_ids: &[NodeId],
    remaining_indices: &[usize],
    provider: &mut P,
) -> Option<NodeId> {

    // Iterate remaining_indices
    //   call abstract_interpret_dt for each root_ids[j] where j is from remaining_indices
    // Aggregate results into safe_votes
    // Calculate free_votes (unsure)
    // Check if winner is guaranteed using _FAST_get_top_two_votes_with_safe
    // If guaranteed, return Some(provider.intern_leaf(winning_class)), else None

    let results = remaining_indices
        .iter() 
        .map(|&j| {
            let root_id = root_ids[j];
            abstract_interpret_dt(provider, root_id, bounds_map)
        });

    // Last index (NUM_CLASSES) is the "unsure" class
    let mut safe_votes = vec![0u32; NUM_CLASSES + 1];
    for result_class in results {
        safe_votes[result_class] += 1;
    }

    let free_votes = safe_votes[NUM_CLASSES]; // "unsure" votes
    // Don't worry, _FAST_get_top_two_votes_with_safe won't care about this last index in safe_votes
    let (idx, max_total, second_max_total) = _FAST_get_top_two_votes_with_safe(current_votes, &safe_votes);
    let second_max_potential = second_max_total + free_votes;
    if max_total > second_max_potential || (max_total == second_max_potential && idx == 0) {
        Some(provider.intern_leaf(idx))
    } else {
        None
    }
}
