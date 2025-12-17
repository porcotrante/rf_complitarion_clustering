//! Contains helper functions related to the partitioning strategy, primarily for 
//! managing node representation across different stages or parallel computations.
//! This includes logic for transferring and deduplicating nodes between arenas.

use crate::tree::{Arena, NodeId};
use std::collections::HashMap;


/// Represents the relevant range of pre-sorted global thresholds for each feature within a
/// specific partition. Used by the partitioning heuristic.
pub(crate) type PartitionThresholdBounds = Vec<(usize, usize)>; // feature_idx -> [start, end)


/// Recursively copies a subtree rooted at `local_node_id` from a source `local_arena` 
/// into a destination `final_arena`, ensuring that nodes are correctly interned 
/// (deduplicated) in the destination arena.
///
/// This function is crucial for the "Assemble Results" step (Section 4.3, Step 5) of the 
/// partitioning strategy. After solving subproblems (potentially in parallel, resulting 
/// in compiled subtrees possibly residing in separate temporary arenas or simplified 
/// within the same arena but needing integration), this function merges these results 
/// back into the main `final_arena`. It rebuilds the connections between nodes while 
/// leveraging the `final_arena`'s interning mechanism (hash-consing) to avoid creating 
/// duplicate nodes and control the size of the final compiled tree.
///
/// A `memoization_cache` (mapping local NodeIds to their corresponding NodeIds in the 
/// `final_arena`) is used to handle the potentially graph-like structure of decision 
/// trees (shared subtrees) and avoid redundant copying and interning work.
///
/// # Arguments
/// * `local_arena` - The source arena containing the node data to copy.
/// * `local_node_id` - The root ID of the subtree to copy from `local_arena`.
/// * `final_arena` - The destination arena where the copied and interned nodes will be stored.
/// * `memoization_cache` - A mutable HashMap to track which `local_arena` nodes have already 
///   been copied and what their corresponding IDs are in `final_arena`.
///
/// # Returns
/// The `NodeId` of the root of the copied and interned subtree within the `final_arena`.
pub(crate) fn _copy_and_intern_nodes(
    local_arena: &Arena,
    local_node_id: NodeId,
    final_arena: &mut Arena,
    memoization_cache: &mut HashMap<NodeId, NodeId>, // Maps local_id -> final_id
) -> NodeId {
    // Check cache first to avoid recomputing for already copied nodes/subtrees.
    if let Some(final_id) = memoization_cache.get(&local_node_id) {
        return *final_id;
    }

    // Node hasn't been copied yet. Determine if leaf or internal in the local arena.
    let final_id = if local_arena.is_leaf(local_node_id) {
        // Base Case: Leaf node. Intern a corresponding leaf in the final arena.
        let class = local_arena.get_leaf_class(local_node_id);
        final_arena.intern_leaf(class)
    } else {
        // Recursive Step: Internal node.
        let feature = local_arena.get_feature_usize(local_node_id);
        let threshold_bits = local_arena.get_threshold_bits(local_node_id);
        let local_true_id = local_arena.get_true_id(local_node_id);
        let local_false_id = local_arena.get_false_id(local_node_id);

        // Recursively copy/intern the children first. The results (final_left_id, final_right_id)
        // are the NodeIds of the children *within the final_arena*.
        let final_left_id = _copy_and_intern_nodes(local_arena, local_true_id, final_arena, memoization_cache);
        let final_right_id = _copy_and_intern_nodes(local_arena, local_false_id, final_arena, memoization_cache);

        // Intern the current internal node in the final_arena, using the *final* IDs 
        // of its children. The arena handles deduplication internally.
        final_arena.intern_internal(feature, threshold_bits, final_left_id, final_right_id)
    };

    // Store the mapping from the local node ID to the newly created/found final node ID 
    // in the cache *before* returning, to handle cycles/shared nodes correctly.
    memoization_cache.insert(local_node_id, final_id);
    final_id
}
