//! Orchestrates the transformation of a Random Forest (represented initially in a 
//! format like scikit-learn's) into a single semantically equivalent Decision Tree 
//! stored efficiently using an `Arena` with hash-consing (node deduplication).
//!
//! Provides two main entry points:
//! 1. `transform_rf`: Implements the baseline RF-to-DT compilation based on [17], 
//!    merging trees sequentially with early stopping. (Equivalent to the k=0 case).
//! 2. `transform_rf_partitioned`: Implements the partitioning-based strategy 
//!    described in the paper (Sections 4 & 6) to accelerate the compilation for k > 0. 
//!    It recursively divides the problem, simplifies subproblems, solves them 
//!    (potentially in parallel), and assembles the results.

mod common;
mod data;
mod simplify;
mod early_stopping;
mod builder;
mod partition;
mod predict;

pub use data::{load_rf_data, RandomForestData};
pub use early_stopping::EarlyStoppingStrategy;
pub use predict::{predict_with_merged_tree, predict_with_original_rf};

use crate::tree::{Arena, NodeId};
use crate::export::calculate_tree_height;
use common::BoundsMap;
use crate::{NUM_FEATURES, NUM_CLASSES};
use builder::{ArenaProvider, DualArenaProvider, SingleArenaProvider, TreeBuilder};
use early_stopping::count_reachable_leaves;
use partition::{PartitionThresholdBounds, _copy_and_intern_nodes};
pub(crate) use simplify::{simplify_single_tree, resimplify_from_arena};

use std::time::{Instant, Duration};
use std::collections::HashMap;
use std::ops::Add;

// --- Benchmark Stats Struct ---

/// Stores timing information for different phases of the partitioned transformation, 
/// allowing for performance analysis and comparison between strategies or k values.
/// Also includes optional detailed node counts when the "detailed-stats" feature is enabled.
#[derive(Debug, Default, Clone, Copy)]
pub struct BenchmarkStats {
    /// Time spent calculating the initial global thresholds and bounds template (k>0 only).
    pub template_bounds_duration: Duration,
    /// Time spent processing partitions. 
    /// For k=0: Time for the single sequential merge (`TreeBuilder`).
    /// For k>0: Includes time for recursive simplification (`resimplify_from_arena`) 
    ///          and the base-case merges (`TreeBuilder` at k=0 leaves).
    pub partition_processing_duration: Duration, 
    /// Time spent assembling results from child partitions (k>0 only), primarily 
    /// copying nodes between arenas (`_copy_and_intern_nodes`) and interning the 
    /// final split node for the current level.
    pub assembly_duration: Duration,
    /// Time spent explicitly dropping arenas during the recursive cleanup (k>0 only).
    pub drop_duration: Duration,
    /// [Detailed Stats] Count of nodes pruned during simplification phases.
    /// For k=0: Nodes pruned during the initial `simplify_single_tree` (often 0 if initial bounds are infinite).
    /// For k>0: Includes initial global simplification AND recursive `resimplify_from_arena`.
    #[cfg(feature = "detailed-stats")]
    pub simplify_pruning_count: usize,
    /// [Detailed Stats] Count of nodes pruned during the merge phase (`TreeBuilder`) 
    /// due to path conditions making a child branch unreachable.
    #[cfg(feature = "detailed-stats")]
    pub merge_pruning_count: usize, 
    /// [Detailed Stats] Count of nodes (including leaves) that were traversed but not pruned during 
    /// simplification or merge phases (i.e., condition was undetermined).
    #[cfg(feature = "detailed-stats")]
    pub kept_count: usize, 
}

/// Allows adding `BenchmarkStats` instances together, useful for accumulating 
/// stats recursively or across parallel branches.
impl Add for BenchmarkStats {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            template_bounds_duration: self.template_bounds_duration + other.template_bounds_duration,
            partition_processing_duration: self.partition_processing_duration + other.partition_processing_duration,
            assembly_duration: self.assembly_duration + other.assembly_duration,
            drop_duration: self.drop_duration + other.drop_duration,
            #[cfg(feature = "detailed-stats")]
            simplify_pruning_count: self.simplify_pruning_count + other.simplify_pruning_count,
            #[cfg(feature = "detailed-stats")]
            merge_pruning_count: self.merge_pruning_count + other.merge_pruning_count,
            #[cfg(feature = "detailed-stats")]
            kept_count: self.kept_count + other.kept_count,
        }
    }
}


// --- Baseline transform_rf (k=0 case) ---

/// Transforms a Random Forest into a single Decision Tree using the baseline approach 
/// (our re-implementation of the method in [17]; the k=0 case of the partitioned strategy).
///
/// It performs the following steps:
/// 1. Initializes an empty `Arena`.
/// 2. Interns all original trees from `rf_data` into the `Arena` using `simplify_single_tree` 
///    (typically with non-restrictive initial bounds, so mainly just conversion and initial deduplication).
/// 3. Creates a `TreeBuilder` using a `SingleArenaProvider` (reading and writing to the same `Arena`).
/// 4. Executes the recursive merge process (`_build_merged_tree_recursive_*`) according 
///    to the specified `strategy` (ES, AbsES, HEUR, ORD), accumulating votes and performing 
///    early stopping checks.
/// 5. Calculates the height of the resulting merged tree.
/// 6. Returns the final `Arena`, the root `NodeId` of the merged tree, benchmark statistics 
///    (primarily merge time for k=0), and the tree height.
///
/// # Arguments
/// * `rf_data` - The Random Forest data structure containing the original trees.
/// * `strategy` - The early stopping strategy to use during the merge process.
///
/// # Returns
/// A tuple: `(Arena, NodeId, BenchmarkStats, usize)` containing the arena with the 
/// compiled tree, the root node ID, performance stats, and the tree height.
pub fn transform_rf(rf_data: &RandomForestData, strategy: EarlyStoppingStrategy) -> (Arena, NodeId, BenchmarkStats, usize) { // Return BenchmarkStats and height

    let n_total_trees = rf_data.n_total_trees;

    let mut initial_votes = vec![0u32; NUM_CLASSES];
    // Initial bounds are typically non-restrictive for the k=0 case
    let mut bounds_map: BoundsMap = vec![(f64::NEG_INFINITY, f64::INFINITY); NUM_FEATURES];

    // --- Arena for holding all nodes ---
    let mut arena = Arena::new(); // Single arena for reading and writing nodes

    // --- Stats Counters (Conditional) ---
    #[cfg(feature = "detailed-stats")]
    let mut simplify_count = 0; // Nodes pruned during initial interning (step 2)
    let mut kept_count = 0; // Nodes (including leaves) kept across all phases ("no simplification")
    let mut merge_count = 0; // Nodes pruned during the merge (step 4)

    // --- Step 2: Intern original trees into the Arena ---
    // This converts array format to Arena format and performs initial hash-consing.
    let root_ids: Vec<NodeId> = (0..n_total_trees)
        .map(|tree_idx| {
            simplify_single_tree(
                &mut arena,
                &rf_data.features[tree_idx],
                &rf_data.thresholds[tree_idx],
                &rf_data.thresholds_bits[tree_idx],
                &rf_data.children_left[tree_idx],
                &rf_data.children_right[tree_idx],
                &rf_data.leaf_classes[tree_idx],
                &mut bounds_map, // Non-restricting bounds
                #[cfg(feature = "detailed-stats")]
                &mut simplify_count,
                #[cfg(feature = "detailed-stats")]
                &mut kept_count,
            )
        })
        .collect();

    // --- ORD's setup for step 4: Handle ORD strategy's first tree selection ---
    let ord_first_tree_info = if strategy == EarlyStoppingStrategy::ORD && n_total_trees > 0 {
        let mut initial_remaining_indices: Vec<usize> = (0..n_total_trees).collect();
        let mut min_leaves = usize::MAX;
        let mut first_tree_idx_in_root_ids = usize::MAX;
        let mut first_pos_in_remaining = usize::MAX;

        // Find first tree using count_reachable_leaves
        // Create a temporary provider to call count_reachable_leaves
        let temp_provider = SingleArenaProvider(&mut arena);
        for (pos, &tree_idx) in initial_remaining_indices.iter().enumerate() {
            let current_root_id = root_ids[tree_idx];
            let reachable_leaves = count_reachable_leaves(&temp_provider, current_root_id, &mut bounds_map);

            if reachable_leaves < min_leaves {
                min_leaves = reachable_leaves;
                first_tree_idx_in_root_ids = tree_idx;
                first_pos_in_remaining = pos;
            }
        }
        drop(temp_provider);

        if first_pos_in_remaining == usize::MAX {
            panic!("ORD (k=0): Could not find first tree index.");
        }
        // Remove the first tree index from the list that will be passed to the builder
        initial_remaining_indices.remove(first_pos_in_remaining);
        Some((first_tree_idx_in_root_ids, root_ids[first_tree_idx_in_root_ids], initial_remaining_indices))
    } else {
        None
    };

    // --- Step 3: Create the TreeBuilder with a SingleArenaProvider ---
    let provider = SingleArenaProvider(&mut arena);
    let mut builder = TreeBuilder::new(provider, &root_ids);

    // --- Step 4: Execute Merge ---
    let merge_start = Instant::now();
    let start_node_id = if n_total_trees > 0 { root_ids[0] } else { builder.provider.intern_leaf(0) };

    // Merge according to the strategy
    let root_id = match strategy {
        EarlyStoppingStrategy::Standard => builder._build_merged_tree_recursive_standard(
            0,
            start_node_id,
            &mut initial_votes,
            &mut bounds_map,
            #[cfg(feature = "detailed-stats")]
            &mut merge_count,
            #[cfg(feature = "detailed-stats")]
            &mut kept_count,
        ),
        EarlyStoppingStrategy::Abstract => builder._build_merged_tree_recursive_abstract(
            0,
            start_node_id,
            &mut initial_votes,
            &mut bounds_map,
        ),
        EarlyStoppingStrategy::HEUR => builder._build_merged_tree_recursive_heur(
            0,
            start_node_id,
            &mut initial_votes,
            &mut bounds_map,
        ),
        EarlyStoppingStrategy::ORD => {
            if n_total_trees == 0 {
                 builder.handle_leaf_transition_ord(&mut Vec::new(), &mut initial_votes, &mut bounds_map)
            } else {
                let (first_tree_idx, first_node_id, mut remaining_indices) = ord_first_tree_info.expect("ORD info should be present for n_total_trees > 0");
                builder._build_merged_tree_recursive_ord(
                    first_tree_idx, first_node_id, &mut remaining_indices,
                    &mut initial_votes, &mut bounds_map,
                )
            }
        }
    };
    let merge_duration = merge_start.elapsed();

    // --- Step 5: Calculate Height ---
    let height = calculate_tree_height(&arena, root_id);

    // Conditionally print detailed stats
    #[cfg(feature = "detailed-stats")]
    {
        println!("[INFO] Total simplified nodes (k=0): {}", simplify_count);
        println!("[INFO] Total pruned nodes (k=0): {}", merge_count);
    }

    // --- Step 6: Collate Stats ---
    let stats = BenchmarkStats {
        partition_processing_duration: merge_duration, // Duration of the merge process
        #[cfg(feature = "detailed-stats")]
        simplify_pruning_count: simplify_count, // "partition simplification" count
        #[cfg(feature = "detailed-stats")]
        merge_pruning_count: merge_count, // "merge simplification" count
        #[cfg(feature = "detailed-stats")]
        kept_count: kept_count, // "no simplification" count
        ..Default::default() // Other durations are zero
    };

    (arena, root_id, stats, height) // return arena by value
}


// --- Partitioned Transformation (k > 0) ---

/// Core recursive function for the partitioned RF-to-DT transformation (k > 0).
/// Implements the "Divide, Simplify, Recurse, Assemble" strategy from the paper.
/// Manages arena ownership across recursive calls.
///
/// # Arguments
/// * `k` - The remaining partitioning depth.
/// * `current_arena` - The `Arena` containing the simplified tree nodes *for this partition*. Ownership is taken.
/// * `current_root_ids` - Slice of root `NodeId`s within `current_arena` for this partition.
/// * `global_sorted_thresholds` - Read-only map of feature -> sorted unique thresholds across the entire RF.
/// * `threshold_bounds` - Tracks the relevant *range* of indices within `global_sorted_thresholds` for the current partition. Used by the split heuristic.
/// * `bounds_map` - The current feature value bounds `(min, max)` defining this partition.
/// * `strategy` - The early stopping strategy to use in the base case (k=0).
///
/// # Returns
/// A tuple: `(Arena, NodeId, BenchmarkStats)` containing the *owned* Arena holding the 
/// compiled result for this partition, the root NodeId within that Arena, and the 
/// benchmark stats accumulated within this branch of the recursion.
fn _transform_rf_partitioned_recursive(
    k: usize,
    current_arena: Arena, // Takes ownership of the arena for this level
    current_root_ids: &[NodeId], // Root IDs within current_arena
    global_sorted_thresholds: &HashMap<usize, Vec<f64>>,
    threshold_bounds: PartitionThresholdBounds,
    bounds_map: &BoundsMap, // Bounds relevant to current_arena
    strategy: EarlyStoppingStrategy,
) -> (Arena, NodeId, BenchmarkStats) { // Returns owned Arena

    // Clone bounds for local use/modification
    let mut bounds_map = bounds_map.clone();

    // --- Base Case: k = 0 ---
    // No further partitioning. Compile the trees in current_arena sequentially.
    if k == 0 {
        // This is the base case for the partitioning approach.
        // It's subtly different from the baseline k = 0 (without using partitioning),
        // because the base case will be reached for each partition (i.e., 2^k times, with k > 0).

        // Here, we don't do "partition simplification", since we assume the trees are already simplified.
        // We don't simply call transform_rf (which handles the baseline k = 0) because transform_rf 
        // simplifies the trees again.

        // For instance, bounds_map is crucial for each partition when k reaches 0,
        // while transform_rf starts with unrestricted bounds.

        // Nonetheless, both call _build_merged_tree_recursive_* to start the merge process,
        // so the merging logic RF -> DT is the same.

        // Also, note that the code here (current k = 0) only executes for the partitions
        // (i.e., when initial k > 0).
        // When running a benchmark for initial k = 0 (baseline), transform_rf_partitioned calls
        // transform_rf directly so that the original RF is (potentially) simplified before merging.
        let part_proc_start = Instant::now();
        let mut merge_count = 0; // Base case "merge simplification" counter
        let mut kept_count = 0; // Base case "no simplification" counter

        // Create a *new* arena to store the merged result of this base case partition.
        let mut final_local_arena = Arena::new(); // Arena to write final merged result
        // Use a DualArenaProvider: read from the input `current_arena`, write to `final_local_arena`.
        let provider = DualArenaProvider { read: &current_arena, write: &mut final_local_arena };
        
        // --- Handle ORD strategy pre-calculation for base case ---
        let n_total_trees = current_root_ids.len(); // Number of trees in this partition
        let ord_first_tree_info = if strategy == EarlyStoppingStrategy::ORD && n_total_trees > 0 {
            let mut initial_remaining_indices: Vec<usize> = (0..n_total_trees).collect();
            let mut min_leaves: usize = usize::MAX;
            let mut first_tree_idx_in_root_ids = usize::MAX;
            let mut first_pos_in_remaining = usize::MAX;

            for (pos, &tree_idx) in initial_remaining_indices.iter().enumerate() {
                let current_root_id = current_root_ids[tree_idx];
                let reachable_leaves = count_reachable_leaves(&provider, current_root_id, &mut bounds_map);

                if reachable_leaves < min_leaves {
                    min_leaves = reachable_leaves;
                    first_tree_idx_in_root_ids = tree_idx;
                    first_pos_in_remaining = pos;
                }
            }

            if first_pos_in_remaining == usize::MAX {
                panic!("ORD (k=0): Could not find first tree index.");
            }
            initial_remaining_indices.remove(first_pos_in_remaining);
            Some((first_tree_idx_in_root_ids, current_root_ids[first_tree_idx_in_root_ids], initial_remaining_indices))
        } else {
            None
        };

        // Create the TreeBuilder for the base case merge.
        let mut builder = TreeBuilder::new(provider, &current_root_ids);

        let mut initial_votes = vec![0u32; NUM_CLASSES];
        let start_node_id = if !current_root_ids.is_empty() { current_root_ids[0] } else { builder.provider.intern_leaf(0) };

        // Perform the merge using the appropriate TreeBuilder recursive function.
        let merged_root_id =
            match strategy {
                EarlyStoppingStrategy::Standard => builder._build_merged_tree_recursive_standard(
                    0, start_node_id, &mut initial_votes, &mut bounds_map,
                    #[cfg(feature = "detailed-stats")]
                    &mut merge_count,
                    #[cfg(feature = "detailed-stats")]
                    &mut kept_count,
                ),
                EarlyStoppingStrategy::Abstract => builder._build_merged_tree_recursive_abstract(
                    0, start_node_id, &mut initial_votes, &mut bounds_map,
                ),
                EarlyStoppingStrategy::HEUR => builder._build_merged_tree_recursive_heur(
                    0, start_node_id, &mut initial_votes, &mut bounds_map,
                ),
                EarlyStoppingStrategy::ORD => {
                    let (first_tree_idx_in_part, first_node_id, mut remaining_indices) = ord_first_tree_info.expect("ORD info missing");
                    builder._build_merged_tree_recursive_ord(
                        first_tree_idx_in_part, first_node_id, &mut remaining_indices,
                        &mut initial_votes, &mut bounds_map,
                    )
                }
            };

        let part_proc_duration = part_proc_start.elapsed();
        let stats = BenchmarkStats {
            partition_processing_duration: part_proc_duration,
            #[cfg(feature = "detailed-stats")]
            merge_pruning_count: merge_count,
            #[cfg(feature = "detailed-stats")]
            kept_count: kept_count,
            ..Default::default()
        };

        // Return the newly created arena containing the final merged result for this partition.
        // `current_arena` goes out of scope and is dropped here.
        return (final_local_arena, merged_root_id, stats);
    }

    // --- Recursive Step: k > 0 ---
    let mut current_stats = BenchmarkStats::default();

    // 1. Find Split Feature/Threshold (Heuristic: max remaining thresholds)
    let template_bounds_start = Instant::now();
    let mut best_feature = None;
    let mut max_threshold_count = 0;
    for (feature_idx, &(start, end)) in threshold_bounds.iter().enumerate() {
        let current_threshold_count = end.saturating_sub(start);
        if current_threshold_count > 0 {
            if current_threshold_count > max_threshold_count {
                max_threshold_count = current_threshold_count;
                best_feature = Some(feature_idx);
            }
        }
    }

    // Handle case where no split is possible (e.g., no more thresholds to split on)
    if best_feature.is_none() {
        // Treat as a base case: call self recursively with k=0.
        // Pass ownership of current_arena down.
        println!("[WARN] k={}: No features with remaining thresholds found. Treating as base case.", k);
        current_stats.template_bounds_duration += template_bounds_start.elapsed();
        let (arena, node_id, base_case_stats) = _transform_rf_partitioned_recursive(0, current_arena, current_root_ids, global_sorted_thresholds, threshold_bounds, &bounds_map, strategy);
        return (arena, node_id, current_stats + base_case_stats); // Combine stats
    }
    let feature_to_split = best_feature.unwrap();

    // 2. Prepare Bounds for Child Partitions
    // Calculate left_bounds_map, left_threshold_bounds, right_bounds_map, right_threshold_bounds
    let (start, end) = threshold_bounds[feature_to_split];
    let threshold_slice = &global_sorted_thresholds[&feature_to_split][start..end];
    let median_val = threshold_slice[threshold_slice.len() / 2];
    let first_median_slice_idx = threshold_slice.partition_point(|&x| x < median_val);
    let end_median_slice_idx = threshold_slice.partition_point(|&x| x <= median_val);

    let global_left_end_idx = start + first_median_slice_idx;
    let global_right_start_idx = start + end_median_slice_idx;
    current_stats.template_bounds_duration += template_bounds_start.elapsed();

    let mut left_bounds_map = bounds_map.clone();
    let mut left_threshold_bounds = threshold_bounds.clone();
    left_bounds_map[feature_to_split].1 = left_bounds_map[feature_to_split].1.min(median_val);
    left_threshold_bounds[feature_to_split].1 = global_left_end_idx;

    let mut right_bounds_map = bounds_map;
    let mut right_threshold_bounds = threshold_bounds;
    right_bounds_map[feature_to_split].0 = right_bounds_map[feature_to_split].0.max(median_val);
    right_threshold_bounds[feature_to_split].0 = global_right_start_idx;

    // 3. Simplify Trees for Child Partitions (Incremental Simplification)
    // Create new arenas for the simplified results of the left and right children.
    let simplify_start = Instant::now();
    let mut left_simplified_arena = Arena::new();
    let mut right_simplified_arena = Arena::new();
    let mut left_simplify_pruning_count = 0;
    let mut left_kept_count = 0;
    let mut right_simplify_pruning_count = 0;
    let mut right_kept_count = 0;

    // Simplify into left_simplified_arena using left_bounds_map
    let left_simplified_root_ids = resimplify_from_arena(
        &current_arena, // Read from the arena passed to this level
        current_root_ids,
        &mut left_bounds_map, // Apply left bounds
        &mut left_simplified_arena, // Write to new left arena
        #[cfg(feature = "detailed-stats")]
        &mut left_simplify_pruning_count,
        #[cfg(feature = "detailed-stats")]
        &mut left_kept_count,
    );

    // Simplify into right_simplified_arena using right_bounds_map
    let right_simplified_root_ids = resimplify_from_arena(
        &current_arena, // Read from the arena passed to this level
        current_root_ids,
        &mut right_bounds_map, // Apply right bounds
        &mut right_simplified_arena, // Write to new right arena
        #[cfg(feature = "detailed-stats")]
        &mut right_simplify_pruning_count,
        #[cfg(feature = "detailed-stats")]
        &mut right_kept_count,
    );
    let _simplify_duration = simplify_start.elapsed();
    #[cfg(feature = "detailed-stats")] // Conditionally add counts to stats
    {
        current_stats.simplify_pruning_count += left_simplify_pruning_count + right_simplify_pruning_count;
        current_stats.kept_count += left_kept_count + right_kept_count;
    }

    // current_arena is no longer needed
    let drop_start = Instant::now();
    drop(current_arena);
    current_stats.drop_duration += drop_start.elapsed();

    // 4. Recurse Left and Right (Potentially in Parallel)
    // Define closures (tasks) for the recursive calls. Crucially, these closures 
    // capture ownership of the newly created simplified arenas.
    let left_task = || {
        _transform_rf_partitioned_recursive(
            k - 1,
            left_simplified_arena, // Pass ownership
            &left_simplified_root_ids, // Pass new root IDs
            global_sorted_thresholds,
            left_threshold_bounds,
            &left_bounds_map,
            strategy,
        )
    };
    let right_task = || {
        _transform_rf_partitioned_recursive(
            k - 1,
            right_simplified_arena, // Pass ownership
            &right_simplified_root_ids, // Pass new root IDs
            global_sorted_thresholds,
            right_threshold_bounds,
            &right_bounds_map,
            strategy,
        )
    };

    // Execute tasks (sequentially or in parallel using rayon::join)
    // The results contain the owned arenas returned by the child calls.
    #[cfg(feature = "non-parallel")]
    let ((left_result_arena, left_result_root_id, left_stats), (right_result_arena, right_result_root_id, right_stats)) = (left_task(), right_task());
    #[cfg(not(feature = "non-parallel"))] // Use rayon for parallel execution
    let ((left_result_arena, left_result_root_id, left_stats), (right_result_arena, right_result_root_id, right_stats)) = rayon::join(left_task, right_task);

    // 5. Merge Stats from Children
    current_stats = current_stats + left_stats + right_stats;

    // 6. Assemble Results
    // Combine the results from the left and right children into a single tree for this level.
    let assembly_start = Instant::now();
    let mut right_copy_cache = HashMap::new(); // Cache for copying nodes from right to left arena

    // Choose one arena (e.g., left) to be the final arena for this level. Move ownership.
    let mut final_arena = left_result_arena; // Move ownership
    let final_left_id = left_result_root_id; // This ID is already valid in final_arena
    // Copy the relevant nodes from the *other* result arena (right_result_arena) into final_arena.
    // `_copy_and_intern_nodes` handles deduplication within final_arena.
    let final_right_id = _copy_and_intern_nodes(&right_result_arena, right_result_root_id, &mut final_arena, &mut right_copy_cache);

    // Intern the new split node that connects the left and right results.
    let threshold_bits = median_val.to_bits();
    let final_root_id = final_arena.intern_internal(feature_to_split, threshold_bits, final_left_id, final_right_id);
    current_stats.assembly_duration += assembly_start.elapsed();

    // 7. Cleanup: Drop Unneeded Arenas
    // Drop the result arena that was copied from `right_result_arena`.
    // `left_result_arena`'s ownership was moved to `final_arena`.
    let drop_start = Instant::now();
    drop(right_result_arena);
    // left_result_arena was moved into final_arena, so it's not dropped here
    // current_arena was dropped earlier
    current_stats.drop_duration += drop_start.elapsed();

    // Return the assembled arena (which originated from the left child and now contains merged results)
    (final_arena, final_root_id, current_stats)
}


/// Entry point for transforming a Random Forest using the partitioning strategy (k > 0).
/// Handles setup, calls the recursive helper, and returns the final result.
///
/// Steps:
/// 1. **Initial Global Simplification:** Converts all original trees into a `base_arena` 
///    using `simplify_single_tree` with non-restrictive bounds. This performs initial 
///    hash-consing.
/// 2. **Global Threshold Collection:** Collects all unique split thresholds used across 
///    the entire forest for each feature and sorts them.
/// 3. **Initialize Bounds:** Sets up the initial numerical bounds (`BoundsMap`) and 
///    threshold index bounds (`PartitionThresholdBounds`).
/// 4. **Recursive Partitioning:** Calls `_transform_rf_partitioned_recursive` to start 
///    the partitioning process, passing ownership of the `base_arena`.
/// 5. **Final Result:** Calculates the height of the final compiled tree and returns 
///    the final arena, root ID, accumulated benchmark stats, and height.
///
/// # Arguments
/// * `rf_data` - The Random Forest data.
/// * `k` - The desired partitioning depth (must be > 0, otherwise calls `transform_rf`).
/// * `strategy` - The early stopping strategy to use in the base cases (k=0 leaves).
///
/// # Returns
/// A tuple: `(Arena, NodeId, BenchmarkStats, usize)` containing the final compiled tree, 
/// its root ID, performance stats, and height.
pub fn transform_rf_partitioned(
    rf_data: &RandomForestData,
    k: usize,
    strategy: EarlyStoppingStrategy,
) -> (Arena, NodeId, BenchmarkStats, usize) {
    if k == 0 {
        // Fallback to the non-partitioned version if k=0
        return transform_rf(rf_data, strategy);
    }
    let n_total_trees = rf_data.n_total_trees;

    // --- Step 1: Initial Global Simplification ---
    // Create a base_arena containing the initial Arena representation of all trees.
    #[cfg(feature = "detailed-stats")]
    let simplify_start = Instant::now();
    let mut initial_base_arena = Arena::new();
    let mut initial_bounds_global: BoundsMap = vec![(f64::NEG_INFINITY, f64::INFINITY); NUM_FEATURES];
    #[cfg(feature = "detailed-stats")]
    let mut global_simplify_count = 0;
    #[cfg(feature = "detailed-stats")]
    let mut kept_count = 0;
    let initial_base_root_ids: Vec<NodeId> = (0..n_total_trees)
        .map(|tree_idx| {
            simplify_single_tree(
                &mut initial_base_arena,
                &rf_data.features[tree_idx],
                &rf_data.thresholds[tree_idx],
                &rf_data.thresholds_bits[tree_idx],
                &rf_data.children_left[tree_idx],
                &rf_data.children_right[tree_idx],
                &rf_data.leaf_classes[tree_idx],
                &mut initial_bounds_global,
                #[cfg(feature = "detailed-stats")]
                &mut global_simplify_count,
                #[cfg(feature = "detailed-stats")]
                &mut kept_count,
            )
        })
        .collect();
    
    #[cfg(feature = "detailed-stats")]
    {
        // Log initial simplification stats...
        let initial_simplify_duration = simplify_start.elapsed();
        println!("[INFO] Initial global simplification done in {:.3} s, pruned {} nodes.", initial_simplify_duration.as_secs_f64(), global_simplify_count);
    }


    // --- Step 2: Collect and Sort Global Thresholds ---
    let mut global_sorted_thresholds = HashMap::new();
    for tree_idx in 0..rf_data.n_total_trees {
        for node_idx in 0..rf_data.features[tree_idx].len() {
            let feature_idx_signed = rf_data.features[tree_idx][node_idx];
            if feature_idx_signed >= 0 {
                let feature_idx = feature_idx_signed as usize;
                let threshold = rf_data.thresholds[tree_idx][node_idx];
                global_sorted_thresholds.entry(feature_idx).or_insert_with(Vec::new).push(threshold);
            }
        }
    }
    for thresholds in global_sorted_thresholds.values_mut() {
        thresholds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    }


    // --- Step 3: Initialize Bounds for Top-Level Call ---
    let initial_bounds: BoundsMap = vec![(f64::NEG_INFINITY, f64::INFINITY); NUM_FEATURES];
    let mut initial_indices: PartitionThresholdBounds = vec![(0, 0); NUM_FEATURES];
    for (&feature_idx, thresholds) in &global_sorted_thresholds {
        if feature_idx < NUM_FEATURES {
            initial_indices[feature_idx] = (0, thresholds.len());
        }
    }


    // --- Step 4: Start Recursive Partitioning ---
    // Pass ownership of the initial_base_arena to the recursive function.
    let (
        final_arena, 
        final_root_id, 
        mut stats
    ) = _transform_rf_partitioned_recursive(
        k,
        initial_base_arena, // Pass ownership
        &initial_base_root_ids,
        &global_sorted_thresholds,
        initial_indices,
        &initial_bounds,
        strategy,
    );

    // --- Step 5: Calculate Height & Finalize Stats ---
    let height = calculate_tree_height(&final_arena, final_root_id);

    // Add stats from the initial global simplification step if tracking details
    #[cfg(feature = "detailed-stats")]
    {
        // Note: The initial simplification is counted as a "partition simplification".
        stats.simplify_pruning_count += global_simplify_count;
        stats.kept_count += kept_count;

        println!("[INFO] Total node checks optimized during ALL SIMPLIFY phases (initial + recursive): {}", stats.simplify_pruning_count);
        println!("[INFO] Total node checks optimized during BASE CASE MERGE phase (k=0): {}", stats.merge_pruning_count);
        println!("[INFO] Total node checks kept during all phases: {}", stats.kept_count);
    }

    (
        final_arena,
        final_root_id,
        stats,
        height,
    )
}
