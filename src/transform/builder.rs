//! Implements the core recursive logic for compiling a sequence of decision trees 
//! from a Random Forest into a single equivalent decision tree. This involves 
//! traversing the trees, accumulating votes, applying path condition simplification, 
//! and utilizing various early stopping strategies (ES, AbsES, HEUR, ORD) as 
//! described in the referenced paper [17] and its enhancement (this work).
//! 
//! This module has performance critical code. Safety checks are NOT allowed!
//! 
use crate::tree::{Arena, NodeId, ConditionStatus, FEATURE_LEAF_SENTINEL}; // Import FEATURE_LEAF_SENTINEL
use crate::utils::{_FAST_get_majority_class, _FAST_get_top_two_votes, _FAST_get_max_vote}; // Import the specific function
use super::common::{BoundsMap, check_condition_bounds};
use super::early_stopping::{abstract_early_stopping, abstract_early_stopping_ord, count_reachable_leaves};

// --- Arena Provider Trait ---

/// Abstracts the access to tree node storage (`Arena`).
/// This allows the tree building logic to work seamlessly whether it's reading 
/// from and writing to the same arena (common during the main merge) or reading 
/// from one arena (containing simplified input trees) and writing to a new one 
/// (used in partitioning/assembly steps).
/// The lifetime 'a ensures the provider and the arenas it references are valid 
/// for the duration of the build process segment they are used in.
pub trait ArenaProvider<'a> {
    /// Provides immutable access to the arena containing the node structures 
    /// of the trees being processed (read-only).
    fn read_arena(&self) -> &Arena;

    /// Provides mutable access to the arena where the resulting merged/compiled 
    /// tree nodes should be stored (interned). Mutable access is required for 
    /// node interning (deduplication).
    fn write_arena(&mut self) -> &mut Arena;

    // Convenience methods to access node data via the provider
    fn is_leaf(&self, id: NodeId) -> bool { self.read_arena().is_leaf(id) }
    fn get_feature_raw(&self, id: NodeId) -> i64 { self.read_arena().get_feature_raw(id) }
    fn get_feature_usize(&self, id: NodeId) -> usize { self.read_arena().get_feature_usize(id) }
    fn get_threshold(&self, id: NodeId) -> f64 { self.read_arena().get_threshold(id) }
    fn get_threshold_bits(&self, id: NodeId) -> u64 { self.read_arena().get_threshold_bits(id) }
    fn get_true_id(&self, id: NodeId) -> NodeId { self.read_arena().get_true_id(id) }
    fn get_false_id(&self, id: NodeId) -> NodeId { self.read_arena().get_false_id(id) }
    fn get_leaf_class(&self, id: NodeId) -> usize { self.read_arena().get_leaf_class(id) }

    /// Interns a leaf node with the given class label into the write arena.
    /// Returns the unique NodeId for this leaf.
    fn intern_leaf(&mut self, class: usize) -> NodeId { self.write_arena().intern_leaf(class) }

    /// Interns an internal node with the given properties into the write arena.
    /// Checks for existing identical nodes to ensure deduplication (hash-consing).
    /// Returns the unique NodeId for this internal node configuration.
    fn intern_internal(&mut self, feature: usize, threshold_bits: u64, true_id: NodeId, false_id: NodeId) -> NodeId {
        self.write_arena().intern_internal(feature, threshold_bits, true_id, false_id)
    }
}

/// An `ArenaProvider` implementation where the read and write operations 
/// target the same underlying `Arena`.
pub struct SingleArenaProvider<'a>(pub &'a mut Arena);

impl<'a> ArenaProvider<'a> for SingleArenaProvider<'a> {
    fn read_arena(&self) -> &Arena {
        self.0
    }
    fn write_arena(&mut self) -> &mut Arena {
        self.0
    }
}

/// An `ArenaProvider` implementation that reads from one `Arena` and writes 
/// (interns) results into a different `Arena`. This might be used when 
/// simplifying input trees before the main merge, storing the simplified 
/// versions in a separate arena.
pub struct DualArenaProvider<'a> {
    pub read: &'a Arena,
    pub write: &'a mut Arena,
}

impl<'a> ArenaProvider<'a> for DualArenaProvider<'a> {
    fn read_arena(&self) -> &Arena {
        self.read
    }
    fn write_arena(&mut self) -> &mut Arena {
        self.write
    }
}


// --- Generic Tree Builder ---

/// Orchestrates the recursive compilation (merging) of a sequence of decision trees.
/// It holds references to the input tree roots, manages access to the node arenas 
/// via an `ArenaProvider`, and contains parameters relevant to the merging process, 
/// particularly for early stopping heuristics.
/// 
/// Generic over `P: ArenaProvider<'a>` to support different arena configurations.
pub struct TreeBuilder<'a, P: ArenaProvider<'a>> {
    /// Provides access to the read and write arenas.
    pub provider: P, 
    /// Slice containing the root `NodeId`s of the sequence of trees to be merged.
    root_ids: &'a [NodeId],
    /// Precomputed index marking the halfway point in the sequence of trees. 
    /// Used by the early stopping heuristics.
    halfway_point: usize,
    /// The total number of trees in the input sequence (`root_ids.len()`).
    /// Precomputed for avoiding repeated calls to `len()` during the merge process.
    n_total_trees: usize,
}

impl<'a, P: ArenaProvider<'a>> TreeBuilder<'a, P> {

    /// Creates a new `TreeBuilder`.
    ///
    /// # Arguments
    /// * `provider` - An instance implementing `ArenaProvider` to manage node storage.
    /// * `root_ids` - A slice containing the root `NodeId`s of the trees to be merged, in order.
    pub fn new(provider: P, root_ids: &'a [NodeId]) -> Self {
        let n_total_trees = root_ids.len();
        TreeBuilder {
            provider,
            root_ids,
            halfway_point: n_total_trees / 2,
            n_total_trees,
        }
    }

    // --- Leaf Transition Handlers ---
    // The methods below (handle_leaf_transition_*) are invoked when the recursive traversal reaches
    // a leaf node in the *current* tree being processed (identified by `tree_index` or implicitly 
    // in ORD). Their primary role is to decide whether to stop the merge process 
    // early based on the accumulated votes and the specific early stopping strategy, 
    // or to continue the recursion with the next tree(s) in the sequence.
    // They are performance-critical.

    /// Handles the transition after reaching a leaf node using the Standard Early Stopping (ES) strategy.
    /// ES checks if the current majority class is guaranteed to win even if all 
    /// remaining trees vote for the second-best class. This check is typically 
    /// only performed after processing half the trees (`halfway_point`) for efficiency.
    ///
    /// **Performance Critical:** Optimized for speed, avoids unnecessary (safety) checks.
    ///
    /// # Arguments
    /// * `tree_index` - Index of the tree whose leaf was just reached.
    /// * `current_votes` - Mutable slice tracking votes accumulated along the current path.
    /// * `bounds_map` - Feature bounds derived from the path taken so far.
    /// * `num_pruned_nodes`/`num_kept_nodes` - Counters for detailed statistics (conditional compilation).
    ///
    /// # Returns
    /// `NodeId` of the resulting leaf (if stopped early) or the root of the 
    /// subtree merged from the remaining trees.
    pub fn handle_leaf_transition_standard(
        &mut self,
        tree_index: usize,
        current_votes: &mut [u32],
        bounds_map: &mut BoundsMap,
        #[cfg(feature = "detailed-stats")]
        num_pruned_nodes: &mut usize,
        #[cfg(feature = "detailed-stats")]
        num_kept_nodes: &mut usize,
    ) -> NodeId {
        let tree_index_plus_1 = tree_index + 1;
        // Check if all trees processed
        if tree_index_plus_1 >= self.n_total_trees {
            return self.provider.intern_leaf(_FAST_get_majority_class(current_votes));
        }
        // Perform standard early stopping check (after halfway point)
        if tree_index_plus_1 >= self.halfway_point {
            let remaining_trees = self.n_total_trees - tree_index_plus_1;
            let (winning_class, max_votes, second_max_votes) = _FAST_get_top_two_votes(current_votes);
            let second_max_potential = second_max_votes + remaining_trees as u32;
            if max_votes > second_max_potential || (max_votes == second_max_potential && winning_class == 0) {
                return self.provider.intern_leaf(winning_class);
            }
        }
        // If not stopped, recursively call _build_merged_tree_recursive_standard for the next tree
        let next_root = self.root_ids[tree_index_plus_1];
        self._build_merged_tree_recursive_standard(
            tree_index_plus_1, next_root, current_votes, bounds_map,
            #[cfg(feature = "detailed-stats")]
            num_pruned_nodes,
            #[cfg(feature = "detailed-stats")]
            num_kept_nodes,
        )
    }

    /// Handles the transition after reaching a leaf node using the Abstract Early Stopping (AbsES) strategy.
    /// AbsES uses Abstract Interpretation (`abstract_early_stopping` function) to 
    /// estimate the *guaranteed* minimum votes each class can receive from the 
    /// remaining unprocessed trees, given the current path constraints (`bounds_map`).
    /// This allows for potentially earlier stopping than the Standard ES strategy.
    ///
    /// **Performance Critical:** Optimized for speed, avoids unnecessary (safety) checks.
    ///
    /// # Arguments
    /// * `tree_index` - Index of the tree whose leaf was just reached.
    /// * `current_votes` - Mutable slice tracking votes accumulated along the current path.
    /// * `bounds_map` - Feature bounds derived from the path taken so far.
    ///
    /// # Returns
    /// `NodeId` of the resulting leaf (if stopped early via abstract interpretation) 
    /// or the root of the subtree merged from the remaining trees.
    pub fn handle_leaf_transition_abstract(
        &mut self,
        tree_index: usize,
        current_votes: &mut [u32],
        bounds_map: &mut BoundsMap,
    ) -> NodeId {
        let tree_index_plus_1 = tree_index + 1;
        // Check if all trees processed
        if tree_index_plus_1 >= self.n_total_trees {
            return self.provider.intern_leaf(_FAST_get_majority_class(current_votes));
        }
        // Perform abstract early stopping check
        if let Some(early_stop_result) = abstract_early_stopping(tree_index, current_votes, bounds_map, self.root_ids, &mut self.provider) {
            return early_stop_result;
        }
        // If not stopped, recursively call _build_merged_tree_recursive_abstract for the next tree
        let next_root = self.root_ids[tree_index_plus_1];
        self._build_merged_tree_recursive_abstract(
            tree_index_plus_1, next_root, current_votes, bounds_map,
        )
    }

    /// Handles the transition after reaching a leaf node using the Heuristic Early Stopping (HEUR) strategy.
    /// HEUR aims for a balance between the speed of Standard ES and the power of AbsES. 
    /// It applies the cheaper Standard ES check after the `halfway_point`. Before that, 
    /// it uses a heuristic (e.g., based on the current vote margin) to decide whether 
    /// to invoke the more expensive AbsES check.
    ///
    /// **Performance Critical:** Optimized for speed, avoids unnecessary (safety) checks.
    ///
    /// # Arguments
    /// * `tree_index` - Index of the tree whose leaf was just reached.
    /// * `current_votes` - Mutable slice tracking votes accumulated along the current path.
    /// * `bounds_map` - Feature bounds derived from the path taken so far.
    ///
    /// # Returns
    /// `NodeId` of the resulting leaf (if stopped early by either standard or abstract check)
    /// or the root of the subtree merged from the remaining trees.
    pub fn handle_leaf_transition_heur(
        &mut self,
        tree_index: usize,
        current_votes: &mut [u32],
        bounds_map: &mut BoundsMap,
    ) -> NodeId {
        let tree_index_plus_1 = tree_index + 1;
        // Check if all trees processed
        if tree_index_plus_1 >= self.n_total_trees {
            return self.provider.intern_leaf(_FAST_get_majority_class(current_votes));
        }
        // Perform standard check (if after halfway point)
        if tree_index_plus_1 >= self.halfway_point {
            let remaining_trees = self.n_total_trees - tree_index_plus_1;
            let (winning_class, max_votes, second_max_votes) = _FAST_get_top_two_votes(current_votes);
            let second_max_potential = second_max_votes + remaining_trees as u32;
            if max_votes > second_max_potential || (max_votes == second_max_potential && winning_class == 0) {
                return self.provider.intern_leaf(winning_class);
            }
        } else {
            // Perform heuristic check + potential abstract check (if before halfway point)
            // Currently winning class must have received at least 70% of the possible votes so far
            let max_votes = _FAST_get_max_vote(current_votes);
            if 10 * (max_votes as u64) >= 7 * (tree_index_plus_1 as u64) {
                if let Some(early_stop_node_id) = abstract_early_stopping(tree_index, current_votes, bounds_map, self.root_ids, &mut self.provider) {
                    return early_stop_node_id;
                }
            }
        }
        // If not stopped, recursively call _build_merged_tree_recursive_heur for the next tree
        let next_root = self.root_ids[tree_index_plus_1];
        self._build_merged_tree_recursive_heur(
            tree_index_plus_1, next_root, current_votes, bounds_map,
        )
    }

    /// Handles the transition after reaching a leaf node using the Ordered Early Stopping (ORD) strategy.
    /// ORD extends HEUR by dynamically selecting the *next* tree to process from the 
    /// remaining pool (`remaining_indices`). It prioritizes trees estimated to be 
    /// "simpler" under the current path constraints (`bounds_map`), often measured 
    /// by the number of reachable leaves (`count_reachable_leaves`). The goal is to 
    /// potentially prune branches earlier by processing simpler, more constrained trees first.
    /// It uses the same early stopping logic (Standard + heuristic Abstract) as HEUR.
    ///
    /// **Performance Critical:** Optimized for speed, avoids unnecessary (safety) checks.
    ///
    /// # Arguments
    /// * `remaining_indices` - Mutable vector of indices into `self.root_ids` for trees yet to be processed. Modified during tree selection.
    /// * `current_votes` - Mutable slice tracking votes accumulated along the current path.
    /// * `bounds_map` - Feature bounds derived from the path taken so far.
    ///
    /// # Returns
    /// `NodeId` of the resulting leaf (if stopped early) or the root of the subtree 
    /// merged from the remaining trees (processed in the dynamically determined order).
    pub fn handle_leaf_transition_ord(
        &mut self,
        remaining_indices: &mut Vec<usize>,
        current_votes: &mut [u32],
        bounds_map: &mut BoundsMap,
    ) -> NodeId {
        let remaining_count = remaining_indices.len();
        // Check if all trees processed
        if remaining_count == 0 {
            return self.provider.intern_leaf(_FAST_get_majority_class(current_votes));
        }

        // Perform HEUR-like early stopping checks (standard + abstract)
        let trees_processed_count = self.n_total_trees - remaining_count;
        if trees_processed_count >= self.halfway_point { // Standard check
            let (winning_class, max_votes, second_max_votes) = _FAST_get_top_two_votes(current_votes);
            let second_max_potential = second_max_votes + remaining_count as u32;
            if max_votes > second_max_potential || (max_votes == second_max_potential && winning_class == 0) {
                return self.provider.intern_leaf(winning_class);
            }
        } else { // Abstract check (heuristic)
            let max_votes = _FAST_get_max_vote(current_votes);
            // Currently winning class must have received at least 70% of the possible votes so far
            if 10 * (max_votes as u64) >= 7 * (trees_processed_count as u64) {
                if let Some(early_stop_node_id) = abstract_early_stopping_ord(current_votes, bounds_map, self.root_ids, remaining_indices, &mut self.provider) {
                    return early_stop_node_id;
                }
            }
        }

        // If not stopped:
        //  1. Find the best next tree based on reachable leaves heuristic
        //  2. Remove the chosen tree's index from remaining_indices
        //  3. Recursively call _build_merged_tree_recursive_ord for the selected tree
        //  4. Restore remaining_indices (insert back the index) for backtracking
        let mut min_leaves = usize::MAX;
        let mut best_next_tree_idx_in_root_ids = usize::MAX;
        let mut best_pos_in_remaining = usize::MAX;
        
        // Step 1: Find the best next tree based on reachable leaves heuristic
        for (pos, &tree_idx_in_root_ids) in remaining_indices.iter().enumerate() {
            let root_node_id = self.root_ids[tree_idx_in_root_ids];
            let reachable_leaves = count_reachable_leaves(&self.provider, root_node_id, bounds_map);

            if reachable_leaves < min_leaves || (reachable_leaves == min_leaves && tree_idx_in_root_ids < best_next_tree_idx_in_root_ids) {
                min_leaves = reachable_leaves;
                best_next_tree_idx_in_root_ids = tree_idx_in_root_ids;
                best_pos_in_remaining = pos;
            }
        }
        if best_pos_in_remaining == usize::MAX {
            return self.provider.intern_leaf(_FAST_get_majority_class(current_votes));
        }

        // Step 2: Remove the chosen tree's index from remaining_indices
        let next_tree_idx = remaining_indices.remove(best_pos_in_remaining);

        // Step 3: Recursively call _build_merged_tree_recursive_ord for the selected tree
        let next_root = self.root_ids[next_tree_idx];
        let result_node_id = self._build_merged_tree_recursive_ord(
            next_tree_idx, next_root, remaining_indices, current_votes, bounds_map,
        );

        // Step 4: Restore remaining_indices (insert back the index) for backtracking
        remaining_indices.insert(best_pos_in_remaining, next_tree_idx);
        result_node_id
    }


    // --- Recursive Build Helpers ---
    // The methods below (_build_merged_tree_recursive_*) are the core recursive functions for each strategy.
    // They traverse a single tree (`node_id`) from the input sequence (`root_ids`). Path conditions are 
    // propagated via `bounds_map` to prune unreachable branches (redundant predicate 
    // elimination). When a leaf of the current tree is reached, the corresponding 
    // `handle_leaf_transition_*` function is called to decide the next step.
    // They are performance-critical.

    /// Recursively builds the merged tree using Standard Early Stopping (ES).
    /// Traverses the tree specified by `tree_index` and `node_id`.
    ///
    /// **Performance Critical:** Optimized for speed, avoids unnecessary (safety) checks.
    ///
    /// # Arguments
    /// * `tree_index` - Index in `self.root_ids` of the tree currently being processed.
    /// * `node_id` - The current node being visited in the `tree_index`-th tree.
    /// * `current_votes` - Accumulated votes for the path taken so far.
    /// * `bounds_map` - Current path constraints (feature bounds).
    /// * `num_pruned_nodes`/`num_kept_nodes` - Stats counters (conditional compilation).
    ///
    /// # Returns
    /// `NodeId` of the compiled subtree rooted at this point in the recursion.
    pub(crate) fn _build_merged_tree_recursive_standard(
        &mut self,
        tree_index: usize,
        node_id: NodeId,
        current_votes: &mut [u32],
        bounds_map: &mut BoundsMap,
        #[cfg(feature = "detailed-stats")]
        num_pruned_nodes: &mut usize,
        #[cfg(feature = "detailed-stats")]
        num_kept_nodes: &mut usize,
    ) -> NodeId {
        let is_leaf = self.provider.is_leaf(node_id);

        let computed_node_id: NodeId = if is_leaf {
            // Base Case: Leaf node ->
            //  update votes, call handle_leaf_transition_standard, revert votes.

            #[cfg(feature = "detailed-stats")]
            { *num_kept_nodes += 1; }

            let leaf_class = self.provider.get_leaf_class(node_id);
            current_votes[leaf_class] += 1;
            let next_node_result = self.handle_leaf_transition_standard(
                tree_index, current_votes, bounds_map,
                #[cfg(feature = "detailed-stats")]    
                num_pruned_nodes,
                #[cfg(feature = "detailed-stats")]
                num_kept_nodes,
            );
            current_votes[leaf_class] -= 1;
            next_node_result
        } else {
            // Recursive Step: Internal node -> 
            //  Check condition against bounds_map (AlwaysTrue, AlwaysFalse, Undetermined)
            //  If determined, recurse on the single reachable child.
            //  If undetermined, update bounds, recurse on both children, restore bounds, intern new node.

            let feature = self.provider.get_feature_usize(node_id);
            let threshold = self.provider.get_threshold(node_id);
            let threshold_bits = self.provider.get_threshold_bits(node_id);
            let left_child = self.provider.get_true_id(node_id);
            let right_child = self.provider.get_false_id(node_id);

            let condition_status = check_condition_bounds(bounds_map, feature, threshold);

            match condition_status {
                ConditionStatus::AlwaysTrue => {
                    #[cfg(feature = "detailed-stats")]
                    { *num_pruned_nodes += 1; }
                    self._build_merged_tree_recursive_standard(
                        tree_index, left_child, current_votes, bounds_map, 
                        #[cfg(feature = "detailed-stats")]
                        num_pruned_nodes,
                        #[cfg(feature = "detailed-stats")]
                        num_kept_nodes
                    )
                }
                ConditionStatus::AlwaysFalse => {
                    #[cfg(feature = "detailed-stats")]
                    { *num_pruned_nodes += 1; }
                    self._build_merged_tree_recursive_standard(
                        tree_index, right_child, current_votes, bounds_map, 
                        #[cfg(feature = "detailed-stats")]
                        num_pruned_nodes,
                        #[cfg(feature = "detailed-stats")]
                        num_kept_nodes
                    )
                }
                ConditionStatus::Undetermined => {
                    #[cfg(feature = "detailed-stats")]
                    { *num_kept_nodes += 1; }

                    let original_bound = bounds_map[feature];
                    bounds_map[feature].1 = original_bound.1.min(threshold);
                    let true_id = self._build_merged_tree_recursive_standard(
                        tree_index, left_child, current_votes, bounds_map, 
                        #[cfg(feature = "detailed-stats")]
                        num_pruned_nodes,
                        #[cfg(feature = "detailed-stats")]
                        num_kept_nodes
                    );
                    bounds_map[feature].1 = original_bound.1;

                    bounds_map[feature].0 = original_bound.0.max(threshold);
                    let false_id = self._build_merged_tree_recursive_standard(
                        tree_index, right_child, 
                        current_votes, 
                        bounds_map, 
                        #[cfg(feature = "detailed-stats")]
                        num_pruned_nodes,
                        #[cfg(feature = "detailed-stats")]
                        num_kept_nodes
                    );
                    bounds_map[feature].0 = original_bound.0;

                    self.provider.intern_internal(feature, threshold_bits, true_id, false_id)
                }
            }
        };
        computed_node_id
    }

    /// Recursively builds the merged tree using Abstract Early Stopping (AbsES).
    /// Traverses the tree specified by `tree_index` and `node_id`.
    ///
    /// **Performance Critical:** Optimized for speed, avoids unnecessary (safety) checks.
    ///
    /// # Arguments
    /// * `tree_index` - Index in `self.root_ids` of the tree currently being processed.
    /// * `node_id` - The current node being visited in the `tree_index`-th tree.
    /// * `current_votes` - Accumulated votes for the path taken so far.
    /// * `bounds_map` - Current path constraints (feature bounds).
    ///
    /// # Returns
    /// `NodeId` of the compiled subtree rooted at this point in the recursion.
    pub(crate) fn _build_merged_tree_recursive_abstract(
        &mut self,
        tree_index: usize,
        node_id: NodeId,
        current_votes: &mut [u32],
        bounds_map: &mut BoundsMap,
    ) -> NodeId {
        let feature_idx_signed = self.provider.get_feature_raw(node_id);

        if feature_idx_signed == FEATURE_LEAF_SENTINEL {
            // Base Case: Leaf node ->
            //  update votes, call handle_leaf_transition_abstract, revert votes.
            let leaf_class = self.provider.get_leaf_class(node_id);
            current_votes[leaf_class] += 1;
            let next_node_result = self.handle_leaf_transition_abstract(tree_index, current_votes, bounds_map);
            current_votes[leaf_class] -= 1;
            return next_node_result;
        }

        // Recursive Step: Internal node -> 
        //  Check condition against bounds_map (AlwaysTrue, AlwaysFalse, Undetermined)
        //  If determined, recurse on the single reachable child.
        //  If undetermined, update bounds, recurse on both children, restore bounds, intern new node.
        let feature = self.provider.get_feature_usize(node_id);
        let threshold = self.provider.get_threshold(node_id);
        let threshold_bits = self.provider.get_threshold_bits(node_id);
        let left_child = self.provider.get_true_id(node_id);
        let right_child = self.provider.get_false_id(node_id);

        let condition_status = check_condition_bounds(bounds_map, feature, threshold);
        match condition_status {
            ConditionStatus::AlwaysTrue => self._build_merged_tree_recursive_abstract(tree_index, left_child, current_votes, bounds_map),
            ConditionStatus::AlwaysFalse => self._build_merged_tree_recursive_abstract(tree_index, right_child, current_votes, bounds_map),
            ConditionStatus::Undetermined => {
                let original_bound = bounds_map[feature];
                bounds_map[feature].1 = original_bound.1.min(threshold);
                let true_id = self._build_merged_tree_recursive_abstract(tree_index, left_child, current_votes, bounds_map);
                bounds_map[feature].1 = original_bound.1;
                bounds_map[feature].0 = original_bound.0.max(threshold);
                let false_id = self._build_merged_tree_recursive_abstract(tree_index, right_child, current_votes, bounds_map);
                bounds_map[feature].0 = original_bound.0;
                self.provider.intern_internal(feature, threshold_bits, true_id, false_id)
            }
        }
    }

    /// Recursively builds the merged tree using Heuristic Early Stopping (HEUR).
    /// Traverses the tree specified by `tree_index` and `node_id`.
    ///
    /// **Performance Critical:** Optimized for speed, avoids unnecessary (safety) checks.
    ///
    /// # Arguments
    /// * `tree_index` - Index in `self.root_ids` of the tree currently being processed.
    /// * `node_id` - The current node being visited in the `tree_index`-th tree.
    /// * `current_votes` - Accumulated votes for the path taken so far.
    /// * `bounds_map` - Current path constraints (feature bounds).
    ///
    /// # Returns
    /// `NodeId` of the compiled subtree rooted at this point in the recursion.
     pub(crate) fn _build_merged_tree_recursive_heur(
        &mut self,
        tree_index: usize,
        node_id: NodeId,
        current_votes: &mut [u32],
        bounds_map: &mut BoundsMap,
    ) -> NodeId {
        let feature_idx_signed: i64 = self.provider.get_feature_raw(node_id);

        if feature_idx_signed == FEATURE_LEAF_SENTINEL {
            // Base Case: Leaf node ->
            //  update votes, call handle_leaf_transition_heur, revert votes.
            let leaf_class = self.provider.get_leaf_class(node_id);
            current_votes[leaf_class] += 1;
            let next_node_result = self.handle_leaf_transition_heur(tree_index, current_votes, bounds_map);
            current_votes[leaf_class] -= 1;
            return next_node_result;
        }

        // Recursive Step: Internal node -> 
        //  Check condition against bounds_map (AlwaysTrue, AlwaysFalse, Undetermined)
        //  If determined, recurse on the single reachable child.
        //  If undetermined, update bounds, recurse on both children, restore bounds, intern new node.
        let feature = self.provider.get_feature_usize(node_id);
        let threshold = self.provider.get_threshold(node_id);
        let threshold_bits = self.provider.get_threshold_bits(node_id);
        let left_child = self.provider.get_true_id(node_id);
        let right_child = self.provider.get_false_id(node_id);

        let condition_status = check_condition_bounds(bounds_map, feature, threshold);
        match condition_status {
            ConditionStatus::AlwaysTrue => self._build_merged_tree_recursive_heur(tree_index, left_child, current_votes, bounds_map),
            ConditionStatus::AlwaysFalse => self._build_merged_tree_recursive_heur(tree_index, right_child, current_votes, bounds_map),
            ConditionStatus::Undetermined => {
                let original_bound = bounds_map[feature];
                bounds_map[feature].1 = original_bound.1.min(threshold);
                let true_id = self._build_merged_tree_recursive_heur(tree_index, left_child, current_votes, bounds_map);
                bounds_map[feature].1 = original_bound.1;
                bounds_map[feature].0 = original_bound.0.max(threshold);
                let false_id = self._build_merged_tree_recursive_heur(tree_index, right_child, current_votes, bounds_map);
                bounds_map[feature].0 = original_bound.0;
                self.provider.intern_internal(feature, threshold_bits, true_id, false_id)
            }
        }
    }

    /// Recursively builds the merged tree using Ordered Early Stopping (ORD).
    /// Traverses the tree specified by `current_tree_idx_in_root_ids` and `node_id`.
    /// Note that the *next* tree to process is determined dynamically in `handle_leaf_transition_ord`.
    ///
    /// **Performance Critical:** Optimized for speed, avoids unnecessary (safety) checks.
    ///
    /// # Arguments
    /// * `current_tree_idx_in_root_ids` - Index in `self.root_ids` of the tree currently being processed.
    /// * `node_id` - The current node being visited in that tree.
    /// * `remaining_indices` - Indices of trees yet to be processed (passed down for leaf transitions).
    /// * `current_votes` - Accumulated votes for the path taken so far.
    /// * `bounds_map` - Current path constraints (feature bounds).
    ///
    /// # Returns
    /// `NodeId` of the compiled subtree rooted at this point in the recursion.
    pub(crate) fn _build_merged_tree_recursive_ord(
        &mut self,
        current_tree_idx_in_root_ids: usize,
        node_id: NodeId,
        remaining_indices: &mut Vec<usize>,
        current_votes: &mut [u32],
        bounds_map: &mut BoundsMap,
    ) -> NodeId {
        let feature_idx_signed = self.provider.get_feature_raw(node_id);

        if feature_idx_signed == FEATURE_LEAF_SENTINEL {
            // Base Case: Leaf node ->
            //  update votes, call handle_leaf_transition_ord, revert votes.
            let leaf_class = self.provider.get_leaf_class(node_id);
            current_votes[leaf_class] += 1;
            let next_node_result = self.handle_leaf_transition_ord(remaining_indices, current_votes, bounds_map);
            current_votes[leaf_class] -= 1;
            return next_node_result;
        }

        // Recursive Step: Internal node -> 
        //  Check condition against bounds_map (AlwaysTrue, AlwaysFalse, Undetermined)
        //  If determined, recurse on the single reachable child.
        //  If undetermined, update bounds, recurse on both children, restore bounds, intern new node.
        let feature = self.provider.get_feature_usize(node_id);
        let threshold = self.provider.get_threshold(node_id);
        let threshold_bits = self.provider.get_threshold_bits(node_id);
        let left_child = self.provider.get_true_id(node_id);
        let right_child = self.provider.get_false_id(node_id);

        let condition_status = check_condition_bounds(bounds_map, feature, threshold);
        match condition_status {
            ConditionStatus::AlwaysTrue => self._build_merged_tree_recursive_ord(current_tree_idx_in_root_ids, left_child, remaining_indices, current_votes, bounds_map),
            ConditionStatus::AlwaysFalse => self._build_merged_tree_recursive_ord(current_tree_idx_in_root_ids, right_child, remaining_indices, current_votes, bounds_map),
            ConditionStatus::Undetermined => {
                let original_bound = bounds_map[feature];
                bounds_map[feature].1 = original_bound.1.min(threshold);
                let true_id = self._build_merged_tree_recursive_ord(current_tree_idx_in_root_ids, left_child, remaining_indices, current_votes, bounds_map);
                bounds_map[feature].1 = original_bound.1;
                bounds_map[feature].0 = original_bound.0.max(threshold);
                let false_id = self._build_merged_tree_recursive_ord(current_tree_idx_in_root_ids, right_child, remaining_indices, current_votes, bounds_map);
                bounds_map[feature].0 = original_bound.0;
                self.provider.intern_internal(feature, threshold_bits, true_id, false_id)
            }
        }
    }
}
