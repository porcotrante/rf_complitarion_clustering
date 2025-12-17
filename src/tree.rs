//! Defines the core data structures for the merged decision tree,
//! including the node representation, the arena allocator for efficient
//! node storage and deduplication (hash-consing), and helper enums.

use rustc_hash::{FxBuildHasher, FxHashMap};
use std::{collections::HashSet, mem};
use crate::NUM_CLASSES;

/// Type alias for a node identifier within the `Arena`. Remains usize.
pub type NodeId = usize;

// --- Sklearn-Style Sentinels ---
/// Sentinel value for NodeId fields (true_id, false_id) indicating a leaf node. Stored as i64.
pub const CHILD_LEAF_SENTINEL: i64 = -1;
/// Sentinel value for feature index indicating a leaf node. Stored as i64.
pub const FEATURE_LEAF_SENTINEL: i64 = -2;
/// Sentinel value for threshold bits indicating a leaf node. Stored as u64. Use transmute for const context.
pub const THRESHOLD_LEAF_SENTINEL_BITS: u64 = unsafe { mem::transmute(-2.0f64) };
/// Sentinel f64 value for threshold indicating a leaf node.
pub const THRESHOLD_LEAF_SENTINEL_F64: f64 = -2.0;
/// Sentinel value for leaf class indicating an internal node. Stored as usize.
pub const INTERNAL_NODE_CLASS_SENTINEL: usize = usize::MAX;
// --- End Sentinels ---

/// Internal key used for hash-consing nodes within the `Arena`.
/// Note: NodeKey::Leaf is no longer used by the Arena's main cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKey {
    /// Key variant for an `Internal` node, based on its feature (as usize), threshold (as bits),
    /// and the `NodeId`s (as usize) of its children.
    Internal(usize, u64, NodeId, NodeId), // Uses usize for NodeId here
}

/// An arena allocator for merged tree nodes using Structure of Arrays (SoA).
/// Uses sentinel values inspired by scikit-learn and pre-fills leaf nodes.
#[derive(Debug, Clone)]
pub struct Arena {
    // --- SoA Data Storage ---
    features: Vec<i64>,
    thresholds_bits: Vec<u64>,
    thresholds_f64: Vec<f64>,
    true_ids: Vec<i64>,
    false_ids: Vec<i64>,
    leaf_classes: Vec<usize>,
    // --- Hash Consing (Internal Nodes Only) ---
    cache: FxHashMap<NodeKey, NodeId>,
}

impl Arena {
    /// Creates a new Arena with default initial capacity, pre-filling leaf nodes.
    pub fn new() -> Self {
        let mut arena = Arena {
            features: Vec::new(),
            thresholds_bits: Vec::new(),
            thresholds_f64: Vec::new(),
            true_ids: Vec::new(),
            false_ids: Vec::new(),
            leaf_classes: Vec::new(),
            cache: FxHashMap::default(),
        };
        arena.prefill_leaves_internal();
        arena
    }

    /// Creates a new Arena with specified initial capacities, pre-filling leaf nodes.
    ///
    /// # Arguments
    /// * `node_capacity` - Estimated total number of nodes (internal + leaves).
    /// * `cache_capacity` - Estimated number of unique internal nodes for the cache.
    pub fn with_capacity(node_capacity: usize, cache_capacity: usize) -> Self {
        // Ensure node_capacity accounts for the leaves we are about to add
        let actual_node_capacity = node_capacity.max(NUM_CLASSES);
        let mut arena = Arena {
            features: Vec::with_capacity(actual_node_capacity),
            thresholds_bits: Vec::with_capacity(actual_node_capacity),
            thresholds_f64: Vec::with_capacity(actual_node_capacity),
            true_ids: Vec::with_capacity(actual_node_capacity),
            false_ids: Vec::with_capacity(actual_node_capacity),
            leaf_classes: Vec::with_capacity(actual_node_capacity),
            cache: FxHashMap::with_capacity_and_hasher(cache_capacity, FxBuildHasher::default()),
        };
        arena.prefill_leaves_internal();
        arena
    }

    /// Internal helper to pre-fill all leaf nodes.
    fn prefill_leaves_internal(&mut self) {
        // bulk‑fill sentinel arrays in one go
        self.features.resize(NUM_CLASSES, FEATURE_LEAF_SENTINEL);
        self.thresholds_bits.resize(NUM_CLASSES, THRESHOLD_LEAF_SENTINEL_BITS);
        self.thresholds_f64.resize(NUM_CLASSES, THRESHOLD_LEAF_SENTINEL_F64);
        self.true_ids.resize(NUM_CLASSES, CHILD_LEAF_SENTINEL);
        self.false_ids.resize(NUM_CLASSES, CHILD_LEAF_SENTINEL);

        // fill leaf_classes with 0..NUM_CLASSES
        self.leaf_classes.extend(0..NUM_CLASSES);
    }


    /// Returns the pre-interned `NodeId` for a `Leaf` node, which is simply the class index itself.
    /// Assumes `class` is valid and leaves were pre-filled.
    /// This is currently a no-op operation (just returning the argument).
    #[inline(always)]
    pub fn intern_leaf(&self, class: usize) -> NodeId {
        debug_assert!(class < NUM_CLASSES, "Invalid class index {} (NUM_CLASSES is {})", class, NUM_CLASSES);
        // The NodeId for the leaf node representing `class` is simply `class`.
        class
    }

    /// Interns an `Internal` node.
    /// Leaf nodes are assumed to be pre-filled and handled by `intern_leaf`.
    #[inline(always)]
    pub fn intern_internal(&mut self, feature: usize, threshold_bits: u64, true_id: NodeId, false_id: NodeId) -> NodeId {
        // Convert usize feature to i64 for storage, ensuring it's not the sentinel.
        let feature_i64 = feature as i64;
        debug_assert!(feature_i64 != FEATURE_LEAF_SENTINEL, "Attempted to intern internal node with leaf feature sentinel value");
        debug_assert!(threshold_bits != THRESHOLD_LEAF_SENTINEL_BITS, "Attempted to intern internal node with leaf threshold sentinel");
        // Convert usize NodeIds to i64 for storage, ensuring they aren't the sentinel equivalent.
        let true_id_i64 = true_id as i64;
        let false_id_i64 = false_id as i64;
        debug_assert!(true_id_i64 != CHILD_LEAF_SENTINEL, "Attempted to intern internal node with leaf true_id sentinel value");
        debug_assert!(false_id_i64 != CHILD_LEAF_SENTINEL, "Attempted to intern internal node with leaf false_id sentinel value");

        // In 32-bit systems, the operations above are signed-extends.
        // In 64-bit systems, the operations above are no-ops.

        // Optimization: If children are identical, just return the child ID
        if true_id == false_id {
            return true_id;
        }

        // NodeKey uses the bit representation of the threshold for consistency in hashing
        let key = NodeKey::Internal(feature, threshold_bits, true_id, false_id);

        // Check if the key already exists.
        if let Some(&id) = self.cache.get(&key) {
            return id;
        }

        // If not, create the node and then insert it into the cache.
        let id = self.nodes_len(); // ID will be the current length
        self.features.push(feature_i64); // Store as i64
        self.thresholds_bits.push(threshold_bits); // Pushes the actual threshold bits
        self.thresholds_f64.push(f64::from_bits(threshold_bits)); // Push computed f64
        self.true_ids.push(true_id_i64); // Store as i64
        self.false_ids.push(false_id_i64); // Store as i64
        self.leaf_classes.push(INTERNAL_NODE_CLASS_SENTINEL); // Store sentinel for internal

        // Now insert into the cache after other mutable borrows are done.
        self.cache.insert(key, id);
        id
    }

    // --- SoA Accessor Methods (using sklearn sentinels) ---

    /// Checks if the node with the given ID is a leaf.
    /// # Panics
    /// Panics if the `id` is out of bounds.
    #[inline(always)]
    pub fn is_leaf(&self, id: NodeId) -> bool {
        // A node is a leaf if its feature index is the leaf sentinel (-2).
        self.features[id] == FEATURE_LEAF_SENTINEL
    }

    /// Gets the raw feature index (i64) for the node. Returns FEATURE_LEAF_SENTINEL (-2) for leaves.
    /// # Panics
    /// Panics if the `id` is out of bounds.
    #[inline(always)]
    pub fn get_feature_raw(&self, id: NodeId) -> i64 {
        self.features[id]
    }

    /// Gets the feature index as usize for internal nodes.
    /// # Panics
    /// Panics if the node `id` is a leaf or out of bounds.
    #[inline(always)]
    pub fn get_feature_usize(&self, id: NodeId) -> usize {
        let feature_raw = self.features[id];
        debug_assert!(feature_raw != FEATURE_LEAF_SENTINEL, "Called get_feature_usize on a leaf node");
        feature_raw as usize
    }

    /// Gets the threshold bits (u64) for the node. Returns THRESHOLD_LEAF_SENTINEL_BITS for leaves.
    /// # Panics
    /// Panics if the `id` is out of bounds.
    #[inline(always)]
    pub fn get_threshold_bits(&self, id: NodeId) -> u64 {
        self.thresholds_bits[id]
    }

     /// Gets the precomputed threshold value (f64) for the node.
     /// Returns THRESHOLD_LEAF_SENTINEL_F64 (-2.0) for leaves.
     /// # Panics
     /// Panics if the `id` is out of bounds.
    #[inline(always)]
    pub fn get_threshold(&self, id: NodeId) -> f64 {
        // Directly return the precomputed f64 value.
        self.thresholds_f64[id]
    }

    /// Gets the raw true child ID (i64) for the node. Returns CHILD_LEAF_SENTINEL (-1) for leaves.
    /// # Panics
    /// Panics if the `id` is out of bounds.
    #[inline(always)]
    pub fn get_true_id_raw(&self, id: NodeId) -> i64 {
        self.true_ids[id]
    }

    /// Gets the true child ID as NodeId (usize).
    /// # Panics
    /// Panics if the node `id` is a leaf or out of bounds.
    #[inline(always)]
    pub fn get_true_id(&self, id: NodeId) -> NodeId {
        let child_raw = self.true_ids[id];
        debug_assert!(child_raw != CHILD_LEAF_SENTINEL, "Called get_true_id on a leaf node");
        child_raw as NodeId // Cast i64 to usize
    }

    /// Gets the raw false child ID (i64) for the node. Returns CHILD_LEAF_SENTINEL (-1) for leaves.
    /// # Panics
    /// Panics if the `id` is out of bounds.
    #[inline(always)]
    pub fn get_false_id_raw(&self, id: NodeId) -> i64 {
        self.false_ids[id]
    }

    /// Gets the false child ID as NodeId (usize).
    /// # Panics
    /// Panics if the node `id` is a leaf or out of bounds.
    #[inline(always)]
    pub fn get_false_id(&self, id: NodeId) -> NodeId {
        let child_raw = self.false_ids[id];
        debug_assert!(child_raw != CHILD_LEAF_SENTINEL, "Called get_false_id on a leaf node");
        child_raw as NodeId // Cast i64 to usize
    }

    /// Gets the predicted class for the node. Returns INTERNAL_NODE_CLASS_SENTINEL for internal nodes.
    /// # Panics
    /// Panics if the `id` is out of bounds.
    #[inline(always)]
    pub fn get_leaf_class(&self, id: NodeId) -> usize {
        let class = self.leaf_classes[id];
        debug_assert!((class == INTERNAL_NODE_CLASS_SENTINEL) || (self.features[id] == FEATURE_LEAF_SENTINEL), "Inconsistent state: Node has leaf class but non-leaf feature");
        class
    }

    // --- Arena Info Methods ---

    /// Returns the number of unique nodes currently stored in the arena.
    #[inline(always)]
    pub fn unique_node_count(&self) -> usize {
        self.cache.len()
    }

    /// Returns the current capacity of the internal node vectors.
    #[inline(always)]
    pub fn nodes_capacity(&self) -> usize {
        // Capacity should be the same for all parallel vectors
        // Using features as the reference, assuming all have same capacity.
        self.features.capacity()
    }

    /// Returns the current number of nodes stored in the arena (length of vectors).
    #[inline(always)]
    pub fn nodes_len(&self) -> usize {
        // Length should be the same for all parallel vectors
        // Using features as the reference, assuming all have same length.
        self.features.len()
    }

    /// Returns the current capacity of the internal cache HashMap.
    #[inline(always)]
    pub fn cache_capacity(&self) -> usize {
        self.cache.capacity()
    }
}

/// Represents the possible outcomes of checking a split condition against feature bounds.
/// This is used during the tree transformation to prune branches that are provably
/// unreachable given the path taken so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionStatus {
    /// The condition (`feature <= threshold`) is always true given the current bounds.
    AlwaysTrue,
    /// The condition (`feature <= threshold`) is always false given the current bounds.
    AlwaysFalse,
    /// The condition's outcome cannot be determined solely from the current bounds.
    Undetermined,
}


pub fn print_forest(arena: &Arena, roots: &[NodeId]) {
    println!("\n\nForest Structure ({} trees):", roots.len());
    for (i, &root_id) in roots.iter().enumerate() {
        println!("\n--- Tree {} (Root ID: {}) ---", i, root_id);
        // Use a visited set for each tree to handle shared nodes correctly within that tree's printout
        let mut visited_in_tree = HashSet::new();
        print_node_recursive(arena, root_id, "", &mut visited_in_tree);
    }
}

/// Recursive helper function to print a node and its children.
///
/// # Arguments
/// * `arena` - A reference to the `Arena`.
/// * `node_id` - The ID of the current node to print.
/// * `prefix` - The string prefix for indentation and connection lines.
/// * `visited` - A mutable set to track visited nodes *within the current tree traversal*
///               to detect shared subtrees and prevent infinite loops (though unlikely with typical tree structures).
fn print_node_recursive(arena: &Arena, node_id: NodeId, prefix: &str, visited: &mut HashSet<NodeId>) {
    // Check bounds just in case, although valid IDs should always be passed
    if node_id >= arena.nodes_len() {
        println!("{}Invalid Node ID: {}", prefix, node_id);
        return;
    }

    if arena.is_leaf(node_id) {
        let class = arena.get_leaf_class(node_id);
        // Ensure class is not the internal node sentinel before printing
        if class != INTERNAL_NODE_CLASS_SENTINEL {
            println!("{}Leaf(class = {}) [ID: {}]", prefix, class, node_id);
        } else {
            // This case should ideally not happen if is_leaf is true, but good to handle
             println!("{}Leaf(Error: Internal Sentinel Class) [ID: {}]", prefix, node_id);
        }
    } else {
        // Handle internal nodes
        // Check if we've already visited this node *in this specific tree's print traversal*
        if !visited.insert(node_id) {
            // We've encountered this node before in this tree traversal, indicate it's shared.
            println!("{}-> Shared Node [ID: {}]", prefix, node_id);
            return; // Don't recurse further down this shared branch
        }

        let feature = arena.get_feature_usize(node_id);
        let threshold = arena.get_threshold(node_id); // Use the f64 representation for printing
        let true_id = arena.get_true_id(node_id);
        let false_id = arena.get_false_id(node_id);

        println!("{}Node(feature[{}] <= {:.4}?) [ID: {}]", prefix, feature, threshold, node_id);

        // Prepare prefixes for children
        let child_prefix_true = format!("{}  |-- True: ", prefix);
        let child_prefix_false = format!("{}  `-- False:", prefix); // Use backtick for the last child

        // Recurse for true branch
        print_node_recursive(arena, true_id, &child_prefix_true, visited);

        // Recurse for false branch
        print_node_recursive(arena, false_id, &child_prefix_false, visited);
    }
}