//! Common types, constants, and utility functions used across the transformation modules.

use crate::tree::ConditionStatus;

/// Type alias for the bounds tracking.
///
/// Represents `(lower_bound, upper_bound)`, where `feature > lower_bound` and `feature <= upper_bound`.
/// Uses `f64::NEG_INFINITY` and `f64::INFINITY` to represent unbounded sides.
pub type BoundsMap = Vec<(f64, f64)>;


/// Checks if a split condition is always true, always false, or undetermined
/// based on the current feature bounds.
/// Condition being checked: `feature <= threshold`.
/// 
/// **Performance Critical:** Optimized for speed, avoids unnecessary (safety) checks.
///
/// # Arguments
/// * `bounds_map` - The current bounds established for each feature.
/// * `feature` - The index of the feature being tested by the split.
/// * `threshold` - The threshold value of the split.
/// 
/// # Returns
/// * `ConditionStatus::AlwaysTrue` if `upper_bound <= threshold`.
/// * `ConditionStatus::AlwaysFalse` if `lower_bound >= threshold`.
/// * `ConditionStatus::Undetermined` otherwise.
///
/// # Panics
/// Panics if `feature` index is out of bounds (`>= NUM_FEATURES`).
pub fn check_condition_bounds(bounds_map: &BoundsMap, feature: usize, threshold: f64) -> ConditionStatus {
    // Direct indexing with array. Assumes feature < NUM_FEATURES.
    // (No safety checks for performance reasons)
    let (lower_bound, upper_bound) = bounds_map[feature];

    // If the known upper bound is already <= threshold, the condition must be true.
    if upper_bound <= threshold {
        return ConditionStatus::AlwaysTrue;
    }
    // If the known lower bound is already >= threshold, the feature must be > lower_bound >= threshold,
    // so the condition (feature <= threshold) must be false.
    if lower_bound >= threshold {
        return ConditionStatus::AlwaysFalse;
    }
    // Otherwise, the outcome depends on the specific feature value.
    ConditionStatus::Undetermined
}
