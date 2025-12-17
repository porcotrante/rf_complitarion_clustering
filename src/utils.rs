//! Utility functions used during the Random Forest transformation,
//! primarily for handling vote aggregation and tie-breaking.

use std::time::Duration;

/// Determines the majority class from a slice of votes.
///
/// Finds the class index with the highest vote count. Ties are broken
/// by choosing the class with the lower index, mirroring the behavior
/// often seen in libraries like scikit-learn.
/// 
/// However, note that sklearn predicts by computing the class probabilities means:
/// 
/// "The predicted class of an input sample is a vote by the trees in the forest,
/// weighted by their probability estimates. That is, the predicted class is
/// the one with highest mean probability estimate across the trees."
/// [Source](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict)
/// 
/// Therefore, for proper comparisons in Python, manually compute the votes of
/// each tree from a RandomForestClassifier
/// (e.g., find the majority class in `[t.predict(...) for t in rf.estimators_]`)
///
/// # Arguments
/// * `votes` - A slice where `votes[i]` is the number of votes for class `i`.
///
/// # Returns
/// The index of the majority class. Returns 0 if the `votes` slice is empty.
#[allow(non_snake_case, reason="Must be FAST!")]
pub fn _FAST_get_majority_class(votes: &[u32]) -> usize {
    // [_FAST!] Assume votes are not empty
    let mut max_votes = 0;
    let mut winning_index = 0;

    for (index, &count) in votes.iter().enumerate() {
        if index == 0 {
            max_votes = count;
            // winning_index is already 0
        } else if count > max_votes {
            // New winner (tie broken by lower index)
            max_votes = count;
            winning_index = index;
        }
        // Tie-breaking: prefer lower index, so do nothing if count == max_votes
        // as the current winning_index is already lower or equal.
    }
    winning_index
}

/// Finds the winning class index, its vote count, and the second highest vote count.
///
/// This function efficiently determines the top two vote counts and the index
/// of the absolute winner (handling ties by lower index) in a single pass.
/// Useful for early stopping optimizations during tree merging.
///
/// # Arguments
/// * `votes` - A slice where `votes[i]` is the number of votes for class `i`.
///
/// # Returns
/// A tuple `(winner_index, winner_votes, second_highest_votes)`.
/// If `votes` is empty, returns `(0, 0, 0)`.
/// If `votes` has one element `v`, returns `(0, v, 0)`.
#[allow(non_snake_case, reason="Must be FAST!")]
pub fn _FAST_get_top_two_votes(votes: &[u32]) -> (usize, u32, u32) {
    // [_FAST!] Assume votes are not empty
    let mut top_idx = 0;
    let mut top = 0;
    let mut second = 0;
    for (i, &v) in votes.iter().enumerate() {
        if i == 0 {
            top = v;
            // top_idx is already 0
        } else if v > top {
            // New winner (or tie broken by lower index)
            second = top;
            top = v;
            top_idx = i;
        } else if v > second {
            // New second place
            second = v;
        }
    }
    (top_idx, top, second)
}

/// Finds the winning class index, its total vote count (current + safe), and the second highest total vote count.
///
/// Used for Abstract Early Stopping. Calculates total votes for each class before comparison.
/// Ties for the winner are broken by lower class index.
///
/// # Arguments
/// * `current_votes` - Slice of votes accumulated so far for the current path.
/// * `safe_votes` - Slice of votes guaranteed from unprocessed trees based on abstract interpretation.
///
/// # Returns
/// A tuple `(winner_index, winner_total_votes, second_highest_total_votes)`.
/// Returns `(0, 0, 0)` if `current_votes` (and thus `safe_votes`) is empty.
#[allow(non_snake_case, reason="Must be FAST!")]
pub fn _FAST_get_top_two_votes_with_safe(current_votes: &[u32], safe_votes: &[u32]) -> (usize, u32, u32) {
    // [_FAST!] Assume current_votes and safe_votes are not empty
    // [_FAST!] and that safe_votes has at least the same length as current_votes
    // [_FAST!] (this might happen because safe_votes[n_classes] stores free votes)
    let mut top_idx = 0;
    let mut top_total = 0;
    let mut second_total = 0;

    for i in 0..current_votes.len() {
        let total_votes = current_votes[i] + safe_votes[i];

        if i == 0 {
            top_total = total_votes;
            // top_idx is already 0
        } else if total_votes > top_total {
            // New winner (or tie broken by lower index)
            second_total = top_total;
            top_total = total_votes;
            top_idx = i;
        } else if total_votes > second_total {
            // New second place
            second_total = total_votes;
        }
    }
    (top_idx, top_total, second_total)
}

/// Finds the maximum vote count in a slice.
///
/// # Arguments
/// * `votes` - A slice where `votes[i]` is the number of votes for class `i`.
///
/// # Returns
/// The highest vote count found in the slice. Returns 0 if the slice is empty.
/// 
/// # Benchmark
/// Check `bench/max_benchmark.rs` for performance comparison with other implementations.
#[allow(non_snake_case, reason="Must be FAST!")]
pub fn _FAST_get_max_vote(votes: &[u32]) -> u32 {
    let mut max = 0;
    for &vote in votes {
        if vote > max {
            max = vote;
        }
    }
    max
}

/// Calculates min, median, max, mean, and standard deviation for a slice of Durations.
///
/// # Arguments
/// * `durations` - A slice of `std::time::Duration` values.
///
/// # Returns
/// A tuple `(min_sec, median_sec, max_sec, mean_sec, std_dev_sec)`.
/// Returns `(0.0, 0.0, 0.0, 0.0, 0.0)` if the slice is empty.
/// For a single element slice, min=median=max=mean, and std_dev=0.
pub fn calculate_stats(durations: &[Duration]) -> (f64, f64, f64, f64, f64) {
    if durations.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let mut times_sec: Vec<f64> = durations.iter().map(|d| d.as_secs_f64()).collect();
    let n = times_sec.len();
    let n_f64 = n as f64;

    // Calculate Mean
    let mean = times_sec.iter().sum::<f64>() / n_f64;

    // Calculate Std Dev (Population)
    let variance = times_sec.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / n_f64;
    let std_dev = variance.sqrt();

    // Calculate Min, Max, Median
    // Sort the vector to find min, max, and median
    times_sec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min_val = times_sec[0];
    let max_val = times_sec[n - 1];

    let median_val = if n % 2 == 1 {
        // Odd number of elements
        times_sec[n / 2]
    } else {
        // Even number of elements: average of the two middle ones
        (times_sec[n / 2 - 1] + times_sec[n / 2]) / 2.0
    };

    (min_val, median_val, max_val, mean, std_dev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_get_majority() {
        assert_eq!(_FAST_get_majority_class(&[1, 5, 2]), 1);
        // Tie-breaking check: [5, 1, 5]. Index 0 has 5, Index 2 has 5. Lower index (0) wins.
        assert_eq!(_FAST_get_majority_class(&[5, 1, 5]), 0);
        // All equal check: [1, 1, 1]. Index 0 has 1, Index 1 has 1, Index 2 has 1. Lower index (0) wins.
        assert_eq!(_FAST_get_majority_class(&[1, 1, 1]), 0);
        assert_eq!(_FAST_get_majority_class(&[]), 0);
        assert_eq!(_FAST_get_majority_class(&[0, 0, 1]), 2);
        assert_eq!(_FAST_get_majority_class(&[3, 9, 9]), 1); // Tie, index 1 wins over 2
    }

    #[test]
    fn test_get_top_two() {
        assert_eq!(_FAST_get_top_two_votes(&[1, 5, 2]), (1, 5, 2));
        assert_eq!(_FAST_get_top_two_votes(&[5, 1, 5]), (0, 5, 5)); // Tie for first, lower index wins
        assert_eq!(_FAST_get_top_two_votes(&[1, 1, 1]), (0, 1, 1)); // All equal
        assert_eq!(_FAST_get_top_two_votes(&[10]), (0, 10, 0));      // Single element
        assert_eq!(_FAST_get_top_two_votes(&[]), (0, 0, 0));         // Empty
        assert_eq!(_FAST_get_top_two_votes(&[0, 0, 0]), (0, 0, 0));  // All zero
        assert_eq!(_FAST_get_top_two_votes(&[7, 0, 3]), (0, 7, 3));
        assert_eq!(_FAST_get_top_two_votes(&[7, 0, 7]), (0, 7, 7)); // Tie for first, second is equal
        assert_eq!(_FAST_get_top_two_votes(&[3, 7, 7]), (1, 7, 7)); // Tie for first, second is equal
        assert_eq!(_FAST_get_top_two_votes(&[7, 7, 7]), (0, 7, 7)); // Tie for first, second is equal
        assert_eq!(_FAST_get_top_two_votes(&[5, 9, 2]), (1, 9, 5)); // Second appears before first
    }

    #[test]
    fn test_get_top_two_with_safe() {
        // Basic case
        assert_eq!(_FAST_get_top_two_votes_with_safe(&[1, 5, 2], &[1, 0, 1]), (1, 5, 3)); // Totals: [2, 5, 3]

        // Tie for first (total), lower index wins
        assert_eq!(_FAST_get_top_two_votes_with_safe(&[5, 1, 3], &[0, 4, 2]), (0, 5, 5)); // Totals: [5, 5, 5] -> idx 0 wins

        // All equal total
        assert_eq!(_FAST_get_top_two_votes_with_safe(&[1, 1, 1], &[1, 1, 1]), (0, 2, 2)); // Totals: [2, 2, 2] -> idx 0 wins

        // Single element
        assert_eq!(_FAST_get_top_two_votes_with_safe(&[10], &[2]), (0, 12, 0)); // Totals: [12]

        // Empty
        assert_eq!(_FAST_get_top_two_votes_with_safe(&[], &[]), (0, 0, 0));

        // All zero
        assert_eq!(_FAST_get_top_two_votes_with_safe(&[0, 0, 0], &[0, 0, 0]), (0, 0, 0));

        // Safe votes change winner
        assert_eq!(_FAST_get_top_two_votes_with_safe(&[7, 8, 3], &[2, 0, 6]), (0, 9, 9)); // Totals: [9, 8, 9] -> idx 0 wins (tie break) - Correction: Original test was wrong, this is correct
        assert_eq!(_FAST_get_top_two_votes_with_safe(&[7, 8, 3], &[2, 0, 7]), (2, 10, 9)); // Totals: [9, 8, 10] -> idx 2 wins

        // Second place changes
         assert_eq!(_FAST_get_top_two_votes_with_safe(&[10, 2, 3], &[0, 5, 4]), (0, 10, 7)); // Totals: [10, 7, 7] -> idx 0 wins, second is 7
    }

    #[test]
    fn test_get_max_vote() {
        assert_eq!(_FAST_get_max_vote(&[1, 5, 2]), 5);
        assert_eq!(_FAST_get_max_vote(&[5, 1, 5]), 5);
        assert_eq!(_FAST_get_max_vote(&[1, 1, 1]), 1);
        assert_eq!(_FAST_get_max_vote(&[10]), 10);
        assert_eq!(_FAST_get_max_vote(&[]), 0);
        assert_eq!(_FAST_get_max_vote(&[0, 0, 0]), 0);
        assert_eq!(_FAST_get_max_vote(&[7, 0, 3]), 7);
    }

    #[test]
    fn test_calculate_stats_multiple() {
        let durations = vec![
            Duration::from_secs_f64(1.0), // sorted: 0.9, 1.0, 1.0, 1.1
            Duration::from_secs_f64(1.1),
            Duration::from_secs_f64(0.9),
            Duration::from_secs_f64(1.0),
        ];
        let (min_val, median_val, max_val, mean, std_dev) = calculate_stats(&durations);

        assert!((min_val - 0.9).abs() < 1e-9);
        assert!((median_val - 1.0).abs() < 1e-9); // Median of (1.0, 1.0)
        assert!((max_val - 1.1).abs() < 1e-9);
        assert!((mean - 1.0).abs() < 1e-9);
        // Expected variance = 0.005, std_dev = sqrt(0.005) approx 0.07071
        assert!((std_dev - 0.070710678).abs() < 1e-9);
    }

     #[test]
    fn test_calculate_stats_multiple_odd() {
        let durations = vec![
            Duration::from_secs_f64(1.0), // sorted: 0.9, 1.0, 1.1
            Duration::from_secs_f64(1.1),
            Duration::from_secs_f64(0.9),
        ];
        let n = durations.len() as f64;
        let (min_val, median_val, max_val, mean, std_dev) = calculate_stats(&durations);

        assert!((min_val - 0.9).abs() < 1e-9);
        assert!((median_val - 1.0).abs() < 1e-9); // Middle element
        assert!((max_val - 1.1).abs() < 1e-9);
        assert!((mean - 1.0).abs() < 1e-9); // (0.9 + 1.0 + 1.1) / 3 = 1.0
        // Variance = ((-0.1)^2 + 0^2 + (0.1)^2) / 3 = (0.01 + 0 + 0.01) / 3 = 0.02 / 3
        let expected_variance = 0.02 / n;
        assert!((std_dev - expected_variance.sqrt()).abs() < 1e-9);
    }


    #[test]
    fn test_calculate_stats_empty() {
        let empty_durations: Vec<Duration> = vec![];
        let (min_val, median_val, max_val, mean, std_dev) = calculate_stats(&empty_durations);
        assert_eq!(min_val, 0.0);
        assert_eq!(median_val, 0.0);
        assert_eq!(max_val, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(std_dev, 0.0);
    }

    #[test]
    fn test_calculate_stats_single() {
        let single_duration = vec![Duration::from_secs_f64(5.0)];
         let (min_val, median_val, max_val, mean, std_dev) = calculate_stats(&single_duration);
         assert_eq!(min_val, 5.0);
         assert_eq!(median_val, 5.0);
         assert_eq!(max_val, 5.0);
         assert_eq!(mean, 5.0);
         assert_eq!(std_dev, 0.0); // Std dev of one item is 0
    }

    // Test specifically for the old calculate_stats signature (mean, std_dev)
    #[test]
    fn test_calculate_stats_old_signature_check() {
        let durations = vec![
            Duration::from_secs_f64(1.0),
            Duration::from_secs_f64(1.1),
            Duration::from_secs_f64(0.9),
            Duration::from_secs_f64(1.0),
        ];
        // Destructure all 5 values, but only check mean and std_dev for this test
        let (_min_val, _median_val, _max_val, mean, std_dev) = calculate_stats(&durations);
        assert!((mean - 1.0).abs() < 1e-9);
        // Expected variance = ((0)^2 + (0.1)^2 + (-0.1)^2 + (0)^2) / 4 = (0 + 0.01 + 0.01 + 0) / 4 = 0.02 / 4 = 0.005
        // Expected std_dev = sqrt(0.005) approx 0.07071
        assert!((std_dev - 0.070710678).abs() < 1e-9);

        let empty_durations: Vec<Duration> = vec![];
        // Destructure all 5 values
        let (_min_empty, _median_empty, _max_empty, mean_empty, std_dev_empty) = calculate_stats(&empty_durations);
        assert_eq!(mean_empty, 0.0);
        assert_eq!(std_dev_empty, 0.0);

        let single_duration = vec![Duration::from_secs_f64(5.0)];
         // Destructure all 5 values
         let (_min_single, _median_single, _max_single, mean_single, std_dev_single) = calculate_stats(&single_duration);
         assert_eq!(mean_single, 5.0);
         assert_eq!(std_dev_single, 0.0); // Std dev of one item is 0
    }
}