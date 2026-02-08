use std::{error::Error, fs::File, path::PathBuf, time::{Duration, Instant}};

use csv::ReaderBuilder;
use transform::resimplify_from_arena;

use crate::{config::{K_VALUES, NUM_BENCHMARK_RUNS, NUM_FEATURES, STRATEGIES_TO_TEST}, cpu_time::get_cpu_time, export, paths::{distance_between_trees, extract_paths_from_arena, extract_paths_from_arena_for_instances, mean_feature_overlap_distance}, results::BenchmarkResult, transform::{self, BenchmarkStats, EarlyStoppingStrategy, RandomForestData, predict_with_merged_tree, predict_with_original_rf, transform_rf_partitioned}, tree::{Arena, NodeId}, utils::calculate_stats};

/// Performs the benchmark for a single configuration (k, strategy) using the provided RF data and prediction file.
///
/// Runs the `transform_rf_partitioned` function multiple times (`NUM_BENCHMARK_RUNS`),
/// collects timing information (wall and CPU), calculates statistics (min, median, max, mean, stddev),
/// verifies the compiled tree's predictions against the provided CSV, and returns the results.
///
/// # Arguments
/// * `rf_data` - Loaded data for the original Random Forest model.
/// * `partition_depth_k` - The partitioning depth `k` to test.
/// * `current_strategy` - The early stopping strategy to use.
/// * `prediction_path_buf` - Path to the CSV file containing original RF predictions for verification.
///
/// # Returns
/// A `Result` containing the `BenchmarkResult` struct on success, or an error.
pub fn benchmark_single_config(
    rf_data: &RandomForestData,
    partition_depth_k: usize,
    current_strategy: EarlyStoppingStrategy,
    prediction_path_buf: &PathBuf,
    seed: u32
) -> Result<BenchmarkResult, Box<dyn Error>> {
    println!("\n--- Benchmarking for k = {}, Strategy = {} ---", partition_depth_k, current_strategy);
    println!("[BENCH] Running Partitioned Transform...");
    let mut total_transform_durations: Vec<Duration> = Vec::with_capacity(NUM_BENCHMARK_RUNS);
    let mut cpu_transform_durations: Vec<Duration> = Vec::with_capacity(NUM_BENCHMARK_RUNS);
    // Vectors to store component timings from BenchmarkStats across runs
    let mut assembly_durations: Vec<Duration> = Vec::with_capacity(NUM_BENCHMARK_RUNS);
    let mut partition_processing_durations: Vec<Duration> = Vec::with_capacity(NUM_BENCHMARK_RUNS); // Base-case work
    let mut transform_result_option: Option<(Arena, NodeId)> = None; // Store arena/root from first run
    let mut final_nodes = 0;
    let mut final_height = 0;
    let mut first_run_stats = BenchmarkStats::default(); // Store detailed stats from the first run
    let mut acc: f64 = 0.0;
    let mut global_id: usize = 0;

    // --- Run the transformation multiple times for timing ---
    for i in 0..NUM_BENCHMARK_RUNS {
        let cpu_start = get_cpu_time();
        let total_transform_start = Instant::now();

        // Execute the core transformation function
        let (
            current_arena, current_root_id,
            current_stats,
            current_height
        ) = transform_rf_partitioned(rf_data, partition_depth_k, current_strategy);

        //puting the information into the global variables (only makes sense if run once)
        global_id = current_root_id;

        let total_transform_duration = total_transform_start.elapsed();
        let cpu_end = get_cpu_time();
        let cpu_transform_duration = cpu_end.checked_sub(cpu_start).unwrap_or(Duration::ZERO);
        acc = export::get_tree_acc(&current_arena, current_root_id, seed as u16);

        // Store timings
        total_transform_durations.push(total_transform_duration);
        cpu_transform_durations.push(cpu_transform_duration);
        assembly_durations.push(current_stats.assembly_duration);
        partition_processing_durations.push(current_stats.partition_processing_duration);

        // Keep results from the first run for verification and stats
        if i == 0 {
            // Compact the arena (remove unreachable nodes)
            // before computing the final node count
            let mut local_arena = Arena::new();
            let mut bounds_map = vec![(-f64::INFINITY, f64::INFINITY); NUM_FEATURES];
            let mut dummy1 = 0usize;
            let mut dummy2 = 0usize;
            resimplify_from_arena(&current_arena, &[current_root_id], &mut bounds_map, &mut local_arena,
                #[cfg(feature = "detailed-stats")]
                &mut dummy1,
                #[cfg(feature = "detailed-stats")]
                &mut dummy2,
            );

            final_nodes = local_arena.unique_node_count();
            final_height = current_height;
            transform_result_option = Some((current_arena, current_root_id));
            first_run_stats = current_stats;
        }
    }
    // Unwrap the result from the first run (arena now owned here)
    let (arena, root_id) = transform_result_option.ok_or(format!("Failed to transform RF data for k={}, strategy={}", partition_depth_k, current_strategy))?;

    // --- Calculate Timing Statistics ---
    // Calculate "Other" time by subtracting measured components from total time
    let mut other_durations: Vec<Duration> = Vec::with_capacity(NUM_BENCHMARK_RUNS);
    for i in 0..NUM_BENCHMARK_RUNS {
        let sum_components = assembly_durations[i] + partition_processing_durations[i];
        let other_duration = total_transform_durations[i].checked_sub(sum_components).unwrap_or(Duration::ZERO);
        other_durations.push(other_duration);
    }

    // Use utils::calculate_stats for each duration vector
    let (total_time_min, total_time_median, total_time_max, total_time_mean, total_time_std_dev) = calculate_stats(&total_transform_durations);
    let (cpu_time_min, cpu_time_median, cpu_time_max, cpu_time_mean, cpu_time_std_dev) = calculate_stats(&cpu_transform_durations);
    let (assembly_time_min, assembly_time_median, assembly_time_max, assembly_time_mean, assembly_time_std_dev) = calculate_stats(&assembly_durations);
    let (partition_processing_time_min, partition_processing_time_median, partition_processing_time_max, partition_processing_time_mean, partition_processing_time_std_dev) = calculate_stats(&partition_processing_durations); // Base case work
    let (other_time_min, other_time_median, other_time_max, other_time_mean, other_time_std_dev) = calculate_stats(&other_durations);

    println!("[RESULT] Transform: min={:.3}s med={:.3}s max={:.3}s mean={:.3}s std={:.3}s",
             total_time_min, total_time_median, total_time_max, total_time_mean, total_time_std_dev);

    // --- Prediction Verification ---
    let mut final_mismatches = 0;
    let mut final_total_predictions = 0;
    // Open and read the prediction CSV
    let file = File::open(prediction_path_buf)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let headers = rdr.headers()?.clone();
    let rf_pred_col_index = headers.iter().position(|h| h == "rf_pred")
        .ok_or_else(|| format!("'rf_pred' column not found in CSV header of {:?}", prediction_path_buf))?;
    let feature_count = rf_pred_col_index;
    for (row_idx, result) in rdr.records().enumerate() {
        let record = result?;
        final_total_predictions += 1;
        let mut features: Vec<f64> = Vec::with_capacity(feature_count);
        for j in 0..feature_count {
            let val_str = record.get(j).ok_or_else(|| format!("Missing feature value at row {}, col {}", row_idx + 1, j + 1))?;
            let val: f64 = val_str.trim().parse()
                .map_err(|e| format!("Failed to parse feature at row {}, col {}: {}", row_idx + 1, j + 1, e))?;
            features.push(val);
        }
        let expected_pred_str = record.get(rf_pred_col_index).ok_or_else(|| format!("Missing 'rf_pred' value at row {}", row_idx + 1))?;
        let expected_prediction: usize = expected_pred_str.trim().parse()
             .map_err(|e| format!("Failed to parse 'rf_pred' at row {}: {}", row_idx + 1, e))?;
        // Predict using the compiled tree
        let merged_tree_prediction = predict_with_merged_tree(&arena, root_id, &features);
        // Compare and log mismatches
        if merged_tree_prediction != expected_prediction {
            final_mismatches += 1;
            // Re-predict with original RF for detailed mismatch logging
            let (original_rf_prediction, _original_votes, _individual_preds) = predict_with_original_rf(rf_data, &features);
            if final_mismatches <= 5 {
                println!(
                    "[MISMATCH] Row {}: Features: {:?}, Expected(CSV): {}, Merged: {}, OriginalRF: {}",
                    row_idx + 1, features, expected_prediction, merged_tree_prediction, original_rf_prediction // Use the extracted prediction
                );
            } else if final_mismatches == 6 {
                 println!("[MISMATCH] ... (further mismatches omitted)");
            }
        }
    }
    println!("[RESULT] Mismatches: {} / {}", final_mismatches, final_total_predictions);


    // Construct and return the final result structure
    Ok(BenchmarkResult {
        seed: seed, // Seed is set in the outer loop
        k: partition_depth_k,
        strategy: current_strategy,
        total_time_min, total_time_median, total_time_max, total_time_mean, total_time_std_dev,
        cpu_time_min, cpu_time_median, cpu_time_max, cpu_time_mean, cpu_time_std_dev,
        assembly_time_min, assembly_time_median, assembly_time_max, assembly_time_mean, assembly_time_std_dev, // Assembly (k>0)
        partition_processing_time_min, partition_processing_time_median, partition_processing_time_max, partition_processing_time_mean, partition_processing_time_std_dev, // Base Case Work (k=0)
        other_time_min, other_time_median, other_time_max, other_time_mean, other_time_std_dev, // Other
        mismatches: final_mismatches,
        nodes: final_nodes,
        height: final_height,
        accuracy: acc,
        tree_arena: arena,
        tree_id: global_id,
        // Use detailed stats from the first run
        #[cfg(feature = "detailed-stats")]
        simplify_pruning_count: first_run_stats.simplify_pruning_count,
        #[cfg(feature = "detailed-stats")]
        merge_pruning_count: first_run_stats.merge_pruning_count,
        #[cfg(feature = "detailed-stats")]
        kept_count: first_run_stats.kept_count,
    })
}

/// Runs the benchmarks for all configured k values and strategies for a single loaded RF model.
///
/// # Arguments
/// * `rf_data` - The loaded Random Forest data for the current seed.
/// * `prediction_path_buf` - Path to the corresponding prediction CSV file.
///
/// # Returns
/// A `Result` containing a vector of `BenchmarkResult` structs on success, or an error.
pub fn run_benchmarks(rf_data: &RandomForestData, prediction_path_buf: &PathBuf, seed: u32) -> Result<Vec<BenchmarkResult>, Box<dyn Error>> { // Pass rf_data
    let mut all_results: Vec<BenchmarkResult> = Vec::with_capacity(K_VALUES.len() * STRATEGIES_TO_TEST.len());

    // Loop over K_VALUES and STRATEGIES_TO_TEST
    for &partition_depth_k in K_VALUES.iter() {
        for &current_strategy in STRATEGIES_TO_TEST.iter() {
            // Call the single config benchmark function
            let result = benchmark_single_config(rf_data, partition_depth_k, current_strategy, prediction_path_buf, seed)?;
            all_results.push(result);
        }
    }
    Ok(all_results)
}

pub struct TreeComparison {
    pub seed: u32,
    pub distances: Vec<f64>,
}

pub fn compare_trees(
    seed: u32,
    dist: bool,
    normal_arena: &Arena,
    egap_arena: &Arena,
    normal_id: usize,
    egap_id: usize,
    feat_num: usize,
) -> f64 {
        let distance: f64;
        if dist {
            distance = compare_trees_per_strat(normal_arena, normal_id, egap_arena, egap_id);
        }
        else {
            println!("Usando Jaccard");
            distance = mean_feature_overlap_distance(normal_arena, normal_id, egap_arena, egap_id, feat_num);
        }
        
        distance
    }

pub fn compare_trees_per_strat(normal_arena: &Arena, normal_id: usize, egap_arena: &Arena, egap_id: usize) -> f64 {
    let complete_paths = extract_paths_from_arena(normal_arena, normal_id);
    println!("Caminhos da árvore normal extraídos");
    let egap_paths = extract_paths_from_arena(egap_arena, egap_id);
    println!("Caminhos da árvore egap extraídos");

    return distance_between_trees(&complete_paths, &egap_paths);
}

pub fn compare_trees_per_strat_instance(normal_arena: &Arena, normal_id: usize, egap_arena: &Arena, egap_id: usize, normal_buf: &PathBuf, egap_buf: &PathBuf) -> f64 {
    let complete_paths = extract_paths_from_arena_for_instances(normal_arena, normal_id, normal_buf);
    println!("Caminhos da árvore normal extraídos");
    let egap_paths = extract_paths_from_arena_for_instances(egap_arena, egap_id, egap_buf);
    println!("Caminhos da árvore egap extraídos");

    return distance_between_trees(&complete_paths, &egap_paths);
}