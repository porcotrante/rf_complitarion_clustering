use std::{error::Error, fs::{File, OpenOptions}, path::PathBuf};

use csv::WriterBuilder;

use crate::{benchmark::TreeComparison, config::DATASET_NAME, transform::EarlyStoppingStrategy};

pub struct BenchmarkResult {
    pub seed: u32,
    pub k: usize,
    pub strategy: EarlyStoppingStrategy,
    // --- Wall Time Metrics (seconds) ---
    pub total_time_min: f64,    // Minimum wall time over NUM_BENCHMARK_RUNS
    pub total_time_median: f64, // Median wall time
    pub total_time_max: f64,    // Maximum wall time
    pub total_time_mean: f64,   // Mean wall time
    pub total_time_std_dev: f64,// Standard deviation of wall time
    // --- CPU Time Metrics (seconds) ---
    pub cpu_time_min: f64,      // Minimum CPU time
    pub cpu_time_median: f64,   // Median CPU time
    pub cpu_time_max: f64,      // Maximum CPU time
    pub cpu_time_mean: f64,     // Mean CPU time
    pub cpu_time_std_dev: f64,  // Standard deviation of CPU time
    // --- Component Wall Time Metrics (seconds) ---
    // Timings captured from the BenchmarkStats returned by transform_rf_partitioned
    pub assembly_time_min: f64, // Min time spent assembling results (k>0)
    pub assembly_time_median: f64,
    pub assembly_time_max: f64,
    pub assembly_time_mean: f64,
    pub assembly_time_std_dev: f64,
    pub partition_processing_time_min: f64, // Min time spent in base case merge (k=0) or partition simplification/merge (k>0)
    pub partition_processing_time_median: f64,
    pub partition_processing_time_max: f64,
    pub partition_processing_time_mean: f64,
    pub partition_processing_time_std_dev: f64,
    pub other_time_min: f64, // Min time for non-measured components (e.g., initial simplify, threshold sort) calculated by subtraction
    pub other_time_median: f64,
    pub other_time_max: f64,
    pub other_time_mean: f64,
    pub other_time_std_dev: f64,
    // --- Other Results ---
    pub mismatches: usize, // Number of predictions differing from the original RF's CSV output
    pub nodes: usize,      // Number of unique nodes in the final compiled tree
    pub height: usize,     // Height of the final compiled tree
    pub accuracy: f64,
    // --- Detailed Stats (Conditional) ---
    #[cfg(feature = "detailed-stats")]
    pub simplify_pruning_count: usize, // Nodes pruned during all simplification phases
    #[cfg(feature = "detailed-stats")]
    pub merge_pruning_count: usize,    // Nodes pruned during merge/base-case phases
    #[cfg(feature = "detailed-stats")]
    pub kept_count: usize,             // Nodes traversed but not pruned
}

/// Prints a formatted summary table of benchmark results to the console.
pub fn print_summary_table(results: &[BenchmarkResult]) {
    println!(
        "\n======== BENCHMARK SUMMARY ({}) - Last Seed Shown if Multiple ========",
        DATASET_NAME
    );
    println!(
        "Seed | k | Strategy | Cluster | Wall Time (min, s) | CPU Time (min, s) | Miss  | Nodes    | Height | Accuracy"
    );
    println!(
        "-----|---|----------|---------|--------------------|-------------------|-------|----------|--------|-------"
    );
    for (idx, result) in results.iter().enumerate() {
        let cluster = if (idx / 4) % 2 == 0 { "NO" } else { "YES" };

        println!(
            "{:>4} |{:>2} | {:<8} | {:<7} | {:>18.3} | {:>17.3} | {:>5} | {:>8} | {:>6} | {:>6.2}",
            result.seed,
            result.k,
            result.strategy.to_string(),
            cluster,
            result.total_time_min,
            result.cpu_time_min,
            result.mismatches,
            result.nodes,
            result.height,
            result.accuracy
        );
    }
    println!(
        "=================================================================================================="
    );
}

pub fn print_tree_distance_table(results: &[TreeComparison]) {
    println!(
        "\n======== TREE SIMILARITY SUMMARY ({}) ========",
        DATASET_NAME
    );

    for comparison in results {
        println!("\nSeed: {}", comparison.seed);
        println!("--------------------------------------------");
        println!("Strategy | Distance to Original");
        println!("---------|----------------------");

        // Ordem fixa esperada
        let strategy_names = ["ES", "AbsES", "HEUR", "ORD"];

        for (i, strategy) in strategy_names.iter().enumerate() {
            let distance = comparison
                .distances
                .get(i)
                .copied()
                .unwrap_or(f64::NAN);

            println!(
                "{:<8} | {:>20.6}",
                strategy,
                distance
            );
        }
    }

    println!(
        "\n============================================"
    );
}

/// Writes only the header row to the specified CSV file. Creates the file if needed.
pub fn write_csv_header(csv_path: &PathBuf) -> Result<(), Box<dyn Error>> {
    let file = File::create(csv_path)?; // Create or truncate
    let mut wtr = WriterBuilder::new().from_writer(file);

    // --- Build Header Row Conditionally (Simplified) ---
    let mut headers = vec![
        "Seed", "k", "Strategy",
        // Total Wall Time
        "TotalTime_Min", "TotalTime_Median", "TotalTime_Max", "TotalTime_Mean", "TotalTime_StdDev",
        // CPU Time
        "CpuTime_Min", "CpuTime_Median", "CpuTime_Max", "CpuTime_Mean", "CpuTime_StdDev",
        // Assembly Time
        "AssemblyTime_Min", "AssemblyTime_Median", "AssemblyTime_Max", "AssemblyTime_Mean", "AssemblyTime_StdDev",
        // Partition Processing Time (Base Case)
        "PartitionProcessingTime_Min", "PartitionProcessingTime_Median", "PartitionProcessingTime_Max", "PartitionProcessingTime_Mean", "PartitionProcessingTime_StdDev",
        // Other Time
        "OtherTime_Min", "OtherTime_Median", "OtherTime_Max", "OtherTime_Mean", "OtherTime_StdDev",
        // Other Metrics
        "Mismatches", "Nodes", "Height", "Accuracy"
    ];

    #[cfg(feature = "detailed-stats")]
    {
        headers.push("SimplifyPruningCount");
        headers.push("MergePruningCount");
        headers.push("KeptCount");
    }

    wtr.write_record(&headers)?;
    wtr.flush()?;
    Ok(())
}

/// Appends a slice of `BenchmarkResult` structs as data rows to an existing CSV file.
pub fn append_results_to_csv(results: &[BenchmarkResult], csv_path: &PathBuf) -> Result<(), Box<dyn Error>> {
    let file = OpenOptions::new()
        .append(true)
        .open(csv_path)?; // Open in append mode
    let mut wtr = WriterBuilder::new()
        .has_headers(false) // Don't write headers again
        .from_writer(file);

    // --- Write Data Rows Conditionally (Simplified) ---
    for result in results {
        let mut record = vec![
            result.seed.to_string(),
            result.k.to_string(),
            result.strategy.to_string(),
            // Total Wall Time
            format!("{:.6}", result.total_time_min), format!("{:.6}", result.total_time_median), format!("{:.6}", result.total_time_max), format!("{:.6}", result.total_time_mean), format!("{:.6}", result.total_time_std_dev),
            // CPU Time
            format!("{:.6}", result.cpu_time_min), format!("{:.6}", result.cpu_time_median), format!("{:.6}", result.cpu_time_max), format!("{:.6}", result.cpu_time_mean), format!("{:.6}", result.cpu_time_std_dev),
            // Assembly Time
            format!("{:.6}", result.assembly_time_min), format!("{:.6}", result.assembly_time_median), format!("{:.6}", result.assembly_time_max), format!("{:.6}", result.assembly_time_mean), format!("{:.6}", result.assembly_time_std_dev),
            // Partition Processing Time (Base Case)
            format!("{:.6}", result.partition_processing_time_min), format!("{:.6}", result.partition_processing_time_median), format!("{:.6}", result.partition_processing_time_max), format!("{:.6}", result.partition_processing_time_mean), format!("{:.6}", result.partition_processing_time_std_dev),
            // Other Time
            format!("{:.6}", result.other_time_min), format!("{:.6}", result.other_time_median), format!("{:.6}", result.other_time_max), format!("{:.6}", result.other_time_mean), format!("{:.6}", result.other_time_std_dev),
            // Other Metrics
            result.mismatches.to_string(), result.nodes.to_string(), result.height.to_string(), format!("{:.3}", result.accuracy)
        ];
        
        #[cfg(feature = "detailed-stats")]
        {
            record.push(result.simplify_pruning_count.to_string());
            record.push(result.merge_pruning_count.to_string());
            record.push(result.kept_count.to_string());
        }

        wtr.write_record(&record)?;
    }

    wtr.flush()?;
    Ok(())
}

// Exports the benchmark results to a CSV file. DEPRECATED in favor of incremental writing.
// Kept for reference or potential future use if needed to write all at once.
// pub fn _export_all_results_to_csv(results: &[BenchmarkResult], dataset_name: &str) -> Result<PathBuf, Box<dyn Error>> {
//     let results_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("results");
//     std::fs::create_dir_all(&results_dir)?;
//     let csv_filename = results_dir.join(format!("benchmark_results_{}_all_seeds_COMPLETE.csv", dataset_name)); // Changed name slightly
//     println!("\n[EXPORT] Writing ALL results for all seeds to {:?}", csv_filename);

//     // Write header first (using the helper)
//     write_csv_header(&csv_filename)?;
//     // Append all results (using the helper)
//     append_results_to_csv(results, &csv_filename)?;

//     println!("[EXPORT] All results successfully written.");
//     Ok(csv_filename)
// }