use std::fmt;

use crate::transform::EarlyStoppingStrategy;

// --- Dataset Configurations ---
// Defines constants for different datasets used in benchmarks.
// Format: (dataset_name_string, number_of_features, number_of_classes)
const BANKNOTE_CONFIG: (&str, usize, usize) = ("banknote", 4, 2);
const MAGIC_CONFIG: (&str, usize, usize) = ("magic", 10, 2);
const ECOLI_CONFIG: (&str, usize, usize) = ("ecoli", 7, 5);
const GLASS2_CONFIG: (&str, usize, usize) = ("glass2", 9, 2);
const IONOSPHERE_CONFIG: (&str, usize, usize) = ("ionosphere", 34, 2);
const IRIS_CONFIG: (&str, usize, usize) = ("iris", 4, 3);
const SEGMENTATION_CONFIG: (&str, usize, usize) = ("segmentation", 19, 7);
const SHUTTLE_CONFIG: (&str, usize, usize) = ("shuttle", 7, 7);
const HEART2_CONFIG: (&str, usize, usize) = ("heart2", 22, 2);
const ADULT_CONFIG: (&str, usize, usize) = ("adult", 107, 2);
const MUSHROOM_CONFIG: (&str, usize, usize) = ("mushroom", 111, 2);
const DEFAULT_CREDIT_CONFIG: (&str, usize, usize) = ("default-credit", 32, 2); // Default dataset for credit card fraud detection

// --- Benchmarking Configuration ---
pub const CHOSEN_CONFIG: (&str, usize, usize) = IRIS_CONFIG; // Change this to switch datasets
pub const K_VALUES: [usize; 1] = [0]; // k values to test (remember to change [usize; N] to the correct number)
pub const NUM_BENCHMARK_RUNS: usize = 1; // Number of times to run each timed operation (per seed/k/strategy)
pub const BASE_SEED: u32 = 42; // Starting seed, matching train.py
pub const NUM_SEEDS_TO_RUN: u32 = 1; // Number of seeds to run, matching train.py

// --- Strategies to test ---
// (remember to change the number N in [EarlyStoppingStrategy; N])
pub const STRATEGIES_TO_TEST: [EarlyStoppingStrategy; 4] = [
    EarlyStoppingStrategy::Standard,
     EarlyStoppingStrategy::Abstract,
     EarlyStoppingStrategy::HEUR,
     EarlyStoppingStrategy::ORD,
];

// Derived information from the chosen configuration
pub const DATASET_NAME: &str = CHOSEN_CONFIG.0; // Name of the benchmark
pub const NUM_FEATURES: usize = CHOSEN_CONFIG.1; // Number of features in the dataset
pub const NUM_CLASSES: usize = CHOSEN_CONFIG.2; // Number of classes in the dataset

// Implement Display for EarlyStoppingStrategy for table printing
impl fmt::Display for EarlyStoppingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EarlyStoppingStrategy::Standard => write!(f, "ES"),  
            EarlyStoppingStrategy::Abstract => write!(f, "AbsES"),
            EarlyStoppingStrategy::HEUR => write!(f, "HEUR"),
            EarlyStoppingStrategy::ORD => write!(f, "ORD"),
        }
    }
}