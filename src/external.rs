use std::{error::Error, path::{Path, PathBuf}, process::Command};

/// Attempts to execute the `visualize_benchmarks.py` script to generate plots from the CSV data.
pub fn run_visualization(csv_path: &Path, dataset_name: &str, results_dir: &Path) -> Result<(), Box<dyn Error>> {
    println!("[VISUALIZE] Attempting to generate plot...");
    let script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("visualize_benchmarks.py");
    let csv_path_str = csv_path.to_str().ok_or("CSV path is not valid UTF-8")?;
    let output_dir_str = results_dir.to_str().ok_or("Results directory path is not valid UTF-8")?;
    let script_path_str = script_path.to_str().ok_or("Script path is not valid UTF-8")?;

    let python_executable = if cfg!(windows) { "python" } else { "python3" };
    let mut cmd = Command::new(python_executable);
    cmd.arg(script_path_str)
       .arg("-d")
       .arg(dataset_name);

    let output = cmd.output();

    match output {
        Ok(output_data) => {
            if output_data.status.success() {
                println!("[VISUALIZE] Python script executed successfully.");
            } else {
                eprintln!("[VISUALIZE] Python script execution failed with status: {}", output_data.status);
                if !output_data.stderr.is_empty() {
                    eprintln!("Script error:\n{}", String::from_utf8_lossy(&output_data.stderr));
                }
                 // Try falling back to 'python' if 'python3' failed (non-Windows)
                 if !cfg!(windows) && python_executable == "python3" {
                    println!("[VISUALIZE] Retrying with 'python'...");
                    let mut cmd_fallback = Command::new("python");
                    cmd_fallback.arg(script_path_str)
                       .arg("--csv_path")
                       .arg(csv_path_str)
                       .arg("--dataset_name")
                       .arg(dataset_name)
                       .arg("--output_dir")
                       .arg(output_dir_str);
                    let output_fallback = cmd_fallback.output();
                    match output_fallback {
                         Ok(fallback_data) => {
                             if fallback_data.status.success() {
                                 println!("[VISUALIZE] Python script executed successfully with 'python'.");
                             } else {
                                 eprintln!("[VISUALIZE] Python script execution failed again with 'python'. Status: {}", fallback_data.status);
                                 if !fallback_data.stderr.is_empty() {
                                     eprintln!("Script error:\n{}", String::from_utf8_lossy(&fallback_data.stderr));
                                 }
                             }
                         },
                         Err(e) => eprintln!("[VISUALIZE] Failed to execute 'python': {}", e),
                    }
                 }
            }
        }
        Err(e) => {
            eprintln!("[VISUALIZE] Failed to execute Python script '{}': {}", python_executable, e);
            eprintln!("Ensure Python is installed and '{}' or 'python' is in your system PATH.", python_executable);
        }
    }
    Ok(())
}
/// Attempts to execute the `benchmarks_by_comparison.py` script to generate plots from the CSV data.
pub fn run_comparison(original_path: &Path, egap_path: &Path, dataset_name: &str, results_dir: &Path) -> Result<(), Box<dyn Error>> {
    println!("[VISUALIZE] Attempting to generate plot...");
    let script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks_by_comparison.py");
    let original_path_str = original_path.to_str().ok_or("CSV path is not valid UTF-8")?;
    let egap_path_str = egap_path.to_str().ok_or("CSV path is not valid UTF-8")?;
    let output_dir_str = results_dir.to_str().ok_or("Results directory path is not valid UTF-8")?;
    let script_path_str = script_path.to_str().ok_or("Script path is not valid UTF-8")?;

    let python_executable = if cfg!(windows) { "python" } else { "python3" };
    let mut cmd = Command::new(python_executable);
    cmd.arg(script_path_str)
       .arg("-d")
       .arg(dataset_name);

    let output = cmd.output();

    match output {
        Ok(output_data) => {
            if output_data.status.success() {
                println!("[VISUALIZE] Python script executed successfully.");
            } else {
                eprintln!("[VISUALIZE] Python script execution failed with status: {}", output_data.status);
                if !output_data.stderr.is_empty() {
                    eprintln!("Script error:\n{}", String::from_utf8_lossy(&output_data.stderr));
                }
                 // Try falling back to 'python' if 'python3' failed (non-Windows)
                 if !cfg!(windows) && python_executable == "python3" {
                    println!("[VISUALIZE] Retrying with 'python'...");
                    let mut cmd_fallback = Command::new("python");
                    cmd_fallback.arg(script_path_str)
                       .arg("--original_path")
                       .arg(original_path_str)
                       .arg("--egap_path")
                       .arg(egap_path_str)
                       .arg("--dataset_name")
                       .arg(dataset_name)
                       .arg("--output_dir")
                       .arg(output_dir_str);
                    let output_fallback = cmd_fallback.output();
                    match output_fallback {
                         Ok(fallback_data) => {
                             if fallback_data.status.success() {
                                 println!("[VISUALIZE] Python script executed successfully with 'python'.");
                             } else {
                                 eprintln!("[VISUALIZE] Python script execution failed again with 'python'. Status: {}", fallback_data.status);
                                 if !fallback_data.stderr.is_empty() {
                                     eprintln!("Script error:\n{}", String::from_utf8_lossy(&fallback_data.stderr));
                                 }
                             }
                         },
                         Err(e) => eprintln!("[VISUALIZE] Failed to execute 'python': {}", e),
                    }
                 }
            }
        }
        Err(e) => {
            eprintln!("[VISUALIZE] Failed to execute Python script '{}': {}", python_executable, e);
            eprintln!("Ensure Python is installed and '{}' or 'python' is in your system PATH.", python_executable);
        }
    }
    Ok(())
}