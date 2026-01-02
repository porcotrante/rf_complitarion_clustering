import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

from visualize_benchmarks import aggregate_results

STRATEGY_AESTHETICS = {
    'ES': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
    'AbsES': {'color': 'green', 'marker': 's', 'linestyle': '--'},
    'HEUR': {'color': 'red', 'marker': '^', 'linestyle': ':'},
    'ORD': {'color': 'purple', 'marker': 'D', 'linestyle': '-.'}
}

def plot_strategy_time_comparison(
    original_agg_df,
    egap_agg_df,
    dataset_name,
    output_dir,
    metric_col='TotalTime_Min_mean'
):
    """
    Compares Original vs EGAP aggregated results by Strategy.
    Assumes k is constant and data is already aggregated.
    """

    # --- Align dataframes by Strategy ---
    df_plot = pd.merge(
        original_agg_df[['Strategy', metric_col]],
        egap_agg_df[['Strategy', metric_col]],
        on='Strategy',
        suffixes=('_Original', '_EGAP')
    ).sort_values('Strategy')

    strategies = df_plot['Strategy'].astype(str).tolist()
    x = np.arange(len(strategies))
    bar_width = 0.35

    # --- Plot ---
    plt.figure(figsize=(6 + 2 * len(strategies), 5))
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.bar(
        x - bar_width / 2,
        df_plot[f'{metric_col}_Original'],
        width=bar_width,
        label='Original'
    )

    plt.bar(
        x + bar_width / 2,
        df_plot[f'{metric_col}_EGAP'],
        width=bar_width,
        label='EGAP'
    )

    plt.xticks(x, strategies)
    plt.xlabel('Strategy', fontsize=12)
    plt.ylabel('Mean Min Wall Time (seconds)', fontsize=12)

    plt.title(
        f'Mean Min Wall Time per Strategy\nOriginal vs EGAP — Dataset: {dataset_name}',
        fontsize=16
    )

    plt.legend()
    plt.tight_layout()

    # --- Save ---
    plot_filename = os.path.join(
        output_dir,
        f'benchmark_strategy_comparison_{dataset_name}.svg'
    )

    plt.savefig(plot_filename)
    print(f"Strategy comparison plot saved as '{plot_filename}'")
    plt.close()

def plot_strategy_accuracy_comparison(
    original_agg_df,
    egap_agg_df,
    dataset_name,
    output_dir,
    metric_col='Accuracy_mean'
):
    """
    Compares accuracy between Original and EGAP aggregated by Strategy.
    Assumes 'accuracy' column exists in both aggregated dataframes.
    """

    # --- Align dataframes by Strategy ---
    df_plot = pd.merge(
        original_agg_df[['Strategy', metric_col]],
        egap_agg_df[['Strategy', metric_col]],
        on='Strategy',
        suffixes=('_Original', '_EGAP')
    ).sort_values('Strategy')

    strategies = df_plot['Strategy'].astype(str).tolist()
    x = np.arange(len(strategies))
    bar_width = 0.35

    # --- Plot ---
    plt.figure(figsize=(6 + 2 * len(strategies), 5))
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.bar(
        x - bar_width / 2,
        df_plot[f'{metric_col}_Original'],
        width=bar_width,
        label='Original'
    )

    plt.bar(
        x + bar_width / 2,
        df_plot[f'{metric_col}_EGAP'],
        width=bar_width,
        label='EGAP'
    )

    plt.xticks(x, strategies)
    plt.xlabel('Strategy', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)

    plt.title(
        f'Accuracy per Strategy\nOriginal vs EGAP — Dataset: {dataset_name}',
        fontsize=16
    )

    plt.ylim(0, 1.0)  # acurácia normalizada
    plt.legend()
    plt.tight_layout()

    # --- Save ---
    plot_filename = os.path.join(
        output_dir,
        f'benchmark_strategy_accuracy_comparison_{dataset_name}.svg'
    )

    plt.savefig(plot_filename)
    print(f"Strategy accuracy comparison plot saved as '{plot_filename}'")
    plt.close()

def visualize_comparison(original_path, egap_path, dataset_name, base_output_dir):
    if not os.path.exists(original_path):
        print(f"Error: CSV file not found at '{original_path}'")
        return
    
    if not os.path.exists(egap_path):
        print(f"Error: CSV file not found at '{egap_path}'")
        return

    try:
        original_df = pd.read_csv(original_path)
    except Exception as e:
        print(f"Error reading Original CSV file: {e}")
        return

    if original_df.empty:
        print("Error: Oginial CSV file is empty or contains no data.")
        return

    try:
        egap_df = pd.read_csv(egap_path)
    except Exception as e:
        print(f"Error reading Egap CSV file: {e}")
        return

    if egap_df.empty:
        print("Error: Egap CSV file is empty or contains no data.")
        return
    
    output_dir = os.path.join(base_output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")

    original_agg_df = aggregate_results(original_df.copy())
    egap_agg_df = aggregate_results(egap_df.copy())

    strategy_order = ['ES', 'AbsES', 'HEUR', 'ORD']

    present_strategies = original_agg_df['Strategy'].unique()
    ordered_present_strategies = [s for s in strategy_order if s in present_strategies]
    
    if not ordered_present_strategies:
        print("Error: None of the specified strategies found in the 'Strategy' column of aggregated data.")
        return
    
    original_agg_df['Strategy'] = pd.Categorical(original_agg_df['Strategy'], categories=ordered_present_strategies, ordered=True)
    egap_agg_df['Strategy'] = pd.Categorical(egap_agg_df['Strategy'], categories=ordered_present_strategies, ordered=True)

    plot_strategy_time_comparison(
        original_agg_df,
        egap_agg_df,
        dataset_name,
        output_dir
    )

    plot_strategy_accuracy_comparison(
        original_agg_df,
        egap_agg_df,
        dataset_name,
        output_dir
    )

    
if __name__ == "__main__":
    # Usage Example: python visualize_benchmarks.py -d banknote
    parser = argparse.ArgumentParser(description="Visualize compile-rf benchmark results, inferring paths.")
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Name of the dataset (e.g., magic, banknote). Used to infer CSV path.')

    args = parser.parse_args()

    # Infer paths based on dataset name
    dataset_name = args.dataset
    base_output_dir = "results" # Base directory for results
    original_filename = f"benchmark_results_{dataset_name}_all_seeds.csv"
    egap_filename = f"egap_benchmark_results_{dataset_name}_all_seeds.csv"
    # Assume CSV is directly in the base_output_dir
    original_path = os.path.join(base_output_dir, original_filename)
    egap_path = os.path.join(base_output_dir, egap_filename)

    print(f"Dataset: {dataset_name}")
    print(f"Inferred Original CSV Path: {original_path}")
    print(f"Inferred Egap CSV Path: {egap_path}")
    print(f"Base Output Directory: {base_output_dir}")

    # Call the visualization function with inferred/fixed paths
    visualize_comparison(original_path, egap_path, dataset_name, base_output_dir)