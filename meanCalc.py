import pandas as pd
from dataclasses import dataclass
from typing import List

# =========================
# CONFIGURAÇÃO
# =========================
DATASET_NAME = "iris"  # <-- altere aqui
ORIGINAL_CSV = "results/benchmark_results_iris_all_seeds.csv"
EGAP_CSV = "results/egap_benchmark_results_iris_all_seeds.csv"

# =========================
# STRUCT (dataclass)
# =========================
@dataclass
class Comparison:
    dataset_name: str
    original_mean_time: List[float]
    egap_mean_time: List[float]
    original_mean_accuracy: List[float]
    egap_mean_accuracy: List[float]

# =========================
# FUNÇÕES
# =========================
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Strategy")[["TotalTime_Mean", "Accuracy"]]
          .mean()
          .reset_index()
          .sort_values("Strategy")  # garante alinhamento
    )

# =========================
# PIPELINE
# =========================
# Lê os CSVs
df_original = pd.read_csv(ORIGINAL_CSV)
df_egap = pd.read_csv(EGAP_CSV)

# Calcula métricas
metrics_original = compute_metrics(df_original)
metrics_egap = compute_metrics(df_egap)

# Merge para comparação e print
comparison_df = metrics_original.merge(
    metrics_egap,
    on="Strategy",
    suffixes=("_original", "_egap")
)
print(DATASET_NAME)
print(comparison_df)

# =========================
# CRIA A STRUCT
# =========================
comparison = Comparison(
    dataset_name=DATASET_NAME,
    original_mean_time=metrics_original["TotalTime_Mean"].tolist(),
    egap_mean_time=metrics_egap["TotalTime_Mean"].tolist(),
    original_mean_accuracy=metrics_original["Accuracy"].tolist(),
    egap_mean_accuracy=metrics_egap["Accuracy"].tolist(),
)

# (opcional) visualizar a struct
#print("\nStruct Comparison:")
#print(comparison)