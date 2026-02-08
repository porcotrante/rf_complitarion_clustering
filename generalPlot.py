# =========================
# CRIA A STRUCT
# =========================
from typing import List
from dataclasses import dataclass


@dataclass
class Comparison:
    dataset_name: str
    original_mean_time: List[float]
    egap_mean_time: List[float]
    original_mean_accuracy: List[float]
    egap_mean_accuracy: List[float]

comparisons = [

    Comparison(
        dataset_name="banknote",
        original_mean_time=[23.371246, 25.059254, 15.097239, 4.591239],
        egap_mean_time=[10.037311, 9.798459, 6.651473, 3.970800],
        original_mean_accuracy=[96.8146, 96.8146, 96.8146, 96.8146],
        egap_mean_accuracy=[97.037, 97.037, 97.037, 97.037],
    ),

    Comparison(
        dataset_name="ecoli",
        original_mean_time=[418.261051, 202.242370, 187.877388, 64.420287],
        egap_mean_time=[121.394482, 54.833138, 52.577212, 28.682314],
        original_mean_accuracy=[85.606, 85.606, 85.606, 85.606],
        egap_mean_accuracy=[85.606, 85.606, 85.606, 85.606],
    ),

    Comparison(
        dataset_name="glass2",
        original_mean_time=[45.638739, 23.585665, 22.253137, 16.102551],
        egap_mean_time=[23.671755, 11.666625, 11.513025, 8.041798],
        original_mean_accuracy=[76.9696, 76.9696, 76.9696, 76.9696],
        egap_mean_accuracy=[75.7576, 75.7576, 75.7576, 75.7576],
    ),

    Comparison(
        dataset_name="iris",
        original_mean_time=[0.546056, 0.315266, 0.207078, 0.181927],
        egap_mean_time=[0.142001, 0.054570, 0.060331, 0.087005],
        original_mean_accuracy=[94.0, 94.0, 94.0, 94.0],
        egap_mean_accuracy=[94.6667, 94.6667, 94.6667, 94.6667],
    ),

    Comparison(
        dataset_name="magic",
        original_mean_time=[328.888145, 243.474653, 242.395150, 217.818362],
        egap_mean_time=[13.797572, 6.769391, 6.949963, 6.108970],
        original_mean_accuracy=[80.997, 80.997, 80.997, 80.997],
        egap_mean_accuracy=[81.0635, 81.0635, 81.0635, 81.0635],
    ),

    Comparison(
        dataset_name="segmentation",
        original_mean_time=[70.447441, 42.400211, 32.143327, 37.339492],
        egap_mean_time=[4.962065, 2.415110, 2.325787, 2.356564],
        original_mean_accuracy=[85.714, 85.714, 85.714, 85.714],
        egap_mean_accuracy=[85.714, 85.714, 85.714, 85.714],
    ),

    Comparison(
        dataset_name="shuttle",
        original_mean_time=[16.657736, 11.527301, 10.308572, 4.681688],
        egap_mean_time=[1.287089, 0.631073, 0.591181, 0.529137],
        original_mean_accuracy=[0.1186, 0.1186, 0.1186, 0.1186],
        egap_mean_accuracy=[0.1259, 0.1259, 0.1259, 0.1259],
    ),

    Comparison(
        dataset_name="default-credit",
        original_mean_time=[69.618340, 41.175384, 40.275893, 40.655634],
        egap_mean_time=[1.354086, 0.567144, 0.528345, 0.887602],
        original_mean_accuracy=[81.567, 81.567, 81.567, 81.567],
        egap_mean_accuracy=[82.05, 82.05, 82.05, 82.05],
    ),
]

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 20,            # tamanho base
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "legend.title_fontsize": 22
})

STRATEGIES = ["AbsES", "ES", "HEUR", "ORD"]
n_strategies = len(STRATEGIES)
n_datasets = len(comparisons)

# largura das barras e dos grupos
bar_width = 0.18
group_spacing = 0.3
group_width = n_strategies * bar_width + group_spacing

# posições dos grupos
x_groups = np.arange(n_datasets) * group_width

fig, ax = plt.subplots(figsize=(18, 6))

# cores consistentes por estratégia
colors = plt.cm.tab10.colors  # ou defina manualmente

for j, strategy in enumerate(STRATEGIES):
    speedups = [
        comp.original_mean_time[j] / comp.egap_mean_time[j]
        for comp in comparisons
    ]

    ax.bar(
        x_groups + j * bar_width,
        speedups,
        width=bar_width,
        label=strategy,
        color=colors[j]
    )

# eixo Y
ax.set_yscale("log", base=2)
ax.axhline(1.0, linestyle="--", linewidth=1, color="gray")
ax.set_ylabel("Speedup (Original / EGAP)")

# eixo X → nomes dos datasets centralizados nos grupos
ax.set_xticks(x_groups + (n_strategies - 1) * bar_width / 2)
ax.set_xticklabels([comp.dataset_name for comp in comparisons])

# legenda
ax.legend(title="Strategy", loc="upper left", bbox_to_anchor=(1.01, 1))

# título
ax.set_title("Speedup of EGAP over Original")

plt.tight_layout()
plt.savefig("speedup.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
axes = axes.flatten()

x = np.arange(len(STRATEGIES))
width = 0.35

axes[0].legend()
fig.suptitle("Average Execution Time (Log Scale)", fontsize=16)
plt.tight_layout()
plt.savefig("average_time_log.pdf", format="pdf")
plt.close(fig)

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
axes = axes.flatten()

for ax, comp in zip(axes, comparisons):
    ax.plot(STRATEGIES, comp.original_mean_accuracy, marker="o", label="Original")
    ax.plot(STRATEGIES, comp.egap_mean_accuracy, marker="o", label="EGAP")

    ax.set_title(comp.dataset_name)
    ax.set_ylabel("Accuracy (%)")

fig.legend(
    loc="center left",
    bbox_to_anchor=(1.0, 1),
    fontsize=10,
    frameon=False
)

fig.suptitle("Accuracy Comparison", fontsize=16)
plt.tight_layout()
plt.savefig("accuracy.pdf", format="pdf")
plt.close(fig)

import math

print("=" * 80)
print("SPEEDUP REPORT (Original / EGAP)")
print("=" * 80)

for comp in comparisons:
    print(f"\nDataset: {comp.dataset_name}")
    print("-" * 80)
    print(f"{'Strategy':<10} | {'Raw':>10} | {'log2':>10} | {'% Gain':>10}")
    print("-" * 80)

    for j, strategy in enumerate(STRATEGIES):
        raw_speedup = comp.original_mean_time[j] / comp.egap_mean_time[j]
        log_speedup = math.log2(raw_speedup)
        percent_gain = (raw_speedup - 1.0) * 100.0

        print(
            f"{strategy:<10} | "
            f"{raw_speedup:>10.3f} | "
            f"{log_speedup:>10.3f} | "
            f"{percent_gain:>9.1f}%"
        )

print("\nDone.")