from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass
class Path:
    splits: list          # lista de dicts: { "feature": int, "threshold": float }
    predicted_class: int

def extract_paths_from_tree(decision_tree):
    """
    Extrai todos os caminhos da árvore.
    Cada split carrega feature, threshold e direção.
    """
    tree = decision_tree.tree_
    paths = []

    def traverse(node_id, splits):
        # Nó folha
        if tree.children_left[node_id] == -1 and tree.children_right[node_id] == -1:
            predicted_class = np.argmax(tree.value[node_id])
            paths.append(Path(splits.copy(), predicted_class))
            return

        # Nó interno
        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]

        # Caminho para a esquerda (<= threshold)
        left_split = {
            "feature": feature,
            "threshold": threshold,
            "direction": "left"
        }
        traverse(
            tree.children_left[node_id],
            splits + [left_split]
        )

        # Caminho para a direita (> threshold)
        right_split = {
            "feature": feature,
            "threshold": threshold,
            "direction": "right"
        }
        traverse(
            tree.children_right[node_id],
            splits + [right_split]
        )

    traverse(0, [])
    return paths


def extract_paths_from_forest(rf: RandomForestClassifier):
    """
    Retorna uma lista de listas de caminhos (um Path[] para cada árvore).
    """
    all_paths = []
    for estimator in rf.estimators_:
        all_paths.append(extract_paths_from_tree(estimator))
    return all_paths

def extract_feature_intervals(splits):
    """
    Retorna um dicionário:
        feature -> (lower_bound, upper_bound)
    """
    intervals = {}

    for s in splits:
        f = s["feature"]
        t = s["threshold"]
        direction = s["direction"]

        if f not in intervals:
            intervals[f] = [-np.inf, np.inf]

        if direction == "left":      # feature <= t
            intervals[f][1] = min(intervals[f][1], t)
        else:                        # feature > t
            intervals[f][0] = max(intervals[f][0], t)

    return intervals

def interval_distance(i1, i2):
    a1, b1 = i1
    a2, b2 = i2

    # sobreposição
    if b1 >= a2 and b2 >= a1:
        return 0.0

    # distância mínima entre bordas
    return min(abs(a2 - b1), abs(a1 - b2))

def distance_between_paths(p1: Path, p2: Path):
    """
    Distância baseada em intervalos induzidos pelos splits.
    """

    splits1 = p1.splits
    splits2 = p2.splits

    F1 = len(splits1)
    if F1 == 0:
        return 0.0

    intervals1 = extract_feature_intervals(splits1)
    intervals2 = extract_feature_intervals(splits2)

    distances = []

    for feature, interval1 in intervals1.items():
        if feature in intervals2:
            interval2 = intervals2[feature]
            d = interval_distance(interval1, interval2)
            distances.append(d)
        else:
            # penalidade por feature ausente
            distances.append(1.0)

    return sum(distances) / len(intervals1)

def distance_between_trees(paths1, paths2):
    """
    Calcula a distância entre duas árvores de decisão
    representadas por listas de objetos Path.
    """

    if len(paths1) == 0:
        return 0.0

    total = 0.0

    # Indexar paths2 por classe para otimizar
    class_index = {}
    for p in paths2:
        class_index.setdefault(p.predicted_class, []).append(p)

    for p1 in paths1:
        same_class_paths = class_index.get(p1.predicted_class, None)

        if not same_class_paths:
            # Nenhum caminho com mesma classe → penalidade máxima
            total += 1.0
            continue

        # Calcular distâncias e pegar a menor
        best = min(distance_between_paths(p1, p2) for p2 in same_class_paths)
        total += best

    return total / len(paths1)