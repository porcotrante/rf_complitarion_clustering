from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


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


def compute_feature_vector(tree: DecisionTreeClassifier, num_features: int):
    """
    Compute a structural feature vector for a scikit-learn DecisionTreeClassifier.

    Parameters
    ----------
    tree : DecisionTreeClassifier
        A fitted decision tree.
    num_features : int
        Total number of input features.

    Returns
    -------
    np.ndarray
        Feature vector of shape (num_features,)
    """
    tree_ = tree.tree_

    values = np.zeros(num_features, dtype=float)
    max_depth = np.zeros(num_features, dtype=int)

    # stack: (node_id, depth)
    stack = [(0, 1)]  # root node is always 0

    while stack:
        node_id, depth = stack.pop()

        feature = tree_.feature[node_id]

        # Leaf node
        if feature == -2:
            continue

        weight = 1.0 / (2 ** (depth - 1))
        values[feature] += weight
        max_depth[feature] = max(max_depth[feature], depth)

        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]

        stack.append((left, depth + 1))
        stack.append((right, depth + 1))

    # Normalize by the maximum depth at which the feature appears
    for f in range(num_features):
        if max_depth[f] > 0:
            values[f] /= max_depth[f]

    return values


def tree_structural_distance(
    tree1: DecisionTreeClassifier,
    tree2: DecisionTreeClassifier,
    num_features: int,
) -> float:
    """
    Compute the squared Euclidean distance between the structural
    feature vectors of two decision trees.
    """
    v1 = compute_feature_vector(tree1, num_features)
    v2 = compute_feature_vector(tree2, num_features)

    diff = v1 - v2
    return np.sum(diff ** 2)

def average_intracluster_distance(
    labels: dict[int, int],
    rf: RandomForestClassifier,
    num_features: int,
) -> float:
    """
    Computes the weighted average intracluster structural distance
    between trees in a Random Forest.

    Parameters
    ----------
    labels : dict[int, int]
        Mapping from original tree id to cluster id.
    rf : RandomForestClassifier
        Trained random forest model from scikit-learn.
    num_features : int
        Number of features used by the model.

    Returns
    -------
    float
        Weighted average intracluster distance.
    """

    # Agrupa árvores por cluster
    clusters: dict[int, list[DecisionTreeClassifier]] = defaultdict(list)

    for tree_id, cluster_id in labels.items():
        clusters[cluster_id].append(rf.estimators_[tree_id])

    total_weighted_distance = 0.0
    total_trees = 0

    for cluster_id, trees in clusters.items():
        cluster_size = len(trees)

        # Clusters com menos de 2 árvores não contribuem
        if cluster_size < 2:
            continue

        # Distância média intracluster
        distances = [
            tree_structural_distance(t1, t2, num_features)
            for t1, t2 in combinations(trees, 2)
        ]

        mean_cluster_distance = sum(distances) / len(distances)

        # Ponderação pelo tamanho do cluster
        total_weighted_distance += mean_cluster_distance * cluster_size
        total_trees += cluster_size

    if total_trees == 0:
        return 0.0

    return total_weighted_distance / total_trees