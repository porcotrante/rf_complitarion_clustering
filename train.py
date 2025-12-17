# !pip install ucimlrepo # if in Colab
from collections import Counter, defaultdict
import csv
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.tree._tree import Tree
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn_extra.cluster import KMedoids
from ucimlrepo import fetch_ucirepo
import json
import os
import argparse
import sys
import numpy as np
import pandas as pd
from typing import List, Set, Union, Callable, Optional
from sklearn.preprocessing import LabelEncoder

from paths import distance_between_trees, extract_paths_from_forest

# --- Configuration ---
BASE_SEED = 42
NUM_SEEDS_TO_RUN = 10 # Number of different RFs to train and export
CLUSNUM = 10

# --- Preprocessing Functions ---
def create_binary_preprocessor(dataset_name: str, classes_to_keep: Union[List[int], Set[int]], zero_class: int):
    """
    Factory function to create a binary classification preprocessor.
    Accepts and returns (pd.DataFrame, np.ndarray).
    Filters rows based on original y and relabels y.
    """
    if zero_class not in classes_to_keep:
        raise ValueError(f"zero_class ({zero_class}) must be in classes_to_keep ({classes_to_keep})")

    # Ensure classes_to_keep is a set for efficient lookup
    _classes_to_keep = set(classes_to_keep)

    def _preprocessor(X_df: pd.DataFrame, y: np.ndarray):
        """The actual preprocessing function returned by the factory."""
        print(f"Applying {dataset_name} binary preprocessing...")
        y = y.astype(int) # Ensure y is integer type

        # Create mask based on y to keep only specified classes
        keep_mask = np.isin(y, list(_classes_to_keep))

        # Filter both X DataFrame and y array using the mask
        X_df_filtered = X_df.loc[keep_mask].reset_index(drop=True)
        y_filtered = y[keep_mask]

        # Relabel: zero_class -> 0; other kept classes -> 1
        y_relabeled = np.where(y_filtered == zero_class, 0, 1)

        print(f"{dataset_name} preprocessing: Kept {len(y_relabeled)} instances out of {len(y)}.")
        print(f"New class distribution: {np.unique(y_relabeled, return_counts=True)}")
        # Return filtered DataFrame and relabeled y array
        return X_df_filtered, y_relabeled

    return _preprocessor

def create_class_filter_preprocessor(dataset_name: str, classes_to_remove: Union[List[int], Set[int]]):
    """
    Factory function to create a preprocessor that removes specified classes.
    Accepts and returns (pd.DataFrame, np.ndarray).
    Filters rows based on original y values *before* any relabeling.
    Important: This should typically run *before* generic LabelEncoding if using original class labels/indices.
    """
    _classes_to_remove = set(classes_to_remove) # Ensure set for efficient lookup

    def _filter_preprocessor(X_df: pd.DataFrame, y: np.ndarray):
        """The actual filtering function returned by the factory."""
        print(f"Applying {dataset_name} class filtering: Removing classes {_classes_to_remove}...")
        original_y_dtype = y.dtype
        print(f"  Input y dtype: {original_y_dtype}, unique values: {np.unique(y)}")

        # Create mask to *keep* rows NOT in the removal set
        # Ensure y is compatible for comparison (e.g., integer if classes_to_remove are ints)
        try:
            # Attempt conversion if y is not already suitable for comparison
            if not np.can_cast(y.dtype, np.min_scalar_type(list(_classes_to_remove))):
                 print(f"  Attempting to cast y from {y.dtype} for comparison...")
                 y_comparable = y.astype(np.min_scalar_type(list(_classes_to_remove)))
            else:
                 y_comparable = y
        except Exception as e:
             print(f"  Warning: Could not safely cast y for comparison ({e}). Filtering might fail.")
             y_comparable = y # Proceed with original y, hoping for the best

        keep_mask = ~np.isin(y_comparable, list(_classes_to_remove))

        # Filter both X DataFrame and y array using the mask
        X_df_filtered = X_df.loc[keep_mask].reset_index(drop=True)
        y_filtered = y[keep_mask] # Keep original y values for now, let subsequent steps handle relabeling

        print(f"  {dataset_name} filtering: Kept {len(y_filtered)} instances out of {len(y)}.")
        print(f"  Remaining unique y values (before potential relabeling): {np.unique(y_filtered)}")
        # Return filtered DataFrame and filtered y array (still with original labels)
        return X_df_filtered, y_filtered

    return _filter_preprocessor

def preprocess_adult_target(X_df: pd.DataFrame, y: np.ndarray):
    """
    Cleans and maps the target variable for the 'Adult' dataset.
    Removes trailing dots and maps '<=50K' -> 0, '>50K' -> 1.
    Accepts (pd.DataFrame, np.ndarray) and returns (pd.DataFrame, np.ndarray).
    """
    print("Applying Adult dataset target preprocessing...")
    if y.dtype == 'object':
        # Remove trailing dots and whitespace
        y_cleaned = np.char.strip(y.astype(str), ' .')
        # Map to 0 and 1
        le = LabelEncoder()
        y_processed = le.fit_transform(y_cleaned) # Should map '<=50K' to 0 and '>50K' to 1 if those are the only unique values after cleaning
        print(f"  Cleaned target labels. Original unique: {np.unique(y)}, Cleaned unique: {np.unique(y_cleaned)}")
        print(f"  Mapped target labels to: {np.unique(y_processed)}")
        if not np.array_equal(le.classes_, ['<=50K', '>50K']):
             print(f"  Warning: Unexpected classes found after cleaning: {le.classes_}. Mapping might be incorrect.")
        return X_df, y_processed
    else:
        print("  Target variable is not object type. Skipping Adult target cleaning.")
        return X_df, y


def preprocess_features_sklearn(X_df: pd.DataFrame, y: np.ndarray, categorical_features: Optional[List[str]] = None):
    """
    Applies imputation and one-hot encoding using sklearn's ColumnTransformer.
    - Numeric features: Median imputation.
    - Categorical features: Mode imputation followed by One-Hot Encoding.
      * Binary categorical features are encoded into a single 0/1 column.
      * Multi-category features are one-hot encoded as usual.
    Uses provided categorical_features if not None/empty, otherwise auto-detects object/category types.
    Accepts (pd.DataFrame, np.ndarray) and returns (pd.DataFrame, np.ndarray).
    """
    print("Applying Sklearn Preprocessing (Imputation + OHE with binary optimization)...")

    numeric_features = list(X_df.select_dtypes(include=np.number).columns)
    
    # Determine categorical features: Use provided list or auto-detect
    if categorical_features: # If a list is provided (even if empty, though that's unlikely use)
        valid_categorical_features = [col for col in categorical_features if col in X_df.columns]
        print(f"  Using provided categorical features: {valid_categorical_features}")
        # Remove these from numeric features if they were accidentally included
        numeric_features = [col for col in numeric_features if col not in valid_categorical_features]
    else:
        # Auto-detect object/category columns if no list is provided
        valid_categorical_features = list(X_df.select_dtypes(include=['object', 'category']).columns)
        print(f"  Auto-detecting categorical features: {valid_categorical_features}")
        # Numeric features are already correctly identified in this case

    print(f"  Numeric features to process: {numeric_features}")
    print(f"  Categorical features to process: {valid_categorical_features}")

    # Create preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # Use drop='if_binary' to handle binary features efficiently
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32, drop='if_binary'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, valid_categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any)
    )

    # Fit and transform the data
    try:
        X_processed = preprocessor.fit_transform(X_df)
        print(f"  Shape after ColumnTransformer: {X_processed.shape}")

        # Get feature names after transformation
        # Note: Feature names might change significantly after OHE
        feature_names = preprocessor.get_feature_names_out()
        print(f"  New feature names (sample): {feature_names[:10]}...") # Print sample names

        # Convert the processed NumPy array back to a DataFrame with new feature names
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names, dtype=np.float32) # Ensure float32

    except Exception as e:
        print(f"Error during ColumnTransformer execution: {e}")
        # Optionally re-raise or return original data
        raise

    return X_processed_df, y




# Dataset metadata
dataset_metadata = {
    'banknote': {
        'id': 267,
        'test_size': 270,
        'n_trees': 100,
        'max_depth': 4
    },
    'ecoli': {
        'id': 39,
        'test_size': 66,
        'n_trees': 100,
        'max_depth': 4,
        'preprocess': [
            create_class_filter_preprocessor(
                dataset_name='Ecoli',
                # these classes are very rare (summed, they amount < 2.7%)
                # the baseline paper [17] removes 3 classes from ecoli
                # I believe these are the removed classes
                classes_to_remove={'imL', 'imS', 'omL'}
            ),
        ]
    },
    'magic': {
        'id': 159,
        'test_size': 3781,
        'n_trees': 25,
        'max_depth': 4,
    },
    'glass2': {
        'id': 42,
        'test_size': 33,
        'n_trees': 25,
        'max_depth': 4,
        'preprocess': create_binary_preprocessor(
            dataset_name='Glass2',
            classes_to_keep=[1, 2, 3],
            zero_class=2
        )
    },
    'ionosphere': {
        'id': 52,
        'test_size': 70,
        'n_trees': 15,
        'max_depth': 4
    },
    'iris': {
        'id': 53,
        'test_size': 30,
        'n_trees': 100,
        'max_depth': 4
    },
    'segmentation': {
        'id': 50,
        'test_size': 42,
        'n_trees': 15,
        'max_depth': 4
    },
    'shuttle': {
        'id': 148,
        'test_size': 11600,
        'n_trees': 50,
        'max_depth': 4
    },
    'waveform-1': {
        'id': 107,
        'test_size': 1000,
        'n_trees': 15,
        'max_depth': 4
    },
    'wine': {
        'id': 109,
        'test_size': 36,
        'n_trees': 25,
        'max_depth': 4
    },
    'heart2': {
        'id': 45,
        'test_size': 60,
        'n_trees': 15,
        'max_depth': 4,
        # Define preprocessing as a list (pipeline) - Sklearn preprocessor first, then binary filter/relabel
        'preprocess': [
            # Use the combined sklearn preprocessor
            lambda X_df, y: preprocess_features_sklearn(
                X_df, y,
                categorical_features=['cp', 'restecg', 'slope', 'thal'] # Pass categorical feature names
            ),
            # Then apply the binary preprocessor (which expects a DataFrame)
            create_binary_preprocessor(
                dataset_name='Heart2',
                classes_to_keep=[0, 1, 2, 3, 4], # These are original y values before any processing
                zero_class=0
            )
        ]
    },
    'adult': {
        'id': 2,
        'test_size': 9768,
        'n_trees': 15,
        'max_depth': 4,
        'preprocess': [
            preprocess_adult_target, # Clean the target variable first
            lambda X_df, y: preprocess_features_sklearn(X_df, y),
        ]
    },
    'mushroom': {
        'id': 73,
        'test_size': 1624,
        'n_trees': 15,
        'max_depth': 4,
        'preprocess': preprocess_features_sklearn
    },
    'default-credit': {
        'id': 350,
        'test_size': 6000,
        'n_trees': 15,
        'max_depth': 4,
        'preprocess': [
            lambda X_df, y: preprocess_features_sklearn(
                X_df, y,
                categorical_features=['X2', 'X3', 'X4']
            ),
        ]
    }
}


def preprocess_target_variable(X_df: pd.DataFrame, y: np.ndarray):
    """
    Preprocesses the target variable y.
    - If y is object type, uses LabelEncoder to convert to integers.
    - If y is numeric, ensures it is integer type.
    Accepts (pd.DataFrame, np.ndarray) and returns (pd.DataFrame, np.ndarray).
    """
    print("Applying generic target variable preprocessing...")
    if y.dtype == 'object':
        print(f"  Target variable is object type. Original unique values: {np.unique(y)}. Applying LabelEncoder.")
        le = LabelEncoder()
        try:
            y_processed = le.fit_transform(y)
            print(f"  LabelEncoder mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
            print(f"  Processed target unique values: {np.unique(y_processed)}")
            return X_df, y_processed.astype(int) # Ensure integer type after encoding
        except Exception as e:
            print(f"  Error during LabelEncoder transformation: {e}")
            raise ValueError("Failed to encode object type target variable.") from e
    elif pd.api.types.is_numeric_dtype(y):
        # If it's already numeric, ensure it's integer
        if pd.api.types.is_integer_dtype(y):
            print("  Target variable is already numeric integer type.")
            return X_df, y
        else:
            print(f"  Target variable is numeric but not integer ({y.dtype}). Attempting conversion to int.")
            try:
                # Attempt conversion, check for NaNs which can cause issues
                if np.isnan(y).any():
                     print("  Warning: NaNs found in numeric target variable. Cannot convert directly to int. Consider imputation or filtering.")
                     raise ValueError("NaNs found in numeric target variable, cannot convert to int.")
                y_processed = y.astype(int)
                # Verify conversion if possible (e.g., check if float values were truncated)
                if not np.array_equal(y, y_processed.astype(y.dtype)):
                     print("  Warning: Conversion from float to int might have truncated values.")
                print(f"  Successfully converted numeric target to int. Unique values: {np.unique(y_processed)}")
                return X_df, y_processed
            except Exception as e:
                print(f"  Error converting numeric target to int: {e}")
                raise ValueError("Failed to convert numeric target variable to integer.") from e
    else:
        print(f"  Target variable has an unexpected dtype: {y.dtype}. Skipping target preprocessing.")
        return X_df, y

def obter_classificacoes_por_arvore(floresta, X_club, Y_club):
    resultados = {}
    for i, arvore in enumerate(floresta.estimators_):
        predicoes = arvore.predict(X_club)
        acc = accuracy_score(Y_club, predicoes)
        resultados[i] = [predicoes, acc]
    return resultados

def get_trained_rf(config, train_test_data, current_seed: int):
    """Trains RF with a specific seed and returns classifier and accuracy."""
    print(f"Training RF with seed {current_seed}...")
    n_trees = config['n_trees']
    max_depth = config['max_depth']
    X_train, X_test, y_train, y_test = train_test_data # Unpack the split data
    # Use current_seed for RF training
    rf_classifier = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, random_state=current_seed)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    preds = obter_classificacoes_por_arvore(rf_classifier, X_test, y_test)
    print(f"RF Training complete. Test accuracy [seed={current_seed}]: {score * 100:.2f}%")
    return rf_classifier, score, preds

def get_rf_stats(rf: RandomForestClassifier):
    """Extracts node count, unique predicates, and max depth from a trained RF."""
    total_nodes = 0
    unique_predicates = set()
    max_depth = 0 # Initialize max depth for the forest
    if not hasattr(rf, 'estimators_') or not rf.estimators_:
        return total_nodes, unique_predicates, max_depth

    for estimator in rf.estimators_:
        tree_ = estimator.tree_
        if tree_ is None:
            continue
        total_nodes += tree_.node_count
        # Update forest max depth if current tree is deeper
        if hasattr(estimator, 'get_depth'): # Use get_depth() if available
             current_tree_depth = estimator.get_depth()
        elif hasattr(tree_, 'max_depth'): # Fallback to internal attribute
             current_tree_depth = tree_.max_depth
        else: # Estimate if necessary (less accurate)
             # This part might be complex to implement accurately without helper functions
             # For simplicity, we rely on the standard attributes/methods first.
             # If neither exists, we might skip or warn.
             print("Warning: Could not determine tree depth accurately.")
             current_tree_depth = 0 # Default or estimate
        max_depth = max(max_depth, current_tree_depth)

        # Iterate through nodes to find decision nodes (feature != -2)
        for i in range(tree_.node_count):
            feature_index = tree_.feature[i]
            # Use -2 to check for leaf nodes (no feature/threshold)
            if feature_index != -2: # It's a decision node
                threshold = tree_.threshold[i]
                unique_predicates.add((feature_index, threshold))
    # Return total nodes, unique predicates set, and the maximum depth found
    return total_nodes, unique_predicates, max_depth

def export_rf_to_json(rf: RandomForestClassifier, filename: str):
    """Exports RF structure to JSON."""
    if not hasattr(rf, 'estimators_') or not rf.estimators_:
        raise ValueError("The RandomForestClassifier object has no estimators (trees). Is it trained?")
    if not hasattr(rf, 'n_classes_'):
         raise ValueError("The RandomForestClassifier object doesn't have n_classes_. Is it trained?")

    n_nodes = sum(tree.tree_.node_count for tree in rf.estimators_)
    max_depth = max(tree.max_depth for tree in rf.estimators_)

    print(f"Exporting RF: n_trees={len(rf.estimators_)}, n_nodes={n_nodes}, max_depth={max_depth}, n_classes={rf.n_classes_}")

    all_features = []
    all_thresholds = []
    all_children_left = []
    all_children_right = []
    all_values = []

    for i, estimator in enumerate(rf.estimators_):
        if not isinstance(estimator, DecisionTreeClassifier):
            print(f"Warning: Estimator {i} is not a DecisionTreeClassifier, skipping.")
            continue
        tree_ = estimator.tree_
        if tree_ is None:
             print(f"Warning: Estimator {i} has no tree_ attribute, skipping.")
             continue

        features_list = tree_.feature.tolist()
        thresholds_list = tree_.threshold.tolist()
        children_left_list = tree_.children_left.tolist()
        children_right_list = tree_.children_right.tolist()
        values_list = tree_.value.squeeze(axis=1).tolist()

        all_features.append(features_list)
        all_thresholds.append(thresholds_list)
        all_children_left.append(children_left_list)
        all_children_right.append(children_right_list)
        all_values.append(values_list)

    data_to_export = {
        "n_total_trees": len(all_features),
        "features": all_features,
        "thresholds": all_thresholds,
        "children_left": all_children_left,
        "children_right": all_children_right,
        "values": all_values,
    }

    print(f"Writing data to {filename}...")
    try:
        with open(filename, 'w') as f:
            json.dump(data_to_export, f) # Use minimal JSON (no indent) for Rust
        print("Export successful!")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")
    except TypeError as e:
        print(f"Error serializing data to JSON: {e}")

def predict_by_majority_vote(rf, X):
    """Predicts using majority vote."""
    n_samples = X.shape[0]
    n_classes = rf.n_classes_
    votes = np.zeros((n_samples, n_classes))

    # Ensure X is float32 (should be from get_prediction_df)
    if X.dtype != np.float32:
        print(f"[PRED Warning] Input X dtype is {X.dtype}, expected float32. Casting.")
        X = X.astype(np.float32)

    for tree_idx, t in enumerate(rf.estimators_):
        pred = t.predict(X) # Get class prediction for each tree for all samples
        # Increment vote count for the predicted class for all samples
        np.add.at(votes, (np.arange(n_samples), pred.astype(int)), 1)

    # Final majority vote for all samples
    final_predictions = np.argmax(votes, axis=1)

    return final_predictions


def get_prediction_df(rf: RandomForestClassifier, X_full: np.ndarray) -> pd.DataFrame:
    """Generates prediction dataframe for the entire dataset, ensuring float32 dtype."""
    # Ensure X_full is float32
    if X_full.dtype != np.float32:
        print(f"[PRED DF Warning] Input X_full dtype is {X_full.dtype}, expected float32. Casting.")
        X = X_full.astype(np.float32)
    else:
        X = X_full

    # Create DataFrame with float32 dtype
    X_df = pd.DataFrame(X, columns=[f'X[{i}]' for i in range(X.shape[1])], dtype=np.float32) # Use float32

    # Use the custom majority vote prediction
    rf_predict = predict_by_majority_vote(rf, X) # X is now float32
    df = pd.DataFrame(rf_predict, columns=['rf_pred'])

    # Concatenate feature columns and prediction column
    df_final = pd.concat([X_df, df], axis=1)
    return df_final

def kmeans(forest_paths, classificacoes_por_arvore, n_clusters, max_iter=50):
    """
    Clusteriza árvores usando K-Means baseado na distância personalizada entre árvores
    e retorna o resultado NO MESMO FORMATO do algoritmo original:
    
        labels: {id_arvore_original: cluster_id}
        acc_table: {id_arvore_original: valor_original}
    """

    # Ordenar índices como no algoritmo original
    indices = sorted(classificacoes_por_arvore.keys())

    # Quantidade de árvores
    n_arvores = len(indices)

    # === 1. Inicialização: sorteia k árvores como centróides (por índice ordenado) ===
    escolhidos = np.random.choice(n_arvores, size=n_clusters, replace=False)
    centroid_indices = [indices[i] for i in escolhidos]  # índices reais das árvores

    for _ in range(max_iter):

        # === 2. Atribuição: cluster de cada árvore ===
        clusters_tmp = defaultdict(list)

        for real_idx in indices:

            # distâncias entre a árvore real_idx e cada centro
            distancias = [
                distance_between_trees(
                    forest_paths[real_idx],
                    forest_paths[centro]
                )
                for centro in centroid_indices
            ]

            melhor_cluster = int(np.argmin(distancias))
            clusters_tmp[melhor_cluster].append(real_idx)

        # === 3. Atualização dos centróides (tipo medoid) ===
        novos_centros = []

        for k_id in range(n_clusters):

            membros = clusters_tmp.get(k_id, [])

            # se cluster está vazio -> escolhe um centro aleatório
            if not membros:
                novos_centros.append(np.random.choice(indices))
                continue

            # acha o medoid: árvore com menor soma de distâncias internas
            melhor_medoid = None
            melhor_soma = float("inf")

            for a in membros:
                soma = 0
                for b in membros:
                    soma += distance_between_trees(
                        forest_paths[a],
                        forest_paths[b]
                    )
                if soma < melhor_soma:
                    melhor_soma = soma
                    melhor_medoid = a

            novos_centros.append(melhor_medoid)

        # === 4. Critério de convergência ===
        if novos_centros == centroid_indices:
            break

        centroid_indices = novos_centros

    # === Final: construir labels no mesmo formato do seu algoritmo ===
    labels = {}
    for cluster_id, membros in clusters_tmp.items():
        for m in membros:
            labels[m] = cluster_id
        print(f"Cluster {cluster_id} - tamanho {len(membros)}")

    acc_table = {i: classificacoes_por_arvore[i][1] for i in classificacoes_por_arvore.keys()}

    return labels, acc_table

def criar_working_idle_clusters(clusters, cluster_num):
    """
    Cria clusters 'working' e 'idle' a partir de um dicionário de clusters.

    Parâmetros:
        clusters (dict): mapeia índice da árvore -> índice do cluster.
        cluster_num (int): número total de clusters originais.

    Retorna:
        dict: dicionário no formato
              {
                "working": {cluster_id: [índices de árvores]},
                "idle": {cluster_id: [índices de árvores]}
              }
    """
    # Agrupar as árvores por cluster
    agrupados = defaultdict(list)
    for arvore_idx, cluster_idx in clusters.items():
        agrupados[cluster_idx].append(arvore_idx)
    
    # Encontrar o tamanho do menor cluster
    t = min(len(arvores) for arvores in agrupados.values())
    print(f"Tamanho do menor cluster: {t}")
    
    # Criar os clusters working e idle
    working_clusters = {}
    idle_clusters = {}
    
    for c in range(cluster_num):
        arvores = agrupados[c]
        # embaralhar para aleatorizar a seleção
        random.shuffle(arvores)
        working_clusters[c] = arvores[:t]
        idle_clusters[c] = arvores[t:]
    
    return {
        "working": working_clusters,
        "idle": idle_clusters
    }

def avg_acc_cluster(cluster, acc_table):
    return sum(acc_table[i] for i in cluster) / len(cluster)

def acc_all_clusters(working_clusters, acc_table):
    return sum(avg_acc_cluster(c, acc_table) for c in working_clusters.values()) / len(working_clusters)

def idle_to_working(working, idle, acc_table):
    if idle:
        best = max(idle, key=lambda i: acc_table[i])
        idle.remove(best)
        working.append(best)
    return working, idle

def working_to_idle(working, idle, acc_table):
    if len(working) > 1:
        worst = min(working, key=lambda i: acc_table[i])
        working.remove(worst)
        idle.append(worst)
    return working, idle

def RD_execution(rd_iter, clusters, acc_table):
    for _ in range(rd_iter):
        for i in clusters['working'].keys():
            working = clusters['working'][i]
            idle = clusters['idle'][i]

            # Calcula acurácia média do cluster atual e geral
            avg_cluster_acc = avg_acc_cluster(working, acc_table)
            general_acc = acc_all_clusters(clusters['working'], acc_table)

            # Decide se troca elementos entre working e idle
            if avg_cluster_acc >= general_acc:
                working, idle = idle_to_working(working, idle, acc_table)
            else:
                working, idle = working_to_idle(working, idle, acc_table)

            # Atualiza os clusters após as trocas
            clusters['working'][i] = working
            clusters['idle'][i] = idle

    # Após todas as iterações, junta todos os índices dos working clusters
    final_working_indices = []
    print("\nTamanho final dos working clusters:")
    for idx, cluster in clusters['working'].items():
        print(f"Working Cluster {idx} - tamanho {len(cluster)}")
        final_working_indices.extend(cluster)

    return final_working_indices


def construir_subfloresta(clf_original, selected_indices):
    floresta_podada = RandomForestClassifier(
        n_estimators=len(selected_indices),
        max_depth=clf_original.max_depth,
        random_state=clf_original.random_state,
        criterion=clf_original.criterion,
        max_features=clf_original.max_features,
        min_samples_split=clf_original.min_samples_split,
        min_samples_leaf=clf_original.min_samples_leaf,
        bootstrap=clf_original.bootstrap
    )

    # Copiar apenas os estimadores medoides
    floresta_podada.estimators_ = [clf_original.estimators_[i] for i in selected_indices]

    # Herdar também os metadados importantes (classes e n_features)
    floresta_podada.n_features_in_ = clf_original.n_features_in_
    floresta_podada.n_classes_ = clf_original.n_classes_
    floresta_podada.classes_ = clf_original.classes_

    return floresta_podada

def save_test_data_csv(x_test, y_test, current_seed, dir):
    """
    Salva x_test e y_test em um único arquivo CSV.
    O y_test é salvo como a última coluna.
    """
    # Garantir que y_test seja coluna
    y_test = np.asarray(y_test)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)

    # Concatenar features + label
    data = np.hstack((x_test, y_test))

    filename = os.path.join(dir,f"test_data_{current_seed}.csv")

    np.savetxt(
        filename,
        data,
        delimiter=",",
        fmt="%.6f"
    )

    print(f"Test data saved to {filename}")

# --- Main Execution Logic ---
def main(args):
    dataset_name = args.dataset
    if dataset_name not in dataset_metadata:
        print(f"Error: Dataset '{dataset_name}' not found in metadata.")
        print(f"Available datasets: {list(dataset_metadata.keys())}")
        return

    dataset_config = dataset_metadata[dataset_name]
    test_size = dataset_config['test_size'] # Get test_size for splitting later

    # --- Load and Preprocess Data ONCE ---
    print(f"\n--- Loading and Preprocessing Dataset: {dataset_name} ---")
    try:
        dataset_uci = fetch_ucirepo(id=dataset_config['id'])
        X_df_orig = dataset_uci.data.features
        y_orig = dataset_uci.data.targets.squeeze().values

        print(f"Original X shape: {X_df_orig.shape}, Original y shape: {y_orig.shape}, Original y dtype: {y_orig.dtype}")

        # Initialize processed variables with original data
        X_processed_df = X_df_orig
        y_processed = y_orig

        # --- Apply Dataset-Specific Preprocessing Pipeline FIRST ---
        preprocess_pipeline = dataset_config.get('preprocess', None)
        if preprocess_pipeline:
            if not isinstance(preprocess_pipeline, list):
                preprocess_pipeline = [preprocess_pipeline]
            print("Applying dataset-specific preprocessing pipeline...")
            for i, func in enumerate(preprocess_pipeline):
                print(f"Pipeline step {i+1}: Running {getattr(func, '__name__', repr(func))}...")
                # Pass the current state of X and y to the function
                X_processed_df, y_processed = func(X_processed_df, y_processed)
                print(f"  -> Output X shape: {X_processed_df.shape}, Output y shape: {y_processed.shape}")
                print(f"  -> Output X type: {type(X_processed_df)}, Output y type: {type(y_processed)}, y dtype: {y_processed.dtype}") # Added y type/dtype check
                # Note: Feature preprocessing (like OHE) might happen here if included in the pipeline
                # Ensure X remains a DataFrame if subsequent steps expect it
                if not isinstance(X_processed_df, pd.DataFrame) and i < len(preprocess_pipeline) - 1:
                     print(f"  -> Warning: Output X type changed to {type(X_processed_df)}. Attempting conversion back to DataFrame.")
                     # This might be needed if a function returns NumPy but the next expects DataFrame

        # --- Apply Generic Target Preprocessing AFTER dataset-specific steps ---
        # This ensures y is numeric and integer before splitting/training
        try:
            # Pass the potentially modified X and y from the specific pipeline
            # Note: X might have changed shape/type if feature processing was in the pipeline
            X_processed_df, y_processed = preprocess_target_variable(X_processed_df, y_processed)
            print(f"After generic target preprocessing - y shape: {y_processed.shape}, y dtype: {y_processed.dtype}")
        except Exception as e:
             print(f"Error during generic target preprocessing: {e}")
             import traceback
             traceback.print_exc()
             sys.exit(1)


        # Final conversion of X to float32 NumPy array
        # X_processed_df now holds the result after all relevant preprocessing steps (specific and potentially feature processing)
        print(f"Converting final X DataFrame/Array (shape {X_processed_df.shape}) to float32 NumPy array.")
        if isinstance(X_processed_df, pd.DataFrame):
            non_numeric_cols = X_processed_df.select_dtypes(exclude=np.number).columns
            if not non_numeric_cols.empty:
                print(f"Warning: Non-numeric columns found before final conversion: {list(non_numeric_cols)}. Attempting conversion.")
                try:
                    X_final = X_processed_df.astype(np.float32).values
                except Exception as e:
                    print(f"Error converting DataFrame to float32 NumPy array: {e}")
                    raise ValueError("Could not convert final features to numeric float32 array.") from e
            else:
                X_final = X_processed_df.values.astype(np.float32)
        elif isinstance(X_processed_df, np.ndarray):
             # If already a NumPy array (e.g., from sklearn pipeline), ensure it's float32
             if X_processed_df.dtype != np.float32:
                 print(f"Converting final X NumPy array from {X_processed_df.dtype} to float32.")
                 X_final = X_processed_df.astype(np.float32)
             else:
                 X_final = X_processed_df
        else:
            raise TypeError(f"Unexpected type for final X features: {type(X_processed_df)}")


        y_final = y_processed # y should now be a NumPy array from preprocessing

        # Ensure y_final is integer type before proceeding
        if not pd.api.types.is_integer_dtype(y_final):
            print(f"Warning: Final y array dtype is {y_final.dtype}. Attempting conversion to integer.")
            try:
                # Attempt conversion, handling potential errors if mixed types remain
                # This is a fallback if preprocessing didn't fully handle it
                y_final = y_final.astype(int)
            except ValueError as e:
                print(f"Error converting final y to integer: {e}. Check preprocessing steps for target variable.")
                # Check for specific problematic values if possible
                try:
                    problem_values = y_final[pd.to_numeric(y_final, errors='coerce').isna()]
                    print(f"Problematic values in y_final: {np.unique(problem_values)}")
                except Exception:
                    print("Could not identify specific problematic values in y_final.")
                raise ValueError("Could not convert final target variable to integer.") from e


        print(f"Final X NumPy array shape: {X_final.shape}, dtype: {X_final.dtype}")
        print(f"Final y NumPy array shape: {y_final.shape}, dtype: {y_final.dtype}") # Should now show int dtype
        print("--- Dataset Loading and Preprocessing Complete ---")

    except Exception as e:
        print(f"Error during initial data loading/preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) # Exit if initial loading fails

    # --- Stats Printing Logic (Uses preprocessed data) ---
    if args.stats:
        print(f"\n--- Dataset Statistics: {dataset_name} (Post-Preprocessing) ---")
        # Use X_final and y_final for stats, but need feature names if X was DataFrame
        # Reconstruct DataFrame for stats if needed, using generic names if OHE changed them
        if isinstance(X_processed_df, pd.DataFrame):
             X_df_for_stats = X_processed_df # Use the DataFrame before final numpy conversion
        else:
             # If X_final is numpy, create a temporary DataFrame for stats display
             feature_names_for_stats = [f'feature_{i}' for i in range(X_final.shape[1])]
             X_df_for_stats = pd.DataFrame(X_final, columns=feature_names_for_stats)

        y_for_stats = y_final # y_final holds the final processed target

        num_features_after_preprocessing = X_df_for_stats.shape[1]
        unique_classes, counts = np.unique(y_for_stats, return_counts=True)
        num_classes = len(unique_classes)
        class_distribution = dict(zip(unique_classes, counts))

        print(f"  Number of Features: {num_features_after_preprocessing}")
        print(f"  Number of Classes: {num_classes}")
        print("\n  Class Distribution:")
        total_instances = sum(counts)
        for cls, count in class_distribution.items():
            percentage = (count / total_instances) * 100 if total_instances > 0 else 0
            print(f"    - Class '{cls}': {count} instances ({percentage:.2f}%)")

        print("\n  Feature Types and Statistics:")
        for i, col in enumerate(X_df_for_stats.columns):
            dtype = X_df_for_stats[col].dtype
            if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                 print(f"    - Feature {i} ('{col}'): Categorical/Object ({dtype})")
            elif pd.api.types.is_numeric_dtype(dtype):
                stats = X_df_for_stats[col].describe()
                is_binary_like = np.allclose(stats['min'], 0) and np.allclose(stats['max'], 1) and len(X_df_for_stats[col].unique()) <= 2
                tag = " (Binary)" if is_binary_like else ""
                is_integer_like = pd.api.types.is_integer_dtype(dtype) or np.allclose(X_df_for_stats[col], X_df_for_stats[col].round())
                type_desc = "Integer" if is_integer_like else "Continuous"
                print(f"    - Feature {i} ('{col}'): {type_desc} ({dtype}){tag} | "
                      f"Min: {stats['min']:.3f}, Median: {stats['50%']:.3f}, Max: {stats['max']:.3f}, "
                      f"Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")
            else:
                print(f"    - Feature {i} ('{col}'): Other ({dtype})")
        sys.exit(0) # Exit after printing stats

    # --- Default Training & Export Logic ---
    output_dir = "src/rf" # Directory where Rust expects the files
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Processing Dataset for Training & Export: {dataset_name} ---")
    # ... (print config details) ...
    print(f"  Number of Trees (n_estimators): {dataset_config['n_trees']}")
    print(f"  Max Depth: {dataset_config['max_depth']}")
    print(f"  Test Set Size: {test_size}")
    print(f"  Number of Seeds Run: {NUM_SEEDS_TO_RUN}")
    print(f"  Base Seed: {BASE_SEED}")


    # Initialize lists/sets for aggregated stats
    all_accuracies = []
    all_node_counts = []
    all_unique_predicate_counts = []
    all_max_depths = [] # New list to store max depth per RF run

    print("\n--- Per-Seed Training & Export ---")
    for i in range(NUM_SEEDS_TO_RUN):
        current_seed = BASE_SEED + i
        print(f"\n  Seed: {current_seed} ({i+1}/{NUM_SEEDS_TO_RUN})")
        try:
            # 1. Split data with current seed using the preprocessed X_final, y_final
            print(f"Splitting preprocessed data with seed {current_seed}...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_final, y_final, test_size=test_size, random_state=current_seed
            )
            print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

            #X_test, X_preds, y_test, y_preds = train_test_split(
                #X_test, y_test, test_size=0.5, random_state=current_seed
            #)

            # Ensure split data is float32 (redundant if X_final is already float32, but safe)
            if X_train.dtype != np.float32: X_train = X_train.astype(np.float32)
            if X_test.dtype != np.float32: X_test = X_test.astype(np.float32)

            # 2. Train RF with current seed (pass the split tuple)
            rf_classifier, accuracy, preds = get_trained_rf(dataset_config, (X_train, X_test, y_train, y_test), current_seed)

            # 3. Get RF stats (including max_depth)
            node_count, unique_predicates, max_depth_run = get_rf_stats(rf_classifier) # Get max_depth_run
            num_unique_predicates_this_run = len(unique_predicates)

            # 4. Collect stats for aggregation
            all_accuracies.append(accuracy)
            all_node_counts.append(node_count)
            all_unique_predicate_counts.append(num_unique_predicates_this_run)
            all_max_depths.append(max_depth_run) # Store max depth for this run

            # Print per-seed stats immediately
            print(f"    Accuracy: {accuracy * 100:.2f}%")
            print(f"    Total Nodes: {node_count}")
            print(f"    Unique Predicates in this RF: {num_unique_predicates_this_run}")
            print(f"    Max Depth (Height) in this RF: {max_depth_run}") # Print max depth for the run

            #clusterizing forest and applying egap
            paths = extract_paths_from_forest(rf_classifier)
            clusters, acc_table = kmeans(paths, preds, CLUSNUM)
            working_idle = criar_working_idle_clusters(clusters, CLUSNUM)
            selected = RD_execution(100, working_idle, acc_table)
            egapForest = construir_subfloresta(rf_classifier, selected)

            print(f"Tamanho da Floresta Selecionada pelo Egap: {len(egapForest.estimators_)}")

            # 5. Define seed-specific filenames
            rf_json_filename = os.path.join(output_dir, f"rf-{dataset_name}-seed{current_seed}.json")
            pred_csv_filename = os.path.join(output_dir, f"pred-{dataset_name}-seed{current_seed}.csv")

            egap_rf_json_filename = os.path.join(output_dir, f"egap-rf-{dataset_name}-seed{current_seed}.json")
            egap_pred_csv_filename = os.path.join(output_dir, f"egap-pred-{dataset_name}-seed{current_seed}.csv")

            # 6. Export RF to JSON
            export_rf_to_json(rf_classifier, rf_json_filename)
            export_rf_to_json(egapForest, egap_rf_json_filename)

            # 7. Generate and save prediction CSV with higher precision
            print(f"Generating predictions for {pred_csv_filename}...")
            pred_df = get_prediction_df(rf_classifier, X_final)
            pred_df.to_csv(pred_csv_filename, index=False, float_format='%.17g')
            print(f"Predictions saved to {pred_csv_filename}")

            print(f"Generating predictions for {egap_pred_csv_filename}...")
            pred_df = get_prediction_df(egapForest, X_final)
            pred_df.to_csv(egap_pred_csv_filename, index=False, float_format='%.17g')
            print(f"Predictions saved to {egap_pred_csv_filename}")

            save_test_data_csv(X_test, y_test, current_seed, output_dir)

        except Exception as e:
            print(f"    Error during processing for seed {current_seed}: {e}")
            all_accuracies.append(np.nan)
            all_node_counts.append(np.nan)
            all_unique_predicate_counts.append(np.nan)
            all_max_depths.append(np.nan)


    # --- Print Aggregated Statistics After Loop ---
    print("\n--- Aggregated Training Statistics ---")
    num_successful_runs = len([acc for acc in all_accuracies if not np.isnan(acc)])
    print(f"  Based on {num_successful_runs} successful runs out of {NUM_SEEDS_TO_RUN}.")

    # Accuracy aggregation
    if num_successful_runs > 0:
        mean_accuracy = np.nanmean(all_accuracies)
        std_accuracy = np.nanstd(all_accuracies)
        print(f"  Accuracy: Mean={mean_accuracy * 100:.2f}%, StdDev={std_accuracy * 100:.2f}%")
    else:
        print("  Accuracy: No successful runs to aggregate.")

    # Node count aggregation
    if num_successful_runs > 0:
         min_nodes = np.nanmin(all_node_counts)
         median_nodes = np.nanmedian(all_node_counts)
         max_nodes = np.nanmax(all_node_counts)
         mean_nodes = np.nanmean(all_node_counts)
         std_nodes = np.nanstd(all_node_counts)
         print(f"  Total Nodes per RF: Min={min_nodes:.0f}, Median={median_nodes:.0f}, Max={max_nodes:.0f}, Mean={mean_nodes:.2f}, StdDev={std_nodes:.2f}")
    else:
         print("  Total Nodes per RF: No successful runs to aggregate.")

    # Unique Predicate Count aggregation
    if num_successful_runs > 0:
         mean_predicates = np.nanmean(all_unique_predicate_counts)
         std_predicates = np.nanstd(all_unique_predicate_counts)
         # Min/Median/Max might also be interesting here, but sticking to Mean/Std as before for now
         print(f"  Unique Predicates per RF: Mean={mean_predicates:.2f}, StdDev={std_predicates:.2f}")
    else:
         print("  Unique Predicates per RF: No successful runs to aggregate.")

    # Max Depth (Height) aggregation
    if num_successful_runs > 0:
         min_depth = np.nanmin(all_max_depths)
         median_depth = np.nanmedian(all_max_depths)
         max_depth_agg = np.nanmax(all_max_depths)
         mean_depth = np.nanmean(all_max_depths)
         std_depth = np.nanstd(all_max_depths)
         print(f"  Max Depth (Height) per RF: Min={min_depth:.0f}, Median={median_depth:.0f}, Max={max_depth_agg:.0f}, Mean={mean_depth:.2f}, StdDev={std_depth:.2f}")
    else:
         print("  Max Depth (Height) per RF: No successful runs to aggregate.")


    print(f"\n--- Finished processing dataset {dataset_name} for {NUM_SEEDS_TO_RUN} seeds ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/export RFs or display dataset stats.")
    parser.add_argument('--dataset', '-d', type=str, required=True, choices=dataset_metadata.keys(),
                        help='Name of the dataset to process.')
    parser.add_argument('--stats', action='store_true',
                        help='If set, display dataset metadata stats and exit (no training).')

    args = parser.parse_args()
    main(args)
