use std::{collections::{HashMap, HashSet}, f64::{INFINITY, NEG_INFINITY}, fs::File, io::{BufRead, BufReader}, path::PathBuf};

use crate::tree::{Arena, NodeId};

#[derive(Clone, Debug)]
pub enum Direction {
    Left,  // <= threshold
    Right, // > threshold
}

#[derive(Clone, Debug)]
pub struct Split {
    pub feature: i64,
    pub threshold: f64,
    pub direction: Direction,
}

#[derive(Clone, Debug)]
pub struct Path {
    pub splits: Vec<Split>,
    pub predicted_class: usize,
}

pub fn extract_paths_from_arena(
    arena: &Arena,
    root_id: NodeId,
) -> Vec<Path> {
    let mut paths = Vec::new();
    let mut current_splits = Vec::new();

    fn dfs(
        arena: &Arena,
        node_id: NodeId,
        current_splits: &mut Vec<Split>,
        paths: &mut Vec<Path>,
    ) {
        // === Caso folha ===
        if arena.is_leaf(node_id) {
            let predicted_class = arena.get_leaf_class(node_id);
            paths.push(Path {
                splits: current_splits.clone(),
                predicted_class,
            });
            return;
        }

        let feature = arena.get_feature_raw(node_id);
        let threshold = arena.get_threshold(node_id);

        // === LEFT branch (<= threshold) ===
        current_splits.push(Split {
            feature,
            threshold,
            direction: Direction::Left,
        });
        let left_child = arena.get_true_id(node_id);
        dfs(arena, left_child, current_splits, paths);
        current_splits.pop();

        // === RIGHT branch (> threshold) ===
        current_splits.push(Split {
            feature,
            threshold,
            direction: Direction::Right,
        });
        let right_child = arena.get_false_id(node_id);
        dfs(arena, right_child, current_splits, paths);
        current_splits.pop();
    }

    dfs(arena, root_id, &mut current_splits, &mut paths);
    paths
}

pub fn extract_paths_from_arena_for_instances(
    arena: &Arena,
    root_id: NodeId,
    data_path: &PathBuf,
) -> Vec<Path> {
    let file = File::open(data_path)
        .expect("Erro ao abrir arquivo de dados");

    let reader = BufReader::new(file);

    let mut paths = Vec::new();

    for line in reader.lines().skip(1) {
        let line = line.expect("Erro ao ler linha");

        // Ignora linhas vazias
        if line.trim().is_empty() {
            continue;
        }

        // Parse da linha: X[0], X[1], ..., X[n], rf_pred
        let values: Vec<f64> = line
            .split(',')
            .map(|v| v.trim().parse::<f64>().expect("Erro ao parsear valor"))
            .collect();

        // Ignora a última coluna (rf_pred)
        let features = &values[..values.len() - 1];

        let mut current_splits = Vec::new();
        let mut node_id = root_id;

        // Percorre a árvore seguindo APENAS esse exemplo
        while !arena.is_leaf(node_id) {
            let feature = arena.get_feature_raw(node_id) as i64;
            let threshold = arena.get_threshold(node_id);

            let go_left = features[feature as usize] <= threshold;

            let direction = if go_left {
                Direction::Left
            } else {
                Direction::Right
            };

            current_splits.push(Split {
                feature,
                threshold,
                direction,
            });

            node_id = if go_left {
                arena.get_true_id(node_id)
            } else {
                arena.get_false_id(node_id)
            };
        }

        // Chegou na folha
        let predicted_class = arena.get_leaf_class(node_id);

        paths.push(Path {
            splits: current_splits,
            predicted_class,
        });
    }

    paths
}

fn compute_feature_vector(
    arena: &Arena,
    root: NodeId,
    num_features: usize,
) -> Vec<f64> {
    let mut values = vec![0.0; num_features];
    let mut max_depth = vec![0usize; num_features];

    // DFS explícito: (node_id, depth)
    let mut stack = Vec::new();
    stack.push((root, 1usize));

    while let Some((node_id, depth)) = stack.pop() {
        if arena.is_leaf(node_id) {
            continue;
        }

        let feature = arena.get_feature_raw(node_id) as usize;

        let weight = 1.0 / 2f64.powi((depth - 1) as i32);
        values[feature] += weight;
        max_depth[feature] = max_depth[feature].max(depth);

        let left = arena.get_true_id(node_id);
        let right = arena.get_false_id(node_id);

        stack.push((left, depth + 1));
        stack.push((right, depth + 1));
    }

    // Normalização pelo nível mais profundo da feature
    for f in 0..num_features {
        if max_depth[f] > 0 {
            values[f] /= max_depth[f] as f64;
        }
    }

    values
}

pub fn mean_feature_overlap_distance(
    arena_rf: &Arena,
    root_rf: NodeId,
    arena_egap: &Arena,
    root_egap: NodeId,
    num_features: usize,
) -> f64 {
    let v_rf = compute_feature_vector(arena_rf, root_rf, num_features);
    let v_egap = compute_feature_vector(arena_egap, root_egap, num_features);

    v_rf.iter()
        .zip(v_egap.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum()
}

pub fn distance_between_paths(p1: &Path, p2: &Path) -> f64 {
    if p1.splits.is_empty() {
        return 0.0;
    }

    let intervals1 = extract_feature_intervals(&p1.splits);
    let intervals2 = extract_feature_intervals(&p2.splits);

    let mut total_distance = 0.0;

    for (feature, interval1) in &intervals1 {
        if let Some(interval2) = intervals2.get(feature) {
            total_distance += interval_distance(*interval1, *interval2);
        } else {
            // Penalidade por feature ausente
            total_distance += 1.0;
        }
    }

    total_distance / intervals1.len() as f64
}

pub fn distance_between_trees(paths1: &[Path], paths2: &[Path]) -> f64 {
    if paths1.is_empty() {
        return 0.0;
    }

    let mut total = 0.0;

    // Indexar paths2 por classe
    let mut class_index: HashMap<usize, Vec<&Path>> = HashMap::new();
    for p in paths2 {
        class_index
            .entry(p.predicted_class)
            .or_default()
            .push(p);
    }

    for p1 in paths1 {
        match class_index.get(&p1.predicted_class) {
            None => {
                // Nenhum caminho com mesma classe → penalidade máxima
                total += 1.0;
            }
            Some(same_class_paths) => {
                let mut best = f64::INFINITY;

                for p2 in same_class_paths {
                    let d = distance_between_paths(p1, p2);
                    if d < best {
                        best = d;
                    }
                }

                total += best;
            }
        }
    }

    total / paths1.len() as f64
}

fn extract_feature_intervals(splits: &[Split]) -> HashMap<i64, (f64, f64)> {
    let mut intervals: HashMap<i64, (f64, f64)> = HashMap::new();

    for s in splits {
        let entry = intervals
            .entry(s.feature)
            .or_insert((NEG_INFINITY, INFINITY));

        match s.direction {
            Direction::Left => {
                // feature <= threshold
                entry.1 = entry.1.min(s.threshold);
            }
            Direction::Right => {
                // feature > threshold
                entry.0 = entry.0.max(s.threshold);
            }
        }
    }

    intervals
}

fn interval_distance(i1: (f64, f64), i2: (f64, f64)) -> f64 {
    let (a1, b1) = i1;
    let (a2, b2) = i2;

    // Sobreposição
    if b1 >= a2 && b2 >= a1 {
        return 0.0;
    }

    // Distância mínima entre bordas
    (a2 - b1).abs().min((a1 - b2).abs())
}

//pub fn generate_paths_from_data(seed)