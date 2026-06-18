"""
CityMind – Challenge 5: Crime Risk Prediction and Integration
Algorithms:
  Step 1 – K-Means clustering (unsupervised, no pre-labeled data)
  Step 2 – Decision Tree classifier on synthetic crime dataset
  Step 3 – Feed predicted risk back into shared graph (adjusts edge costs)
"""

import math
import random
from city_graph import CityGraph


# ─────────────────────────────────────────
#  K-Means Clustering
# ─────────────────────────────────────────
def _euclidean(a: list, b: list) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def kmeans(data: list[list[float]], k: int = 3, max_iter: int = 100,
           seed: int = 0) -> list[int]:
    """
    Assigns each data point to one of k clusters.
    Returns list of cluster labels (0..k-1) aligned with data.
    """
    random.seed(seed)
    n = len(data)
    if n == 0:
        return []

    # Initialise centroids via k-means++ style (spread out)
    centroids = [random.choice(data)]
    for _ in range(k - 1):
        dists = [min(_euclidean(pt, c) for c in centroids) for pt in data]
        total = sum(dists)
        r = random.uniform(0, total)
        cumul = 0
        for i, d in enumerate(dists):
            cumul += d
            if cumul >= r:
                centroids.append(data[i])
                break
        else:
            centroids.append(data[-1])

    labels = [0] * n
    for _ in range(max_iter):
        # Assign
        new_labels = [
            min(range(k), key=lambda ci: _euclidean(pt, centroids[ci]))
            for pt in data
        ]
        if new_labels == labels:
            break
        labels = new_labels

        # Update centroids
        for ci in range(k):
            members = [data[i] for i, l in enumerate(labels) if l == ci]
            if members:
                centroids[ci] = [sum(col) / len(members) for col in zip(*members)]

    return labels


# ─────────────────────────────────────────
#  Synthetic Crime Dataset Generation
# ─────────────────────────────────────────
def _industrial_proximity(graph: CityGraph, nid: int) -> float:
    """Manhattan distance to nearest Industrial node (0 if none)."""
    industrial = graph.nodes_by_type("Industrial")
    if not industrial:
        return 0.0
    n = graph.nodes[nid]
    return min(abs(n.x - graph.nodes[i].x) + abs(n.y - graph.nodes[i].y)
               for i in industrial)


def generate_synthetic_dataset(graph: CityGraph, seed: int = 42) -> list[dict]:
    """
    For each node, generate a record:
      features: [population_density, industrial_proximity, cluster_label]
      label:    "Low" | "Medium" | "High"

    Crime likelihood logic (justifiable):
      - High density + near industrial → higher risk
      - Low density + far industrial → lower risk
    """
    random.seed(seed)

    # Step 1: cluster on [population_density, industrial_proximity]
    node_ids = list(graph.nodes.keys())
    features = []
    for nid in node_ids:
        n = graph.nodes[nid]
        prox = _industrial_proximity(graph, nid)
        features.append([n.population_density, prox])

    # Normalise features to [0,1]
    max_density = max(f[0] for f in features) or 1
    max_prox    = max(f[1] for f in features) or 1
    normed = [[f[0] / max_density, f[1] / max_prox] for f in features]

    labels = kmeans(normed, k=3, seed=seed)

    # Map cluster → rough risk (heuristic: higher density cluster → more risk)
    cluster_avg_density = {}
    for ci in range(3):
        members = [features[i][0] for i, l in enumerate(labels) if l == ci]
        cluster_avg_density[ci] = sum(members) / len(members) if members else 0

    rank = sorted(cluster_avg_density, key=cluster_avg_density.get)
    cluster_risk = {rank[0]: "Low", rank[1]: "Medium", rank[2]: "High"}

    dataset = []
    for idx, nid in enumerate(node_ids):
        n = graph.nodes[nid]
        prox = features[idx][1]
        cluster = labels[idx]

        # Synthetic incident rate with noise
        base_risk = {"Low": 0, "Medium": 1, "High": 2}[cluster_risk[cluster]]
        # Boost risk if near industrial and dense
        if prox < 2 and n.population_density > 800:
            base_risk = min(2, base_risk + 1)
        # Industrial/PowerPlant zones themselves have lower resident crime
        if n.location_type in ("Industrial", "PowerPlant"):
            base_risk = 0

        # Add small noise
        noise = random.choice([-1, 0, 0, 0, 1])
        final_risk = max(0, min(2, base_risk + noise))
        risk_label = ["Low", "Medium", "High"][final_risk]

        dataset.append({
            "node_id":            nid,
            "population_density": round(n.population_density, 2),
            "industrial_proximity": round(prox, 2),
            "cluster":            cluster,
            "risk_label":         risk_label,
        })

    return dataset


# ─────────────────────────────────────────
#  Decision Tree (from scratch)
# ─────────────────────────────────────────
class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.label     = label   # leaf label


def _gini(labels: list) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def _best_split(X: list[list], y: list, features: list) -> tuple:
    best_gini = float("inf")
    best_feat, best_thresh = None, None
    n = len(y)

    for fi in features:
        values = sorted(set(row[fi] for row in X))
        thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
        for thresh in thresholds:
            left_y  = [y[i] for i in range(n) if X[i][fi] <= thresh]
            right_y = [y[i] for i in range(n) if X[i][fi] > thresh]
            if not left_y or not right_y:
                continue
            g = (len(left_y) * _gini(left_y) + len(right_y) * _gini(right_y)) / n
            if g < best_gini:
                best_gini, best_feat, best_thresh = g, fi, thresh

    return best_feat, best_thresh


def _majority(labels: list) -> str:
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return max(counts, key=counts.get)


def _build_tree(X: list[list], y: list, depth: int, max_depth: int = 6) -> DecisionNode:
    if not y or depth >= max_depth or len(set(y)) == 1:
        return DecisionNode(label=_majority(y) if y else "Low")

    features = list(range(len(X[0])))
    feat, thresh = _best_split(X, y, features)
    if feat is None:
        return DecisionNode(label=_majority(y))

    n = len(y)
    left_mask  = [X[i][feat] <= thresh for i in range(n)]
    right_mask = [not m for m in left_mask]

    left_X  = [X[i] for i in range(n) if left_mask[i]]
    left_y  = [y[i] for i in range(n) if left_mask[i]]
    right_X = [X[i] for i in range(n) if right_mask[i]]
    right_y = [y[i] for i in range(n) if right_mask[i]]

    left_node  = _build_tree(left_X, left_y, depth + 1, max_depth)
    right_node = _build_tree(right_X, right_y, depth + 1, max_depth)
    return DecisionNode(feature=feat, threshold=thresh, left=left_node, right=right_node)


def _predict_one(node: DecisionNode, x: list) -> str:
    if node.label is not None:
        return node.label
    if x[node.feature] <= node.threshold:
        return _predict_one(node.left, x)
    return _predict_one(node.right, x)


# ─────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────
def run_crime_prediction(graph: CityGraph, seed: int = 42) -> dict:
    """
    Full pipeline:
      1. Generate synthetic dataset (includes K-Means clustering)
      2. Train Decision Tree
      3. Predict risk for every node
      4. Update shared graph with risk levels and recompute effective costs
    Returns report dict.
    """
    dataset = generate_synthetic_dataset(graph, seed=seed)

    # Prepare full feature matrix and labels
    X = [[d["population_density"], d["industrial_proximity"], d["cluster"]]
         for d in dataset]
    y = [d["risk_label"] for d in dataset]

    # Normalise X to [0, 1]
    max_vals = [max(row[i] for row in X) or 1 for i in range(len(X[0]))]
    X_norm = [[row[i] / max_vals[i] for i in range(len(row))] for row in X]

    # ── Train / Test split (80 / 20) ──────────────────────────────────────
    # The decision tree is trained on 80 % of the data and evaluated on the
    # remaining 20 %, giving a meaningful generalisation accuracy figure.
    # After reporting accuracy the model is retrained on all 100 % of nodes
    # so that final risk predictions cover the entire city graph.
    rng_split = random.Random(seed)
    indices   = list(range(len(dataset)))
    rng_split.shuffle(indices)
    split_at  = int(0.8 * len(indices))
    train_idx = indices[:split_at]
    test_idx  = indices[split_at:]

    X_train = [X_norm[i] for i in train_idx]
    y_train = [y[i]      for i in train_idx]
    X_test  = [X_norm[i] for i in test_idx]
    y_test  = [y[i]      for i in test_idx]

    # Train on 80 % subset
    tree_eval = _build_tree(X_train, y_train, depth=0, max_depth=6)

    # Measure accuracy on held-out 20 %
    train_correct = sum(
        1 for i in range(len(X_train)) if _predict_one(tree_eval, X_train[i]) == y_train[i]
    )
    test_correct  = sum(
        1 for i in range(len(X_test))  if _predict_one(tree_eval, X_test[i])  == y_test[i]
    )
    train_accuracy = round(train_correct / len(y_train), 3) if y_train else 0.0
    test_accuracy  = round(test_correct  / len(y_test),  3) if y_test  else 0.0

    # Retrain on all data for final city-wide predictions
    tree = _build_tree(X_norm, y, depth=0, max_depth=6)

    predictions = {}
    cluster_assignments = {}   # node_id → cluster (0/1/2)
    for i, d in enumerate(dataset):
        pred = _predict_one(tree, X_norm[i])
        predictions[d["node_id"]] = pred
        cluster_assignments[d["node_id"]] = d["cluster"]
        graph.set_risk(d["node_id"], pred)
        # Store cluster label directly on the node for UI access
        graph.nodes[d["node_id"]].cluster = d["cluster"]

    counts = {"Low": 0, "Medium": 0, "High": 0}
    for level in predictions.values():
        counts[level] += 1

    # Store cluster assignments on the graph for UI rendering
    graph.cluster_assignments = cluster_assignments

    # Deploy police based on updated risk predictions
    police_report = deploy_police_officers(graph, num_officers=10)

    return {
        "predictions":         predictions,
        "counts":              counts,
        "cluster_assignments": cluster_assignments,
        "tree_depth":          6,
        "samples":             len(dataset),
        "train_size":          len(train_idx),
        "test_size":           len(test_idx),
        "train_accuracy":      train_accuracy,
        "test_accuracy":       test_accuracy,
        "police":              police_report,
    }


# ─────────────────────────────────────────
#  Police Officer Deployment
# ─────────────────────────────────────────
def deploy_police_officers(graph: CityGraph, num_officers: int = 10) -> dict:
    """
    Allocate police officers to nodes based on predicted crime risk.

    Strategy (greedy priority, guaranteed full deployment):
      Phase 1 – High-risk nodes: 3 officers each (densest first)
      Phase 2 – Medium-risk nodes: 1 officer each (densest first)
      Phase 3 – Any remaining officers cycle back through ALL nodes
                 (densest first) until every single officer is placed.

    Result written to graph.police_deployment: {node_id: num_officers}
    """
    # Sort node pools by population density descending
    high_nodes = sorted(
        [nid for nid, n in graph.nodes.items() if n.risk_level == "High"],
        key=lambda nid: graph.nodes[nid].population_density,
        reverse=True,
    )
    medium_nodes = sorted(
        [nid for nid, n in graph.nodes.items() if n.risk_level == "Medium"],
        key=lambda nid: graph.nodes[nid].population_density,
        reverse=True,
    )
    # Phase 3 fallback: non-Low-risk nodes only, densest first.
    # Cycling through ALL nodes (including Low-risk PowerPlants etc.) is not
    # justified as "intelligent" deployment. Only High/Medium nodes are used.
    all_nodes = sorted(
        [nid for nid, n in graph.nodes.items() if n.risk_level in ("High", "Medium")],
        key=lambda nid: graph.nodes[nid].population_density,
        reverse=True,
    )
    # Final safety net: if somehow all officers are already placed (Phase 1+2
    # exhausted the count), all_nodes may be empty; fall back to any node.
    if not all_nodes:
        all_nodes = sorted(
            list(graph.nodes.keys()),
            key=lambda nid: graph.nodes[nid].population_density,
            reverse=True,
        )

    deployment: dict = {}
    remaining = num_officers

    # Phase 1: 3 officers each to High-risk nodes
    for nid in high_nodes:
        if remaining <= 0:
            break
        assign = min(3, remaining)
        deployment[nid] = deployment.get(nid, 0) + assign
        remaining -= assign

    # Phase 2: 1 officer each to Medium-risk nodes
    for nid in medium_nodes:
        if remaining <= 0:
            break
        deployment[nid] = deployment.get(nid, 0) + 1
        remaining -= 1

    # Phase 3: cycle through ALL nodes until every officer is placed
    idx = 0
    while remaining > 0:
        nid = all_nodes[idx % len(all_nodes)]
        deployment[nid] = deployment.get(nid, 0) + 1
        remaining -= 1
        idx += 1

    # Write back to shared graph so all modules can see it
    graph.police_deployment = deployment

    total_assigned = sum(deployment.values())
    return {
        "deployment":     deployment,
        "total_assigned": total_assigned,
        "unassigned":     num_officers - total_assigned,
        "high_covered":   sum(1 for nid in high_nodes   if nid in deployment),
        "medium_covered": sum(1 for nid in medium_nodes if nid in deployment),
    }