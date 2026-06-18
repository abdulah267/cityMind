"""
CityMind – Challenge 3: Ambulance Placement
Algorithm: Simulated Annealing

Objective: minimise the worst-case response time (max shortest-path distance
           from any citizen node to its nearest ambulance).
"""

import math
import random
from city_graph import CityGraph


NUM_AMBULANCES = 3


# ─────────────────────────────────────────
#  Objective function
# ─────────────────────────────────────────
def _objective(graph: CityGraph, placement: list[int], citizen_nodes: list[int]) -> float:
    """
    Returns the maximum (worst-case) response distance across all citizens.
    Ambulances call Dijkstra from their own node; each citizen takes nearest.
    """
    if not citizen_nodes:
        return 0.0

    # Precompute Dijkstra from each ambulance position
    dists_from_amb = []
    for amb in placement:
        d, _ = graph.dijkstra(amb)
        dists_from_amb.append(d)

    worst = 0.0
    for c in citizen_nodes:
        nearest = min(d[c] for d in dists_from_amb)
        if nearest == float("inf"):
            nearest = 1e9
        worst = max(worst, nearest)

    return worst


# ─────────────────────────────────────────
#  Simulated Annealing
# ─────────────────────────────────────────
def place_ambulances(graph: CityGraph, seed: int = 0) -> dict:
    """
    Returns:
      {
        "placement": [node_id, node_id, node_id],
        "worst_response": float,
        "iterations": int,
      }
    """
    random.seed(seed)
    all_nodes = list(graph.nodes.keys())
    citizen_nodes = [nid for nid, n in graph.nodes.items()
                     if n.location_type in ("Residential", "Hospital", "School")]

    if len(all_nodes) < NUM_AMBULANCES:
        placement = all_nodes[:NUM_AMBULANCES]
        return {"placement": placement, "worst_response": 0.0, "iterations": 0}

    # Initial random placement
    current = random.sample(all_nodes, NUM_AMBULANCES)
    current_score = _objective(graph, current, citizen_nodes)
    best = list(current)
    best_score = current_score

    # Annealing schedule
    T = 100.0
    T_min = 0.1
    alpha = 0.97       # cooling rate
    iterations = 0
    max_iter = 2000

    while T > T_min and iterations < max_iter:
        # Generate neighbour: swap one ambulance to a random node
        neighbour = list(current)
        swap_idx = random.randrange(NUM_AMBULANCES)
        new_node = random.choice(all_nodes)
        neighbour[swap_idx] = new_node

        # Avoid duplicates
        if len(set(neighbour)) < NUM_AMBULANCES:
            T *= alpha
            iterations += 1
            continue

        neighbour_score = _objective(graph, neighbour, citizen_nodes)
        delta = neighbour_score - current_score

        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbour
            current_score = neighbour_score

        if current_score < best_score:
            best = list(current)
            best_score = current_score

        T *= alpha
        iterations += 1

    # Store on graph for other modules to use
    graph.ambulance_positions = best

    return {
        "placement":      best,
        "worst_response": round(best_score, 3),
        "iterations":     iterations,
    }
