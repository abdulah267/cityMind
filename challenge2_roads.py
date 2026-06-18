"""
CityMind – Challenge 2: Road Network Optimization
Algorithm: Kruskal's Minimum Spanning Tree + redundancy edge

Ensures:
  1. All locations connected at minimum total cost
  2. At least two independent paths between Hospital and AmbulanceDepot
"""

from city_graph import CityGraph


# ─────────────────────────────────────────
#  Union-Find (Disjoint Set)
# ─────────────────────────────────────────
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


# ─────────────────────────────────────────
#  Kruskal's MST
# ─────────────────────────────────────────
def _kruskal_mst(nodes: list[int], edges_list: list[tuple]) -> list[tuple]:
    """
    edges_list: [(cost, u, v), ...]
    Returns list of (cost, u, v) edges in the MST.
    """
    id_map = {nid: idx for idx, nid in enumerate(nodes)}
    uf = UnionFind(len(nodes))
    mst = []

    for cost, u, v in sorted(edges_list):
        pu, pv = id_map[u], id_map[v]
        if uf.union(pu, pv):
            mst.append((cost, u, v))
            if len(mst) == len(nodes) - 1:
                break

    return mst


# ─────────────────────────────────────────
#  Path independence check (edge-disjoint BFS)
# ─────────────────────────────────────────
def _count_independent_paths(graph: CityGraph, src: int, dst: int) -> int:
    """
    Count edge-disjoint paths from src to dst using Ford-Fulkerson BFS.
    Each edge has capacity 1 in both directions.
    """
    # Build residual graph
    residual: dict[int, dict[int, int]] = {nid: {} for nid in graph.nodes}
    for (u, v), edge in graph.edges.items():
        if not edge.blocked:
            residual[u][v] = residual[u].get(v, 0) + 1
            residual[v][u] = residual[v].get(u, 0) + 1

    flow = 0
    while True:
        # BFS to find augmenting path
        parent = {src: None}
        queue = [src]
        found = False
        while queue and not found:
            next_queue = []
            for u in queue:
                for v, cap in residual[u].items():
                    if cap > 0 and v not in parent:
                        parent[v] = u
                        if v == dst:
                            found = True
                            break
                        next_queue.append(v)
            queue = next_queue

        if not found:
            break

        # Augment along path
        v = dst
        while v != src:
            u = parent[v]
            residual[u][v] -= 1
            residual[v][u] += 1
            v = u
        flow += 1

    return flow


# ─────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────
def optimize_road_network(graph: CityGraph) -> dict:
    """
    Uses Kruskal's to build MST on graph edges, then ensures two
    independent paths between Hospital and AmbulanceDepot.

    Returns a report dict.
    """
    nodes = list(graph.nodes.keys())
    edges_list = [(edge.base_cost, edge.u, edge.v) for edge in graph.edges.values()]

    mst_edges = _kruskal_mst(nodes, edges_list)
    mst_cost  = sum(c for c, _, _ in mst_edges)

    # Mark all non-MST edges as initially "non-road" by flagging them
    # (we keep them in the graph for potential blocking/redundancy use,
    #  but track which are primary MST roads)
    mst_set = {(min(u, v), max(u, v)) for _, u, v in mst_edges}

    report = {
        "mst_total_cost": round(mst_cost, 2),
        "mst_edges":      len(mst_edges),
        "redundancy_added": False,
        "redundancy_edge":  None,
        "independent_paths": 0,
    }

    # Find Hospital and AmbulanceDepot nodes
    hospitals = graph.nodes_by_type("Hospital")
    depots    = graph.nodes_by_type("AmbulanceDepot")

    if not hospitals or not depots:
        report["warning"] = "No Hospital or AmbulanceDepot found; skipping redundancy check."
        return report

    # Use the explicitly designated Primary Hospital and Primary Depot set by
    # plan_city_layout (the most central hospital). This ensures Challenge 2
    # checks redundancy for the correct, named pair — not an arbitrary one.
    hospital_id = graph.primary_hospital if graph.primary_hospital is not None else hospitals[0]
    depot_id    = graph.primary_depot    if graph.primary_depot    is not None else depots[0]

    # Check independent paths in MST subgraph
    # Temporarily limit edges to MST
    all_edges_backup = dict(graph.edges)
    for key in list(graph.edges.keys()):
        if key not in mst_set:
            graph.edges[key].blocked = True

    paths_in_mst = _count_independent_paths(graph, hospital_id, depot_id)

    # Restore
    for key, edge in all_edges_backup.items():
        graph.edges[key].blocked = edge.blocked

    if paths_in_mst < 2:
        # Add the cheapest non-MST edge that creates a second path
        # between hospital and depot (or any edge if not directly available)
        best_extra = None
        best_cost  = float("inf")

        for (u, v), edge in graph.edges.items():
            key = (min(u, v), max(u, v))
            if key not in mst_set and edge.base_cost < best_cost:
                # Temporarily add this edge and check paths
                mst_set.add(key)
                for k in graph.edges:
                    if k not in mst_set:
                        graph.edges[k].blocked = True

                paths = _count_independent_paths(graph, hospital_id, depot_id)

                for k in graph.edges:
                    if k not in mst_set:
                        graph.edges[k].blocked = False

                mst_set.discard(key)

                if paths >= 2 and edge.base_cost < best_cost:
                    best_cost  = edge.base_cost
                    best_extra = (u, v)

        # Restore all edges to unblocked (simulation manages blocking)
        for edge in graph.edges.values():
            edge.blocked = False

        if best_extra:
            mst_set.add((min(best_extra[0], best_extra[1]), max(best_extra[0], best_extra[1])))
            report["redundancy_added"] = True
            report["redundancy_edge"]  = best_extra
            report["mst_total_cost"]  += best_cost
        else:
            # If no existing grid edge can create a second path, we add a direct
            # emergency corridor between the Primary Hospital and Primary Depot
            # at cost 1.5. This represents a dedicated emergency access road —
            # justified because the spec mandates two independent routes must
            # always exist, and the grid topology may not provide one naturally.
            graph.add_edge_with_cost(hospital_id, depot_id, 1.5)
            mst_set.add((min(hospital_id, depot_id), max(hospital_id, depot_id)))
            report["redundancy_added"] = True
            report["redundancy_edge"]  = (hospital_id, depot_id)
    else:
        # Restore
        for edge in graph.edges.values():
            edge.blocked = False

    # ── Prune non-MST edges from the graph ───────────────────────────────
    # Only roads in mst_set are actually "built". Remove all others so the
    # UI and pathfinding only see real roads, not every possible grid connection.
    non_mst = [(u, v) for (u, v) in list(graph.edges.keys()) if (u, v) not in mst_set]
    for u, v in non_mst:
        graph.remove_edge(u, v)

    report["independent_paths"] = _count_independent_paths(graph, hospital_id, depot_id)
    graph.primary_hospital = hospital_id
    graph.primary_depot    = depot_id

    return report