"""
CityMind – Shared City Graph
Single source of truth for the entire system.
All modules read from and write to this graph directly.
"""

import heapq
from collections import defaultdict, deque

# ─── Location types ────────────────────────────────────────────────────────
LOCATION_TYPES = ["Residential", "Hospital", "School",
                  "Industrial", "PowerPlant", "AmbulanceDepot", "Empty"]

# ─── Risk levels and their effective-cost multipliers ─────────────────────
RISK_LEVELS     = ["Low", "Medium", "High"]
RISK_MULTIPLIER = {"Low": 1.0, "Medium": 1.3, "High": 1.6}

# ─── Base road costs (per project spec) ───────────────────────────────────
BASE_COST_STANDARD    = 1.0   # all non-residential roads
BASE_COST_RESIDENTIAL = 0.8   # roads where BOTH endpoints are residential


class Node:
    def __init__(self, node_id: int, x: int, y: int,
                 location_type: str = "Residential",
                 population_density: float = 100.0):
        self.node_id           = node_id
        self.x                 = x
        self.y                 = y
        self.location_type     = location_type
        self.population_density = population_density
        self.risk_level        = "Low"   # updated by Challenge 5 ML module
        self.accessibility     = True    # set False if isolated by blockages
        self.cluster           = -1      # K-Means cluster id (set by Challenge 5)

    def __repr__(self):
        return f"Node({self.node_id}, {self.location_type}, ({self.x},{self.y}))"


class Edge:
    """
    Undirected edge between nodes u and v.
    base_cost: 0.8 if BOTH endpoints are Residential, else 1.0.
    effective_cost: base_cost × max risk multiplier of the two endpoints.
    """
    def __init__(self, u: int, v: int, base_cost: float = BASE_COST_STANDARD):
        self.u         = u
        self.v         = v
        self.base_cost = base_cost
        self.blocked   = False          # True if road flooded / accident

    def effective_cost(self, graph: "CityGraph") -> float:
        if self.blocked:
            return float("inf")
        risk_u = RISK_MULTIPLIER[graph.nodes[self.u].risk_level]
        risk_v = RISK_MULTIPLIER[graph.nodes[self.v].risk_level]
        return self.base_cost * max(risk_u, risk_v)

    def __repr__(self):
        s = "BLOCKED" if self.blocked else f"cost={self.base_cost:.1f}"
        return f"Edge({self.u}↔{self.v}, {s})"


class CityGraph:
    def __init__(self, rows: int = 6, cols: int = 8):
        self.rows  = rows
        self.cols  = cols
        self.nodes: dict[int, Node]           = {}
        self.edges: dict[tuple, Edge]         = {}   # (min_id, max_id) → Edge
        self.adjacency: dict[int, list[int]]  = defaultdict(list)
        self._observers: list                 = []

        # Set by modules for cross-module access
        self.ambulance_positions: list[int]   = []
        self.primary_hospital: int | None     = None
        self.primary_depot: int | None        = None
        self.police_deployment: dict[int,int] = {}   # node_id → num officers
        self.cluster_assignments: dict[int,int] = {}  # node_id → cluster (0/1/2)

    # ── Observer pattern (so A* auto-replans on edge changes) ─────────────
    def register_observer(self, callback):
        self._observers.append(callback)

    def _notify(self, event: str, data=None):
        for cb in self._observers:
            cb(event, data)

    # ── Node management ───────────────────────────────────────────────────
    def add_node(self, node: Node):
        self.nodes[node.node_id] = node

    def nodes_by_type(self, loc_type: str) -> list[int]:
        return [nid for nid, n in self.nodes.items()
                if n.location_type == loc_type]

    def node_at(self, x: int, y: int) -> int | None:
        for nid, n in self.nodes.items():
            if n.x == x and n.y == y:
                return nid
        return None

    # ── Edge management ───────────────────────────────────────────────────
    def _key(self, u: int, v: int) -> tuple:
        return (min(u, v), max(u, v))

    def add_edge(self, u: int, v: int):
        """Add edge; cost = 0.8 if both endpoints Residential, else 1.0."""
        key = self._key(u, v)
        if key in self.edges:
            return
        n_u = self.nodes[u]
        n_v = self.nodes[v]
        if n_u.location_type == "Residential" and n_v.location_type == "Residential":
            cost = BASE_COST_RESIDENTIAL
        else:
            cost = BASE_COST_STANDARD
        self.edges[key] = Edge(u, v, cost)
        self.adjacency[u].append(v)
        self.adjacency[v].append(u)

    def add_edge_with_cost(self, u: int, v: int, cost: float):
        """Force a specific cost (used for redundancy edges)."""
        key = self._key(u, v)
        if key in self.edges:
            return
        self.edges[key] = Edge(u, v, cost)
        self.adjacency[u].append(v)
        self.adjacency[v].append(u)

    def remove_edge(self, u: int, v: int):
        """Permanently remove an edge (used by road optimiser to delete non-MST roads)."""
        key = self._key(u, v)
        if key not in self.edges:
            return
        del self.edges[key]
        if v in self.adjacency[u]:
            self.adjacency[u].remove(v)
        if u in self.adjacency[v]:
            self.adjacency[v].remove(u)

    def get_edge(self, u: int, v: int) -> "Edge | None":
        return self.edges.get(self._key(u, v))

    def block_edge(self, u: int, v: int):
        """Block a road (flood / accident). Notifies all observers."""
        edge = self.get_edge(u, v)
        if edge and not edge.blocked:
            edge.blocked = True
            # Update accessibility flags
            self._update_accessibility()
            self._notify("edge_blocked", {"u": u, "v": v})

    def unblock_edge(self, u: int, v: int):
        edge = self.get_edge(u, v)
        if edge and edge.blocked:
            edge.blocked = False
            self._update_accessibility()
            self._notify("edge_unblocked", {"u": u, "v": v})

    def _update_accessibility(self):
        """Mark nodes as inaccessible if completely cut off."""
        reachable = set()
        if not self.nodes:
            return
        start = next(iter(self.nodes))
        queue = deque([start])
        while queue:
            nid = queue.popleft()
            if nid in reachable:
                continue
            reachable.add(nid)
            for nb in self.adjacency[nid]:
                edge = self.get_edge(nid, nb)
                if edge and not edge.blocked and nb not in reachable:
                    queue.append(nb)
        for nid, node in self.nodes.items():
            node.accessibility = (nid in reachable)

    def effective_cost(self, u: int, v: int) -> float:
        edge = self.get_edge(u, v)
        return edge.effective_cost(self) if edge else float("inf")

    def neighbors(self, nid: int, respect_blocked: bool = True) -> list[int]:
        result = []
        for nb in self.adjacency[nid]:
            edge = self.get_edge(nid, nb)
            if edge:
                if not respect_blocked or not edge.blocked:
                    result.append(nb)
        return result

    # ── Risk update (called by Challenge 5) ───────────────────────────────
    def set_risk(self, nid: int, level: str):
        assert level in RISK_LEVELS
        self.nodes[nid].risk_level = level
        self._notify("risk_updated", {"node": nid, "level": level})

    # ── Dijkstra (used by Challenge 3 for coverage distances) ─────────────
    def dijkstra(self, source: int) -> tuple[dict, dict]:
        dist = {nid: float("inf") for nid in self.nodes}
        prev = {nid: None for nid in self.nodes}
        dist[source] = 0.0
        pq = [(0.0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v in self.neighbors(u):
                nd = d + self.effective_cost(u, v)
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v]  = u
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    # ── A* (used by Challenge 4 emergency routing) ────────────────────────
    def astar(self, src: int, dst: int) -> tuple[list[int], float]:
        """
        A* with admissible heuristic: 0.8 × Manhattan distance.
        Admissibility proof: the minimum possible edge cost in the graph is
        0.8 (residential-to-residential road, risk multiplier = 1.0).
        Therefore h(n) = 0.8 × Manhattan_distance never overestimates the
        true path cost, guaranteeing that A* returns the shortest path.
        Using plain Manhattan distance (h = 1.0 × steps) would overestimate
        on residential edges (cost 0.8 < 1.0), violating admissibility.
        """
        def heuristic(a: int, b: int) -> float:
            na, nb = self.nodes[a], self.nodes[b]
            return 0.8 * (abs(na.x - nb.x) + abs(na.y - nb.y))

        g_score  = {nid: float("inf") for nid in self.nodes}
        g_score[src] = 0.0
        came_from: dict[int, int | None] = {src: None}
        open_set  = [(heuristic(src, dst), 0.0, src)]

        while open_set:
            _, g, current = heapq.heappop(open_set)
            if g > g_score[current]:
                continue
            if current == dst:
                # Reconstruct
                path = []
                c = dst
                while c is not None:
                    path.append(c)
                    c = came_from.get(c)
                path.reverse()
                return path, g_score[dst]
            for nb in self.neighbors(current):
                tentative = g_score[current] + self.effective_cost(current, nb)
                if tentative < g_score[nb]:
                    g_score[nb]   = tentative
                    came_from[nb] = current
                    f = tentative + heuristic(nb, dst)
                    heapq.heappush(open_set, (f, tentative, nb))

        return [], float("inf")

    # ── Utility ───────────────────────────────────────────────────────────
    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def is_fully_connected(self) -> bool:
        if not self.nodes:
            return True
        visited = set()
        queue   = deque([next(iter(self.nodes))])
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            for nb in self.neighbors(nid, respect_blocked=False):
                if nb not in visited:
                    queue.append(nb)
        return len(visited) == len(self.nodes)