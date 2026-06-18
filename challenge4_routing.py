"""
CityMind – Challenge 4: Emergency Routing Under Changing Conditions
Algorithm: A* Search with dynamic re-routing

The medical team visits civilians in sequence.
Whenever an edge is blocked mid-journey, route is recomputed immediately.
"""

from city_graph import CityGraph


class EmergencyRouter:
    """
    Manages an active mission: visits a sequence of civilian nodes.
    Recalculates path via A* whenever graph changes.
    """

    def __init__(self, graph: CityGraph, start_node: int, civilians: list[int]):
        self.graph         = graph
        self.current_node  = start_node
        self.civilians     = list(civilians)       # ordered target list
        self.target_idx    = 0                     # index into civilians
        self.current_path  = []                    # current planned path (list of node ids)
        self.path_cost     = 0.0
        self.log           = []

        # Register for graph changes so we auto-recompute
        graph.register_observer(self._on_graph_change)

        # Plan first path
        self._plan_next_path()

    # ── internal ──────────────────────────────
    def _on_graph_change(self, event: str, data):
        if event == "edge_blocked":
            u, v = data["u"], data["v"]
            # If blocked edge is on our current planned path, recompute
            if self._path_uses_edge(u, v):
                msg = f"Road {u}↔{v} blocked — recalculating route."
                self.log.append(msg)
                self._plan_next_path()

    def _path_uses_edge(self, u: int, v: int) -> bool:
        for i in range(len(self.current_path) - 1):
            a, b = self.current_path[i], self.current_path[i + 1]
            if {a, b} == {u, v}:
                return True
        return False

    def _plan_next_path(self):
        """Compute A* path from current position to next civilian."""
        if self.target_idx >= len(self.civilians):
            self.current_path = []
            self.path_cost    = 0.0
            return

        target = self.civilians[self.target_idx]
        path, cost = self.graph.astar(self.current_node, target)

        if not path:
            msg = (f"No path from {self.current_node} to civilian "
                   f"{target}! Skipping.")
            self.log.append(msg)
            self.target_idx += 1
            self._plan_next_path()
            return

        self.current_path = path
        self.path_cost    = cost
        msg = (f"A* route planned: {self.current_node} → {target} "
               f"via {path}, cost={cost:.2f}")
        self.log.append(msg)

    # ── public API ────────────────────────────
    def step(self) -> str | None:
        """
        Advance one node along current path.
        Returns a log message, or None if mission complete.
        """
        if self.mission_complete():
            return None

        if len(self.current_path) <= 1:
            # Arrived at current target
            if self.target_idx < len(self.civilians):
                civilian = self.civilians[self.target_idx]
                msg = f"Reached civilian at node {civilian}."
                self.log.append(msg)
                self.target_idx += 1
                self._plan_next_path()
                return msg
            return None

        # Move one step forward
        self.current_path.pop(0)
        self.current_node = self.current_path[0]
        msg = f"Team moved to node {self.current_node}."
        self.log.append(msg)
        return msg

    def mission_complete(self) -> bool:
        return self.target_idx >= len(self.civilians) and not self.current_path

    def get_current_target(self) -> int | None:
        if self.target_idx < len(self.civilians):
            return self.civilians[self.target_idx]
        return None

    def status(self) -> dict:
        return {
            "current_node":   self.current_node,
            "target":         self.get_current_target(),
            "civilians_left": len(self.civilians) - self.target_idx,
            "path":           list(self.current_path),
            "complete":       self.mission_complete(),
        }
