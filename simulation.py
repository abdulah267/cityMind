"""
CityMind – Simulation Engine
20-step simulation runner.
Orchestrates all five modules through the shared city graph.
"""

import random
from city_graph import CityGraph
from challenge1_layout import plan_city_layout, get_layout_report
from challenge2_roads   import optimize_road_network
from challenge3_ambulance import place_ambulances
from challenge4_routing  import EmergencyRouter
from challenge5_crime    import run_crime_prediction


class CityMindSimulation:
    """
    Full CityMind simulation system.
    Initialise once, then call run_simulation() or step() iteratively.
    """

    def __init__(self, rows: int = 7, cols: int = 7, seed: int = 42, max_steps: int = 20):
        self.rows   = rows
        self.cols   = cols
        self.seed   = seed
        self.step_num = 0
        self.max_steps = max(20, max_steps)  # minimum 20 steps
        self.event_log: list[str] = []
        self.graph: CityGraph | None = None
        self.router: EmergencyRouter | None = None
        self._step_data: list[dict] = []   # per-step snapshot for UI
        self._initialized = False

    # ── setup ──────────────────────────────────────
    def initialize(self):
        """Build city, run all static modules, set up simulation."""
        self.graph = CityGraph(self.rows, self.cols)
        rng = random.Random(self.seed)

        # Challenge 1: Layout
        success, violations = plan_city_layout(self.graph, seed=self.seed)
        layout_report = get_layout_report(self.graph)
        if success:
            self._log(f"[INIT] City layout planned. CSP backtracking found a VALID solution — all constraints satisfied.")
        else:
            self._log(f"[INIT] City layout: CSP backtracking could NOT find a perfect solution. "
                      f"Minimum-conflict fallback applied.")
            constraint_names = {
                "C1": "C1 (Industrial must not be adjacent to School/Hospital)",
                "C2": "C2 (Residential must be within hop-range of a Hospital)",
                "C3": "C3 (PowerPlant must be within 2 hops of Industrial)",
            }
            for k, vlist in violations.items():
                if vlist:
                    self._log(f"  ► Violated rule: {constraint_names[k]}")
                    self._log(f"    Affected cells: {vlist[:3]}{'...' if len(vlist) > 3 else ''}")
                    if k == "C1":
                        self._log(f"    Repair: relocated Industrial zones away from sensitive locations.")
                    elif k == "C2":
                        self._log(f"    Repair: added extra Hospital(s) near uncovered Residential cells.")
                    elif k == "C3":
                        self._log(f"    Repair: relocated PowerPlant(s) closer to Industrial zones.")
            self._log(f"  ► Minimum-conflict solution applied: layout is near-optimal with fewest rule violations.")
        self._log(layout_report)

        # Challenge 2: Roads
        road_report = optimize_road_network(self.graph)
        self._log(f"[INIT] Road network built. MST cost={road_report['mst_total_cost']}, "
                  f"edges={road_report['mst_edges']}, "
                  f"independent paths Hospital↔Depot={road_report['independent_paths']}.")
        if road_report["redundancy_added"]:
            self._log(f"[INIT] Redundancy edge added: {road_report['redundancy_edge']}.")

        # Challenge 5: Crime prediction (before ambulance placement so risk is set)
        crime_report = run_crime_prediction(self.graph, seed=self.seed)
        self._log(f"[INIT] Crime risk predicted — "
                  f"Low:{crime_report['counts']['Low']}, "
                  f"Medium:{crime_report['counts']['Medium']}, "
                  f"High:{crime_report['counts']['High']}.")
        p = crime_report['police']
        self._log(f"[INIT] Police deployed — {p['total_assigned']}/10 officers assigned "
                  f"({p['high_covered']} High-risk nodes, {p['medium_covered']} Medium-risk nodes covered).")

        # Challenge 3: Ambulance placement
        amb_report = place_ambulances(self.graph, seed=self.seed)
        risk_info = ", ".join(
            f"node {n}[{self.graph.nodes[n].risk_level}]" for n in amb_report['placement']
        )
        self._log(f"[INIT] Ambulances placed (Simulated Annealing, risk-weighted Dijkstra): "
                  f"{amb_report['placement']} | {risk_info} — "
                  f"worst response={amb_report['worst_response']} "
                  f"(SA iterations={amb_report['iterations']}). "
                  f"High-risk zones increase effective edge cost by ×1.6.")

        # Challenge 4: Emergency routing
        # Pick 4 random civilian nodes as targets
        civilian_candidates = [nid for nid, n in self.graph.nodes.items()
                                if n.location_type in ("Residential", "School")]
        random.seed(self.seed)
        civilians = random.sample(civilian_candidates, min(4, len(civilian_candidates)))
        start_node = self.graph.nodes_by_type("Hospital")[0]

        self.router = EmergencyRouter(self.graph, start_node, civilians)
        self._log(f"[INIT] Emergency team dispatched from node {start_node} "
                  f"to civilians at {civilians}.")

        # Schedule random flooding events (every ~4 steps: 3, 7, 11, 15, 19, 23, ...)
        all_edges = list(self.graph.edges.keys())
        rng.shuffle(all_edges)
        self._flood_events: dict[int, tuple] = {}
        flood_steps = list(range(3, self.max_steps + 1, 4))
        for i, step in enumerate(flood_steps):
            if i < len(all_edges):
                self._flood_events[step] = all_edges[i]

        # Schedule ambulance re-evaluation (every ~6 steps: 8, 14, 20, 26, ...)
        self._amb_reevaluate_steps = set(range(8, self.max_steps + 1, 6))

        # Schedule risk re-prediction (every ~10 steps: 10, 20, 30, ...)
        self._risk_restep = set(range(10, self.max_steps + 1, 10))

        self._initialized = True
        self.step_num = 0
        self._snapshot()

    # ── simulation step ────────────────────────────
    def step(self) -> dict:
        """Advance simulation by one step. Returns step summary."""
        if not self._initialized:
            self.initialize()

        if self.step_num >= self.max_steps:
            return {"done": True, "step": self.step_num}

        self.step_num += 1
        step_events = []

        # Flooding event?
        if self.step_num in self._flood_events:
            u, v = self._flood_events[self.step_num]
            self.graph.block_edge(u, v)
            msg = f"[Step {self.step_num}] FLOOD: Road {u}↔{v} blocked."
            self._log(msg)
            step_events.append(msg)

        # Emergency team moves
        if self.router and not self.router.mission_complete():
            move_msg = self.router.step()
            if move_msg:
                msg = f"[Step {self.step_num}] ROUTER: {move_msg}"
                self._log(msg)
                step_events.append(msg)

            # Also relay any auto-recompute messages the router logged
            if self.router.log:
                last = self.router.log[-1]
                if "recalculating" in last:
                    msg = f"[Step {self.step_num}] ROUTER: {last}"
                    self._log(msg)
                    step_events.append(msg)

        # Risk re-prediction
        if self.step_num in self._risk_restep:
            old_positions = list(getattr(self.graph, "ambulance_positions", []))
            crime_report = run_crime_prediction(self.graph, seed=self.seed + self.step_num)
            msg = (f"[Step {self.step_num}] RISK UPDATE: "
                   f"Low:{crime_report['counts']['Low']}, "
                   f"Medium:{crime_report['counts']['Medium']}, "
                   f"High:{crime_report['counts']['High']}. "
                   f"Edge costs updated — High-risk zones now cost ×1.6, Medium ×1.3.")
            self._log(msg)
            step_events.append(msg)
            p = crime_report['police']
            msg2 = (f"[Step {self.step_num}] POLICE REDEPLOYED: "
                    f"{p['total_assigned']}/10 officers — "
                    f"{p['high_covered']} High-risk, {p['medium_covered']} Medium-risk nodes covered.")
            self._log(msg2)
            step_events.append(msg2)

        # Ambulance re-evaluation
        if self.step_num in self._amb_reevaluate_steps:
            old_positions = list(getattr(self.graph, "ambulance_positions", []))
            amb_report = place_ambulances(self.graph, seed=self.seed + self.step_num)
            new_positions = amb_report['placement']
            moved = [n for n in new_positions if n not in old_positions]
            # Annotate which new positions are in High/Medium risk zones
            risk_info = ", ".join(
                f"node {n}[{self.graph.nodes[n].risk_level}]" for n in new_positions
            )
            msg = (f"[Step {self.step_num}] AMBULANCE REPOSITIONED (risk-weighted SA): "
                   f"→ {new_positions} | {risk_info} | "
                   f"worst_response={amb_report['worst_response']} "
                   f"(Dijkstra uses risk-adjusted edge costs).")
            if moved:
                msg += f" Moved away from: {[n for n in old_positions if n not in new_positions]}."
            self._log(msg)
            step_events.append(msg)

        self._snapshot()
        return {
            "done":   self.step_num >= self.max_steps,
            "step":   self.step_num,
            "events": step_events,
        }

    def run_simulation(self) -> list[dict]:
        """Run all 20 steps. Returns list of step summaries."""
        if not self._initialized:
            self.initialize()
        results = []
        while self.step_num < self.max_steps:
            result = self.step()
            results.append(result)
            if result.get("done"):
                break
        return results

    def _snapshot(self):
        """Capture current state for UI rendering."""
        snap = {
            "step":      self.step_num,
            "nodes":     {nid: {
                              "x":       n.x,
                              "y":       n.y,
                              "type":    n.location_type,
                              "risk":    n.risk_level,
                              "density": n.population_density,
                              "cluster": n.cluster,
                          }
                          for nid, n in self.graph.nodes.items()},
            "edges":     [{"u": e.u, "v": e.v,
                           "blocked": e.blocked,
                           "cost":    e.base_cost}
                          for e in self.graph.edges.values()],
            "ambulances":          getattr(self.graph, "ambulance_positions", []),
            "police_deployment":   dict(getattr(self.graph, "police_deployment", {})),
            "cluster_assignments": dict(getattr(self.graph, "cluster_assignments", {})),
            "router": self.router.status() if self.router else {},
            "log":    list(self.event_log[-10:]),   # last 10 entries for UI
        }
        self._step_data.append(snap)

    def _log(self, msg: str):
        self.event_log.append(msg)

    # ── accessors ──────────────────────────────────
    def current_snapshot(self) -> dict:
        return self._step_data[-1] if self._step_data else {}

    def get_step_data(self, step: int) -> dict:
        if 0 <= step < len(self._step_data):
            return self._step_data[step]
        return {}

    def full_log(self) -> list[str]:
        return list(self.event_log)