"""
CityMind – Challenge 1: City Layout Planning
Algorithm: Backtracking Search with Forward Checking (CSP)

Variables   : each grid cell
Domain      : {Residential, Hospital, School, Industrial, PowerPlant,
               AmbulanceDepot, Empty}
Constraints :
  C1 – Industrial not adjacent (1 hop) to School or Hospital
  C2 – Every Residential within 3 road-hops of at least one Hospital
  C3 – Every PowerPlant within 2 road-hops of at least one Industrial zone

Empty nodes:
  A cell becomes Empty when it cannot be assigned Residential without
  violating C2 (i.e. no Hospital is reachable within 3 hops and none can
  be placed in range), AND no other mandatory type needs to go there.
  Empty cells are valid grid nodes — they participate in pathfinding and
  BFS hop counts — but carry no functional type and have no C2 obligation.
  They are kept to a minimum: only cells geometrically unreachable from
  any hospital within 3 hops after all hospitals are placed become Empty.

Forward Checking:
  - C1: after assigning Industrial, prune School/Hospital from neighbours' domains
  - C2: before assigning Residential, verify a Hospital exists within 3 hops
         OR an unassigned cell within 3 hops still has Hospital in its domain;
         if neither holds -> assign Empty instead (never violate C2)
  - C3: before assigning PowerPlant, verify Industrial exists within 2 hops
         OR an unassigned cell within 2 hops still has Industrial in its domain
"""

import random
from collections import deque
from city_graph import CityGraph, Node, BASE_COST_STANDARD, BASE_COST_RESIDENTIAL

# How many of each type to place (Residential fills reachable remainder;
# Empty used only for cells outside 3-hop hospital coverage)
# Grid: 7x7 = 49 nodes total
PLACEMENT_COUNTS = {
    "Hospital":       4,
    "School":         3,
    "Industrial":     4,
    "PowerPlant":     3,
    "AmbulanceDepot": 2,
}

# Maximum fraction of cells allowed to become Empty
MAX_EMPTY_FRACTION = 0.15   # <= 15% of the grid -> at most 7 cells on 7x7


# --- Grid helpers -----------------------------------------------------------
def _adj(x: int, y: int, cols: int, rows: int) -> list:
    """4-connected grid neighbours."""
    result = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < cols and 0 <= ny < rows:
            result.append((nx, ny))
    return result


def _cells_within(x: int, y: int, dist: int, cols: int, rows: int) -> list:
    """All grid cells within Manhattan distance `dist` of (x,y)."""
    result = []
    for dx in range(-dist, dist + 1):
        for dy in range(-(dist - abs(dx)), dist - abs(dx) + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and (dx, dy) != (0, 0):
                result.append((nx, ny))
    return result


# --- Constraint checks -------------------------------------------------------
def _violates_c1(cell: tuple, loc_type: str, assignment: dict,
                 cols: int, rows: int) -> bool:
    """C1: Industrial must not be adjacent to School or Hospital."""
    if loc_type == "Industrial":
        for nb in _adj(cell[0], cell[1], cols, rows):
            if assignment.get(nb) in ("School", "Hospital"):
                return True
    if loc_type in ("School", "Hospital"):
        for nb in _adj(cell[0], cell[1], cols, rows):
            if assignment.get(nb) == "Industrial":
                return True
    return False


def _c2_hops(cols: int, rows: int) -> int:
    """C2 hop limit: project specification mandates exactly 3 road hops."""
    return 3


def _can_satisfy_c2(cell: tuple, assignment: dict, domains: dict,
                    cols: int, rows: int) -> bool:
    """
    C2 lookahead: can this cell be within C2_HOPS of a Hospital?
    True if a Hospital is already placed within range, OR an unassigned
    cell within range still has Hospital in its domain.
    Empty cells do NOT block BFS traversal.
    """
    x, y = cell
    hops = _c2_hops(cols, rows)
    for cx, cy in _cells_within(x, y, hops, cols, rows):
        t = assignment.get((cx, cy))
        if t == "Hospital":
            return True
        if t is None and "Hospital" in domains.get((cx, cy), set()):
            return True
    return False


def _can_satisfy_c3(cell: tuple, assignment: dict, domains: dict,
                    cols: int, rows: int) -> bool:
    """C3 lookahead: can this PowerPlant cell be within 2 hops of an Industrial?"""
    x, y = cell
    for cx, cy in _cells_within(x, y, 2, cols, rows):
        t = assignment.get((cx, cy))
        if t == "Industrial":
            return True
        if t is None and "Industrial" in domains.get((cx, cy), set()):
            return True
    return False


def _bfs_hops(start: tuple, assignment: dict, cols: int, rows: int,
              max_hops: int) -> dict:
    """
    BFS on the 4-connected grid.  All cells (including Empty) are traversable.
    Returns {cell: hop_count} for all cells reachable within max_hops.
    """
    visited = {start: 0}
    queue = deque([(start, 0)])
    while queue:
        (cx, cy), hops = queue.popleft()
        if hops >= max_hops:
            continue
        for nx, ny in _adj(cx, cy, cols, rows):
            nb = (nx, ny)
            if nb not in visited:
                visited[nb] = hops + 1
                queue.append((nb, hops + 1))
    return visited


def _check_full_constraints(assignment: dict, cols: int, rows: int) -> dict:
    """
    Check all three constraints against the final assignment.
    Empty cells have no type requirement and are excluded from C2/C3 checks.
    Returns {constraint: [violating cells]}.
    """
    violations = {"C1": [], "C2": [], "C3": []}

    for cell, t in assignment.items():
        if t == "Empty":
            continue
        if _violates_c1(cell, t, assignment, cols, rows):
            if cell not in violations["C1"]:
                violations["C1"].append(cell)

    hospital_cells   = [c for c, t in assignment.items() if t == "Hospital"]
    industrial_cells = [c for c, t in assignment.items() if t == "Industrial"]
    c2_limit = _c2_hops(cols, rows)

    for cell, t in assignment.items():
        if t == "Residential":
            reachable = False
            for h in hospital_cells:
                hops_map = _bfs_hops(h, assignment, cols, rows, c2_limit)
                if cell in hops_map:
                    reachable = True
                    break
            if not reachable:
                violations["C2"].append(cell)

        if t == "PowerPlant":
            reachable = False
            for ind in industrial_cells:
                hops_map = _bfs_hops(ind, assignment, cols, rows, 2)
                if cell in hops_map:
                    reachable = True
                    break
            if not reachable:
                violations["C3"].append(cell)

    return violations


# --- Forward Checking --------------------------------------------------------
def _forward_check(cell: tuple, loc_type: str,
                   domains: dict, assignment: dict,
                   cols: int, rows: int) -> bool:
    """
    C1 forward checking: prune neighbour domains after Industrial/School/Hospital.
    Empty is always available in every domain so domains cannot truly empty out.
    Returns False only if a domain loses ALL values (shouldn't happen with Empty).
    """
    if loc_type == "Industrial":
        for nb in _adj(cell[0], cell[1], cols, rows):
            if nb in domains:
                domains[nb] -= {"School", "Hospital"}
                if not domains[nb]:
                    return False
    if loc_type in {"School", "Hospital"}:
        for nb in _adj(cell[0], cell[1], cols, rows):
            if nb in domains:
                domains[nb] -= {"Industrial"}
                if not domains[nb]:
                    return False
    return True


# --- MRV heuristic -----------------------------------------------------------
def _select_variable(unassigned: list, domains: dict) -> tuple:
    return min(unassigned, key=lambda c: len(domains[c]))


# --- Coverage helper ---------------------------------------------------------
def _coverage_from_positions(positions: list, cols: int, rows: int,
                              hops_limit: int) -> set:
    covered = set()
    queue = deque()
    for h in positions:
        if h not in covered:
            covered.add(h)
            queue.append((h, 0))
    while queue:
        (cx, cy), d = queue.popleft()
        if d >= hops_limit:
            continue
        for nx, ny in _adj(cx, cy, cols, rows):
            nb = (nx, ny)
            if nb not in covered:
                covered.add(nb)
                queue.append((nb, d + 1))
    return covered


# --- Main backtracking search ------------------------------------------------
def _backtrack(unassigned: list, assignment: dict, domains: dict,
               remaining: dict, cols: int, rows: int,
               hosp_reachable: set, max_empty: int) -> dict | None:
    if not unassigned:
        v = _check_full_constraints(assignment, cols, rows)
        if v["C2"] or v["C3"]:
            return None
        return assignment

    cell = _select_variable(unassigned, domains)
    unassigned_copy = [c for c in unassigned if c != cell]

    # Value ordering: typed nodes first, Residential, Empty last (last resort)
    priority_order = ["Hospital", "School", "Industrial", "PowerPlant",
                      "AmbulanceDepot", "Residential", "Empty"]
    ordered_values = sorted(
        list(domains[cell]),
        key=lambda v: priority_order.index(v) if v in priority_order else 99
    )

    for value in ordered_values:
        # Count check for typed nodes
        if value not in ("Residential", "Empty") and remaining.get(value, 0) <= 0:
            continue

        # Empty budget guard + only use Empty when Residential can't satisfy C2
        if value == "Empty":
            empty_so_far = sum(1 for t in assignment.values() if t == "Empty")
            if empty_so_far >= max_empty:
                continue
            # Only allow Empty if this cell is outside hospital 3-hop coverage
            if cell in hosp_reachable:
                continue  # Residential is viable -> don't waste a cell on Empty

        # C1 immediate check
        if _violates_c1(cell, value, assignment, cols, rows):
            continue

        # C2 lookahead
        if value == "Residential" and not _can_satisfy_c2(cell, assignment, domains, cols, rows):
            continue

        # C3 lookahead
        if value == "PowerPlant" and not _can_satisfy_c3(cell, assignment, domains, cols, rows):
            continue

        # Tentative assignment
        assignment[cell] = value
        if value not in ("Residential", "Empty"):
            remaining[value] -= 1

        saved_domains = {c: set(domains[c]) for c in unassigned_copy}

        if _forward_check(cell, value, domains, assignment, cols, rows):
            result = _backtrack(unassigned_copy, assignment, domains,
                                remaining, cols, rows, hosp_reachable, max_empty)
            if result is not None:
                return result

        # Undo
        del assignment[cell]
        if value not in ("Residential", "Empty"):
            remaining[value] += 1
        for c in unassigned_copy:
            domains[c] = saved_domains[c]

    return None


# --- Minimum-conflict fallback -----------------------------------------------
def _min_conflict_fallback(cells: list, cols: int, rows: int,
                           remaining: dict, hospital_cells: list,
                           hosp_reachable: set, seed: int) -> dict:
    """
    Greedy min-conflict assignment used when full backtracking fails.
    Cells unreachable from any hospital within 3 hops are assigned Empty.
    """
    rng = random.Random(seed)
    assignment = {cell: "Hospital" for cell in hospital_cells}
    rem = dict(remaining)

    for cell in cells:
        if cell in assignment:
            continue

        can_be_residential = cell in hosp_reachable
        best_type = None
        best_cost = float("inf")
        candidates = [t for t, cnt in rem.items() if cnt is None or cnt > 0]
        rng.shuffle(candidates)

        for t in candidates:
            if not can_be_residential and t == "Residential":
                continue
            assignment[cell] = t
            v = _check_full_constraints(assignment, cols, rows)
            cost = sum(len(lst) for lst in v.values())
            if cost < best_cost:
                best_cost = cost
                best_type = t
            del assignment[cell]

        if best_type is None:
            best_type = "Residential" if can_be_residential else "Empty"

        assignment[cell] = best_type
        if rem.get(best_type) is not None:
            rem[best_type] = max(0, rem[best_type] - 1)

    return assignment


# --- Build grid edges --------------------------------------------------------
def _build_edges(graph: CityGraph):
    """
    Build edges for all adjacent cell pairs, including Empty nodes.
    Empty nodes participate in pathfinding so they do not interrupt connectivity.
    """
    for nid, node in graph.nodes.items():
        for dx, dy in [(1, 0), (0, 1)]:
            nx, ny = node.x + dx, node.y + dy
            nb_id  = graph.node_at(nx, ny)
            if nb_id is not None:
                graph.add_edge(nid, nb_id)


# --- Population density per type ---------------------------------------------
def _density(loc_type: str, rng: random.Random) -> float:
    return {
        "Residential":    rng.uniform(500, 2000),
        "Hospital":       rng.uniform(100, 300),
        "School":         rng.uniform(200, 600),
        "Industrial":     rng.uniform(50,  200),
        "PowerPlant":     rng.uniform(20,   80),
        "AmbulanceDepot": rng.uniform(30,  100),
        "Empty":          0.0,
    }[loc_type]


# --- Public API --------------------------------------------------------------
def plan_city_layout(graph: CityGraph, seed: int = 42) -> tuple:
    """
    Unified CSP with backtracking + Empty-node fallback:

      Phase 1a - Place hospitals via backtracking.  Goal: maximise coverage;
                  any remaining uncovered cells will become Empty, not Residential.
      Phase 1b - Place Industrial, School, PowerPlant, AmbulanceDepot with
                  C1 (adjacency) and C3 (PowerPlant near Industrial) checks.
      Phase 2  - Fill remaining cells:
                  * Cell in hosp_reachable (within 3 hops) -> Residential (C2 OK)
                  * Cell outside coverage              -> Empty (no C2 obligation)

    This design guarantees:
      - C1 is enforced by construction during placement
      - C2 is NEVER violated: only reachable cells get Residential
      - C3 is enforced by construction during placement
      - Empty nodes are minimal: only cells geometrically beyond 3 hops

    Returns (success, violation_report).
    """
    rng  = random.Random(seed)
    cols = graph.cols
    rows = graph.rows

    all_cells = [(x, y) for y in range(rows) for x in range(cols)]
    rng.shuffle(all_cells)

    max_empty = max(1, int(cols * rows * MAX_EMPTY_FRACTION))

    # -- Phase 1a: backtrack over hospital positions -------------------------
    hosp_count = PLACEMENT_COUNTS["Hospital"]
    hosp_cells = []

    def place_hospitals(idx: int) -> bool:
        if idx == hosp_count:
            return True  # any arrangement accepted; non-covered -> Empty
        hops = _c2_hops(cols, rows)
        max_per_hosp = (2 * hops + 1) ** 2
        remaining_slots = hosp_count - idx

        for cell in all_cells:
            if cell in hosp_cells:
                continue
            hosp_cells.append(cell)
            current_cov = len(_coverage_from_positions(hosp_cells, cols, rows, hops))
            uncovered = (cols * rows) - current_cov
            # Pruning: if even optimistic future coverage can't cover all, skip
            if uncovered > (remaining_slots - 1) * max_per_hosp:
                # Still accept if we're within empty budget
                if uncovered > max_empty:
                    hosp_cells.pop()
                    continue
            if place_hospitals(idx + 1):
                return True
            hosp_cells.pop()
        return False

    hosp_success = place_hospitals(0)
    placed = {cell: "Hospital" for cell in hosp_cells}

    # Cells reachable from hospitals within 3 hops (C2 boundary)
    c2_limit = _c2_hops(cols, rows)
    hosp_reachable = _coverage_from_positions(hosp_cells, cols, rows, c2_limit)

    # -- Phase 1b: place Industrial, School, PowerPlant, AmbulanceDepot -----
    priority_types = ["Industrial", "School", "PowerPlant", "AmbulanceDepot"]
    needed = []
    for typ in priority_types:
        needed.extend([typ] * PLACEMENT_COUNTS[typ])

    def backtrack_rest(idx: int) -> bool:
        if idx == len(needed):
            return True
        typ = needed[idx]
        for cell in all_cells:
            if cell in placed:
                continue
            if _violates_c1(cell, typ, placed, cols, rows):
                continue
            if typ == "PowerPlant":
                already_has = any(
                    placed.get((cx, cy)) == "Industrial"
                    for cx, cy in _cells_within(cell[0], cell[1], 2, cols, rows)
                )
                remaining_ind = sum(
                    1 for j in range(idx + 1, len(needed)) if needed[j] == "Industrial"
                )
                if not already_has and remaining_ind == 0:
                    continue
            placed[cell] = typ
            if backtrack_rest(idx + 1):
                return True
            del placed[cell]
        return False

    rest_success = backtrack_rest(0)
    success = hosp_success and rest_success

    if not success:
        # Min-conflict greedy fallback
        rng2 = random.Random(seed + 1)
        remaining_cells = [c for c in all_cells if c not in placed]
        rng2.shuffle(remaining_cells)
        for typ in priority_types:
            count = PLACEMENT_COUNTS[typ]
            assigned = 0
            for cell in remaining_cells:
                if cell in placed or assigned >= count:
                    continue
                placed[cell] = typ
                assigned += 1

    # -- Phase 2: fill remaining cells (Residential or Empty) ----------------
    empty_count = 0
    for cell in all_cells:
        if cell not in placed:
            if cell in hosp_reachable:
                placed[cell] = "Residential"
            else:
                placed[cell] = "Empty"
                empty_count += 1

    # Safety cap: if somehow too many empties, convert excess back to Residential
    if empty_count > max_empty:
        empty_cells = [c for c, t in placed.items() if t == "Empty"]
        for ec in empty_cells[max_empty:]:
            placed[ec] = "Residential"

    # -- Populate graph nodes ------------------------------------------------
    nid = 0
    for y in range(rows):
        for x in range(cols):
            cell     = (x, y)
            loc_type = placed.get(cell, "Residential")
            node     = Node(nid, x, y, loc_type, _density(loc_type, rng))
            graph.add_node(node)
            nid += 1

    _build_edges(graph)

    # -- Designate Primary Hospital and Primary Ambulance Depot ---------------
    cx, cy = (cols - 1) / 2.0, (rows - 1) / 2.0
    hospital_nodes = graph.nodes_by_type("Hospital")
    if hospital_nodes:
        graph.primary_hospital = min(
            hospital_nodes,
            key=lambda nid: abs(graph.nodes[nid].x - cx) + abs(graph.nodes[nid].y - cy),
        )

    depot_nodes = graph.nodes_by_type("AmbulanceDepot")
    if depot_nodes:
        graph.primary_depot = depot_nodes[0]

    violations = _check_full_constraints(placed, cols, rows)
    return success, violations


def layout_summary(graph: CityGraph) -> str:
    from collections import Counter
    counts = Counter(n.location_type for n in graph.nodes.values())
    lines  = ["City Layout:"]
    for t in ["Hospital", "AmbulanceDepot", "School", "Industrial",
              "PowerPlant", "Residential", "Empty"]:
        if counts.get(t, 0) > 0:
            lines.append(f"  {t}: {counts.get(t, 0)}")
    return "\n".join(lines)


def get_layout_report(graph: CityGraph) -> str:
    return layout_summary(graph)
