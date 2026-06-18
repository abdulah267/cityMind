# CityMind –  Implementation

**Language: Python 3.10+**

---

## How to Run

### 1. Install dependencies
```
pip install pygame
```
> No other external packages needed. All algorithms (K-Means, Decision Tree,
> Kruskal's, A*, Simulated Annealing, CSP Backtracking) are implemented from scratch.

### 2. Launch the visual interface
```
python main.py
```

### 3. Controls
| Button | Action |
|--------|--------|
| ▶ Run  | Auto-advance through all 20 simulation steps |
| ⏸ Pause | Pause auto-run |
| ⏭ Step | Manually advance one step at a time |
| ↺ Reset | Restart the full simulation |

### 4. Overlay Toggles (right panel)
- **Road Network** – shows all roads; blocked roads appear in red
- **Ambulance Coverage** – shows radial coverage zones from ambulance positions
- **Crime Heatmap** – colours node background by risk level (green/yellow/red)
- **Router Path** – shows the A* planned route for the emergency team

---

## File Structure

| File | Purpose |
|------|---------|
| `city_graph.py` | Shared city graph — single source of truth |
| `challenge1_layout.py` | CSP backtracking + forward checking city layout |
| `challenge2_roads.py` | Kruskal's MST + Hospital↔Depot redundancy |
| `challenge3_ambulance.py` | Simulated Annealing ambulance placement |
| `challenge4_routing.py` | A* dynamic routing with real-time re-planning |
| `challenge5_crime.py` | K-Means + Decision Tree crime risk prediction |
| `simulation.py` | 20-step simulation orchestrator |
| `main.py` | pygame visual interface |

---

## Algorithms Used

| Challenge | Algorithm | From Scratch? |
|-----------|-----------|---------------|
| Layout Planning | Backtracking + Forward Checking (CSP) | ✅ Yes |
| Road Network | Kruskal's MST + Union-Find | ✅ Yes |
| Ambulance Placement | Simulated Annealing | ✅ Yes |
| Emergency Routing | A* Search (Manhattan heuristic) | ✅ Yes |
| Crime Clustering | K-Means (K-Means++ init) | ✅ Yes |
| Crime Classification | Decision Tree (Gini impurity) | ✅ Yes |

---

## System Integration

All five modules share one `CityGraph` object. Changes propagate immediately:
- When Challenge 5 updates node risk levels → edge effective costs update automatically
- When a road is blocked → EmergencyRouter (Challenge 4) detects this via observer and replans A*
- When risk changes → Simulated Annealing re-evaluates ambulance placement
- The shared graph is the **single source of truth** — no module holds a private copy

---

## Simulation Events (20 Steps)

- **Steps 3, 7, 11, 15** – Random road flooding events (edge blocking)
- **Step 8, 14** – Ambulance positions re-evaluated via Simulated Annealing
- **Step 10** – Crime risk re-predicted and graph weights updated
- All steps – Emergency team advances via A*, replanning whenever a flood blocks its path
