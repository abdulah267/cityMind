"""
CityMind – Visual Interface
pygame-based real-time city grid display with:
  - Road network overlay
  - Ambulance coverage overlay
  - Crime heatmap overlay
  - Toggle buttons for each overlay
  - Live event log panel
  - Step controls: Run / Pause / Reset / Step
"""

import sys
import random
import pygame
import math
from simulation import CityMindSimulation

# ── Colour palette ──────────────────────────────
BG_COLOR        = (15,  20,  35)
PANEL_COLOR     = (25,  32,  50)
PANEL_BORDER    = (50,  70, 110)
TEXT_COLOR      = (210, 220, 240)
TEXT_DIM        = (120, 130, 155)
WHITE           = (255, 255, 255)

NODE_COLORS = {
    "Residential":    (70,  130, 200),
    "Hospital":       (220,  60,  60),
    "School":         (220, 180,  40),
    "Industrial":     (100, 100, 110),
    "PowerPlant":     (160,  70, 200),
    "AmbulanceDepot": (60,  200, 100),
    "Empty":          (45,   50,  65),
}

RISK_HEATMAP = {
    "Low":    (30,  160,  80, 80),
    "Medium": (220, 140,  30, 110),
    "High":   (220,  40,  40, 140),
}

ROAD_COLOR        = (255, 255, 255)   # white — easily visible on dark bg
ROAD_BLOCKED      = (255,  50,  50)   # bright red for blocked roads
AMBULANCE_COLOR   = (60,  220, 140)
ROUTER_PATH_COLOR = (255, 220,  50)
ROUTER_POS_COLOR  = (255, 255, 100)
POLICE_COLOR      = (80,  160, 255)   # bright blue for police officers

BUTTON_NORMAL  = (40,  55,  90)
BUTTON_HOVER   = (60,  80, 130)
BUTTON_ACTIVE  = (30, 100, 200)
BUTTON_TEXT    = (210, 225, 255)

# ── Layout constants ────────────────────────────
WINDOW_W     = 1280
WINDOW_H     = 760
GRID_MARGIN  = 20
GRID_LEFT    = 20
GRID_TOP     = 90
GRID_W       = 760
GRID_H       = 560
PANEL_X      = GRID_LEFT + GRID_W + 20
PANEL_W      = WINDOW_W - PANEL_X - 10
LOG_H        = 130
LOG_Y        = WINDOW_H - LOG_H - 10

NODE_RADIUS  = 14
FONT_SMALL   = 13
FONT_MED     = 15
FONT_LARGE   = 18
FONT_TITLE   = 22


def lerp_color(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


class Button:
    def __init__(self, rect, label, toggle=False, active=False):
        self.rect   = pygame.Rect(rect)
        self.label  = label
        self.toggle = toggle
        self.active = active
        self._hover = False

    def draw(self, surf, font):
        self._hover = self.rect.collidepoint(pygame.mouse.get_pos())
        if self.active and self.toggle:
            color = BUTTON_ACTIVE
        elif self._hover:
            color = BUTTON_HOVER
        else:
            color = BUTTON_NORMAL
        pygame.draw.rect(surf, color, self.rect, border_radius=6)
        pygame.draw.rect(surf, PANEL_BORDER, self.rect, 1, border_radius=6)
        txt = font.render(self.label, True, BUTTON_TEXT)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def handle_click(self, pos) -> bool:
        if self.rect.collidepoint(pos):
            if self.toggle:
                self.active = not self.active
            return True
        return False


class CityMindUI:
    def __init__(self, max_steps=20, seed=None):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))

        self.font_s  = pygame.font.SysFont("Arial", FONT_SMALL)
        self.font_m  = pygame.font.SysFont("Arial", FONT_MED)
        self.font_l  = pygame.font.SysFont("Arial", FONT_LARGE)
        self.font_t  = pygame.font.SysFont("Arial", FONT_TITLE, bold=True)

        self.max_steps = max(20, max_steps)
        self.seed      = seed if seed is not None else random.randint(0, 99999)
        pygame.display.set_caption(f"CityMind – Urban Intelligence System  |  Seed: {self.seed}")
        self.sim     = CityMindSimulation(rows=7, cols=7, seed=self.seed, max_steps=self.max_steps)
        self.running = False
        self.done    = False
        self.clock   = pygame.time.Clock()
        self.step_timer = 0
        self.step_interval = 800   # ms between auto-steps

        # Overlays
        self.show_roads     = Button((PANEL_X,      150, PANEL_W, 32), "Road Network",      toggle=True, active=True)
        self.show_coverage  = Button((PANEL_X,      190, PANEL_W, 32), "Ambulance Coverage",toggle=True, active=False)
        self.show_heatmap   = Button((PANEL_X,      230, PANEL_W, 32), "Crime Heatmap",     toggle=True, active=False)
        self.show_router    = Button((PANEL_X,      270, PANEL_W, 32), "Router Path",       toggle=True, active=True)
        self.show_police    = Button((PANEL_X,      310, PANEL_W, 32), "Police Officers",   toggle=True, active=True)
        self.toggles = [self.show_roads, self.show_coverage, self.show_heatmap,
                        self.show_router, self.show_police]

        # Controls
        self.btn_run   = Button((GRID_LEFT,       50, 100, 32), "Run")
        self.btn_pause = Button((GRID_LEFT + 110, 50, 100, 32), "Pause")
        self.btn_step  = Button((GRID_LEFT + 220, 50, 100, 32), "Step")
        self.btn_reset = Button((GRID_LEFT + 330, 50, 100, 32), "Reset")
        self.controls  = [self.btn_run, self.btn_pause, self.btn_step, self.btn_reset]

        self.sim.initialize()
        self._update_snap()

    def _update_snap(self):
        self.snap = self.sim.current_snapshot()

    def _grid_to_screen(self, x, y):
        """Map grid (x,y) to pixel (px, py)."""
        cols = self.sim.cols
        rows = self.sim.rows
        cell_w = GRID_W / (cols + 1)
        cell_h = GRID_H / (rows + 1)
        px = GRID_LEFT + (x + 1) * cell_w
        py = GRID_TOP  + (y + 1) * cell_h
        return int(px), int(py)

    # ── drawing ─────────────────────────────────
    def _draw_background(self):
        self.screen.fill(BG_COLOR)
        # Grid area border
        pygame.draw.rect(self.screen, PANEL_COLOR,
                         (GRID_LEFT - 5, GRID_TOP - 5, GRID_W + 10, GRID_H + 10),
                         border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER,
                         (GRID_LEFT - 5, GRID_TOP - 5, GRID_W + 10, GRID_H + 10),
                         2, border_radius=8)
        # Title
        title = self.font_t.render("CityMind  –  Urban Intelligence System", True, WHITE)
        self.screen.blit(title, (GRID_LEFT, 10))

        # Seed display
        seed_txt = self.font_s.render(f"Seed: {self.seed}", True, TEXT_DIM)
        self.screen.blit(seed_txt, (GRID_LEFT + 480, 8))

        # Step counter
        step_txt = self.font_l.render(f"Step: {self.snap.get('step', 0)} / {self.max_steps}", True, TEXT_DIM)
        self.screen.blit(step_txt, (GRID_LEFT + 480, 24))

    def _draw_heatmap(self):
        nodes = self.snap.get("nodes", {})
        surf = pygame.Surface((GRID_W, GRID_H), pygame.SRCALPHA)
        for nid_str, nd in nodes.items():
            x, y = nd["x"], nd["y"]
            risk  = nd.get("risk", "Low")
            color = RISK_HEATMAP[risk]
            px, py = self._grid_to_screen(x, y)
            pygame.draw.circle(surf, color, (px - GRID_LEFT, py - GRID_TOP), NODE_RADIUS + 14)
        self.screen.blit(surf, (GRID_LEFT, GRID_TOP))

    def _draw_ambulance_coverage(self):
        ambulances = self.snap.get("ambulances", [])
        nodes = self.snap.get("nodes", {})
        if not ambulances:
            return
        surf = pygame.Surface((GRID_W, GRID_H), pygame.SRCALPHA)
        for amb_id in ambulances:
            nd = nodes.get(str(amb_id)) or nodes.get(amb_id)
            if nd is None:
                continue
            px, py = self._grid_to_screen(nd["x"], nd["y"])
            # Gradient circle for coverage
            for r in range(80, 0, -10):
                alpha = int(50 * (1 - r / 80))
                pygame.draw.circle(surf, (*AMBULANCE_COLOR, alpha),
                                   (px - GRID_LEFT, py - GRID_TOP), r)
        self.screen.blit(surf, (GRID_LEFT, GRID_TOP))

    def _draw_police_officers(self):
        """Render police officer markers on nodes where officers are deployed."""
        police = self.snap.get("police_deployment", {})
        nodes  = self.snap.get("nodes", {})
        if not police:
            return
        for nid, count in police.items():
            nd = nodes.get(str(nid)) or nodes.get(nid)
            if nd is None:
                continue
            px, py = self._grid_to_screen(nd["x"], nd["y"])
            # Outer glow ring — scales slightly with officer count
            radius = NODE_RADIUS + 6 + min(count, 3)
            pygame.draw.circle(self.screen, POLICE_COLOR, (px, py), radius, 2)
            # "P" label above node
            p_lbl = self.font_s.render("P", True, POLICE_COLOR)
            self.screen.blit(p_lbl, (px + NODE_RADIUS + 1, py - NODE_RADIUS - 2))
            # Officer count badge
            badge = self.font_s.render(str(count), True, WHITE)
            self.screen.blit(badge, (px + NODE_RADIUS + 1, py + 2))

    def _draw_router_path(self):
        router = self.snap.get("router", {})
        path = router.get("path", [])
        nodes = self.snap.get("nodes", {})
        if len(path) < 2:
            return
        for i in range(len(path) - 1):
            a_nd = nodes.get(str(path[i])) or nodes.get(path[i])
            b_nd = nodes.get(str(path[i + 1])) or nodes.get(path[i + 1])
            if a_nd and b_nd:
                pa = self._grid_to_screen(a_nd["x"], a_nd["y"])
                pb = self._grid_to_screen(b_nd["x"], b_nd["y"])
                pygame.draw.line(self.screen, ROUTER_PATH_COLOR, pa, pb, 4)

        # Current position highlight
        cur = router.get("current_node")
        if cur is not None:
            nd = nodes.get(str(cur)) or nodes.get(cur)
            if nd:
                px, py = self._grid_to_screen(nd["x"], nd["y"])
                pygame.draw.circle(self.screen, ROUTER_POS_COLOR, (px, py), NODE_RADIUS + 4, 3)

    def _draw_edges(self):
        edges = self.snap.get("edges", [])
        nodes = self.snap.get("nodes", {})
        for e in edges:
            u_nd = nodes.get(str(e["u"])) or nodes.get(e["u"])
            v_nd = nodes.get(str(e["v"])) or nodes.get(e["v"])
            if u_nd and v_nd:
                pu = self._grid_to_screen(u_nd["x"], u_nd["y"])
                pv = self._grid_to_screen(v_nd["x"], v_nd["y"])
                color = ROAD_BLOCKED if e["blocked"] else ROAD_COLOR
                width = 3 if e["blocked"] else 2   # thicker for blocked, 2px for normal
                pygame.draw.line(self.screen, color, pu, pv, width)

    def _draw_nodes(self):
        nodes = self.snap.get("nodes", {})
        ambulances = set(self.snap.get("ambulances", []))
        router = self.snap.get("router", {})
        cur_node = router.get("current_node")

        for nid_str, nd in nodes.items():
            nid = int(nid_str) if isinstance(nid_str, str) else nid_str
            x, y = nd["x"], nd["y"]
            px, py = self._grid_to_screen(x, y)
            loc_type = nd["type"]
            base_color = NODE_COLORS.get(loc_type, (150, 150, 150))

            # Draw node
            pygame.draw.circle(self.screen, base_color, (px, py), NODE_RADIUS)
            pygame.draw.circle(self.screen, WHITE, (px, py), NODE_RADIUS, 1)

            # Ambulance marker
            if nid in ambulances:
                pygame.draw.circle(self.screen, AMBULANCE_COLOR, (px, py), NODE_RADIUS + 5, 2)
                amb_label = self.font_s.render("A", True, AMBULANCE_COLOR)
                self.screen.blit(amb_label, (px - 4, py - 22))

            # Type abbreviation
            abbrev = {"Residential": "R", "Hospital": "H", "School": "S",
                      "Industrial": "I", "PowerPlant": "P", "AmbulanceDepot": "D"}.get(loc_type, "?")
            lbl = self.font_s.render(abbrev, True, WHITE)
            self.screen.blit(lbl, lbl.get_rect(center=(px, py)))

    def _draw_right_panel(self):
        panel_bottom = WINDOW_H - LOG_H - 15
        # Panel background
        pygame.draw.rect(self.screen, PANEL_COLOR,
                         (PANEL_X - 8, 0, PANEL_W + 18, panel_bottom), border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER,
                         (PANEL_X - 8, 0, PANEL_W + 18, panel_bottom), 1, border_radius=8)

        # -- Toggle buttons header and buttons --
        hdr = self.font_l.render("Toggle Views", True, WHITE)
        self.screen.blit(hdr, (PANEL_X, 120))
        for btn in self.toggles:
            btn.draw(self.screen, self.font_m)

        # Toggles end at y=310+32=342; legend starts at 360
        legend_y = 360
        MAX_Y = panel_bottom - 8   # hard lower bound; nothing renders below this

        def draw_line(text, color=TEXT_COLOR, indent=0, dot_color=None):
            nonlocal legend_y
            if legend_y + 16 > MAX_Y:
                return False
            if dot_color:
                pygame.draw.circle(self.screen, dot_color,
                                   (PANEL_X + indent + 8, legend_y + 7), 7)
                txt = self.font_s.render(text, True, color)
                self.screen.blit(txt, (PANEL_X + indent + 20, legend_y))
            else:
                txt = self.font_s.render(text, True, color)
                self.screen.blit(txt, (PANEL_X + indent, legend_y))
            legend_y += 18
            return True

        def draw_header(text, color=WHITE, font=None):
            nonlocal legend_y
            if legend_y + 20 > MAX_Y:
                return False
            f = font or self.font_l
            h = f.render(text, True, color)
            self.screen.blit(h, (PANEL_X, legend_y))
            legend_y += 22
            return True

        # ── Legend ──
        if not draw_header("Legend"):
            return
        for loc_type, color in NODE_COLORS.items():
            if not draw_line(loc_type, TEXT_COLOR, dot_color=color):
                return

        legend_y += 4

        # ── Ambulances ──
        if not draw_header("Ambulances"):
            return
        ambulances = self.snap.get("ambulances", [])
        nodes = self.snap.get("nodes", {})
        for i, amb_id in enumerate(ambulances):
            nd = nodes.get(str(amb_id)) or nodes.get(amb_id)
            coord = f"({nd['x']},{nd['y']})" if nd else ""
            if not draw_line(f"Amb {i+1}: Node {amb_id} {coord}", AMBULANCE_COLOR):
                return

        legend_y += 4

        # ── Police summary (totals only — per-node list too long) ──
        police = self.snap.get("police_deployment", {})
        total_pol = sum(police.values()) if police else 0
        if not draw_header(f"Police: {total_pol}/10 deployed", POLICE_COLOR):
            return
        high_count   = sum(c for nid, c in police.items()
                           if (nodes.get(str(nid)) or nodes.get(nid) or {}).get("risk") == "High")
        medium_count = sum(c for nid, c in police.items()
                           if (nodes.get(str(nid)) or nodes.get(nid) or {}).get("risk") == "Medium")
        low_count    = total_pol - high_count - medium_count
        draw_line(f"High-risk zones:   {high_count} officers", (220, 80, 80))
        draw_line(f"Medium-risk zones: {medium_count} officers", (220, 160, 40))
        draw_line(f"Low-risk zones:    {low_count} officers",   (60, 180, 80))

        legend_y += 4

        # ── Router status ──
        router = self.snap.get("router", {})
        if not draw_header("Router"):
            return
        target = router.get("target", None)
        target_str = str(target) if target is not None else "None"
        draw_line(f"At node: {router.get('current_node', '?')}")
        draw_line(f"Target:  {target_str}")
        draw_line(f"Civilians left: {router.get('civilians_left', 0)}")
        draw_line(f"Complete: {'Yes' if router.get('complete') else 'No'}")

    def _draw_log(self):
        log_rect = pygame.Rect(GRID_LEFT - 5, LOG_Y, WINDOW_W - GRID_LEFT + 5 - 10, LOG_H)
        pygame.draw.rect(self.screen, PANEL_COLOR, log_rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, log_rect, 1, border_radius=6)

        hdr = self.font_m.render("Event Log", True, WHITE)
        self.screen.blit(hdr, (GRID_LEFT, LOG_Y + 6))

        log_lines = self.snap.get("log", [])
        y = LOG_Y + 26
        for line in log_lines[-5:]:
            txt = self.font_s.render(line[:120], True, TEXT_DIM)
            self.screen.blit(txt, (GRID_LEFT, y))
            y += 18

    def _draw_controls(self):
        for btn in self.controls:
            btn.draw(self.screen, self.font_m)
        status = "Running" if self.running and not self.done else ("Done" if self.done else "Paused")
        color  = (60, 200, 60) if self.running else (200, 60, 60)
        lbl = self.font_m.render(status, True, color)
        self.screen.blit(lbl, (GRID_LEFT + 460, 58))

    # ── event handling ──────────────────────────
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                for btn in self.toggles:
                    btn.handle_click(pos)
                if self.btn_run.handle_click(pos):
                    self.running = True
                if self.btn_pause.handle_click(pos):
                    self.running = False
                if self.btn_step.handle_click(pos) and not self.done:
                    result = self.sim.step()
                    self._update_snap()
                    if result.get("done"):
                        self.done = True
                        self.running = False
                if self.btn_reset.handle_click(pos):
                    self.seed = random.randint(0, 99999)
                    pygame.display.set_caption(f"CityMind – Urban Intelligence System  |  Seed: {self.seed}")
                    self.sim = CityMindSimulation(rows=7, cols=7, seed=self.seed, max_steps=self.max_steps)
                    self.sim.initialize()
                    self._update_snap()
                    self.running = False
                    self.done = False
        return True

    # ── main loop ───────────────────────────────
    def run(self):
        while True:
            dt = self.clock.tick(60)

            if not self.handle_events():
                break

            # Auto-step
            if self.running and not self.done:
                self.step_timer += dt
                if self.step_timer >= self.step_interval:
                    self.step_timer = 0
                    result = self.sim.step()
                    self._update_snap()
                    if result.get("done"):
                        self.done = True
                        self.running = False

            # Draw
            self._draw_background()

            if self.show_heatmap.active:
                self._draw_heatmap()
            if self.show_coverage.active:
                self._draw_ambulance_coverage()
            if self.show_roads.active:
                self._draw_edges()
            if self.show_router.active:
                self._draw_router_path()
            if self.show_police.active:
                self._draw_police_officers()

            self._draw_nodes()
            self._draw_right_panel()
            self._draw_log()
            self._draw_controls()

            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    # Ask user for number of simulation steps
    try:
        user_input = input("Enter number of simulation steps (minimum 20, default 20): ").strip()
        if user_input == "":
            steps = 20
        else:
            steps = int(user_input)
            if steps < 20:
                print("Minimum is 20 steps. Using 20.")
                steps = 20
    except ValueError:
        print("Invalid input. Using default 20 steps.")
        steps = 20

    seed = random.randint(0, 99999)
    print(f"Starting CityMind simulation with {steps} steps... (Seed: {seed})")
    ui = CityMindUI(max_steps=steps, seed=seed)
    ui.run()