import pygame
import config as C
import numpy as np
from utils import hsv_to_rgb, sat_from_energy, weight_to_rgb, scalar_to_gray

class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((C.W + C.INSPECT_W, C.H + C.HUD_H))
        pygame.display.set_caption("Dot Evolver Rewrite")
        self.font = pygame.font.SysFont("consolas", 10)
        self.clock = pygame.time.Clock()
        self._inspect_frame = 0

    def draw(self, world, render_entities: bool, extra: str, fps: float):
        self.screen.fill((16, 16, 16), pygame.Rect(0, 0, C.W, C.HUD_H))
        self.screen.fill((0, 0, 0), pygame.Rect(0, C.HUD_H, C.W, C.H))

        alive = int(world.alive.sum().item())
        food_n = len(world.food_xy)

        total_e = float(world.energy[world.alive].sum().item()) + float(world.food_e.sum() if hasattr(world, 'food_e') else (food_n * C.FOOD_ENERGY))
        txt1 = f"ticks={world.ticks:8d}  alive={alive:5d}  food={food_n:5d}  E_total={total_e:10.1f}  resets={world.reset_count:5d}  Baldwin={'ON' if world.baldwin_enabled else 'OFF'}"
        txt2 = f"AsexRate={world.rate_asex():5.2f}/tick  SexRate={world.rate_sex():5.2f}/tick  KillRate={world.rate_kill():5.2f}/tick   {extra}   fps={fps:5.1f}"
        self.screen.blit(self.font.render(txt1, True, (235, 235, 235)), (10, 8))
        self.screen.blit(self.font.render(txt2, True, (210, 210, 210)), (10, 30))

        if render_entities:
            for fx, fy in world.food_xy:
                pygame.draw.rect(
                    self.screen,
                    (0, 255, 0),
                    pygame.Rect(int(fx - C.FOOD_SIZE_PX / 2), int(fy - C.FOOD_SIZE_PX / 2 + C.HUD_H), C.FOOD_SIZE_PX, C.FOOD_SIZE_PX),
                )
            idx = world.alive.nonzero(as_tuple=False).squeeze(1)
            sel = getattr(world, 'selected_idx', None)
            if idx.numel() > 0:
                x = world.x[idx].detach().cpu().numpy()
                y = world.y[idx].detach().cpu().numpy()
                hue = world.hue[idx].detach().cpu().numpy()
                e = world.energy[idx].detach().cpu().numpy()
                em = world.emax[idx].detach().cpu().numpy()
                step = 1
                if len(x) > 8000:
                    step = 2
                for i in range(0, len(x), step):
                    s = sat_from_energy(float(e[i]), float(em[i]))
                    rgb = hsv_to_rgb(float(hue[i]), s, 1.0)
                    px = int(x[i]); py = int(y[i] + C.HUD_H)
                    pygame.draw.circle(self.screen, rgb, (px, py), C.DOT_RADIUS_PX)
                    if sel is not None and int(idx[i].item()) == int(sel):
                        pygame.draw.circle(self.screen, (240, 240, 240), (px, py), C.DOT_RADIUS_PX + 2, 1)

        self._draw_inspector(world)
        pygame.display.flip()


    def _draw_heatmap(self, surf: pygame.Surface, arr: np.ndarray, rect: pygame.Rect, vmax: float = None):
        if arr.ndim != 2:
            return
        h, w = arr.shape
        if h <= 0 or w <= 0:
            return
        if vmax is None:
            vmax = float(np.max(np.abs(arr))) if arr.size else 1.0
        cell_w = max(1, rect.w // w)
        cell_h = max(1, rect.h // h)
        draw_w = cell_w * w
        draw_h = cell_h * h
        ox = rect.x + (rect.w - draw_w) // 2
        oy = rect.y + (rect.h - draw_h) // 2
        for iy in range(h):
            y0 = oy + iy * cell_h
            for ix in range(w):
                x0 = ox + ix * cell_w
                c = weight_to_rgb(float(arr[iy, ix]), vmax)
                pygame.draw.rect(surf, c, pygame.Rect(x0, y0, cell_w, cell_h))

    def _draw_strip(self, surf: pygame.Surface, vec: np.ndarray, rect: pygame.Rect, kind: str = "bias"):
        n = int(vec.shape[0])
        if n <= 0:
            return
        cell_w = max(1, rect.w // n)
        draw_w = cell_w * n
        ox = rect.x + (rect.w - draw_w) // 2
        vmax = float(np.max(np.abs(vec))) if (kind == "bias" and vec.size) else 1.0
        for i in range(n):
            x0 = ox + i * cell_w
            if kind == "bias":
                c = weight_to_rgb(float(vec[i]), vmax)
            else:
                c = scalar_to_gray(float(vec[i]), 0.0, 1.0)
            pygame.draw.rect(surf, c, pygame.Rect(x0, rect.y, cell_w, rect.h))

    def _draw_inspector(self, world):
        px = C.W
        panel = pygame.Rect(px, 0, C.INSPECT_W, C.H + C.HUD_H)
        self.screen.fill((12, 12, 12), panel)

        pad = C.INSPECT_PAD
        x0 = px + pad
        y0 = pad
        w = C.INSPECT_W - 2 * pad

        self.screen.blit(self.font.render("Inspector (click dot)", True, (230, 230, 230)), (x0, y0))
        y0 += 20

        # refresh cap
        self._inspect_frame += 1
        refresh_div = int(getattr(C, "INSPECT_REFRESH_DIV", 1))
        do_refresh = (refresh_div <= 1) or (self._inspect_frame % refresh_div == 0)

        info = None
        if hasattr(world, "inspect_selected"):
            if do_refresh or not hasattr(self, "_inspect_cached"):
                self._inspect_cached = world.inspect_selected()
            info = self._inspect_cached

        if info is None:
            return

        # Dot swatch
        sw = 42
        cx = x0 + w // 2
        cy = y0 + sw // 2 + 4
        rgb = hsv_to_rgb(float(info["hue"]), 1.0, 1.0)
        pygame.draw.circle(self.screen, rgb, (cx, cy), sw // 2)
        pygame.draw.circle(self.screen, (240, 240, 240), (cx, cy), sw // 2, 1)
        y0 += sw + 8

        # Text stats
        lines = [
            f"id={info['idx']}  age={info['age']} ticks",
            f"mass={info['mass']:.2f}  str={info['strength']:.2f}",
            f"hue={info['hue']:.3f}",
            f"E={info['energy']:.1f}/{info['emax']:.1f}",
        ]
        for ln in lines:
            self.screen.blit(self.font.render(ln, True, (210, 210, 210)), (x0, y0))
            y0 += 16
        y0 += 8

        mats = [
            ("W1", info["W1"], "mat"),
            ("b1", info["b1"], "bias"),
            ("W2", info["W2"], "mat"),
            ("b2", info["b2"], "bias"),
            ("W3", info["W3"], "mat"),
            ("b3", info["b3"], "bias"),
            ("y", info["y"], "bias"),
            ("out", info["probs"], "probs"),
        ]

        gap = 8
        bias_h_native = 10
        probs_h_native = 12

        native_items = []
        total_native = 0.0
        for name, arr, kind in mats:
            if kind == "mat":
                h, ww = int(arr.shape[0]), int(arr.shape[1])
                h_native = max(18.0, w * (h / max(1, ww)))
            elif kind == "bias":
                h_native = float(bias_h_native)
            else:
                h_native = float(probs_h_native)
            native_items.append((name, arr, kind, h_native))
            total_native += (14 + h_native + gap)  # label height

        avail = (C.H + C.HUD_H) - y0 - pad
        scale = min(1.0, avail / total_native) if total_native > 1e-6 else 1.0

        for name, arr, kind, h_native in native_items:
            self.screen.blit(self.font.render(name, True, (190, 190, 190)), (x0, y0))
            y0 += int(14 * scale)
            hh = max(6, int(h_native * scale))
            rect = pygame.Rect(x0, y0, w, hh)
            pygame.draw.rect(self.screen, (24, 24, 24), rect)
            pygame.draw.rect(self.screen, (60, 60, 60), rect, 1)
            if kind == "mat":
                self._draw_heatmap(self.screen, arr, rect)
            elif kind == "bias":
                self._draw_strip(self.screen, arr, rect, kind="bias")
            else:
                self._draw_strip(self.screen, arr, rect, kind="probs")
            y0 += hh + int(gap * scale)
