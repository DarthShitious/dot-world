import pygame
import config as C
from utils import hsv_to_rgb, sat_from_energy

class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((C.W, C.H + C.HUD_H))
        pygame.display.set_caption("Dot Evolver Rewrite")
        self.font = pygame.font.SysFont("consolas", 14)
        self.clock = pygame.time.Clock()

    def draw(self, world, render_entities: bool, extra: str, fps: float):
        self.screen.fill((16, 16, 16), pygame.Rect(0, 0, C.W, C.HUD_H))
        self.screen.fill((0, 0, 0), pygame.Rect(0, C.HUD_H, C.W, C.H))

        alive = int(world.alive.sum().item())
        food_n = len(world.food_xy)

        total_e = float(world.energy[world.alive].sum().item()) + float(food_n) * C.FOOD_ENERGY
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
                    pygame.draw.circle(self.screen, rgb, (int(x[i]), int(y[i] + C.HUD_H)), C.DOT_RADIUS_PX)

        pygame.display.flip()
