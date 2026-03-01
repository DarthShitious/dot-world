import time
import pygame
import torch

import config as C
from world import World
from render import Renderer

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = pick_device()
    world = World(device=device)
    ren = Renderer()

    render_on = True
    paused = False
    running = True

    last = time.perf_counter()
    fps_smooth = 0.0

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    mx, my = ev.pos
                    if mx < C.W and C.HUD_H <= my < (C.HUD_H + C.H):
                        if hasattr(world, "pick_dot"):
                            world.pick_dot(float(mx), float(my - C.HUD_H), C.PICK_RADIUS_PX)
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_b:
                    render_on = not render_on
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_l:
                    world.baldwin_enabled = not world.baldwin_enabled
                elif ev.key == pygame.K_r:
                    world.reset(hard=True)

        if render_on:
            if not paused:
                world.step()
            now = time.perf_counter()
            dt = now - last
            last = now
            fps = (1.0 / dt) if dt > 1e-6 else 0.0
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps
            ren.draw(world, True, ("PAUSED (SPACE)" if paused else "Render=ON (B fast-forward)"), fps_smooth)
            ren.clock.tick(C.FPS_RENDER)
        else:
            if paused:
                now = time.perf_counter()
                dt = now - last
                last = now
                fps = (1.0 / dt) if dt > 1e-6 else 0.0
                fps_smooth = 0.9 * fps_smooth + 0.1 * fps
                ren.draw(world, True, "PAUSED (SPACE)", fps_smooth)
                ren.clock.tick(C.FPS_RENDER)
                continue
            start = time.perf_counter()
            steps = 0
            # run flat-out; only stop to process events and occasionally draw HUD
            while True:
                world.step()
                steps += 1
                if steps % 200 == 0:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            running = False
                        elif ev.type == pygame.MOUSEBUTTONDOWN:
                            if ev.button == 1:
                                mx, my = ev.pos
                                if mx < C.W and C.HUD_H <= my < (C.HUD_H + C.H):
                                    if hasattr(world, "pick_dot"):
                                        world.pick_dot(float(mx), float(my - C.HUD_H), C.PICK_RADIUS_PX)
                        elif ev.type == pygame.KEYDOWN:
                            if ev.key == pygame.K_ESCAPE:
                                running = False
                            elif ev.key == pygame.K_b:
                                render_on = not render_on
                            elif ev.key == pygame.K_SPACE:
                                paused = not paused
                            elif ev.key == pygame.K_l:
                                world.baldwin_enabled = not world.baldwin_enabled
                            elif ev.key == pygame.K_r:
                                world.reset(hard=True)
                    if not running or render_on or paused:
                        break
                if time.perf_counter() - start > 0.08:
                    break

            now = time.perf_counter()
            dt = now - last
            last = now
            fps = (1.0 / dt) if dt > 1e-6 else 0.0
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps
            ren.draw(world, False, f"Render=OFF (fast)  steps~{steps}", fps_smooth)

    pygame.quit()

if __name__ == "__main__":
    main()
