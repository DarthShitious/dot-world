import math
import random
from dataclasses import dataclass
from typing import List, Tuple
from collections import deque

import numpy as np
import torch

import config as C
from utils import torus_delta
from brain import BrainBatch, init_brains, forward, mutate_brain, copy_brain, baldwin_update_last_layer

@dataclass
class RecallGenome:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W3: np.ndarray
    b3: np.ndarray
    mass: float
    strength: float
    hue: float

class World:
    def __init__(self, device: torch.device):
        self.device = device
        self.ticks = 0
        self.reset_count = 0

        self.cap = C.MAX_POP
        self.N = 0
        self.selected_idx = None

        self.x = torch.zeros((self.cap,), device=device)
        self.y = torch.zeros((self.cap,), device=device)
        self.mass = torch.ones((self.cap,), device=device)
        self.strength = torch.ones((self.cap,), device=device)
        self.hue = torch.zeros((self.cap,), device=device)
        self.energy = torch.zeros((self.cap,), device=device)
        self.emax = torch.zeros((self.cap,), device=device)
        self.alive = torch.zeros((self.cap,), device=device, dtype=torch.bool)
        self.age = torch.zeros((self.cap,), device=device, dtype=torch.int32)

        self.food_xy = np.zeros((0, 2), dtype=np.float32)
        self.food_e = np.zeros((0,), dtype=np.float32)
        self.food_e = np.zeros((0,), dtype=np.float32)
        self.recall: List[RecallGenome] = []

        self.obs_dim = (C.OBS_SAMPLE_N * C.OBS_SAMPLE_N * 2) + 2
        self.h1 = C.HIDDEN1
        self.h2 = C.HIDDEN2
        self.out_dim = 6
        self.brain = init_brains(self.cap, self.obs_dim, self.h1, self.h2, self.out_dim, device)

        self.baldwin_enabled = C.BALDWIN_ENABLED_DEFAULT

        # Selected dot (for inspector)
        self.selected_idx = None

        # --- Metrics (rates averaged over RATE_WINDOW ticks) ---
        self.rate_window = int(getattr(C, "RATE_WINDOW", 16))
        self._asex_hist = deque([0] * self.rate_window, maxlen=self.rate_window)
        self._sex_hist = deque([0] * self.rate_window, maxlen=self.rate_window)
        self._kill_hist = deque([0] * self.rate_window, maxlen=self.rate_window)
        self._asex_this = 0
        self._sex_this = 0
        self._kill_this = 0

        self.reset(hard=True)

    def reset(self, hard: bool = False):
        self.ticks = 0
        self.reset_count += 1
        self.food_xy = np.zeros((0, 2), dtype=np.float32)
        self.food_e = np.zeros((0,), dtype=np.float32)
        if hard:
            self.recall.clear()

        self.alive[:] = False
        self.age[:] = 0
        self.N = 0

        n = C.INIT_POP
        if (not hard) and len(self.recall) > 0:
            n_buf = int((1.0 - C.RESTART_RANDOM_FRAC) * n)
            n_rand = n - n_buf
        else:
            n_buf = 0
            n_rand = n

        if n_rand > 0:
            idx = self._alloc(n_rand)
            self._rand_init(idx)

        if n_buf > 0:
            idx = self._alloc(n_buf)
            self._spawn_from_recall(idx)

    def _alloc(self, k: int) -> torch.Tensor:
        free = (~self.alive).nonzero(as_tuple=False).squeeze(1)
        if free.numel() < k:
            k = int(free.numel())
        idx = free[:k]
        self.alive[idx] = True
        self.age[idx] = 0
        self.N = int(self.alive.sum().item())
        return idx

    def _rand_init(self, idx: torch.Tensor):
        n = idx.numel()
        self.x[idx] = torch.rand((n,), device=self.device) * C.W
        self.y[idx] = torch.rand((n,), device=self.device) * C.H
        self.mass[idx] = C.MASS_MIN + (C.MASS_MAX - C.MASS_MIN) * torch.rand((n,), device=self.device)
        self.strength[idx] = C.STRENGTH_MIN + (C.STRENGTH_MAX - C.STRENGTH_MIN) * torch.rand((n,), device=self.device)
        self.hue[idx] = torch.rand((n,), device=self.device)
        self.emax[idx] = self.mass[idx] * C.ENERGY_PER_MASS

        self.energy[idx] = self.emax[idx]
        tmp = init_brains(n, self.obs_dim, self.h1, self.h2, self.out_dim, self.device)
        copy_brain(tmp, self.brain, torch.arange(n, device=self.device, dtype=torch.long), idx)

    def _spawn_from_recall(self, idx: torch.Tensor):
        n = idx.numel()
        for k in range(n):
            g = random.choice(self.recall)
            i = int(idx[k].item())
            self.x[i] = random.random() * C.W
            self.y[i] = random.random() * C.H
            self.mass[i] = g.mass
            self.strength[i] = g.strength
            self.hue[i] = g.hue
            self.emax[i] = g.mass * C.ENERGY_PER_MASS
            self.energy[i] = self.emax[i]
            self.brain.W1[i] = torch.tensor(g.W1, device=self.device)
            self.brain.b1[i] = torch.tensor(g.b1, device=self.device)
            self.brain.W2[i] = torch.tensor(g.W2, device=self.device)
            self.brain.b2[i] = torch.tensor(g.b2, device=self.device)
            self.brain.W3[i] = torch.tensor(g.W3, device=self.device)
            self.brain.b3[i] = torch.tensor(g.b3, device=self.device)
        mutate_brain(self.brain, idx, C.MUT_STD)

    def _push_recall(self, idx: int):
        g = RecallGenome(
            W1=self.brain.W1[idx].detach().cpu().numpy().copy(),
            b1=self.brain.b1[idx].detach().cpu().numpy().copy(),
            W2=self.brain.W2[idx].detach().cpu().numpy().copy(),
            b2=self.brain.b2[idx].detach().cpu().numpy().copy(),
            W3=self.brain.W3[idx].detach().cpu().numpy().copy(),
            b3=self.brain.b3[idx].detach().cpu().numpy().copy(),
            mass=float(self.mass[idx].item()),
            strength=float(self.strength[idx].item()),
            hue=float(self.hue[idx].item()),
        )
        self.recall.append(g)
        if len(self.recall) > C.RECALL_BUFFER_SIZE:
            self.recall.pop(0)

    def _wrap(self):
        self.x %= C.W
        self.y %= C.H

    def _spawn_food(self, expected: float):
        if len(self.food_xy) >= C.MAX_FOOD:
            return
        k = int(expected)
        frac = expected - k
        if random.random() < frac:
            k += 1
        if k <= 0:
            return
        xs = np.random.rand(k).astype(np.float32) * C.W
        ys = np.random.rand(k).astype(np.float32) * C.H
        self.food_xy = np.concatenate([self.food_xy, np.stack([xs, ys], axis=1)], axis=0)
        self.food_e = np.concatenate([self.food_e, np.full((k,), C.FOOD_ENERGY, dtype=np.float32)], axis=0)

    def _raster_world(self):
        gw, gh = C.OBS_GRID_W, C.OBS_GRID_H
        hue = np.zeros((gh, gw), dtype=np.float32)
        sat = np.zeros((gh, gw), dtype=np.float32)
        food = np.zeros((gh, gw), dtype=np.float32)

        idx = self.alive.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            x = self.x[idx].detach().cpu().numpy()
            y = self.y[idx].detach().cpu().numpy()
            h = self.hue[idx].detach().cpu().numpy()
            e = self.energy[idx].detach().cpu().numpy()
            em = self.emax[idx].detach().cpu().numpy()
            s = np.clip(0.15 + (1.0 - 0.15) * (e / np.maximum(em, 1e-6)), 0.0, 1.0)
            cx = np.clip((x / C.W * gw).astype(np.int32), 0, gw - 1)
            cy = np.clip((y / C.H * gh).astype(np.int32), 0, gh - 1)
            hue[cy, cx] = h
            sat[cy, cx] = s

        if len(self.food_xy) > 0:
            fx = self.food_xy[:, 0]
            fy = self.food_xy[:, 1]
            cx = np.clip((fx / C.W * gw).astype(np.int32), 0, gw - 1)
            cy = np.clip((fy / C.H * gh).astype(np.int32), 0, gh - 1)
            food[cy, cx] = 1.0

        raster = np.stack([hue, sat, food], axis=0)
        return torch.tensor(raster, device=self.device)

    def _build_obs_batch(self, raster: torch.Tensor):
        idx = self.alive.nonzero(as_tuple=False).squeeze(1)
        n = idx.numel()
        if n == 0:
            return torch.zeros((0, self.obs_dim), device=self.device), idx

        gw, gh = C.OBS_GRID_W, C.OBS_GRID_H
        rx = (self.x[idx] / C.W * gw).to(torch.int64) % gw
        ry = (self.y[idx] / C.H * gh).to(torch.int64) % gh

        N = C.OBS_SAMPLE_N
        range_cx = max(1, int((C.OBS_RANGE_PX / C.W) * gw))
        range_cy = max(1, int((C.OBS_RANGE_PX / C.H) * gh))
        ox = torch.linspace(-range_cx, range_cx, N, device=self.device).to(torch.int64)
        oy = torch.linspace(-range_cy, range_cy, N, device=self.device).to(torch.int64)

        gx = (rx[:, None, None] + ox[None, None, :]) % gw
        gy = (ry[:, None, None] + oy[None, :, None]) % gh

        hue = raster[0][gy, gx]
        sat = raster[1][gy, gx]
        food = raster[2][gy, gx]
        fh = torch.full_like(hue, 0.33)
        fs = torch.full_like(sat, 1.0)
        hue = torch.where(food > 0.5, fh, hue)
        sat = torch.where(food > 0.5, fs, sat)

        flat = torch.stack([hue, sat], dim=3).reshape(n, -1)
        efrac = (self.energy[idx] / torch.clamp(self.emax[idx], min=1e-6)).unsqueeze(1)
        bias = torch.ones((n, 1), device=self.device)
        obs = torch.cat([flat, efrac, bias], dim=1)
        return obs, idx

    def _action_from_outputs(self, y: torch.Tensor):
        # y = [dx, dy, speed, move_logit, sex_logit, attack_logit]
        # direction from tanh(dx,dy), throttle from sigmoid(speed)
        dxdy = torch.tanh(y[:, 0:2])
        norm = torch.sqrt(dxdy[:, 0] ** 2 + dxdy[:, 1] ** 2)
        speed = torch.sigmoid(y[:, 2])
        # If direction norm is tiny, do not move (avoid division blow-up)
        eps = 1e-6
        inv = torch.where(norm > eps, 1.0 / norm, torch.zeros_like(norm))
        dirx = dxdy[:, 0] * inv
        diry = dxdy[:, 1] * inv
        move = torch.stack([dirx * speed, diry * speed], dim=1) * C.MAX_MOVE_PX

        logits = y[:, 3:6] / max(1e-6, C.MODE_TEMPERATURE)
        probs = torch.softmax(logits, dim=1)
        mode = torch.multinomial(probs, num_samples=1).squeeze(1)
        return move, mode, logits

    def step(self):
        if int(self.alive.sum().item()) == 0:
            self.reset(hard=False)
            return

        self.ticks += 1

        self._asex_this = 0
        self._sex_this = 0
        self._kill_this = 0
        self.energy[self.alive] -= C.ENERGY_DECAY_PER_TICK
        self._spawn_food(C.FOOD_SPAWN_RATE)

        raster = self._raster_world()
        obs, idx = self._build_obs_batch(raster)
        n = idx.numel()
        if n == 0:
            return

        sub = BrainBatch(
            self.brain.W1[idx], self.brain.b1[idx],
            self.brain.W2[idx], self.brain.b2[idx],
            self.brain.W3[idx], self.brain.b3[idx],
        )
        h2, y = forward(sub, obs)
        move, mode, logits = self._action_from_outputs(y)

        is_move = (mode == 0)
        if is_move.any():
            m_idx = idx[is_move]
            mv = move[is_move]
            self.x[m_idx] += mv[:, 0]
            self.y[m_idx] += mv[:, 1]
            dist2 = (mv[:, 0] ** 2 + mv[:, 1] ** 2)
            self.energy[m_idx] -= C.MOVE_COST * self.mass[m_idx] * dist2

        self._wrap()
        self.age[idx] += 1

        self._resolve_one_interaction(idx, mode)
        if len(self.food_xy) > 0:
            self._eat_pass()

        dead = (self.alive & (self.energy <= 0.0)).nonzero(as_tuple=False).squeeze(1)
        if dead.numel() > 0:
            self.alive[dead] = False
            self.energy[dead] = 0.0

        if self.baldwin_enabled and C.RL_LR > 0:
            with torch.no_grad():
                rew = (self.energy[idx] / torch.clamp(self.emax[idx], min=1e-6)) - 0.5
                rew = torch.clamp(rew, -1.0, 1.0)
            baldwin_update_last_layer(sub, h2, logits, mode, rew, C.RL_LR)
            self.brain.W3[idx] = sub.W3
            self.brain.b3[idx] = sub.b3

        # Update rate histories (once per tick)

        self._asex_hist.append(self._asex_this)

        self._sex_hist.append(self._sex_this)

        self._kill_hist.append(self._kill_this)

        self.N = int(self.alive.sum().item())

    def _resolve_one_interaction(self, idx: torch.Tensor, mode: torch.Tensor):
        alive_idx = idx.detach().cpu().numpy()
        ax = self.x[idx].detach().cpu().numpy()
        ay = self.y[idx].detach().cpu().numpy()
        mode_cpu = mode.detach().cpu().numpy()

        cell = C.INTERACT_RADIUS_PX
        gw = int(C.W / cell) + 1
        gh = int(C.H / cell) + 1
        buckets = [[] for _ in range(gw * gh)]
        for k, _i in enumerate(alive_idx):
            cx = int(ax[k] / cell) % gw
            cy = int(ay[k] / cell) % gh
            buckets[cy * gw + cx].append(k)

        best = None
        r2 = C.INTERACT_RADIUS_PX * C.INTERACT_RADIUS_PX
        for cy in range(gh):
            for cx in range(gw):
                base = buckets[cy * gw + cx]
                if not base:
                    continue
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        nb = buckets[((cy + dy) % gh) * gw + ((cx + dx) % gw)]
                        if not nb:
                            continue
                        for ka in base:
                            xa, ya = ax[ka], ay[ka]
                            for kb in nb:
                                if kb <= ka:
                                    continue
                                xb, yb = ax[kb], ay[kb]
                                ddx = torus_delta(xa, xb, C.W)
                                ddy = torus_delta(ya, yb, C.H)
                                d2 = ddx * ddx + ddy * ddy
                                if d2 <= r2:
                                    if best is None or d2 < best[0]:
                                        best = (d2, ka, kb)

        if best is None:
            return

        _, ka, kb = best
        iA = int(alive_idx[ka]); iB = int(alive_idx[kb])
        aA = int(mode_cpu[ka]); aB = int(mode_cpu[kb])

        if aA == 2 and aB == 2:
            self._sexual_repro(iA, iB)
        elif aA == 1 and aB == 1:
            self._kill(iA, float(self.energy[iA].item()))
            self._kill(iB, float(self.energy[iB].item()))
        elif (aA == 1 and aB == 2) or (aA == 2 and aB == 1):
            attacker = iA if aA == 1 else iB
            mater = iB if aA == 1 else iA
            self.energy[attacker] -= C.ATTACK_PENALTY
            self._kill(mater, float(self.energy[mater].item()))

    def _kill(self, i: int, killed_energy: float):
        if not bool(self.alive[i].item()):
            return
        self._kill_this += 1
        scatter_e = max(0.0, killed_energy) * C.CORPSE_FRACTION
        pellets = int(scatter_e / C.CORPSE_FOOD_ENERGY)
        if pellets > 0:
            ang = np.random.rand(pellets).astype(np.float32) * (2 * np.pi)
            rad = (np.random.rand(pellets).astype(np.float32) ** 0.5) * C.CORPSE_SPREAD_PX
            xs = (float(self.x[i].item()) + np.cos(ang) * rad) % C.W
            ys = (float(self.y[i].item()) + np.sin(ang) * rad) % C.H
            self.food_xy = np.concatenate([self.food_xy, np.stack([xs, ys], axis=1).astype(np.float32)], axis=0)
            self.food_e = np.concatenate([self.food_e, np.full((pellets,), C.CORPSE_FOOD_ENERGY, dtype=np.float32)], axis=0)
        self.alive[i] = False
        self.energy[i] = 0.0

    def _sexual_repro(self, iA: int, iB: int):
        eA = float(self.energy[iA].item()); eB = float(self.energy[iB].item())
        mA = float(self.emax[iA].item()); mB = float(self.emax[iB].item())
        if eA < C.SEX_MIN_FRAC * mA or eB < C.SEX_MIN_FRAC * mB:
            return
        self._push_recall(iA)
        self._push_recall(iB)

        postA = float((C.SEX_POST_FRAC * self.emax[iA]).item())
        postB = float((C.SEX_POST_FRAC * self.emax[iB]).item())
        budget = max(0.0, (eA + eB) - (postA + postB))
        self.energy[iA] = postA
        self.energy[iB] = postB
        self._sex_this += 1

        idx = self._alloc(C.SEX_OFFSPRING)
        if idx.numel() == 0:
            return
        parents = torch.where(torch.rand((idx.numel(),), device=self.device) < 0.5,
                              torch.tensor(iA, device=self.device),
                              torch.tensor(iB, device=self.device)).to(torch.long)
        copy_brain(self.brain, self.brain, parents, idx)
        mutate_brain(self.brain, idx, C.MUT_STD)

        self.mass[idx] = self.mass[parents] * (1.0 + 0.05 * torch.randn_like(self.mass[idx]))
        self.mass[idx] = torch.clamp(self.mass[idx], C.MASS_MIN, C.MASS_MAX)
        self.strength[idx] = self.strength[parents] * (1.0 + 0.05 * torch.randn_like(self.strength[idx]))
        self.strength[idx] = torch.clamp(self.strength[idx], C.STRENGTH_MIN, C.STRENGTH_MAX)
        self.hue[idx] = (self.hue[parents] + 0.02 * torch.randn_like(self.hue[idx])) % 1.0
        self.emax[idx] = self.mass[idx] * C.ENERGY_PER_MASS
        per = budget / float(idx.numel())
        child_e = torch.clamp(torch.full((idx.numel(),), per, device=self.device), max=self.emax[idx])
        self.energy[idx] = child_e

        px = torch.where(parents == iA, self.x[iA], self.x[iB])
        py = torch.where(parents == iA, self.y[iA], self.y[iB])
        jit = torch.randn((idx.numel(), 2), device=self.device) * 6.0
        self.x[idx] = (px + jit[:, 0]) % C.W
        self.y[idx] = (py + jit[:, 1]) % C.H


    def _asexual_repro(self, i: int, energy_budget: float):
        """Option A (fission): parent is removed and replaced by ASEX_OFFSPRING children.
        Energy is conserved-ish by splitting the parent's available energy_budget among children.
        """
        if not bool(self.alive[i].item()):
            return
        self._push_recall(i)

        # Kill parent (no corpse scatter for asexual; it is reproduction, not 'killed' by attack)
        self.alive[i] = False
        self.energy[i] = 0.0

        idx = self._alloc(C.ASEX_OFFSPRING)
        if idx.numel() == 0:
            return

        self._asex_this += 1

        # Copy genetics from parent and mutate
        src_idx = torch.full((idx.numel(),), i, device=self.device, dtype=torch.long)
        copy_brain(self.brain, self.brain, src_idx, idx)
        mutate_brain(self.brain, idx, C.MUT_STD)

        # Physical trait inheritance with small mutation
        self.mass[idx] = self.mass[i] * (1.0 + 0.05 * torch.randn_like(self.mass[idx]))
        self.mass[idx] = torch.clamp(self.mass[idx], C.MASS_MIN, C.MASS_MAX)
        self.strength[idx] = self.strength[i] * (1.0 + 0.05 * torch.randn_like(self.strength[idx]))
        self.strength[idx] = torch.clamp(self.strength[idx], C.STRENGTH_MIN, C.STRENGTH_MAX)
        self.hue[idx] = (self.hue[i] + 0.02 * torch.randn_like(self.hue[idx])) % 1.0

        self.emax[idx] = self.mass[idx] * C.ENERGY_PER_MASS

        # Split energy budget among children (cap by each child's max)
        per = max(0.0, float(energy_budget)) / float(idx.numel())
        child_e = torch.clamp(torch.full((idx.numel(),), per, device=self.device), max=self.emax[idx])
        self.energy[idx] = child_e

        # Spawn near parent position
        jit = torch.randn((idx.numel(), 2), device=self.device) * 6.0
        self.x[idx] = (self.x[i] + jit[:, 0]) % C.W
        self.y[idx] = (self.y[i] + jit[:, 1]) % C.H

    def _eat_pass(self):
        if len(self.food_xy) == 0:
            return
        alive_idx = self.alive.nonzero(as_tuple=False).squeeze(1).detach().cpu().numpy()
        ax = self.x[self.alive].detach().cpu().numpy()
        ay = self.y[self.alive].detach().cpu().numpy()

        r = float(C.DOT_RADIUS_PX + C.FOOD_SIZE_PX)
        cell = max(6.0, r * 2.0)
        gw = int(C.W / cell) + 1
        gh = int(C.H / cell) + 1

        fb = [[] for _ in range(gw * gh)]
        fx = self.food_xy[:, 0]; fy = self.food_xy[:, 1]
        for fi in range(len(self.food_xy)):
            cx = int(fx[fi] / cell) % gw
            cy = int(fy[fi] / cell) % gh
            fb[cy * gw + cx].append(fi)

        eaten = np.zeros((len(self.food_xy),), dtype=np.bool_)
        rr2 = r * r
        for k, i in enumerate(alive_idx):
            cx = int(ax[k] / cell) % gw
            cy = int(ay[k] / cell) % gh
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    bucket = fb[((cy + dy) % gh) * gw + ((cx + dx) % gw)]
                    for fi in bucket:
                        if eaten[fi]:
                            continue
                        ddx = torus_delta(ax[k], fx[fi], C.W)
                        ddy = torus_delta(ay[k], fy[fi], C.H)
                        if ddx * ddx + ddy * ddy <= rr2:
                            eaten[fi] = True
                            new_e = float(self.energy[i].item()) + float(self.food_e[fi])
                            # Apply food first
                            self.energy[i] = new_e
                            if C.ASEX_TRIGGER_OVERFLOW and new_e > float(self.emax[i].item()):
                                # Option A: fission. Parent is removed and replaced by children.
                                self._asexual_repro(int(i), energy_budget=new_e)
        if eaten.any():
            self.food_xy = self.food_xy[~eaten]
            self.food_e = self.food_e[~eaten]



    def pick_dot(self, wx: float, wy: float, radius_px: float):
        """Pick nearest alive dot within radius (world coords, torus). Sets self.selected_idx."""
        idx = self.alive.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            self.selected_idx = None
            return None
        x = self.x[idx]
        y = self.y[idx]
        dx = x - float(wx)
        dy = y - float(wy)
        dx = torch.where(dx > 0.5 * C.W, dx - C.W, dx)
        dx = torch.where(dx < -0.5 * C.W, dx + C.W, dx)
        dy = torch.where(dy > 0.5 * C.H, dy - C.H, dy)
        dy = torch.where(dy < -0.5 * C.H, dy + C.H, dy)
        d2 = dx * dx + dy * dy
        minv, arg = torch.min(d2, dim=0)
        if float(minv.item()) <= float(radius_px) * float(radius_px):
            sel = int(idx[int(arg.item())].item())
            self.selected_idx = sel
            return sel
        self.selected_idx = None
        return None

    def inspect_selected(self):
        """Return dict for inspector for selected dot, or None."""
        if self.selected_idx is None:
            return None
        i = int(self.selected_idx)
        if i < 0 or i >= self.cap or (not bool(self.alive[i].item())):
            self.selected_idx = None
            return None

        raster = self._raster_world()

        idx = torch.tensor([i], device=self.device, dtype=torch.long)
        gw, gh = C.OBS_GRID_W, C.OBS_GRID_H
        rx = (self.x[idx] / C.W * gw).to(torch.int64) % gw
        ry = (self.y[idx] / C.H * gh).to(torch.int64) % gh
        N = C.OBS_SAMPLE_N
        range_cx = max(1, int((C.OBS_RANGE_PX / C.W) * gw))
        range_cy = max(1, int((C.OBS_RANGE_PX / C.H) * gh))
        ox = torch.linspace(-range_cx, range_cx, N, device=self.device).to(torch.int64)
        oy = torch.linspace(-range_cy, range_cy, N, device=self.device).to(torch.int64)
        gx = (rx[:, None, None] + ox[None, None, :]) % gw
        gy = (ry[:, None, None] + oy[None, :, None]) % gh

        hue = raster[0][gy, gx]
        sat = raster[1][gy, gx]
        food = raster[2][gy, gx]
        fh = torch.full_like(hue, 0.33)
        fs = torch.full_like(sat, 1.0)
        hue = torch.where(food > 0.5, fh, hue)
        sat = torch.where(food > 0.5, fs, sat)

        flat = torch.stack([hue, sat], dim=3).reshape(1, -1)
        efrac = (self.energy[idx] / torch.clamp(self.emax[idx], min=1e-6)).unsqueeze(1)
        bias = torch.ones((1, 1), device=self.device)
        obs = torch.cat([flat, efrac, bias], dim=1)

        sub = BrainBatch(
            self.brain.W1[idx], self.brain.b1[idx],
            self.brain.W2[idx], self.brain.b2[idx],
            self.brain.W3[idx], self.brain.b3[idx],
        )
        h2, y = forward(sub, obs)
        logits = y[:, 3:6] / max(1e-6, C.MODE_TEMPERATURE)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        return {
            "idx": i,
            "mass": float(self.mass[i].item()),
            "strength": float(self.strength[i].item()),
            "hue": float(self.hue[i].item()),
            "energy": float(self.energy[i].item()),
            "emax": float(self.emax[i].item()),
            "age": int(self.age[i].item()),
            "probs": probs,
            "y": y.squeeze(0).detach().cpu().numpy(),
            "W1": self.brain.W1[i].detach().cpu().numpy(),
            "b1": self.brain.b1[i].detach().cpu().numpy(),
            "W2": self.brain.W2[i].detach().cpu().numpy(),
            "b2": self.brain.b2[i].detach().cpu().numpy(),
            "W3": self.brain.W3[i].detach().cpu().numpy(),
            "b3": self.brain.b3[i].detach().cpu().numpy(),
        }
    def rate_asex(self) -> float:
        n = len(self._asex_hist) if hasattr(self, "_asex_hist") else 0
        return (sum(self._asex_hist) / float(n)) if n > 0 else 0.0

    def rate_sex(self) -> float:
        n = len(self._sex_hist) if hasattr(self, "_sex_hist") else 0
        return (sum(self._sex_hist) / float(n)) if n > 0 else 0.0

    def rate_kill(self) -> float:
        n = len(self._kill_hist) if hasattr(self, "_kill_hist") else 0
        return (sum(self._kill_hist) / float(n)) if n > 0 else 0.0
