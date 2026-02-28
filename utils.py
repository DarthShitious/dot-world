import numpy as np

def torus_delta(a: float, b: float, period: float) -> float:
    d = b - a
    if d > 0.5 * period:
        d -= period
    elif d < -0.5 * period:
        d += period
    return d

def hsv_to_rgb(h: float, s: float, v: float = 1.0):
    if s <= 0.0:
        c = int(v * 255)
        return (c, c, c)
    h = (h % 1.0) * 6.0
    i = int(h)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return (int(r * 255), int(g * 255), int(b * 255))

def sat_from_energy(e: float, emax: float, min_sat: float = 0.15) -> float:
    if emax <= 1e-9:
        return min_sat
    frac = max(0.0, min(1.0, e / emax))
    return min_sat + (1.0 - min_sat) * frac
