# Group: Lauren Sdun, Julia Jones, Julia Baumgarten

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def mc_area_hit_or_miss(a, b, N, rng=np.random.default_rng()):
    # Uniform in bounding box, our indicator for being inside ellipse
    X = rng.uniform(-a, a, size=N)
    Y = rng.uniform(-b, b, size=N)
    inside = (X / a) ** 2 + (Y / b) ** 2 <= 1.0
    p = inside.mean()
    box_area = (2 * a) * (2 * b)
    est = box_area * p
    # Binomial variance for hit/miss:
    var_p = p * (1 - p) / N
    std = box_area * np.sqrt(var_p)
    return est, std

def mc_area_1d(a, b, N, rng=np.random.default_rng()):
    x = rng.uniform(-a, a, size=N)
    integrand = 2 * b * np.sqrt(np.maximum(0.0, 1.0 - (x / a) ** 2))
    width = 2 * a
    vals = integrand
    est = width * vals.mean()
    std = width * (vals.std(ddof=1) / np.sqrt(N))
    return est, std

def mc_circumference(a, b, N, rng=np.random.default_rng()):
    # C = ∫_0^{2π} sqrt(a^2 sin^2 θ + b^2 cos^2 θ) dθ
    theta = rng.uniform(0.0, 2.0 * np.pi, size=N)
    vals = np.sqrt((a ** 2) * (np.sin(theta) ** 2) + (b ** 2) * (np.cos(theta) ** 2))
    est = (2.0 * np.pi) * vals.mean()
    std = (2.0 * np.pi) * (vals.std(ddof=1) / np.sqrt(N))
    return est, std

def circumference_ramanujan(a, b):
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return np.pi * (a + b) * (1 + 3*h / (10 + np.sqrt(4 - 3*h)))





