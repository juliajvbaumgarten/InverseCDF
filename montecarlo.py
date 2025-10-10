# Group: Lauren Sdun, Julia Jones, Julia Baumgarten

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def mc_area_hit_or_miss(a, b, N, rng=np.random.default_rng()):
    # Uniform in bounding box, our indicator for being inside ellipse
    X = rng.uniform(-a, a, size=N)
    Y = rng.uniform(-b, b, size=N)
    inside = (X / a) ** 2 + (Y / b) ** 2 <= 1.0 # returns true for ellipse, false otherwise
    p = inside.mean() # probability of hitting the ellipse
    box_area = (2 * a) * (2 * b) # area of boundary recatngle
    est = box_area * p # mc estimate of the ellipse area
    # Binomial variance for hit/miss:
    var_p = p * (1 - p) / N
    std = box_area * np.sqrt(var_p) # 1sigma statistical uncertainty
    return est, std # return estimated area and statistical uncertainty

def mc_area_1d(a, b, N, rng=np.random.default_rng()):
    theta = rng.uniform(0.0, np.pi, size=N)
    x = a * np.cos(theta)
    root = np.sqrt(np.maximum(0.0, 1.0 - (x / a) ** 2))
    g = 2.0 * b * root
    f = (2.0 / (np.pi * a)) * root
    w = g / f
    est = w.mean()
    std = w.std(ddof=1) / np.sqrt(N)
    return est, std # return estimated area and statistical uncertainty

def mc_circumference(a, b, N, rng=np.random.default_rng()):
    # C = ∫_0^{2π} sqrt(a^2 sin^2 θ + b^2 cos^2 θ) dθ
    t = rng.uniform(0.0, 2.0 * np.pi, size=N)
    vals = np.sqrt((a ** 2) * (np.sin(t) ** 2) + (b ** 2) * (np.cos(t) ** 2))
    est = (2.0 * np.pi) * vals.mean()
    std = (2.0 * np.pi) * (vals.std(ddof=1) / np.sqrt(N))
    return est, std # return estimated area and statistical uncertainty

def ramanujan_perimeter(a: float, b: float) -> float:
    """
    Ramanujan's 1st approximation:
        L ≈ π [ 3(a+b) - sqrt{(3a+b)(a+3b)} ]
    """
    return np.pi * (3*(a+b) - np.sqrt((3*a + b)*(a + 3*b)))



def replicate(estimator, a, b, N, n_rep=200, seed=0):
    rng = np.random.default_rng(seed)
    ests = []
    for _ in range(n_rep):
        e, _ = estimator(a, b, N, rng)
        ests.append(e)
    ests = np.asarray(ests)
    return ests.mean(), ests.std(ddof=1)


def scaling_curve(estimator, a, b, N_values, n_rep=200, seed=0):
    means, stds = [], []
    for N in N_values:
        m, s = replicate(estimator, a, b, N, n_rep=n_rep, seed=seed+N)
        means.append(m); stds.append(s)
    return np.asarray(N_values), np.asarray(means), np.asarray(stds)


def demo(a=5.0, b=2.0, show_plots=True):
    rng = np.random.default_rng(42)
    true_area = np.pi * a * b
    ramL = ramanujan_perimeter(a, b)

    est_box, std_box = mc_area_hit_or_miss(a, b, 10_000, rng)
    est_is,  std_is  = mc_area_1d(a, b, 10_000, rng)
    est_c,   std_c   = mc_circumference(a, b, 10_000, rng)

    print(f"True area πab = {true_area:.6f}")
    print(f"Hit-or-miss (box)   : {est_box:.6f} ± {std_box:.6f}")
    print(f"Importance (1D IS)  : {est_is:.6f} ± {std_is:.6e}")
    print(f"Perimeter (MC)      : {est_c:.6f} ± {std_c:.6f}")
    print(f"Perimeter (Ramanujan): {ramL:.6f}")

    # Uncertainty scaling for area (hit/miss)
    N_values = np.unique(np.logspace(2, 5, 12, dtype=int))  # 1e2 .. 1e5
    N_vals, means, stds = scaling_curve(mc_area_hit_or_miss, a, b, N_values, n_rep=200, seed=123)


   # Fit c / sqrt(N)
    coef = (stds * np.sqrt(N_vals)).mean()
    fit = coef / np.sqrt(N_vals)

    plt.figure()
    plt.loglog(N_vals, stds, 'o', label='Empirical std (hit/miss)')
    plt.loglog(N_vals, fit, '-', label='Fit ~ c / sqrt(N)')
    plt.xlabel('N samples'); plt.ylabel('Statistical uncertainty (std of estimates)')
    plt.title(f'Uncertainty scaling for ellipse area (a={a:.1f}, b={b:.1f})')
    plt.legend()
    plt.show()

    # Compare naive vs tailored estimator distributions
    N = 2000
    n_rep = 400
    ests_box = np.array([mc_area_hit_or_miss(a, b, N, rng)[0] for _ in range(n_rep)])
    ests_is  = np.array([mc_area_1d(a, b, N, rng)[0] for _ in range(n_rep)])

    plt.figure()
    plt.hist(ests_box, bins=30, alpha=0.6, label='Hit/miss')
    plt.axvline(true_area, linestyle='--', label='True')
    plt.xlabel('Area estimates'); plt.ylabel('Count')
    plt.title(f'Distribution of area estimates (N={N} each)')
    plt.legend()
    plt.show()

    plt.figure()
    plt.hist(ests_is, bins=30, alpha=0.6, label='Importance (1D)')
    plt.axvline(true_area, linestyle='--', label='True')
    plt.xlabel('Area estimates'); plt.ylabel('Count')
    plt.title(f'Distribution of area estimates (tailored IS, N={N} each)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Run an end-to-end demo with the assignment's a,b
    demo(a=5.0, b=2.0, show_plots=True)


