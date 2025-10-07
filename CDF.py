# Group: Lauren Sdun, Julia Jones, Julia Baumgarten

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Inverse CDF Part
# Truncated exponential on [0,1] with shape beta > 0:
# Inverse CDF: x = - (1/beta) * ln(1 - u*(1 - exp(-beta)))

# inverse CDF
def sample_trunc_exp_icdf(n, beta, rng=np.random.default_rng()):
    u = rng.uniform(0.0, 1.0, size=n)
    return -(1.0 / beta) * np.log(1.0 - u * (1.0 - np.exp(-beta)))

# random sampling
def sample_trunc_exp_rejection(n, beta, rng=np.random.default_rng()):
    M = beta / (1.0 - np.exp(-beta))
    out = []
    while len(out) < n:
        x = rng.uniform(0.0, 1.0)
        u = rng.uniform(0.0, 1.0)
        # Acceptance probability f(x)/(M g(x)) = e^{-beta x}
        if u <= np.exp(-beta * x):
            out.append(x)
    return np.array(out)

def trunc_exp_pdf(x, beta):
    Z = (1.0 - np.exp(-beta))
    return (beta * np.exp(-beta * x)) / Z * ((x >= 0) & (x <= 1))

def trunc_exp_cdf(x, beta):
    x = np.clip(x, 0.0, 1.0)
    return (1.0 - np.exp(-beta * x)) / (1.0 - np.exp(-beta))

n = 100000
beta = 2.0
rng = np.random.default_rng(123)

samples_icdf = sample_trunc_exp_icdf(n, beta, rng)
samples_rej = sample_trunc_exp_rejection(n, beta, rng)

x = np.linspace(0, 1, 200)

# probability density function
pdf = trunc_exp_pdf(x, beta)

plt.figure(figsize=(10,5))
plt.hist(samples_icdf, bins=50, density=True, alpha=0.5, label="Inverse CDF", color="hotpink")
plt.hist(samples_rej, bins=50, density=True, alpha=0.2, label="Rejection Sampling", color="mediumslateblue")
plt.plot(x, pdf, 'k-', lw=1, label="True PDF")
plt.xlabel("x")
plt.ylabel("Density")
plt.title(f"Truncated Exponential Sampling (beta={beta})")
plt.legend()
plt.show()







# Monte Carlo ellipse
