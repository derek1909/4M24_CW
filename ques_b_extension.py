import numpy as np
import matplotlib.pyplot as plt
from functions import *
import os
from tqdm import tqdm

###--- Data Generation ---###
np.random.seed(42)

def refine_idx(idx,factor,D_old):
    xi_old = idx % D_old
    yi_old = idx // D_old

    xi_new = factor * xi_old
    yi_new = factor * yi_old

    D_new = factor * D_old
    return yi_new * D_new + xi_new

# Create results directory if it doesn't exist
results_dir = f"./results/B/extension"
os.makedirs(results_dir, exist_ok=True)

### Set base parameters
D_base = 8  # Start with D=8
refine_factors = [1, 2, 3, 4, 6, 8, 16]  # Refinement factors
n = 10000
l = 0.3

N_base = D_base*D_base
points = [(x, y) for y in np.arange(D_base) for x in np.arange(D_base)]
coords = [(x, y) for y in np.linspace(0, 1, D_base) for x in np.linspace(0, 1, D_base)]
xi, yi = np.array([c[0] for c in points]), np.array([c[1] for c in points])
x, y = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])

### Data grid defining {vi}i=1,N/subsample_factor - subsampled from inference grid
subsample_factor = 4
idx_base = subsample(N_base, subsample_factor)
M = len(idx_base)

### Generate covariance K and sample from GP
K_base = GaussianKernel(coords, l)
Kc_base = np.linalg.cholesky(K_base + 1e-6 * np.eye(N_base))

z_base = np.random.randn(N_base)
u_base = Kc_base @ z_base

### Observation model: v = G(u) + e,   e~N(0,I)
G_base = get_G(N_base, idx_base)
v_base = G_base @ u_base + np.random.randn(M)

acc_rates_grw_dict = {}
acc_rates_pcn_dict = {}

for refine_factor in tqdm(refine_factors, desc="Processing different refinement factors"):
    D = D_base * refine_factor
    N = D*D

    # Generate refined mesh
    points = [(x, y) for y in np.arange(D) for x in np.arange(D)]
    coords = [(x, y) for y in np.linspace(0, 1, D) for x in np.linspace(0, 1, D)]
    
    # Refine indices to match new resolution
    idx_refined = refine_idx(idx_base, refine_factor, D_base)
    G = get_G(N, idx_refined)

    ### Generate covariance K and sample from GP
    K = GaussianKernel(coords, l)
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

    # Initial states
    z0 = np.random.randn(N)
    u0 = Kc @ z0

    ### MCMC Sampling with Different β Values ###
    beta_values = np.logspace(-2, 0, num=20)
    acc_rates_grw = []
    acc_rates_pcn = []

    for beta in tqdm(beta_values, desc="Running MCMC for different betas"):        
        sampled_u_grw, acc_grw = grw(log_continuous_target, u0=u0, data=v_base, K=K, G=G, n_iters=n, beta=beta)
        sampled_u_pcn, acc_pcn = pcn(log_continuous_likelihood, u0=u0, data=v_base, K=K, G=G, n_iters=n, beta=beta)
        
        acc_rates_grw.append(acc_grw)
        acc_rates_pcn.append(acc_pcn)

    acc_rates_grw_dict[D] = acc_rates_grw
    acc_rates_pcn_dict[D] = acc_rates_pcn

### Plot GRW-MH Acceptance Rate vs Beta
plt.figure(figsize=(8, 6), dpi=300)
for D, acc_rates in acc_rates_grw_dict.items():
    plt.plot(beta_values, acc_rates, linestyle='-', label=f"D={D}")
plt.xscale("log")
plt.xlabel("Beta")
plt.ylabel("Acceptance Rate")
plt.title("GRW-MH: Acceptance Rate vs Beta for Different D")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, "acceptance_rate_vs_beta_GRW.png"), bbox_inches='tight')

### Plot pCN Acceptance Rate vs Beta
plt.figure(figsize=(5, 3), dpi=300)
for D, acc_rates in acc_rates_pcn_dict.items():
    plt.plot(beta_values, acc_rates, linestyle='-', label=f"D={D}")
plt.xscale("log")
plt.xlabel("Beta")
plt.ylabel("Acceptance Rate")
plt.title("pCN: Acceptance Rate vs Beta for Different D")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, "acceptance_rate_vs_beta_pCN.png"), bbox_inches='tight')
