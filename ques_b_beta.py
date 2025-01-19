import numpy as np
import matplotlib.pyplot as plt
from functions import *
import os

###--- Data Generation ---###
np.random.seed(42)

### Set parameters
Dx = 16
Dy = 16
n = 10000
l = 0.3

# Create results directory if it doesn't exist
results_dir = f"./results/B/betas"
os.makedirs(results_dir, exist_ok=True)

N = Dx * Dy     # Total number of coordinates
points = [(x, y) for y in np.arange(Dx) for x in np.arange(Dy)]
coords = [(x, y) for y in np.linspace(0,1,Dy) for x in np.linspace(0,1,Dx)]
xi, yi = np.array([c[0] for c in points]), np.array([c[1] for c in points])
x, y = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])

### Data grid defining {vi}i=1,N/subsample_factor - subsampled from inference grid
subsample_factor = 4
idx = subsample(N, subsample_factor)
M = len(idx) 

### Generate covariance K and sample from GP
K = GaussianKernel(coords, l)
Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

z = np.random.randn(N, )
u = Kc @ z

# initial states
z0 = np.random.randn(N, )
u0 = Kc @ z0

### Observation model: v = G(u) + e,   e~N(0,I)
G = get_G(N, idx)
v = G @ u + np.random.randn(M)

### MCMC Sampling with Different β Values ###
beta_values = np.linspace(0.0, 1.0, 11)  # 11 values from 0.0 to 1.0
beta_values = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])  # 11 values from 0.0 to 1.0
acc_rates_grw = []
acc_rates_pcn = []
mean_errors_grw = []
mean_errors_pcn = []

results_file = os.path.join(results_dir, "mcmc_beta_results.txt")
with open(results_file, "w") as f:
    f.write("MCMC Results for Different Beta Values\n")
    f.write("=====================================\n")
    f.write(" Beta   | Acc Rate (GRW-MH) | Acc Rate (pCN) | Mean Error (GRW-MH) | Mean Error (pCN) \n")
    f.write("------------------------------------------------------------------------\n")

    for beta in beta_values:
        print(f"Running MCMC with beta={beta:.2f}...")

        # Run GRW-MH
        sampled_u_grw, acc_grw = grw(log_continuous_target, u0=u0, data=v, K=K, G=G, n_iters=n, beta=beta)
        mean_u_grw = np.mean(sampled_u_grw, axis=0)  # Compute mean inferred u
        error_grw = np.abs(mean_u_grw - u)  # Compute absolute error
        mean_error_grw = np.mean(error_grw)  # Mean absolute error

        # Run pCN
        sampled_u_pcn, acc_pcn = pcn(log_continuous_likelihood, u0=u0, data=v, K=K, G=G, n_iters=n, beta=beta)
        mean_u_pcn = np.mean(sampled_u_pcn, axis=0)  # Compute mean inferred u
        error_pcn = np.abs(mean_u_pcn - u)  # Compute absolute error
        mean_error_pcn = np.mean(error_pcn)  # Mean absolute error

        acc_rates_grw.append(acc_grw)
        acc_rates_pcn.append(acc_pcn)
        mean_errors_grw.append(mean_error_grw)
        mean_errors_pcn.append(mean_error_pcn)

        # Save results to file
        f.write(f" {beta:.2f}   | {acc_grw:.6f}  | {acc_pcn:.6f}  | {mean_error_grw:.6f}  | {mean_error_pcn:.6f} \n")

print(f"Results saved to {results_file}")

### Plot Acceptance Rate ###
plt.figure(figsize=(6,4), dpi=300)
plt.plot(beta_values, acc_rates_grw, marker='o', label='GRW-MH')
plt.plot(beta_values, acc_rates_pcn, marker='s', label='pCN')
plt.xlabel("Beta")
plt.ylabel("Acceptance Rate")
plt.title("Acceptance Rate vs Beta")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, "acceptance_rate_vs_beta.png"),bbox_inches='tight')

### Plot Mean Absolute Error ###
plt.figure(figsize=(6,4), dpi=300)
plt.plot(beta_values, mean_errors_grw, marker='o', label='GRW-MH')
plt.plot(beta_values, mean_errors_pcn, marker='s', label='pCN')
plt.xlabel("Beta")
plt.ylabel("Mean Absolute Error")
plt.title("Mean Absolute Error vs Beta")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, "error_vs_beta.png"),bbox_inches='tight')