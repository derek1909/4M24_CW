import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import *
import os
from tqdm import tqdm

# Create main results directory if it doesn't exist
main_results_dir = "./results/EF/sweep"
os.makedirs(main_results_dir, exist_ok=True)

# Read in the data
df = pd.read_csv('data.csv')

# Generate arrays from dataframe
data = np.array(df["bicycle.theft"])
xi = np.array(df['xi'])
yi = np.array(df['yi'])
N = len(data)
coords = [(xi[i], yi[i]) for i in range(N)]
x, y = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])  

# Subsample the original dataset
subsample_factor = 3
idx = subsample(N, subsample_factor, seed=42)
G = get_G(N, idx)
c = G @ data
M = len(idx)

# MCMC Parameters
n = 10000
beta = 0.2
log_target = log_poisson_target
log_likelihood = log_poisson_likelihood

# Define grid search over l (logarithmically spaced)
l_values = np.logspace(-2, 2, num=100)  # 30 values between 0.01 and 100

mean_errors = {}

# Add tqdm progress bar
for l in tqdm(l_values, desc="Grid Search Progress"):    # print(f"Running MCMC with length-scale l = {l:.4f}")

    # Generate K, the covariance matrix, and sample from N(0, K)
    K = GaussianKernel(coords, l)
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    
    z0 = np.random.randn(N)
    u0 = Kc @ z0

    # Run MCMC sampling
    sampled_u_pcn, acc_pcn = pcn(log_likelihood, u0=u0, data=c, K=K, G=G, n_iters=n, beta=beta)

    # Compute inferred counts
    c_predict = np.mean(np.exp(sampled_u_pcn), axis=0)
    err = np.abs(c_predict - data)
    mean_err = np.mean(err)
    mean_errors[l] = mean_err

    # print(f"Mean Absolute Error for l={l:.4f}: {mean_err:.6f}")

# Save mean errors for all length-scales in a summary text file
summary_file = os.path.join(main_results_dir, "mean_errors_summary.txt")
with open(summary_file, "w") as f:
    f.write("Mean Absolute Errors for different length-scales:\n")
    for l, err in mean_errors.items():
        f.write(f"l={l:.4f}: {err:.6f}\n")

# Plot Mean Absolute Error vs. l
plt.figure(figsize=(4, 4), dpi=300)
plt.plot(mean_errors.keys(), mean_errors.values(), linestyle='-')
plt.xscale("log")
plt.xlabel("Length-scale (l)")
plt.ylabel("Mean Absolute Error")
# plt.title("Grid Search: Length-scale vs. Mean Absolute Error")
plt.grid(True)
plt.savefig(os.path.join(main_results_dir, "lengthscale_vs_error.png"), bbox_inches='tight')
plt.close()

print("Plot saved as lengthscale_vs_error.png")