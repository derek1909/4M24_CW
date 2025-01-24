import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import *
import os

# Create main results directory if it doesn't exist
main_results_dir = "./results/EF"
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

# Try different values of length-scale l
l_values = [0.01, 2.0, 100]  # Change this list to the desired l values

mean_errors = {}

for l in l_values:
    print(f"Running MCMC with length-scale l = {l}")

    # Create a directory for this l value
    results_dir = os.path.join(main_results_dir, f"l_{l}")
    os.makedirs(results_dir, exist_ok=True)

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

    # Save mean error in a per-length-scale text file
    with open(os.path.join(results_dir, "mean_error.txt"), "w") as f:
        f.write(f"Mean Absolute Error for l={l}: {mean_err:.6f}\n")

    # Plot inferred counts
    plt_obj = plot_2D(c_predict, xi, yi, title=f"Inferred Counts (l={l})")
    plt_obj.xlim(xi.min(), xi.max())
    plt_obj.ylim(yi.min(), yi.max())
    plt_obj.savefig(os.path.join(results_dir, "inferred_counts.png"), bbox_inches='tight')
    plt.close()

    # Plot absolute error field
    plt_obj = plot_2D1(err, xi, yi, title=f"Absolute Error Field (l={l})")
    plt_obj.xlim(xi.min(), xi.max())
    plt_obj.ylim(yi.min(), yi.max())
    plt_obj.savefig(os.path.join(results_dir, "absolute_error_field.png"), bbox_inches='tight')
    plt.close()

    print(f"Results saved in {results_dir}")

# Save mean errors for all length-scales in a summary text file
summary_file = os.path.join(main_results_dir, "mean_errors_summary.txt")
with open(summary_file, "w") as f:
    f.write("Mean Absolute Errors for different length-scales:\n")
    for l, err in mean_errors.items():
        f.write(f"l={l}: {err:.6f}\n")

print(f"Summary of mean errors saved in {summary_file}")

# Plot bike theft count data
plt_obj = plot_2D(data, xi, yi, title='Original Count Data')
plt_obj.xlim(xi.min(), xi.max())
plt_obj.ylim(yi.min(), yi.max())
plt_obj.savefig(os.path.join(main_results_dir, "Bike_Theft_Data.png"), bbox_inches='tight')
plt.close()

# Plot subsampled data
plt_obj = plot_2D(c, xi[idx], yi[idx], title='Subsampled Data')
plt_obj.xlim(xi.min(), xi.max())
plt_obj.ylim(yi.min(), yi.max())
plt_obj.savefig(os.path.join(main_results_dir, "Subsampled_Data.png"), bbox_inches='tight')
plt.close()