import numpy as np
import matplotlib.pyplot as plt
from functions import *
import os

###--- Data Generation ---###
np.random.seed(2781)

### Set parameters
Dx = 16
Dy = 16
n = 20000
beta = 0.2
num_experiments = 3  # Number of full experiments
subsample_factor = 4

# Create results directory if it doesn't exist
results_dir = "./results/D/multiple_l-true"
os.makedirs(results_dir, exist_ok=True)

N = Dx * Dy  # Total number of coordinates
coords = [(x, y) for y in np.linspace(0,1,Dy) for x in np.linspace(0,1,Dx)]  # Coordinates for the inference grid

# Define different true length-scale values
# l_true_values = [0.1, 0.3, 0.5]
l_true_values = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7]
l_values = np.logspace(-2, 1, num=50)  # 50 values between 0.01 and 10 (log scale)

# Dictionary to store averaged errors for each true l
error_results = {l_true: np.zeros(len(l_values)) for l_true in l_true_values}

for exp in range(num_experiments):
    print(f"\n========== Running Full Experiment {exp+1}/{num_experiments} ==========\n")

    for l_true in l_true_values:
        print(f"\n--- Running experiments for true length-scale l = {l_true} (Experiment {exp+1}) ---\n")

        errors = []  # Store average errors for each tested length-scale

        # Generate the true covariance matrix
        K_true = GaussianKernel(coords, l_true)
        Kc_true = np.linalg.cholesky(K_true + 1e-6 * np.eye(N))

        # Generate the true latent field
        z_true = np.random.randn(N, )
        u_true = Kc_true @ z_true

        # Subsample for observed data
        idx = subsample(N, subsample_factor)
        M = len(idx)  # Total number of data points
        G = get_G(N, idx)

        # Generate observations
        v = G @ u_true + np.random.randn(M)
        t_data = probit(v)
        t_true = probit(u_true)

        # Grid search over different length-scale values
        for l in l_values:
            print(f"Testing length-scale l = {l:.3f} (Experiment {exp+1})...")
            
            # Generate covariance matrix with new length-scale
            K = GaussianKernel(coords, l)
            Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

            z0 = np.random.randn(N, )
            u0 = Kc @ z0  # Initialize u0

            # MCMC Sampling
            sampled_u_pcn, acc_pcn = pcn(log_probit_likelihood, u0=u0, data=t_data, K=K, G=G, n_iters=n, beta=beta)
            
            # Compute prediction error
            predicted_t_prob = predict_t(sampled_u_pcn)
            predicted_t = (predicted_t_prob >= 0.5).astype(int)
            mean_prediction_error = np.mean(predicted_t != t_true)

            # Store the error
            errors.append(mean_prediction_error)

            print(f"Avg error for l = {l:.3f} (Experiment {exp+1}): {mean_prediction_error:.6f}")

        # Accumulate errors across experiments
        error_results[l_true] += np.array(errors)

# Average errors over all experiments
for l_true in l_true_values:
    error_results[l_true] /= num_experiments  # Compute mean error over all experiments

# Save all results to a text file
results_file = os.path.join(results_dir, "lengthscale_optimization_multiple_experiments.txt")
with open(results_file, "w") as f:
    for l_true, errors in error_results.items():
        f.write(f"True l = {l_true}\n")
        for l, err in zip(l_values, errors):
            f.write(f"{l:.6f} {err:.6f}\n")
        f.write("\n")

# Generate separate plots for each l_true
for l_true, errors in error_results.items():
    inferred_l = l_values[np.argmin(errors)]  # Find the inferred length-scale

    plt.figure(figsize=(4,3), dpi=300)
    
    # Plot mean prediction error vs length-scale
    plt.plot(l_values, errors, linestyle='-')
    
    # Mark the true length-scale
    plt.axvline(l_true, color='red', linestyle='--', label=f"True l = {l_true}")
    
    # Mark the inferred length-scale
    plt.axvline(inferred_l, color='blue', linestyle='--', label=f"Inferred l = {inferred_l:.3f}")
    
    plt.xscale("log")  # Log scale for better visualization
    plt.xlabel("Tested Length-Scale l")
    plt.ylabel("Mean Prediction Error")
    # plt.title(f"Grid Search for l = {l_true} (Inferred l = {inferred_l:.3f})")
    plt.legend()
    plt.grid(True)
    
    # Save the figure separately
    save_path = os.path.join(results_dir, f"lengthscale_vs_error_ltrue_{l_true}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    print(f"Saved plot for l_true = {l_true} at {save_path}")