import numpy as np
import matplotlib.pyplot as plt
from functions import *
import os


###--- Data Generation ---###
np.random.seed(42)

### Set parameters
Dx = 4
Dy = 4
n = 10000
beta = 0.2
l = 0.3

# Create results directory if it doesn't exist
results_dir = f"./results/B/D={Dx}"
os.makedirs(results_dir, exist_ok=True)


N = Dx * Dy     # Total number of coordinates
points = [(x, y) for y in np.arange(Dx) for x in np.arange(Dy)]                # Indexes for the inference grid
coords = [(x, y) for y in np.linspace(0,1,Dy) for x in np.linspace(0,1,Dx)]    # Coordinates for the inference grid
xi, yi = np.array([c[0] for c in points]), np.array([c[1] for c in points])    # Get x, y index lists
x, y = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])      # Get x, y coordinate lists

### Data grid defining {vi}i=1,N/subsample_factor - subsampled from inference grid
subsample_factor = 4
idx = subsample(N, subsample_factor)
M = len(idx)                                                                   # Total number of data points

### Generate K, the covariance of the Gaussian process, and sample from N(0,K) using a stable Cholesky decomposition
K = GaussianKernel(coords, l)
Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

z = np.random.randn(N, )
u = Kc @ z

z0 = np.random.randn(N, )
u0 = Kc @ z0


### Observation model: v = G(u) + e,   e~N(0,I)
G = get_G(N, idx)
v = G @ u + np.random.randn(M)


###--- MCMC ---####



### Set the likelihood and target, for sampling p(u|v)
log_target = log_continuous_target
log_likelihood = log_continuous_likelihood

### Sample from prior for MCMC initialisation

# Run GRW-MH
sampled_u_grw, acc_grw = grw(log_continuous_target, u0=u0, data=v, K=K, G=G, n_iters=10000, beta=0.2)
# Run pCN
sampled_u_pcn, acc_pcn = pcn(log_continuous_likelihood, u0=u0, data=v, K=K, G=G, n_iters=10000, beta=0.2)

# Compute mean inferred u
mean_u_grw = np.mean(sampled_u_grw, axis=0)
mean_u_pcn = np.mean(sampled_u_pcn, axis=0)

# Compute absolute error fields
error_grw = np.abs(mean_u_grw - u)
error_pcn = np.abs(mean_u_pcn - u)

zlim_u = (-3,3)
zlim_error = (0,1.5)
plt_obj = plot_3D(u, x, y, zlim=zlim_u)
plt_obj.savefig(os.path.join(results_dir, "original_u_surface.png"))

plt_obj = plot_result(u, v, x, y, x[idx], y[idx], title="Original Surface w/ data", zlim=zlim_u)
plt_obj.savefig(os.path.join(results_dir, "original_u_with_data.png"))

plt_obj = plot_3D(mean_u_grw, x, y, title="Mean Inferred u (GRW-MH)", zlim=zlim_u)
plt_obj.savefig(os.path.join(results_dir, "mean_inferred_u_grw.png"))

plt_obj = plot_3D(mean_u_pcn, x, y, title="Mean Inferred u (pCN)", zlim=zlim_u)
plt_obj.savefig(os.path.join(results_dir, "mean_inferred_u_pcn.png"))

plt_obj = plot_3D(error_grw, x, y, title="Error Field (GRW-MH)", zlim=zlim_error)
plt_obj.savefig(os.path.join(results_dir, "error_field_grw.png"))

plt_obj = plot_3D(error_pcn, x, y, title="Error Field (pCN)", zlim=zlim_error)
plt_obj.savefig(os.path.join(results_dir, "error_field_pcn.png"))


# Compute mean absolute error
mean_error_grw = np.mean(error_grw)
mean_error_pcn = np.mean(error_pcn)

# Save results to a text file
results_file = os.path.join(results_dir, "mcmc_results.txt")
with open(results_file, "w") as f:
    f.write("MCMC Results\n")
    f.write("===========================\n")
    f.write(f"Mean Absolute Error (GRW-MH): {mean_error_grw:.6f}\n")
    f.write(f"Mean Absolute Error (pCN): {mean_error_pcn:.6f}\n")
    f.write(f"Acceptance Rate (GRW-MH): {acc_grw:.6f}\n")
    f.write(f"Acceptance Rate (pCN): {acc_pcn:.6f}\n")

print(f"Results saved to {results_file}")