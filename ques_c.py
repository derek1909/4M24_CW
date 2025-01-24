import numpy as np
import matplotlib.pyplot as plt
from functions import *
import os

###--- Data Generation ---###
np.random.seed(3429)

### Set parameters
Dx = 16
Dy = 16
n = 10000
beta = 0.2
l = 0.2

# Create results directory if it doesn't exist
results_dir = f"./results/C/D={Dx}"
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

# MCMC initial u
z0 = np.random.randn(N, )
u0 = Kc @ z0

### Observation model: v = G(u) + e,   e~N(0,I)
G = get_G(N, idx)
v = G @ u + np.random.randn(M)
t_data = probit(v)       # Probit transform of data
t_true = probit(u)       # Probit transform of true u

###--- MCMC Sampling ---###
sampled_u_pcn, acc_pcn = pcn(log_probit_likelihood, u0=u0, data=t_data, K=K, G=G, n_iters=10000, beta=0.2)

###--- Monte Carlo Estimation of Predictive Distribution ---###
predicted_t_prob = predict_t(sampled_u_pcn)
predicted_t = (predicted_t_prob >= 0.5).astype(int)

###--- Visualization ---###

t_data_full = np.full(Dx*Dy, np.nan)  # Create Dx x Dy grid filled with NaNs
for i, index in enumerate(idx):
    t_data_full[index] = t_data[i]

plot_2D(t_true, xi, yi, title="True Class Assignments").savefig(os.path.join(results_dir, "true_classes.png"),bbox_inches='tight')
plot_2D(t_data_full, xi, yi, title="Observed Class Assignment").savefig(os.path.join(results_dir, "subsampled_data.png"),bbox_inches='tight')
plot_2D(predicted_t_prob, xi, yi, title="Predictive Probability Distribution").savefig(os.path.join(results_dir, "predictive_probs.png"),bbox_inches='tight')
plot_2D(predicted_t, xi, yi, title="Predicted Class Assignments").savefig(os.path.join(results_dir, "thresholded_predictions.png"),bbox_inches='tight')

mean_prediction_error = np.mean(predicted_t != t_true)

# Save the result
results_file = os.path.join(results_dir, "prediction_error.txt")
with open(results_file, "w") as f:
    f.write(f"Mean Prediction Error: {mean_prediction_error:.6f}\n")

print(f"Mean Prediction Error: {mean_prediction_error:.6f}")