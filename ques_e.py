import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import *
import os

###--- Import spatial data ---###

# Create results directory if it doesn't exist
results_dir = f"./results/E"
os.makedirs(results_dir, exist_ok=True)

### Read in the data
df = pd.read_csv('data.csv')

### Generate the arrays needed from the dataframe
data = np.array(df["bicycle.theft"])
xi = np.array(df['xi'])
yi = np.array(df['yi'])
N = len(data)
coords = [(xi[i],yi[i]) for i in range(N)]
x, y = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])      # Get x, y coordinate lists

### Subsample the original data set
subsample_factor = 3
idx = subsample(N, subsample_factor, seed=42)
G = get_G(N,idx)
c = G @ data
M = len(idx)                                                                   # Total number of data points


###--- MCMC ---####

### Set MCMC parameters
l = 2
n = 10000
beta = 0.2

### Set the likelihood and target, for sampling p(u|c)
log_target = log_poisson_target
log_likelihood = log_poisson_likelihood


# TODO: Complete Spatial Data questions (e), (f).
### Generate K, the covariance of the Gaussian process, and sample from N(0,K) using a stable Cholesky decomposition
K = GaussianKernel(coords, l)
Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

z = np.random.randn(N, )
u = Kc @ z

# MCMC initial u
z0 = np.random.randn(N, )
u0 = Kc @ z0

###--- MCMC Sampling ---###
sampled_u_pcn, acc_pcn = pcn(log_likelihood, u0=u0, data=c, K=K, G=G, n_iters=n, beta=beta)

# Plot bike theft count data
plt_obj = plot_2D(data, xi, yi, title='Bike Theft Data')
plt_obj.xlim(xi.min(), xi.max())  # 直接对 plt_obj 设置 xlim
plt_obj.ylim(yi.min(), yi.max())  # 直接对 plt_obj 设置 ylim
plt_obj.savefig(os.path.join(results_dir, "Bike_Theft_Data.png"), bbox_inches='tight')

# Plot subsampled data
plt_obj = plot_2D(c, xi[idx], yi[idx], title='Subsampled Data')
plt_obj.xlim(xi.min(), xi.max())
plt_obj.ylim(yi.min(), yi.max())
plt_obj.savefig(os.path.join(results_dir, "Subsampled_Data.png"), bbox_inches='tight')

# Plot mean inferred theft
mean_theta = np.exp(np.mean(sampled_u_pcn, axis=0))
plt_obj = plot_2D(mean_theta, xi, yi, title="Mean Inferred Theta")
plt_obj.xlim(xi.min(), xi.max())
plt_obj.ylim(yi.min(), yi.max())
plt_obj.savefig(os.path.join(results_dir, "mean_inferred_theta.png"), bbox_inches='tight')