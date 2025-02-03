import numpy as np
import matplotlib.pyplot as plt
from functions import *
import os
import ipdb

###--- Data Generation ---###
def refine_idx(idx,factor,D_old):
    xi_old = idx % D_old
    yi_old = idx // D_old

    xi_new = factor * xi_old
    yi_new = factor * yi_old

    D_new = factor * D_old
    return yi_new * D_new + xi_new

results_dir = f"./results/A/refine_test"
os.makedirs(results_dir, exist_ok=True)

### Inference grid defining {ui}i=1,Dx*Dy
Dx = 16
Dy = 16
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
l = 0.3
K = GaussianKernel(coords, l)
z = np.random.randn(N, )
Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
u = Kc @ z

### Observation model: v = G(u) + e,   e~N(0,I)
G = get_G(N, idx)
v = G @ u + np.random.randn(M)



plt_obj = plot_2D(u, xi, yi, title='original data')
plt_obj.xlim(xi.min(), xi.max())  # 直接对 plt_obj 设置 xlim
plt_obj.ylim(yi.min(), yi.max())  # 直接对 plt_obj 设置 ylim
plt_obj.savefig(os.path.join(results_dir, "original_u.png"), bbox_inches='tight')



# Plot subsampled data
plt_obj = plot_2D(v, xi[idx], yi[idx], title='Subsampled Data')
plt_obj.xlim(0,16)
plt_obj.ylim(0,16)
plt_obj.savefig(os.path.join(results_dir, "Subsampled_Data.png"), bbox_inches='tight')

# ipdb.set_trace()

# to refine mesh:
D1 = 2*Dx
N1 = D1**2

idx1 = refine_idx(idx,2,Dx)

G = get_G(N1, idx1)
points = [(x, y) for y in np.arange(D1) for x in np.arange(D1)]
coords = [(x, y) for y in np.linspace(0,1,D1) for x in np.linspace(0,1,D1)]
xi, yi = np.array([c[0] for c in points]), np.array([c[1] for c in points])
x, y = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])


# Plot subsampled data
plt_obj = plot_2D(v, xi[idx1], yi[idx1], title='Subsampled Data (refined)')
plt_obj.xlim(0,32)
plt_obj.ylim(0,32)
plt_obj.savefig(os.path.join(results_dir, "Subsampled_Data_refined.png"), bbox_inches='tight')


# ### Plotting examples
# plot_3D(u, x, y)                                      # Plot original u surface
# plot_result(u, v, x, y, x[idx], y[idx])               # Plot original u with data v