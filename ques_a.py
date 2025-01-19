import numpy as np
import matplotlib.pyplot as plt
from functions import *


###--- Data Generation ---###

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

### Plotting examples
plot_3D(u, x, y)                                      # Plot original u surface
plot_result(u, v, x, y, x[idx], y[idx])               # Plot original u with data v