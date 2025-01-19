import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import norm
import matplotlib.cm as cm
import copy


def GaussianKernel(x, l):
    """ Generate Gaussian kernel matrix efficiently using scipy's distance matrix function"""
    D = distance_matrix(x, x)
    return np.exp(-pow(D, 2)/(2*pow(l, 2)))


def subsample(N, factor, seed=None):
    assert factor>=1, 'Subsampling factor must be greater than or equal to one.'
    N_sub = int(np.ceil(N / factor))
    if seed: np.random.seed(seed)
    idx = np.random.choice(N, size=N_sub, replace=False)  # Indexes of the randomly sampled points
    return idx


def get_G(N, idx):
    """Generate the observation matrix based on datapoint locations.
    Inputs:
        N - Length of vector to subsample
        idx - Indexes of subsampled coordinates
    Outputs:
        G - Observation matrix"""
    M = len(idx)
    G = np.zeros((M, N))
    for i in range(M):
        G[i,idx[i]] = 1
    return G


def probit(v):
    return np.array([0 if x <= 0 else 1 for x in v])


def predict_t(samples):
    return np.mean(norm.cdf(samples), axis=0)


###--- Density functions ---###

def log_prior(u, K_inverse):
    return -0.5 * u.T @ K_inverse @ u


def log_continuous_likelihood(u, v, G):
    residual = v - G @ u
    return -0.5 * residual.T @ residual


def log_probit_likelihood(u, t, G):
    phi = norm.cdf(G @ u)
    return np.sum(t * np.log(phi) + (1 - t) * np.log(1 - phi))


def log_poisson_likelihood(u, c, G):
    return # TODO: Return likelihood p(c|u)


def log_continuous_target(u, y, K_inverse, G):
    return log_prior(u, K_inverse) + log_continuous_likelihood(u, y, G)


def log_probit_target(u, t, K_inverse, G):
    return log_prior(u, K_inverse) + log_probit_likelihood(u, t, G)


def log_poisson_target(u, c, K_inverse, G):
    return log_prior(u, K_inverse) + log_poisson_likelihood(u, c, G)


###--- MCMC ---###

def grw(log_target, u0, data, K, G, n_iters, beta):
    """ Gaussian random walk Metropolis-Hastings MCMC method
        for sampling from pdf defined by log_target.
    Inputs:
        log_target - log-target density
        u0 - initial sample
        data - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    X = []
    acc = 0
    u_prev = u0
    

    # Inverse computed before the for loop for speed
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(len(u0)))
    Kc_inverse = np.linalg.inv(Kc)
    # TODO: compute the inverse of K using its Cholesky decomopsition
    K_inverse = Kc_inverse.T @ Kc_inverse

    lt_prev = log_target(u_prev, data, K_inverse, G)

    for i in range(n_iters):

        # TODO: Propose new sample - use prior covariance, scaled by beta
        u_new = u_prev + beta * (Kc @ np.random.randn(len(u0)))

        lt_new = log_target(u_new, data, K_inverse, G)

        # TODO: Calculate acceptance probability based on lt_prev, lt_new
        log_alpha = lt_new - lt_prev
        log_u = np.log(np.random.random())

        # Accept/Reject
        # TODO: Compare log_alpha and log_u to accept/reject sample (accept should be boolean)
        accept = log_u < log_alpha
        if accept:
            acc += 1
            X.append(u_new)
            u_prev = u_new
            lt_prev = lt_new
        else:
            X.append(u_prev)

    return X, acc / n_iters


def pcn(log_likelihood, u0, data, K, G, n_iters, beta):
    """ pCN MCMC method for sampling from pdf defined by log_prior and log_likelihood.
    Inputs:
        log_likelihood - log-likelihood function
        u0 - initial sample
        data - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    X = []
    acc = 0
    u_prev = u0

    ll_prev = log_likelihood(u_prev, data, G)
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(len(u0)))

    for i in range(n_iters):

        # TODO: Propose new sample using pCN proposal
        xi = Kc @ np.random.randn(len(u0))  # Draw from prior
        u_new = np.sqrt(1 - beta**2) * u_prev + beta * xi

        ll_new = log_likelihood(u_new, data, G)

        # TODO: Calculate pCN acceptance probability
        log_alpha = ll_new - ll_prev
        log_u = np.log(np.random.random())

        # Accept/reject step
        # TODO: Compare log_alpha and log_u to accept/reject sample (accept should be boolean)
        accept = log_u < log_alpha
        if accept:
            acc += 1
            X.append(u_new)
            u_prev = u_new
            ll_prev = ll_new
        else:
            X.append(u_prev)

    return X, acc / n_iters


###--- Plotting ---###
def plot_3D(u, x, y, title=None, zlim=None):
    """Plot the latent variable field u given the list of x,y coordinates with optional z-limit"""
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, u, cmap='viridis', linewidth=0, antialiased=False)

    if zlim:  # Apply z-limit if provided
        ax.set_zlim(zlim[0], zlim[1])

    if title:  plt.title(title)
    return plt


def plot_2D(counts, xi, yi, title=None, colors='viridis'):
    """Visualise count data given the index lists"""
    Z = -np.ones((max(yi) + 1, max(xi) + 1))
    for i in range(len(counts)):
        Z[(yi[i], xi[i])] = counts[i]
    my_cmap = copy.copy(cm.get_cmap(colors))
    my_cmap.set_under('k', alpha=0)
    fig, ax = plt.subplots(figsize=(5,4), dpi=300)
    im = ax.imshow(Z, origin='lower', cmap=my_cmap, clim=[-0.1, np.nanmax(counts)])
    fig.colorbar(im)
    if title:  plt.title(title)
    return plt


def plot_result(u, data, x, y, x_d, y_d, title=None, zlim=None):
    """Plot the latent variable field u with the observations,
        using the latent variable coordinate lists x,y and the
        data coordinate lists x_d, y_d"""
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, u, cmap='viridis', linewidth=0, antialiased=False)
    ax.scatter(x_d, y_d, data, marker='x', color='r')

    if zlim:  # Apply z-limit if provided
        ax.set_zlim(zlim[0], zlim[1])

    if title:  plt.title(title)
    return plt
