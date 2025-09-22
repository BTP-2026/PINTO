import numpy as np
import h5py
from pyDOE import lhs


def read_h5_file(filename):
    """
    Reads 1D Poisson equation dataset (p, a, f, bc) from HDF5 file.
    """
    with h5py.File(filename, 'r') as hf:
        p = hf['p'][:]      # solutions, shape (nSample, nx)
        a = hf['a'][:]      # coefficients, shape (nSample, nx)
        f = hf['f'][:]      # rhs forcing terms, shape (nSample, nx)
        bc = hf['bc'][:]    # boundary conditions [g0, gL], shape (nSample, 2)
        nx = hf.attrs['nx']
        L = hf.attrs['length']

    # Spatial discretization (uniform grid assumed)
    x = np.linspace(0, L, nx)
    return p, a, f, bc, x, L


def get_train_data(data_dir, domain_samples, indices, val_indices):
    """
    Prepare training and validation data for 1D Poisson equation.

    Inputs:
      data_dir       : path to HDF5 file
      domain_samples : number of collocation points in the interior domain
      indices        : list of sample indices for training
      val_indices    : list of sample indices for validation

    Outputs (training):
      xd   : collocation points (x) in the interior domain
      pd   : corresponding solution values (for supervised training)
      ad   : coefficients a(x) at collocation points
      fd   : rhs f(x) at collocation points
      x_bc : boundary points (0 and L)
      p_bc : boundary values
      a_bc : coefficients at boundaries
      f_bc : rhs at boundaries
      x_init, p_init, a_init, f_init, bc_init : full grid data (training samples)

    Outputs (validation):
      x_val, p_val, a_val, f_val, bc_val : same as above for validation
    """
    # --- Read data ---
    p_all, a_all, f_all, bc_all, xdisc, L = read_h5_file(data_dir)

    # --- Training data ---
    p = p_all[indices]   # shape (nTrain, nx)
    a = a_all[indices]
    f = f_all[indices]
    bc = bc_all[indices]

    nTrain, nx = p.shape

    # Collocation points in [0,L] using Latin Hypercube Sampling
    grid_loc = L * lhs(1, domain_samples)  # shape (domain_samples, 1)
    grid_loc = grid_loc.reshape(-1)

    # Repeat collocation points for each training sample
    xd = np.tile(grid_loc, nTrain).reshape(nTrain, -1, 1)   # (nTrain, domain_samples, 1)

    # Interpolate solution, coefficient, and rhs at collocation points
    # (using np.interp since data is on uniform grid)
    pd = np.array([np.interp(grid_loc, xdisc, p[i]) for i in range(nTrain)])  # (nTrain, domain_samples)
    ad = np.array([np.interp(grid_loc, xdisc, a[i]) for i in range(nTrain)])
    fd = np.array([np.interp(grid_loc, xdisc, f[i]) for i in range(nTrain)])

    # Boundary conditions
    x_bc = np.array([[0.0, L]] * nTrain)   # (nTrain, 2)
    p_bc = bc                              # given g0, gL
    a_bc = np.stack([a[:, 0], a[:, -1]], axis=1)  # coefficients at boundaries
    f_bc = np.stack([f[:, 0], f[:, -1]], axis=1)  # rhs at boundaries

    # Full training data (on original grid)
    x_init = np.tile(xdisc, (nTrain, 1))  # (nTrain, nx)
    p_init = p
    a_init = a
    f_init = f
    bc_init = bc

    # --- Validation data ---
    p_val = p_all[val_indices]
    a_val = a_all[val_indices]
    f_val = f_all[val_indices]
    bc_val = bc_all[val_indices]

    nVal = len(val_indices)
    x_val = np.tile(xdisc, (nVal, 1))

    return {
        # Training
        "xd": xd, "pd": pd, "ad": ad, "fd": fd,
        "x_bc": x_bc, "p_bc": p_bc, "a_bc": a_bc, "f_bc": f_bc,
        "x_init": x_init, "p_init": p_init, "a_init": a_init, "f_init": f_init, "bc_init": bc_init,
        # Validation
        "x_val": x_val, "p_val": p_val, "a_val": a_val, "f_val": f_val, "bc_val": bc_val,
    }
