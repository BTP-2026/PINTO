import numpy as np
import h5py
import fcntl


def read_h5_file(filename):
    """
    Read Poisson equation data from HDF5 file.
    
    Returns:
        p_data: Solution data (N_samples, N_points)
        a_data: Coefficient data (N_samples, N_points) 
        f_data: Forcing term data (N_samples, N_points)
        bc_data: Boundary condition data (N_samples, 2)
        x: Spatial discretization
        L: Domain length
    """
    with open(filename, 'r') as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        with h5py.File(filename, 'r') as hf:
            p_data = hf['p'][:]  # Solutions
            a_data = hf['a'][:]  # Coefficients
            f_data = hf['f'][:]  # Forcing terms
            bc_data = hf['bc'][:]  # Boundary conditions
            x = hf['x-coordinate'][:]  # Spatial points
            L = hf.attrs.get('L', 1.0)  # Domain length
        fcntl.flock(f, fcntl.LOCK_UN)
    return p_data, a_data, f_data, bc_data, x, L


def get_train_data_poisson(data_dir, context_indices, train_indices, test_indices):
    """
    Prepare training and validation data for Poisson equation PINTO model.
    
    Args:
        data_dir: Path to HDF5 data file
        context_indices: Indices for context points
        train_indices: Training sample indices
        test_indices: Validation sample indices
    
    Returns:
        Training and validation data arrays
    """
    p_data, a_data, f_data, bc_data, xdisc, L = read_h5_file(data_dir)
    
    # Training data
    p_train = p_data[train_indices]  # (N_train, N_points)
    a_train = a_data[train_indices]
    f_train = f_data[train_indices]
    bc_train = bc_data[train_indices]
    
    # Create domain points for training
    N_train = len(train_indices)
    N_points = len(xdisc)
    
    # Expand spatial coordinates for all training samples
    x_train = np.tile(xdisc.reshape(1, -1, 1), (N_train, 1, 1))  # (N_train, N_points, 1)
    
    # Context data for sensors/boundary conditions
    x_context = xdisc[context_indices]  # Context spatial points
    a_context_train = a_train[:, context_indices]  # (N_train, N_context)
    f_context_train = f_train[:, context_indices]
    p_context_train = p_train[:, context_indices] if len(context_indices) > 0 else None
    
    # Boundary data
    x_bc_train = np.array([xdisc[0], xdisc[-1]])  # Boundary points [0, L]
    bc_values_train = bc_train  # Boundary values
    
    # Validation data
    p_val = p_data[test_indices]
    a_val = a_data[test_indices]
    f_val = f_data[test_indices]
    bc_val = bc_data[test_indices]
    
    N_val = len(test_indices)
    x_val = np.tile(xdisc.reshape(1, -1, 1), (N_val, 1, 1))
    
    # Context data for validation
    a_context_val = a_val[:, context_indices]
    f_context_val = f_val[:, context_indices]
    p_context_val = p_val[:, context_indices] if len(context_indices) > 0 else None
    
    # Boundary data for validation
    bc_values_val = bc_val
    
    return {
        'train': {
            'x': x_train.reshape((-1, 1)),  # Flatten for training
            'p': p_train.reshape((-1, 1)),
            'a': a_train.reshape((-1, 1)),
            'f': f_train.reshape((-1, 1)),
            'x_context': x_context,
            'a_context': a_context_train,
            'f_context': f_context_train,
            'p_context': p_context_train,
            'x_bc': x_bc_train,
            'bc_values': bc_values_train
        },
        'val': {
            'x': x_val.reshape((-1, 1)),
            'p': p_val.reshape((-1, 1)),
            'a': a_val.reshape((-1, 1)),
            'f': f_val.reshape((-1, 1)),
            'x_context': x_context,
            'a_context': a_context_val,
            'f_context': f_context_val,
            'p_context': p_context_val,
            'x_bc': x_bc_train,
            'bc_values': bc_values_val
        },
        'metadata': {
            'xdisc': xdisc,
            'L': L,
            'context_indices': context_indices
        }
    }


def prepare_prediction_data(p_data, a_data, f_data, xdisc, context_indices, sample_idx):
    """
    Prepare data for a single sample prediction.
    
    Args:
        p_data: All solution data
        a_data: All coefficient data  
        f_data: All forcing data
        xdisc: Spatial discretization
        context_indices: Context point indices
        sample_idx: Sample index to prepare
    
    Returns:
        Dictionary with prepared data for model prediction
    """
    # Get data for this sample
    p_sample = p_data[sample_idx]
    a_sample = a_data[sample_idx]
    f_sample = f_data[sample_idx]
    
    # Spatial points
    x_points = xdisc.reshape(-1, 1)
    
    # Context data
    a_context = a_sample[context_indices].reshape(1, -1)
    f_context = f_sample[context_indices].reshape(1, -1)
    p_context = p_sample[context_indices].reshape(1, -1)
    
    # Repeat context for all spatial points
    N_points = len(xdisc)
    a_context_repeated = np.tile(a_context, (N_points, 1))
    f_context_repeated = np.tile(f_context, (N_points, 1))
    p_context_repeated = np.tile(p_context, (N_points, 1))
    
    return {
        'x': x_points,
        'a': a_sample.reshape(-1, 1),
        'f': f_sample.reshape(-1, 1),
        'a_context': a_context_repeated,
        'f_context': f_context_repeated,
        'p_context': p_context_repeated,
        'true_solution': p_sample
    }


def compute_poisson_residual(p_pred, a_coeff, f_force, xdisc):
    """
    Compute PDE residual for Poisson equation: -d/dx(a(x) * dp/dx) - f(x) = 0
    
    Args:
        p_pred: Predicted solution
        a_coeff: Coefficient function a(x)
        f_force: Forcing term f(x)
        xdisc: Spatial discretization
    
    Returns:
        residual: PDE residual
    """
    dx = xdisc[1] - xdisc[0]  # Assuming uniform grid
    
    # Compute dp/dx using finite differences
    dp_dx = np.gradient(p_pred, dx)
    
    # Compute d/dx(a * dp/dx)
    a_dp_dx = a_coeff * dp_dx
    d_a_dp_dx = np.gradient(a_dp_dx, dx)
    
    # Compute residual: -d/dx(a * dp/dx) - f = 0
    residual = -d_a_dp_dx - f_force
    
    return residual


def compute_error_metrics(p_pred, p_true):
    """
    Compute various error metrics between predicted and true solutions.
    
    Args:
        p_pred: Predicted solution
        p_true: True solution
    
    Returns:
        Dictionary of error metrics
    """
    # Absolute error
    abs_error = np.abs(p_pred - p_true)
    
    # Relative error (L2 norm)
    rel_error_l2 = np.sqrt(np.mean((p_pred - p_true)**2)) / np.sqrt(np.mean(p_true**2))
    
    # Relative error (pointwise)
    rel_error_pointwise = abs_error / (1 + np.abs(p_true))
    
    # Maximum error
    max_error = np.max(abs_error)
    
    # Mean absolute error
    mae = np.mean(abs_error)
    
    # Root mean square error
    rmse = np.sqrt(np.mean((p_pred - p_true)**2))
    
    return {
        'abs_error': abs_error,
        'rel_error_l2': rel_error_l2,
        'rel_error_pointwise': rel_error_pointwise,
        'max_error': max_error,
        'mae': mae,
        'rmse': rmse
    }


def load_training_history(history_dir):
    """
    Load training history from CSV files.
    
    Args:
        history_dir: Directory containing history.csv and val_history.csv
    
    Returns:
        Dictionary with training and validation history
    """
    import pandas as pd
    import os
    
    try:
        train_history = pd.read_csv(os.path.join(history_dir, 'history.csv'))
        val_history = pd.read_csv(os.path.join(history_dir, 'val_history.csv'))
        
        return {
            'train': train_history,
            'val': val_history,
            'success': True
        }
    except FileNotFoundError as e:
        return {
            'train': None,
            'val': None,
            'success': False,
            'error': str(e)
        }
