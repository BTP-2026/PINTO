import numpy as np
import h5py as h5
import argparse
import time
from scalar import ScalarGenerator1D
from solver import PoissonSolver1D

def main():
    parser = argparse.ArgumentParser(description='Generate 1D Poisson equation training data')
    parser.add_argument('-f', '--filePrefix', type=str, default="psn1d", 
                        help="prefix of the data file's name")
    parser.add_argument('-n', '--nSample', type=int, default=100, 
                        help="number of samples")
    parser.add_argument('-s', '--shape', type=int, default=64, 
                        help="number of grid points")
    parser.add_argument('-L', '--length', type=float, default=1.0,
                        help="domain length")
    parser.add_argument('-p', '--periodic', default=False, action='store_true',  
                        help="generates additional samples with periodic variation")
    parser.add_argument('--check-every', type=int, default=100,
                        help="check solution every N samples")
    
    args = parser.parse_args()
    
    # Domain parameters
    nx = args.shape
    L = args.length
    nSample = args.nSample
    
    print(f"Generating {nSample} samples of 1D Poisson equation data")
    print(f"Domain: [0, {L}] with {nx} grid points")
    print(f"Grid spacing: h = {L/(nx-1):.6f}")
    
    # Create HDF5 file
    filename = f"{args.filePrefix}_{nx}_{args.nSample}.h5"
    dFile = h5.File(filename, 'w')
    dFile.attrs['nSample'] = args.nSample
    dFile.attrs['nx'] = nx
    dFile.attrs['length'] = L
    dFile.attrs['equation'] = '1D Poisson: d/dx(a*dp/dx) = f'
    
    # Create datasets
    # Solution to the Poisson equation
    pData = dFile.create_dataset('p', (nSample, nx), compression='gzip',
                compression_opts=9, dtype='float64', chunks=True)
    
    # Coefficient a in the equation
    aData = dFile.create_dataset('a', (nSample, nx), compression='gzip',
                compression_opts=9, dtype='float64', chunks=True)
    # Right-hand-side f in the Poisson equation
    fData = dFile.create_dataset('f', (nSample, nx), compression='gzip',
                compression_opts=9, dtype='float64', chunks=True)
    # Boundary conditions [g0, gL]
    bcData = dFile.create_dataset('bc', (nSample, 2), compression='gzip',
                compression_opts=9, dtype='float64', chunks=True)
    
    # Initialize generators
    print("Initializing scalar generators...")
    scaGen1D = ScalarGenerator1D(size=L, nCell=nx, nKnot=8)
    
    # Generate coefficients, source terms, and boundary conditions
    print("Generating coefficients and source terms...")
    start_time = time.time()
    
    # Coefficient a (positive, bounded away from zero)
    a = scaGen1D.generate_scalar1d(nSample, valMin=0.1, valMax=1.0, strictMin=True)
    
    # Source term f (can be positive or negative)
    f = scaGen1D.generate_scalar1d(nSample, valMin=-10.0, valMax=10.0)
    
    # Boundary conditions (random values for each sample)
    bc = np.random.uniform(-1.0, 1.0, (nSample, 2))
    
    generation_time = time.time() - start_time
    print(f"Generated random fields in {generation_time:.2f} seconds")
    
    # Initialize solver
    print("Initializing Poisson solver...")
    solver = PoissonSolver1D()
    
    # Solve equations
    print("Solving Poisson equations...")
    p = np.zeros((nSample, nx))
    solve_times = []
    
    start_time = time.time()
    
    for i in range(nSample):
        # Solve Poisson equation: d/dx(a*dp/dx) = f
        p[i, :], solve_time = solver.solve(L, a[i, :], f[i, :], bc[i], [a[i,0], a[i,-1]])
        solve_times.append(solve_time)
        
        # Check solution periodically
        if (i + 1) % args.check_every == 0:
            print(f"\nSample {i + 1}/{nSample}")
            print("Poisson equation check:")
            solver.check_solution(L, p[i, :], a[i, :], f[i, :], bc[i], [a[i,0], a[i,-1]])
           
            avg_solve_time = np.mean(solve_times[-args.check_every:])
            print(f"Average solve time: {avg_solve_time:.4e} seconds")
    
    total_solve_time = time.time() - start_time
    print(f"\nSolved all equations in {total_solve_time:.2f} seconds")
    print(f"Average solve time per sample: {np.mean(solve_times):.4e} seconds")
    
    # Save data to HDF5 file
    print("Saving data to HDF5 file...")
    pData[...] = p
    aData[...] = a
    fData[...] = f
    bcData[...] = bc
    
    dFile.close()
    
    print(f"\nData generation complete!")
    print(f"Generated {nSample} samples saved to: {filename}")
    print(f"Dataset shape: {nx} grid points")
    print(f"Coefficient range: [{np.min(a):.3f}, {np.max(a):.3f}]")
    print(f"Source term range: [{np.min(f):.3f}, {np.max(f):.3f}]")
    print(f"Solution range: [{np.min(p):.3f}, {np.max(p):.3f}]")


if __name__ == "__main__":
    main()