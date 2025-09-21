import numpy as np
import scipy.sparse as sparse
import time


class PoissonSolver1D:
    """
    Solve the 1D variable coefficient Poisson equation:
    d/dx(a(x) * dp/dx) = f(x), x ∈ [0, L]
    with Dirichlet boundary conditions: p(0) = g0, p(L) = gL
    """
    
    def __init__(self, backend=0):
        self.tolerance = 1e-10
        self.backend = backend
    
    def _build_matrix(self, a, coefBc=None):
       
        nx = len(a)
        nNonZero = 3 * nx - 2
        row = np.zeros(nNonZero, dtype=int)
        col = np.zeros(nNonZero, dtype=int)
        val = np.zeros(nNonZero)
        ival = 0

        for j in range(nx):
            if coefBc is not None:
                ajp = _interp_face(a[j] , a[j+1]) if j < nx -1 else coefBc[1]
                ajm = _interp_face(a[j-1], a[j])  if j > 0 else coefBc[0]

            diagVal = ajm + ajp

            # left neighbor
            if j > 0:
                row[ival] = j
                col[ival] = j - 1
                val[ival] = -ajm
                ival += 1
            else:
                diagVal += ajm  # Adjust diagonal for drichlet BC
            
            # right neighbor
            if j < nx - 1:
                row[ival] = j
                col[ival] = j + 1
                val[ival] = -ajp
                ival += 1
            else:
                diagVal += ajp  # Adjust diagonal for drichlet BC
            
            # self / diagnal
            row[ival] = j
            col[ival] = j
            val[ival] = diagVal
            ival += 1
        
        assert ival == nNonZero
        mat = sparse.csr_matrix((val, (row, col)), shape=(nx,nx))

        return mat

    
    def _build_rhs(self, h, bc, rhs, coefBc=None):
        nx = len(rhs)
        assert len(bc) == 2

        b = -h*h * rhs.copy()

        if coefBc is not None:
            b[0]   += 2 * coefBc[0] * bc[0]      # left boundary
            b[-1]  += 2 * coefBc[1] * bc[1]      # right boundary
        
        
        return b
    
    def solve(self, L, a, rhs, bc, coefBc=None):
        """
        Solve the 1D Poisson equation.
        
        Args:
            L: domain length
            a: coefficient array at cell centers
            f: source term array at cell centers
            g0: left boundary condition
            gL: right boundary condition
            
        Returns:
            (solution, solve_time): solution array and time taken
        """
        nx = len(a)
        h = L / nx 
        
        # Build linear system
        A = self._build_matrix(a, coefBc)
        b = self._build_rhs(h, bc, rhs, coefBc)
        
        # Solve system
        start_time = time.perf_counter()
        
        if self.backend == 0:  # scipy direct solver
            solution = sparse.linalg.spsolve(A, b)
        else:
            raise NotImplementedError(f"Backend {self.backend} not implemented")
        
        end_time = time.perf_counter()
        solve_time = end_time - start_time
        
        return solution, solve_time
    
    def compute_rhs_1d(self,L, p, bc, a, coefBc=None):
        N = len(p)
        h = L / N
        invhh = 1.0 / (h * h)
        f = np.zeros_like(p)

        for i in range(N):
            # face coefficients (interpolated)
            if coefBc is not None:
                aip = _interp_face(a[i] + a[i+1]) if i < N-1 else coefBc[1]
                aim = _interp_face(a[i-1] + a[i]) if i > 0   else coefBc[0]

            # neighbor p values (ghost cell if boundary)
            pip = p[i+1] if i < N-1 else 2*bc[1] - p[i]
            pim = p[i-1] if i > 0    else 2*bc[0] - p[i]

            # flux difference
            f[i] = aip*(pip - p[i]) - aim*(p[i] - pim)
            f[i] *= invhh

        return f

    def check_solution(self,L, p, a, b, bc=None, coefBc=None, exact=None):
        """
        Check residual and error for 1D Poisson solution.
        
        L      : domain length
        p      : solution vector (size N)
        a      : coefficient vector (size N)
        b      : RHS vector (size N)
        bc     : [left_value, right_value] boundary conditions
        coefBc : optional coefficient values at boundaries
        exact  : optional exact solution for error
        """
        N = len(p)
        h = L / N
        invhh = 1.0 / (h * h)
        
        avgRes, maxRes = 0.0, 0.0
        
        for i in range(N):
            # face coefficients
            aip = 0.5*(a[i] + a[i+1]) if i < N-1 else (coefBc[1] if coefBc is not None else a[i])
            aim = 0.5*(a[i-1] + a[i]) if i > 0    else (coefBc[0] if coefBc is not None else a[i])
            
            # neighbor p values (ghost cells for boundaries)
            pip = p[i+1] if i < N-1 else 2*bc[1] - p[i]
            pim = p[i-1] if i > 0    else 2*bc[0] - p[i]
            
            # residual
            res = abs((aip*(pip - p[i]) - aim*(p[i] - pim)) * invhh - b[i])
            avgRes += res
            maxRes  = max(maxRes, res)
        
        print(f"Residual L1, Linf: {avgRes:.4e}, {maxRes:.4e}")
        
        if exact is not None:
            err = abs(p - exact)
            avgErr, maxErr = np.mean(err), np.max(err)
            print(f"Error    L1, Linf: {avgErr:.4e}, {maxErr:.4e}")
            return avgRes, maxRes, avgErr, maxErr
        
        return avgRes, maxRes


def setup_psn1d_by_func(L, nx, f_p, f_a, f_f):
    """
    Set up 1D Poisson problem using analytical functions.
    
    Args:
        L: domain length
        nx: number of grid points
        f_p: function for exact solution p(x)
        f_a: function for coefficient a(x)
        f_f: function for source term f(x)
        
    Returns:
        (x, p_exact, a, f, g0, gL): grid, exact solution, coefficient, source, BCs
    """
    # Grid points (node-centered)
    x = np.linspace(0, L, nx)
    
    # Evaluate functions at grid points
    p_exact = np.array([f_p(xi) for xi in x])
    a = np.array([f_a(xi) for xi in x])
    f = np.array([f_f(xi) for xi in x])
    
    # Boundary conditions
    g0 = f_p(0.0)
    gL = f_p(L)
    
    return x, p_exact, a, f, g0, gL

def _interp_face(left, right):
    return 0.5 * (left + right)