import numpy as np
import matplotlib.pyplot as plt


# Create a generic element class
class LocalElement:
    def __init__(self, x1: float, x2: float):
        self.x1 = x1
        self.x2 = x2

        temp = 1 / (x2 - x1)
        self.a11 = temp
        self.a22 = temp
        self.a12 = -temp
        self.a21 = -temp

    def trans(self, xi):
        # Transformation from local to global coordinate - return x
        return self.x1 + (self.x2 - self.x1) * xi


# Gauss quadrature for integration boundaries [a,b] and function f
class GaussQuadrature:
    def __init__(self):
        # define the Gaussian Quadrature for 3 points
        self.eta = [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]
        self.w = [5 / 9, 8 / 9, 5 / 9]

    def calc_local(self, g) -> float:
        return 0.5 * sum(
            [w_i * g(0.5*eta_i + 0.5) for eta_i, w_i in zip(self.eta, self.w)])


# Number of nodes
n = 18
# Number of elements
n_elements = n - 1

# Domain length
L = np.pi

# Discretize the domain in random locations
x = np.linspace(0, L, n)
x.sort()
x[0], x[-1] = 0.0, L  # domain start and end

# Set boundary condition (feel free to play around with the values)
u_0, u_n = -0.1, 0.4
du_0, du_n = 0.2, -0.3

# Set RHS function
f = lambda xc: np.sin(xc)

f_i = np.array([f(xi) for xi in x])

# User select boundary conditions
bc_set = int(input("Select boundary condition type: \n (1) Dirichlet/Dirichlet,"
                   "\n (2) Dirichlet/Neumann,\n (3) Neumann/Dirichlet \n ---> "))
if bc_set == 1:
    print("Apply Dirichlet u(0)={} and Dirichlet u(L)={}".format(u_0, u_n))
elif bc_set == 2:
    print("Apply Dirichlet u(0)={} and Neumann ∂u(L)={}".format(u_0, du_n))
elif bc_set == 3:
    print("Apply Neumann ∂u(0)={} and Dirichlet u(L)={}".format(du_0, u_n))
else:
    raise 'error - no boundary condition type selected'

# Initialize matrix A and vector b with zeros
A = np.zeros((n, n))
b = np.zeros(n)

# Initialize GaussQuadrature class
gauss_quadrature = GaussQuadrature()

# Define shape functions in local coordinate system
N1 = lambda xi: 1 - xi
N2 = lambda xi: xi

# Calculate the elements of A and b for each node
for i in range(n_elements):
    x1, x2 = x[i], x[i+1]
    el = LocalElement(x1, x2)
    
    A[i, i] += el.a11
    A[i+1, i+1] += el.a22
    A[i, i+1] += el.a12
    A[i+1, i] += el.a21
    h = x2 - x1
    fi, fj = f_i[i], f_i[i+1]

    b[i]   += h/6 * (2*fi + fj)
    b[i+1] += h/6 * (fi + 2*fj)

# Applying boundary conditions
if bc_set == 1:
    b[1] += - A[1, 0] * u_0
    b[-2] += - A[-2, -1] * u_n
elif bc_set == 2:
    b[1] += - A[1, 0] * u_0
    b[-1] += du_n
elif bc_set == 3:
    b[0] += -du_0
    b[-2] += - A[-2, -1] * u_n

# Solve Au=b
u = np.zeros(n)
if bc_set == 1:
    u[1:n - 1] = np.dot(np.linalg.inv(A[1:n - 1, 1:n - 1]), b[1:n - 1])
    u[0] = u_0
    u[-1] = u_n
elif bc_set == 2:
    u[1:] = np.dot(np.linalg.inv(A[1:, 1:]), b[1:])
    u[0] = u_0
elif bc_set == 3:
    u[:n - 1] = np.dot(np.linalg.inv(A[:n - 1, :n - 1]), b[:n - 1])
    u[-1] = u_n

# Set the exact solution of the given boundary problem
x_exact = np.linspace(0, L, 101)
if bc_set == 1:
    c_1 = (u_n - u_0 - np.sin(L)) / L
    c_2 = u_0
elif bc_set == 2:
    c_1 = du_n - np.cos(L)
    c_2 = u_0
elif bc_set == 3:
    c_1 = du_0 - 1
    c_2 = u_n - np.sin(L) - c_1 * L
u_exact = np.sin(x_exact) + c_1 * x_exact + c_2

# Plot the finite elements and the exact solution
fig, ax = plt.subplots(dpi=120)
ax.plot(x, u, '-o', linewidth=3, label='FE solution')
ax.plot(x_exact, u_exact, '--', linewidth=1.5, label='Analytical solution')
ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], labels=[r"0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
ax.legend()
ax.grid()
plt.show()