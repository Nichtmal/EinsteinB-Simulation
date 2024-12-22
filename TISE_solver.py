import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags as sp_diags
from scipy.constants import hbar, electron_mass, e, epsilon_0, physical_constants, alpha, c

from timeit import Timer

# Defining simulation parameters
Z_eff = 1  # Effective nuclear charge
B = 0  # Magnetic field strength in Tesla
n = 1  # Principal quantum number
l = 0 # Azimuthal quantum number
s = 1 / 2  # Spin quantum number
j = l + s
# Optional
m_n = 0  # Mass of nucleus

# Constants
m_e = electron_mass  # Electron mass in kg
a_0 = physical_constants['Bohr radius'][0]  # Bohr radius in meters
mu = m_e if m_n == 0 else (m_e + m_n) / (m_e + m_n)
E_n = -mu * c**2 * Z_eff**2 * alpha**2 / (2 * n**2)
mu_B = 9.2740100783e-24  # In J / T

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Defining grid size
dx = 5e-1 * a_0
#dx = 5e-1 * a_0 # For Debugging purposes
x_max = 2 * a_0
gridsize = int(2 * x_max / dx + 1)
print(gridsize)

# Defining functions
def r_f(x):
    r = np.sqrt(x.dot(x))
    return max(r, 0.1 * dx)

m_js = [m / 2 for m in range(int(-2 * j), int(2 * j + 1)) if m / 2 != int(m / 2)]
m_ls = [m - l for m in range(2 * l + 1)]

# Defining axis
x = np.linspace(-x_max, x_max, gridsize)
y = np.linspace(-x_max, x_max, gridsize)
z = np.linspace(-x_max, x_max, gridsize)

# Matrices X, Y, Z
X, Y, Z = np.meshgrid(x, y, z, indexing="ij", sparse=False)

# Flattening grid points
points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
print("Norm of points:", np.linalg.norm(points))

# Defining Operators
def spin_fx(s):
    return s * sigma_x

def spin_fy(s):
    return s * sigma_y

def spin_fz(s):
    return s * sigma_z

def L_fz(m_ls):
    L_z = np.eye(len(m_ls))
    for i in range(len(m_ls)):
        L_z[i, i] = m_ls[i]
    return L_z

def L_fplus(m_ls):
    L_plus = np.zeros((len(m_ls), len(m_ls)))
    l = (len(m_ls) - 1) / 2
    for i in range(len(m_ls) - 1):
        m = m_ls[i]
        L_plus[i, i + 1] = hbar * (l * (l + 1) - m * (m + 1))**(1 / 2)
    return L_plus

def L_fminus(m_ls):
    L_minus = np.zeros((len(m_ls), len(m_ls)))
    l = (len(m_ls) - 1) / 2
    for i in range(1, len(m_ls)):
        m = m_ls[i]
        L_minus[i, i - 1] = hbar * (l * (l + 1) - m * (m - 1))**(1 / 2)
    return L_minus

def L_fx(m_ls):
    return np.add(L_fplus(m_ls), L_fminus(m_ls)) / 2

def L_fy(m_ls):
    return np.add(L_fplus(m_ls), -L_fminus(m_ls)) / (2j)

# Storing operators
L_x = L_fx(m_ls)
L_y = L_fy(m_ls)
L_z = L_fz(m_ls)

spin_x = spin_fx(s)
spin_y = spin_fy(s)
spin_z = spin_fz(s)

# Potential and Hamiltonian
# Coulomb potential
V_const = -Z_eff * e**2 / (4 * np.pi * epsilon_0)

def V(r):
    return (V_const / r) if r != 0 else (V_const / dx)

def D():
    n_points = gridsize
    main_diag = -2.0 * np.ones(n_points)
    off_diag_x = np.ones(n_points - 1)
    
    off_diag_x[gridsize - 1 :: gridsize] = 0

    diagonals = [
        main_diag,
        off_diag_x,
        off_diag_x]
    offsets = [0, -1, 1]
    D = sp_diags(diagonals, offsets, shape=(n_points, n_points))
    
    return D

D_matrix = D()
print("D computed")

def construct_laplacian():
    n_points = points.shape[0]

    # Create main diagonal (center terms)
    main_diag = -6.0 * np.ones(n_points) / dx**2

    # Create off-diagonals (neighbor terms)
    off_diag_x = np.ones(n_points - 1) / dx**2
    off_diag_y = np.ones(n_points - gridsize) / dx**2
    off_diag_z = np.ones(n_points - gridsize**2) / dx**2

    # Boundary conditions are handled in the diagonal arrays directly
    off_diag_x[gridsize - 1 :: gridsize] = 0
    off_diag_y[(gridsize - 1) * gridsize :] = 0
    off_diag_z[(gridsize - 1) * gridsize**2 :] = 0

    # Using diagonals for sparse matrix
    diagonals = [
        main_diag,
        off_diag_x,
        off_diag_x,
        off_diag_y,
        off_diag_y,
        off_diag_z,
        off_diag_z,
    ]
    offsets = [0, -1, 1, -gridsize, gridsize, -gridsize**2, gridsize**2]

    laplacian = sp_diags(diagonals, offsets, shape=(n_points, n_points))

    return laplacian

def build_laplacian_norm():
    return sp.kron(sp.kron(D_matrix, D_matrix), D_matrix)

laplacian = build_laplacian_norm()
print("Laplacian computed")
print("Norm Laplacian:", sp.linalg.norm(laplacian))

# Hamiltonian
def get_Hamiltonian():
    num_points = points.shape[0]

    # Kinetic Energy
    T = -(hbar**2 / (2 * m_e)) * laplacian / dx**2

    print("T computed")

    # Potential Energy
    V_diagonal = np.array([V(np.linalg.norm(point)) for point in points])
    V_matrix = sp_diags(V_diagonal, 0)

    print("V matrix computed")

    # Spin-Orbit Coupling
    L_dot_S = sp.kron(L_x, spin_x) + sp.kron(L_y, spin_y) + sp.kron(L_z, spin_z)
    H_SOC = (hbar / (2 * m_e**2 * c**2)) * L_dot_S

    print("SOC computed")

    # Relativistic Correction
    p_4 = hbar**4 * laplacian @ laplacian / dx**4
    H_rel = -(1 / (8 * m_e**3 * c**2)) * p_4

    print("Rel computed")

    # Darwin Term
    Darwin_correction = Darwin_correction = np.array([np.pi * Z_eff * e**2 * hbar**2 / (2 * m_e**2 * c**2) if r_f(point) < 10 * dx else 0 for point in points])

    H_Darwin = sp_diags(Darwin_correction, 0)

    print("Darwin computed")

    # Zeeman Effect
    H_Zeeman = mu_B * B * (sp.kron(L_z, np.eye(2)) + 2 * sp.kron(np.eye(len(m_ls)), spin_z))
    
    # Adjusting dimensions for returning
    effects = [T, V_matrix, H_SOC, H_rel, H_Darwin, H_Zeeman]
    shapes = []
    
    for i in effects:
        if i.nnz != 0 and i.shape[0] not in shapes:
            shapes.append(i.shape[0])
    
    print("Effects computed")
    print(shapes)
    
    output = 0
    target_shape = 1
    l = 1
    
    for i in shapes:
        target_shape = np.lcm(target_shape, i)
        l *= i
    
    print("This approach is around " + str((l/target_shape)**3) + " as efficient")
    
    identities = [sp.eye(int(target_shape/i)) for i in shapes]
    print("Identities stored")
    
    for i in effects:
        shape = i.shape[0]
        output += sp.kron(i, identities[shapes.index(shape)])
        print("Effect computed with shape before " + str(shape))

    print("Effects computed!")
    print(output.shape)

    # Print the norms of individual terms
    print("Norm of kinetic energy T:", sp.linalg.norm(T))
    print("Norm of potential energy V_matrix:", sp.linalg.norm(V_matrix))
    print("Norm of spin-orbit coupling H_SOC:", sp.linalg.norm(H_SOC))
    print("Norm of relativistic correction H_rel:", sp.linalg.norm(H_rel))
    print("Norm of Darwin correction H_Darwin:", sp.linalg.norm(H_Darwin))
    print("Norm of Zeeman term H_Zeeman:", sp.linalg.norm(H_Zeeman))
    H_normalized = output / np.max(np.abs(output.data))
    return H_normalized


def solve_TISE(H, n_states):
    print("finally at TISE")
    
    if sp.issparse(H):
        print("Hamiltonian is sparse.")
        eigenvalues, eigenvectors = eigsh(H, k=n_states, which="SA")
    else:
        print("Hamiltonian is dense.")
        eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    print("Solved TISE")
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0)
    return eigenvalues, eigenvectors

eigenvalues, eigenvectors = solve_TISE(get_Hamiltonian(), 3)
eigenvalues_eV = eigenvalues / e

print(eigenvalues_eV[:3])
