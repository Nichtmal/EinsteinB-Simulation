import numpy as np
import scipy.sparse as sp
import scipy
import math

from scipy.sparse.linalg import eigsh
from scipy.sparse import diags as sp_diags
from scipy import sparse
from scipy.constants import (
    hbar,
    electron_mass,
    e,
    epsilon_0,
    physical_constants,
    alpha,
    c,
)

# Defining simulation parameters for specific
Z_eff = 2.2  # Effective nuclear charge
B = 0  # Magnetic field strength in Tesla
n = 3  # Principal quantum number
l = 0  # Azimuthal quantum number
s = 1 / 2  # Spin quantum number
j = l + s

# Optional
m_n = 0  # Mass of nucleus

# Constants
m_e = electron_mass  # Electron mass in kg
a_0 = 0.5292e-10  # Bohr radius in meters
A = 1e-10  # Angstrom in meters
mu = m_e if m_n == 0 else (m_e + m_n) / (m_e + m_n)
E_n = -mu * c**2 * Z_eff**2 * alpha**2 / (2 * n**2)
mu_B = 9.2740100783e-24  # In J / T
g = 2  # Landé g-factor


# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


# Defining functions
def m_js(j):
    return [m / 2 for m in range(int(-2 * j), int(2 * j + 1)) if m / 2 != int(m / 2)]


def m_ls(l):
    return [m - l for m in range(2 * l + 1)]


# Defining Operators
def S_x(s):
    return s * hbar * sigma_x


def S_y(s):
    return s * hbar * sigma_y


def S_z(s):
    return s * hbar * sigma_z


def L_z(m_ls):
    L_z = np.eye(len(m_ls))
    for i in range(len(m_ls)):
        L_z[i, i] = hbar * m_ls[i]
    return L_z


def L_plus(m_ls):
    L_plus = np.zeros((len(m_ls), len(m_ls)))
    l = (len(m_ls) - 1) / 2
    for i in range(len(m_ls) - 1):
        m = m_ls[i]
        L_plus[i, i + 1] = hbar * (l * (l + 1) - m * (m + 1)) ** (1 / 2)
    return L_plus


def L_minus(m_ls):
    L_minus = np.zeros((len(m_ls), len(m_ls)))
    l = (len(m_ls) - 1) / 2
    for i in range(1, len(m_ls)):
        m = m_ls[i]
        L_minus[i, i - 1] = hbar * (l * (l + 1) - m * (m - 1)) ** (1 / 2)
    return L_minus


def L_x(m_ls):
    return np.add(L_plus(m_ls), L_minus(m_ls)) / 2


def L_y(m_ls):
    return np.add(L_plus(m_ls), -L_minus(m_ls)) / (2j)


# Effect of Zeeman per Tesla
def Zeeman_perturbation(mls, s, grid):
    L_S = (
        sp.kronsum(L_x(mls), 2 * S_x(s))
        + sp.kronsum(L_y(mls), 2 * S_y(s))
        + sp.kronsum(L_z(mls), 2 * S_z(s))
    )
    factor = g * mu_B / hbar
    dim = np.ceil(grid / L_S.shape[0])
    return factor * truncate_sparse_matrix(sparse.kron(sparse.eye(dim), L_S), grid)


def perturbing(H, psi, eigenvalue, psi_pert, eigenvalue_pert, dx):
    if np.isclose(
        eigenvalue - eigenvalue_pert, 0
    ).any():  # Prevent division by zero for degenerate states
        return np.zeros_like(psi)
    else:
        try:
            perturbation_element = (
                np.conj(psi_pert)
                @ H
                @ psi
                / ((eigenvalue_pert - eigenvalue) / e)
                * psi_pert
            )  # Converting energies into SI units for this step
            norm = np.sqrt((np.sum(np.abs(perturbation_element) ** 2) * dx**3))
        except ValueError:
            print(f"Vector size: {psi_pert.size} \r Matrix size: {H.size}")
            raise ValueError
        return perturbation_element / norm  # Return normalized element


def xi(r, Z_eff):
    factor = hbar * Z_eff * alpha / (2 * m_e**2 * c)
    result = factor / (r**3 + (1e-20))
    return result


def SOC_perturbation(mls, s, Z_eff, r_grid):
    """
    Compute the SOC Hamiltonian with radial dependence.
    :param mls: Angular momentum quantum number l for orbital
    :param s: Spin value (+1/2 or -1/2)
    :param Z_eff: Effective nuclear charge
    :param r_grid: Radial grid
    :return: Spin-orbit coupling Hamiltonian
    """
    # Compute L · S operator in the angular momentum-spin basis
    L_dot_S = (
        sp.kron(L_x(mls), S_x(s))
        + sp.kron(L_y(mls), S_y(s))
        + sp.kron(L_z(mls), S_z(s))
    )

    r_len = r_grid.shape[0]
    dimension = np.ceil(r_len / L_dot_S.shape[0])
    L_S_expanded = sparse.kron(sparse.eye(dimension), L_dot_S)  # Expand L_dot_S
    L_S_expanded = truncate_sparse_matrix(
        L_S_expanded, r_len
    )  # Truncate into grid space (should only delete a small fraction to ensure accuracy)

    radial_factor = sp.diags(xi(r_grid, Z_eff))  # Diagonal matrix in spatial grid
    H_SOC = radial_factor @ L_S_expanded  # Combine position and angular-spin bases

    return H_SOC


def round_sig(x, sig=3):
    """
    Round a number x to sig significant figures.

    """
    if x == 0:
        return 0
    else:
        return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def truncate_sparse_matrix(A, n):
    """
    Truncate a sparse matrix A to its top-left n x n block.

    :param A: Input sparse matrix (CSR format recommended).
    :param n: Desired dimension of the truncated matrix.
    :return: Truncated sparse matrix.
    """
    if int(n) == n:
        if A.shape[0] < n or A.shape[1] < n:
            print(f"Dimension Matrix: {A.shape[0]}, n = {n}")
            raise ValueError(
                "Truncation size must be smaller than the matrix dimensions."
            )
        if A.shape[0] == n and A.shape[1] == n:
            return A

        # Ensure the matrix is in CSR format for efficient row slicing
        A_csr = A.tocsr()

        # Truncate rows
        truncated_data = A_csr.data[: A_csr.indptr[n]]
        truncated_indices = A_csr.indices[: A_csr.indptr[n]]
        truncated_indptr = A_csr.indptr[: n + 1]

        # Truncate columns (filter entries in each row to keep only indices < n)
        mask = truncated_indices < n
        truncated_data = truncated_data[mask]
        truncated_indices = truncated_indices[mask]

        # Adjust indptr to reflect column truncation
        new_indptr = [0]
        for i in range(1, len(truncated_indptr)):
            count = truncated_indptr[i] - truncated_indptr[i - 1]
            new_count = mask[truncated_indptr[i - 1] : truncated_indptr[i]].sum()
            new_indptr.append(new_indptr[-1] + new_count)

        # Create new truncated matrix
        return sp.csr_matrix(
            (truncated_data, truncated_indices, new_indptr), shape=(n, n)
        )
    else:
        raise TypeError


def get_qn(i):  # The wave functions start with n = 2 and l = 1
    """
    Retrieve the quantum numbers n (principal quantum number) and l (orbital angular momentum)
    for a given index i using the properties of quantum states.

    :param i: The index of the state (starting from 0)
    :return: tuple (n, l) - principal quantum number n and orbital angular momentum l
    """
    i = int(i)
    i += 1  # Adjust for 1-based counting
    i += 2  # Accounts for the first  wave functions (1s, 2s) missing

    # Step 1: Find n using the total states formula
    n = 1
    while i > n * (n + 1) * (2 * n + 1) // 6:
        n += 1

    # Step 2: Compute l using the states for each l < n
    remaining_states = (
        i - (n - 1) * n * (2 * (n - 1) + 1) // 6
    )  # Subtract states for lower n
    l = 0
    while remaining_states > 2 * l + 1:
        remaining_states -= 2 * l + 1
        l += 1

    return n, l


def get_mj_values(l, s):
    """
    Generate all possible m_j values for a given l and spin.

    :param l: Orbital angular momentum quantum number
    :param spin: Spin quantum number (+0.5 or -0.5)
    :return: list of m_j values
    """
    m_s = -abs(s)  # Spin projection

    m_l_values = np.arange(-l, l + 2)

    # Orbital magnetic quantum numbers
    m_j_values = [m_l + m_s for m_l in m_l_values]
    return m_j_values


def get_ml_values(l):
    """
    Generate all possible m_j values for a given l and spin.

    :param l: Orbital angular momentum quantum number
    :param spin: Spin quantum number (+0.5 or -0.5)
    :return: list of m_j values
    """
    m_l_values = np.arange(-l, l + 1)  # Orbital magnetic quantum numbers
    return m_l_values


def generate_radial_grid(N_grid, L):
    """
    Generate a radial grid for a cubic grid of size N_grid x N_grid x N_grid with
    physical length L.

    :param N_grid: The number of grid points in each direction
    :param L: The physical length of the cubic grid
    :return: A 1D array of radial distances
    """

    # Create a 3D grid of points from -L/2 to L/2
    x = np.linspace(-L / 2, L / 2, int(N_grid))
    y = np.linspace(-L / 2, L / 2, int(N_grid))
    z = np.linspace(-L / 2, L / 2, int(N_grid))

    # Create meshgrid for the cubic grid in 3D
    X, Y, Z = np.meshgrid(x, y, z)

    # Calculate the radial distance for each point in the grid (from origin)
    r_grid = np.sqrt(X**2 + Y**2 + Z**2)

    r_flat = r_grid.flatten()

    return r_flat


def spin_bin_to_f(bin):
    if bin == 0:
        return -1 / 2
    elif bin == 1:
        return 1 / 2
    else:
        raise TypeError


def correct_wavefunctions(data, length, desired_states, Z_eff, Bs):
    # desired_states is an array of tuples in the form (n, l), Bs is an array
    dim_grid = length
    grid_spacing = data["resolution"]
    L = data["dimension"]
    dx = grid_spacing * 1e-10
    wf_dim = data["states"][0]["wavefunction"].shape[0] ** (1 / 3)
    print(f"Dim grid: {dim_grid}")
    print(f"Dim grid calculated: {L/dx}")
    print(f"Dim grid cubed: {(L/dx)**3}")
    print(f"Wavefunction dimension: {wf_dim}")
    dim_grid = np.round(wf_dim)
    r_grid = generate_radial_grid(int(dim_grid), L)
    B_res = Bs[1] - Bs[0]  # Assuming constant differences
    [B[i] = round_sig(B[i], 5) for i in range(len(B))]
    B = [round_sig(i, 5) for i in B]

    results = {
        "Z_eff": Z_eff,
        "dimension": L,
        "resolution": grid_spacing,
        "B_range": [Bs[0], Bs[-1]],
        "B_res": B_res,
        "states": [],
    }

    for d_state in desired_states:
        n_d, l_d = d_state
        print(n_d, l_d)
        j_down = l_d + 1 / 2
        j_up = l_d - 1 / 2
        i_down = 0
        i_up = 0
        m_js_up = get_mj_values(l_d, 1 / 2)
        m_js_down = get_mj_values(l_d, -1 / 2)
        print(m_js_up)
        print(m_js_down)
        m_ls = get_ml_values(l_d)
        H_SOC_up = SOC_perturbation(m_ls, 1 / 2, Z_eff, r_grid)
        H_SOC_down = SOC_perturbation(m_ls, -1 / 2, Z_eff, r_grid)
        H_Zee_up = Zeeman_perturbation(m_js_down, 1 / 2, dim_grid**3)
        H_Zee_down = Zeeman_perturbation(m_js_up, -1 / 2, dim_grid**3)

        used = []

        for state in data["states"]:
            # Preventing doubling of states
            id = (state["number"], state["spin"], state["atom"])
            if id not in used:
                used.append(id)
                B_fields = []
                Psis = []

                if get_qn(state["number"]) == d_state:
                    n1, l1 = get_qn(state["number"])
                    print(n1, l1)
                    if n1 != 0:
                        if state["spin"] == 0:
                            m_j = m_js_down[i_down]
                            s = -1 / 2
                            i_down += 1
                        elif state["spin"] == 1:
                            m_j = m_js_up[i_up]
                            s = 1 / 2
                            i_up += 1
                        else:
                            raise TypeError

                        psi = state["wavefunction"]

                        eigenvalue = state["eigenvalue"]
                        if state["spin"] == 0:
                            j = j_down
                            for B in Bs:
                                H_down = H_SOC_down + B * H_Zee_down
                                sum_down = 0
                                for state2 in data["states"]:
                                    n, l = get_qn(state2["number"])
                                    spin = state2["spin"]
                                    psi_pert = state2["wavefunction"]
                                    eigenvalue_pert = state2["eigenvalue"]
                                    if spin == 0:
                                        if (
                                            n,
                                            l,
                                        ) != d_state:  # Preventing degenerate states from acting on each other
                                            sum_down += perturbing(
                                                H_down,
                                                psi,
                                                eigenvalue,
                                                psi_pert,
                                                eigenvalue_pert,
                                                dx,
                                            )
                                print(B)
                                norm = np.sum(np.abs(psi + sum_down) ** 2) * dx**3
                                psi_corr = (psi + sum_down) / np.sqrt(
                                    norm
                                )  # Nomalizing the resulting functions
                                # Debugging:
                                print(norm)

                                B_fields.append(B)
                                Psis.append(psi_corr)
                        elif state["spin"] == 1:
                            j = j_up
                            for B in Bs:
                                H_up = H_SOC_up + B * H_Zee_up
                                sum_up = 0
                                for state2 in data["states"]:
                                    n, l = get_qn(state2["number"])
                                    spin = state2["spin"]
                                    psi_pert = state2["wavefunction"]
                                    eigenvalue_pert = state2["eigenvalue"]
                                    if spin == 1:
                                        if (
                                            n,
                                            l,
                                        ) != d_state:  # Preventing degenerate states from acting on each other
                                            sum_up += perturbing(
                                                H_up,
                                                psi,
                                                eigenvalue,
                                                psi_pert,
                                                eigenvalue_pert,
                                                dx,
                                            )
                                norm_corr = np.sqrt(
                                    (np.sum(np.abs(psi + sum_up) ** 2) * dx**3)
                                )
                                print(norm_corr)
                                psi_corr = (psi + sum_up) / norm_corr
                                B_fields.append(B)
                                Psis.append(psi_corr)
                        else:
                            raise TypeError

                        state_data = {
                            "n": n1,
                            "l": l1,
                            "j": j,
                            "m_j": m_j,
                            "s": s,
                            "Bs": np.array(B_fields),  # List
                            "wavefunctions": np.array(Psis),  # List
                            "base_energy": np.array(
                                state["eigenvalue"]
                            ),  # Will be modified by B in other scripts
                        }

                        results["states"].append(state_data)

    return results


if __name__ == "__main__":
    pass
