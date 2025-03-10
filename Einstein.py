from scipy.special import wofz
import numpy as np
from scipy.constants import hbar, alpha, c, k, e, m_e

from Perturbation import generate_radial_grid

A = 1e-10  # 1 Angstrom in m
gamma = e / (2 * m_e)

Efactor = 4 * np.pi * alpha * c / (3 * hbar)

# Setup Voigt function, both constant for f_0 ~ f(B)
delta_nu_D = 1.7e9  # Doppler FWHM (Hz)
delta_nu_L = 3e9  # Pressure FWHM (Hz)


def integral_f(state1, state2, r_grid, dx):
    integrand = np.conj(state1) * state2 * r_grid
    return np.sum(integrand) * dx**3


def ESC_12(state1, state2, r_grid, dx, g1=1, g2=1):
    integral = integral_f(state1, state2, r_grid, dx)
    return g2 / g1 * Efactor * np.abs(integral) ** 2


def is_transition_allowed(state1, state2):
    """
    Determine if an electric dipole transition is allowed between two states
    based on quantum mechanical selection rules.

    Parameters:
    - state1: dict with quantum numbers of the initial state (n, l, s, j, m_j)
    - state2: dict with quantum numbers of the final state (n, l, s, j, m_j)

    Returns:
    - bool: True if the transition is allowed, False otherwise.

    """
    # Extract quantum numbers for initial and final states
    n1, l1, s1, j1, mj1 = (
        state1["n"],
        state1["l"],
        state1["s"],
        state1["j"],
        state1["m_j"],
    )
    n2, l2, s2, j2, mj2 = (
        state2["n"],
        state2["l"],
        state2["s"],
        state2["j"],
        state2["m_j"],
    )

    if abs(l1 - l2) > 1 or (l1 == 0 and l2 == 0):
        # print("Ls dont match")
        # print(l1, l2)
        return False

    if abs(j1 - j2) > 1:
        # print("Js dont match")
        # print(j1, j2)
        return False

    if abs(mj1 - mj2) > 1:
        print("Ms dont match")
        return False

    if s1 != s2:
        # print("Ss dont match")
        # print(s1, s2)
        return False
    return True


def is_transition_desired(state1, state2, desired_transitions):
    """
    Determine if the transition is desired or not relevant to the problem

    Returns Bool

    """

    # Extract quantum numbers for initial and final states
    n1, l1, s1, j1, mj1 = (
        state1["n"],
        state1["l"],
        state1["s"],
        state1["j"],
        state1["m_j"],
    )
    n2, l2, s2, j2, mj2 = (
        state2["n"],
        state2["l"],
        state2["s"],
        state2["j"],
        state2["m_j"],
    )

    if desired_transitions[0] == False:
        return True
    else:
        if type(desired_transitions[1]) != None:
            if (n1, n2) != desired_transitions[1]:
                return False
        if type(desired_transitions[2]) != None:
            if (l1, l2) != desired_transitions[2]:
                return False
            else:
                print(f"States: {(l1, l2)}")
        if type(desired_transitions[3]) != None:
            if (s1, s2) != desired_transitions[3]:
                return False
    print("Desired!")
    return True


def g_j(j, l, s):
    return (
        1 + (j * (j + 1) - l * (l - 1) + s * (s + 1)) / (2 * j * (j + 1))
        if j != 0
        else 0
    )


def h_j(j, l, n):
    return (
        (j * (j + 1) - l * (l + 1) - 3 / 4) / (l * (l + 1) * (l + 1 / 2)) / (4 * n**3)
    )


def delta_Ef(state_attr1, state_attr2, B, Z):
    n1, l1, j1, m_j1, E1 = state_attr1
    s1 = j1 - l1
    n2, l2, j2, m_j2, E2 = state_attr2
    s2 = j2 - l2
    g_j1 = g_j(j1, l1, s1)
    g_j2 = g_j(j2, l2, s2)
    h_j1 = g_j(j1, l1, n1)
    h_j2 = g_j(j2, l2, n1)

    factor_SOC = m_e * c**2 * alpha**4 * Z**4

    return (
        E2
        - E1
        + (hbar * gamma * B * (g_j1 * m_j1 - g_j2 * m_j2)) / e
        + factor_SOC * (h_j2 - h_j1) / e
    )  # In eV


def calculate_ESC(data, Z, desired_transitions):
    """
    Data has states with structure:
    'n': integer
    'l': integer
    'j': float
    'm_j': float
    'Bs': list
    'wavefunctions': list
    'base_energy': float

    Output has structure
    'resolution': float
    'B_range': list of floats
    'B_res': float
    'states': dict

    States (for each B) have structure
    initial_state: list / tuple of n, l, j, m_j
    final_state: list / tuple of n, l, j, m_j
    E12: Einsteincoefficient (float)
    delta_E: positive float
    B: float
    """
    points_n = int(np.round(data["states"][0]["wavefunctions"].shape[1] ** (1 / 3)))

    dx = data["resolution"] * A  # Convert to meters for the integral

    gridsize = dx
    x_max = float(0.5 * dx * points_n)
    print(f"X_max: {x_max}")

    print(points_n)
    r_grid = generate_radial_grid(points_n, 2 * x_max)

    results = {
        "resolution": gridsize,
        "B_range": data["B_range"],
        "B_res": data["B_res"],
        "states": [],
    }

    # Tests for edge-cases
    B_lengths = [len(state["Bs"]) for state in data["states"]]
    if len(set(B_lengths)) > 1:
        raise ValueError("Inconsistent number of magnetic field values across states.")

    for state in data["states"]:

        state1 = state  # Better readibility when multiple states are taken into account
        state1_attr = (
            state1["n"],
            state1["l"],
            state1["j"],
            state1["m_j"],
            state1["base_energy"],
        )

        a = 0

        for i in range(len(state["Bs"])):
            B = state1["Bs"][i]
            psi_1 = state1["wavefunctions"][i]
            for state2 in data["states"]:
                state2_attr = (
                    state2["n"],
                    state2["l"],
                    state2["j"],
                    state2["m_j"],
                    state2["base_energy"],
                )
                psi_2 = state2["wavefunctions"][i]
                if is_transition_desired(state1, state2, desired_transitions):
                    if is_transition_allowed(state1, state2):
                        a += 1
                        delta_E = delta_Ef(state1_attr, state2_attr, B, Z)
                        if abs(delta_E) > 0:
                            E = ESC_12(psi_1, psi_2, r_grid, dx)
                            print(E)
                            R = {
                                "initial_state": state1_attr,
                                "resulting_state": state2_attr,
                                "E12": E,
                                "delta_E": delta_E,  # In eV
                                "B": B,
                            }
                            if R in results["states"]:
                                print("Same same")
                            results["states"].append(R)
    print(f"Amount of Transitions: {a}")
    print("Einsteincoefficients calculated")
    return results
