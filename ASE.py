from ase import Atoms
from gpaw import GPAW
from gpaw.utilities.ps2ae import PS2AE


def run_simulation(
    element,
    ionized=False,
    num_states=10,
    grid_spacing=0.1,
    dimension=15,
    conv_energy=1e-4,
):
    """
    Run GPAW calculation for the given element and optionally ionized state.

    Parameters:
    - element (str): Chemical element symbol (e.g., 'Na')
    - ionized (bool): If True, treat the atom as ionized (Na+ / Ne)
    - num_states (int): Number of eigenstates to compute
    - grid_spacing (float): Grid resolution for wavefunctions

    Returns:
    dict: A dictionary containing the simulated data with the following structure:
        results = {
            'dimension': int,
            'resolution': float,
            'convergence': float,
            'states': [
                    {'eigenvalue': float, 'wavefunction': ndarray, 'number': int, 'spin': int, 'atom': str},
                    ...
                ]
        }

    """
    atom_label = "Na+" if ionized else element
    print(atom_label)

    # Define the atom
    adjusted_dimension = (
        round(dimension / grid_spacing) * grid_spacing
    )  # Adjust the dimension to ensure a cubic grid
    print(
        f"New dimensions: {adjusted_dimension}, with {adjusted_dimension / grid_spacing} points per dimension"
    )
    atom = Atoms(
        atom_label,
        positions=[(0, 0, 0)],
        cell=[adjusted_dimension, adjusted_dimension, adjusted_dimension],
        pbc=False,
    )
    atom.center(vacuum=5.0)

    # GPAW calculator
    print("Setting up calculator")
    calc = GPAW(
        xc="PBE",
        mode="fd",
        h=grid_spacing,
        kpts=(1, 1, 1),
        txt=f"{atom_label}_output.txt",
        convergence={"energy": conv_energy},
        nbands=num_states,
        spinpol=True,  # Include spin-polarization
    )
    atom.calc = calc
    atom.get_potential_energy()

    # Extract wavefunctions and eigenvalues
    print("Calculating wave functions")
    wfs = PS2AE(calc, grid_spacing=grid_spacing)
    eigenvalues = calc.get_eigenvalues()

    # Collect results
    results = {
        "dimension": adjusted_dimension,
        "resolution": grid_spacing,
        "convergence": conv_energy,
        "states": [],
    }

    print("Results collected, saving to database")

    for n in range(num_states):
        for s in [0, 1]:
            state_data = {
                "eigenvalue": eigenvalues[n],
                "wavefunction": wfs.get_wave_function(k=0, n=n, s=s),
                "number": n,
                "spin": s,
                "atom": atom_label,
            }
            results["states"].append(state_data)

    return results
