import h5py
import numpy as np

from ASE import run_simulation
from database_manager import (
    sim_to_db,
    corr_to_db,
    import_corr_from_db,
    einstein_to_db,
    import_einstein_from_db,
)
from Einstein import calculate_ESC
from interpretation import plot_einstein_coefficients

# Connect to database
database_file = (
    "/../Database/Simulation_database.h5"  # Use Linux convention to run from WSL
)

database_file = "/mnt/c/Users/Tiago/Kreuzgasse Onedrive/Desktop/Tiago/Corona Aufgaben/Physik/GYPT Theorie/Database/Simulation_database.h5"

# Split main into actions to only partially recompute values or scripts


def simulation(element, num_states=15, grid_spacing=0.1, dimension=15, conv=1e-4):
    # Run simulations for Na and Na+ / Ne
    for ionized in [False, True]:
        print(f"Ionized: {ionized}")
        simulation_data = run_simulation(
            element,
            ionized=ionized,
            num_states=num_states,
            grid_spacing=grid_spacing,
            dimension=dimension,
            conv_energy=conv,
        )
        sim_to_db(simulation_data, database_file)


def main():
    # Define simulation parameters
    num_states = 10
    grid_spacing = 0.085
    dimension = 15
    conv = 1e-4
    Z_eff = 2.2  # Effective nuclear charge
    Z = 11
    Bs = np.linspace(0, 0.5, 6)  # Magnetic field values for corrections
    print(f"Magnetic Field Strengths: {Bs}")

    # Limiting Transitions in the form of Active, n, l, s
    desired_transitions = (False, (3, 3), (0, 1), None)

    # User inputs
    input1 = input(
        f"(Re-)Compute all values for {num_states} states with a resolution of {grid_spacing} Angstrom? Type Yes / No "
    ).lower()
    input2 = input(
        f"Do you want to inspect the data from {num_states} states in {database_file}? "
    ).lower()
    input3 = input("Perform wavefunction corrections? Type Yes / No ").lower()
    input4 = input("Compute Einstein coefficients? Type Yes / No ").lower()
    input5 = input("Interpolate and Graph? Type Yes / No ").lower()

    # Calculate states of Na and Na+
    if input1 == "yes":
        print("Starting simulation")
        simulation(
            "Na", num_states, grid_spacing, dimension, conv
        )  # The output starts with the 2p orbital
        print(f"Simulation data saved to {database_file}")

    if input2 == "yes":

        with h5py.File(database_file, "r") as db:

            print("Database Structure:")
            db.visit(print)

    dimension = round(dimension / grid_spacing) * grid_spacing

    set_name = f"dim_{dimension}_res_{grid_spacing}_conv_{conv}"
    set_name = set_name.replace(".", "_")
    print(set_name)

    # Perform wavefunction corrections
    if input3 == "yes":
        desired_states = [(3, 0)]  # Sodium 3s
        corr_to_db(
            desired_states, Z_eff, Bs, set_name, database_file=database_file, Atom="Na"
        )
        desired_states = [(3, 1)]  # Na+ 3p
        corr_to_db(
            desired_states, Z_eff, Bs, set_name, database_file=database_file, Atom="Na+"
        )
        print("Wavefunction corrections saved to database.")

    if input4 == "yes":
        data = import_corr_from_db(set_name, database_file, Bs, atoms=["Na", "Na+"])
        results = calculate_ESC(data, Z, desired_transitions)
        einstein_to_db(results, f"einstein_{set_name}_{len(Bs)}B", database_file)

    if input5 == "yes":
        data = import_einstein_from_db(f"einstein_{set_name}_{len(Bs)}B", database_file)
        plot_einstein_coefficients(data, scale="linear")


if __name__ == "__main__":
    main()
