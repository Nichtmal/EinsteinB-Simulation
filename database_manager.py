import h5py
import numpy as np
from Perturbation import correct_wavefunctions


def sim_to_db(data, database_file):
    """
    Initialize or append simulation results into an HDF5 database.

    Parameters:
        database_file (str): Path to the HDF5 database file.
        results (dict): Dictionary containing simulation results with the following structure:
            results = {
                'dimension': float,
                'resolution': float,
                'convergence': float,
                'states': [
                    {'eigenvalue': float, 'wavefunction': ndarray, 'spin': int, 'atom': str},
                    ...
                ]
            }
    """
    results = data
    # Open (or create) the database file
    with h5py.File(database_file, "a") as db:
        # Create or access the main group for simulation results
        if "simulations" not in db:
            sim_group = db.create_group("simulations")
        else:
            sim_group = db["simulations"]

        # Unique identifier for the current results set
        set_name = f"dim_{results['dimension']}_res_{results['resolution']}_conv_{results['convergence']}"
        set_name = set_name.replace(
            ".", "_"
        )  # Avoid issues with special characters in group names

        # Check if this set already exists
        if set_name not in sim_group:
            current_set = sim_group.create_group(set_name)
            current_set.attrs["dimension"] = results["dimension"]
            current_set.attrs["resolution"] = results["resolution"]
            current_set.attrs["convergence"] = results["convergence"]

            # Initialize empty datasets for states
            current_set.create_dataset(
                "eigenvalues", shape=(0,), maxshape=(None,), dtype="f8"
            )
            current_set.create_dataset(
                "wavefunctions",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.vlen_dtype(np.float64),
            )
            current_set.create_dataset(
                "number", shape=(0,), maxshape=(None,), dtype="i4"
            )
            current_set.create_dataset(
                "spins", shape=(0,), maxshape=(None,), dtype="i4"
            )
            dt_atom = h5py.string_dtype(encoding="utf-8")
            current_set.create_dataset(
                "atoms", shape=(0,), maxshape=(None,), dtype=dt_atom
            )
        else:
            current_set = sim_group[set_name]

        # Append data to the datasets
        num_existing_states = current_set["eigenvalues"].shape[0]
        num_new_states = len(results["states"])

        # Resize datasets to accommodate new data
        # Resize datasets to accommodate new data
        current_set["eigenvalues"].resize(
            int(num_existing_states + num_new_states), axis=0
        )
        current_set["wavefunctions"].resize(
            int(num_existing_states + num_new_states), axis=0
        )
        current_set["number"].resize(int(num_existing_states + num_new_states), axis=0)
        current_set["spins"].resize(int(num_existing_states + num_new_states), axis=0)
        current_set["atoms"].resize(int(num_existing_states + num_new_states), axis=0)

        # Write new data
        for i, state in enumerate(results["states"]):
            current_set["eigenvalues"][num_existing_states + i] = state["eigenvalue"]
            current_set["wavefunctions"][num_existing_states + i] = state[
                "wavefunction"
            ].flatten()  # Use np.flatten() for storing function as 1D array
            current_set["number"][num_existing_states + i] = state["number"]
            current_set["spins"][num_existing_states + i] = state["spin"]
            current_set["atoms"][num_existing_states + i] = state["atom"]

        print(f"Results saved to set: {set_name}")


def corr_to_db(desired_states, Z_eff, Bs, set_name, Atom, database_file):
    """
    Retrieve data from the database, apply wavefunction corrections, and save results back to the database.

    Parameters:
    desired_states (list): Selection of wavefunctions to only compute and save the needed.
    Z_eff (float): Effective nuclear charge of the selected atom; This will be later changed to a more sophisticated potential.
    Bs (list): List of all magnetig field strengths to be tested.
    set_name (str): Name used for saving the dataset.
    Atom (str): Name of the atom used in the ASE simulation.
    database_file (str): Path to the HDF5 database file.

    Returns:
        dict: A dictionary containing the corrected data with the following structure:
            {
                'Z_eff': float,
                'resolution': float,
                'B_range': list of floats,
                'B_res': float,
                'states': [
                    {'eigenvalue': float, 'wavefunction': ndarray, 'spin': int, 'atom': str},
                    ...
                ]
            }

            {
            'dimension': int,
            'resolution': float,
            'wavefunctions': list of array of complex floats,
            'states': [
                {'number': int, 'eigenvalue': float, 'wavefunction': array of complex floats, 'spin': int (binary), 'atom': str}
                for i in range(len(numbers))
            ]
        }
    """
    with h5py.File(database_file, "r+") as db:  # Open in read/write mode
        if "simulations" not in db:
            print("No simulation data found in the database.")
            return

        print(f"Processing dataset: {set_name}")
        dataset = db["simulations"][set_name]

        # Extract data
        eigenvalues = dataset["eigenvalues"][:]
        wavefunctions = [np.array(wf) for wf in dataset["wavefunctions"]]
        numbers = dataset["number"][:]
        spins = dataset["spins"][:]

        # Build the input data dictionary for correct_wavefunctions
        data = {
            "dimension": dataset.attrs["dimension"],
            "resolution": dataset.attrs["resolution"],
            "wavefunctions": wavefunctions,
            "states": [
                {
                    "number": numbers[i],
                    "eigenvalue": eigenvalues[i],
                    "wavefunction": wavefunctions[i],
                    "spin": spins[i],
                    "atom": Atom,
                }
                for i in range(len(numbers))
            ],
        }
        length = (
            int(data["dimension"] / data["resolution"])
            if int(data["dimension"] / data["resolution"])
            == data["dimension"] / data["resolution"]
            else -9
        )
        if length == -9:  # Check for edge cases, such as non-integer dimensions
            raise TypeError

        # Perform corrections
        results = correct_wavefunctions(data, length, desired_states, Z_eff, Bs)

        # Save corrected results back to the database
        corr_group_name = f"corrected_{set_name}_{Atom}_{len(Bs)}B"

        n = results["states"][0]["wavefunctions"].shape[0]
        m = results["states"][0]["wavefunctions"].shape[1]

        if corr_group_name not in db["simulations"]:
            corr_group = db["simulations"].create_group(corr_group_name)

        else:
            print(
                f"Corrected dataset '{corr_group_name}' already exists. Replacing dataset..."
            )
            # Replace dataset
            del_corr_set(database_file, f"{set_name}_{Atom}_{len(Bs)}B")
            corr_group = db["simulations"].create_group(corr_group_name)

        corr_group.attrs.update(
            {
                "Z_eff": results["Z_eff"],
                "dimension": results["dimension"],
                "resolution": results["resolution"],
                "B_range": results["B_range"],
                "B_res": results["B_res"],
                "states": [],
            }
        )

        # Initialize empty datasets for corrected states with proper dtypes
        corr_group.create_dataset("n", shape=(0,), maxshape=(None,), dtype="i4")
        corr_group.create_dataset("l", shape=(0,), maxshape=(None,), dtype="i4")
        corr_group.create_dataset("j", shape=(0,), maxshape=(None,), dtype="i4")
        corr_group.create_dataset("m_j", shape=(0,), maxshape=(None,), dtype="i4")
        corr_group.create_dataset(
            "Bs",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=np.dtype("float64")),
        )
        corr_group.create_dataset(
            "wavefunctions",
            shape=(0, n, m),
            maxshape=(None, n, m),
            dtype="complex128",
        )
        corr_group.create_dataset(
            "base_energy", shape=(0,), maxshape=(None,), dtype="float64"
        )

        # Append data to datasets
        for state in results["states"]:
            # Resize datasets to accommodate new data
            corr_group["n"].resize((corr_group["n"].shape[0] + 1,))
            corr_group["l"].resize((corr_group["l"].shape[0] + 1,))
            corr_group["j"].resize((corr_group["j"].shape[0] + 1,))
            corr_group["m_j"].resize((corr_group["m_j"].shape[0] + 1,))
            corr_group["Bs"].resize((corr_group["Bs"].shape[0] + 1,))
            corr_group["wavefunctions"].resize(
                (corr_group["wavefunctions"].shape[0] + 1, n, m)
            )
            corr_group["base_energy"].resize((corr_group["base_energy"].shape[0] + 1,))

            # Convert data to Numpy arrays for consistency
            bs_array = np.asarray(state["Bs"], dtype="float64")
            wavefunctions_array = np.asarray(state["wavefunctions"], dtype="complex64")

            # Write data into the datasets
            corr_group["n"][-1] = state["n"]
            corr_group["l"][-1] = state["l"]
            corr_group["j"][-1] = state["j"]
            corr_group["m_j"][-1] = state["m_j"]
            corr_group["Bs"][-1] = bs_array
            # print(corr_group["wavefunctions"][0])
            # print(wavefunctions_array)
            corr_group["wavefunctions"][-1] = wavefunctions_array
            corr_group["base_energy"][-1] = state["base_energy"]

        print(f"Corrected dataset '{corr_group_name}' updated successfully.")

        print(f"Selected wavefunction: {results['states'][0]['wavefunctions']}")


def import_corr_from_db(set_name1, database_file, Bs, atoms):
    """
    Import corrected results from the database.

    Parameters:
        set_name (str): Name of the corrected dataset to import.
        database_file (str): Path to the HDF5 database file.

    Returns:
        dict: A dictionary containing the corrected data with the following structure:
            {
                'Z_eff': float,
                'resolution': float,
                'B_range': list of floats,
                'B_res': float,
                'states': [
                    {'eigenvalue': float, 'wavefunction': ndarray, 'spin': int, 'atom': str},
                    ...
                ]
            }
    """
    print(atoms)

    states = []
    for atom in atoms:
        set_name = f"corrected_{set_name1}_{atom}_{len(Bs)}B"
        print(set_name)
        with h5py.File(database_file, "r") as db:
            if "simulations" not in db or set_name not in db["simulations"]:
                print(f"Corrected dataset '{set_name}' not found in the database.")
                return None

            dataset = db["simulations"][set_name]

            # Extract attributes
            Z_eff = dataset.attrs.get("Z_eff", None)
            resolution = dataset.attrs.get("resolution", None)
            B_range = dataset.attrs.get("B_range", None)
            B_res = dataset.attrs.get("B_res", None)

            # Extract states data
            states_len = dataset["n"].shape[0]
            print(f"Length: {states_len}")
            for i in range(states_len):
                print(
                    dataset["n"][i],
                    dataset["l"][i],
                    dataset["j"][i],
                    dataset["m_j"][i],
                    dataset["j"][i] - dataset["l"][i],
                )
            print(f"WVF: {dataset['wavefunctions'].shape[0]}")
            print(f"WVFs: {dataset['wavefunctions']}")
            # for i in range(states_len):
            #     dataset['wavefunctions'][i] = np.array(dataset['wavefunctions'][i])

            a = states_len - dataset["wavefunctions"].shape[0]
            done = []
            # Create and return the results dictionary
            for i in range(states_len):
                numbers = (
                    dataset["n"][i],
                    dataset["l"][i],
                    dataset["j"][i],
                    dataset["m_j"][i],
                    dataset["j"][i] - dataset["l"][i],
                )
                if (
                    dataset["n"][i] != 0 and numbers not in done
                ):  # Disregarding non-physical states with quantum numbers 0
                    done.append(numbers)
                    state_data = {
                        "n": dataset["n"][i],
                        "l": dataset["l"][i],
                        "j": dataset["j"][i],
                        "s": dataset["j"][i] - dataset["l"][i],
                        "m_j": dataset["m_j"][i],
                        "Bs": dataset["Bs"][i],  # List
                        "wavefunctions": dataset["wavefunctions"][
                            i - a
                        ],  # adjusting indices
                        "base_energy": dataset["base_energy"][
                            i - a
                        ],  # Will be modified by B in other scripts
                        "atom": atom,
                    }
                    # print(state_data)
                    if not any(
                        d["n"] == state_data["n"]
                        and d["l"] == state_data["l"]
                        and d["j"] == state_data["j"]
                        and d["s"] == state_data["s"]
                        and d["m_j"] == state_data["m_j"]
                        and d["atom"] == state_data["atom"]
                        and np.array_equal(
                            d["wavefunctions"], state_data["wavefunctions"]
                        )
                        and np.array_equal(d["Bs"], state_data["Bs"])
                        and d["base_energy"] == state_data["base_energy"]
                        for d in states
                    ):
                        states.append(state_data)
    r = {
        "Z_eff": Z_eff,
        "resolution": resolution,
        "B_range": list(B_range) if B_range is not None else None,
        "B_res": B_res,
        "states": states,
    }
    print("Corrections imported!")

    return r


def einstein_to_db(data, set_name, database_file):
    """
    Save Einstein coefficients:

    Parameters:
        database_file (str): Path to the HDF5 file.
        set_name (str): Name of the group (table) to store the data.
        data (dict): Dictionary containing the Einstein coefficient data in the format:
                        {
                            'resolution': gridsize,
                            'B_range': list,
                            'B_res': value,
                            'states': [
                                {
                                    'initial_state': dict,
                                    'resulting_state': dict,
                                    'E12': float,
                                    'delta_E': float,
                                    'B': float
                                },
                                ...
                            ]
                        }
    """

    with h5py.File(database_file, "a") as db:
        if set_name not in db["simulations"]:
            table_group = db["simulations"].create_group(set_name)
        else:
            del db["simulations"][set_name]
            table_group = db["simulations"].create_group(set_name)

        # Save metadata as attributes
        table_group.attrs["resolution"] = data["resolution"]
        table_group.attrs["B_range"] = data["B_range"]
        table_group.attrs["B_res"] = data["B_res"]

        # Initialize datasets
        table_group.create_dataset(
            "initial_state",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.vlen_dtype(str),
        )
        table_group.create_dataset(
            "resulting_state",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.vlen_dtype(str),
        )
        table_group.create_dataset("E12", shape=(0,), maxshape=(None,), dtype="int32")
        table_group.create_dataset(
            "delta_E", shape=(0,), maxshape=(None,), dtype="int32"
        )
        table_group.create_dataset("B", shape=(0,), maxshape=(None,), dtype="f8")

        # Extract current sizes of datasets
        current_size = table_group["E12"].shape[0]
        new_size = current_size + len(data["states"])

        # Resize datasets to accommodate new data
        table_group["initial_state"].resize((new_size,))
        table_group["resulting_state"].resize((new_size,))
        table_group["E12"].resize((new_size,))
        table_group["delta_E"].resize((new_size,))
        table_group["B"].resize((new_size,))

        # Append new data
        for i, state in enumerate(data["states"]):
            table_group["initial_state"][current_size + i] = str(state["initial_state"])
            table_group["resulting_state"][current_size + i] = str(
                state["resulting_state"]
            )
            table_group["E12"][current_size + i] = state["E12"]
            table_group["delta_E"][current_size + i] = state["delta_E"]
            table_group["B"][current_size + i] = state["B"]
    print("Einstein coefficients saved")


def import_einstein_from_db(set_name, database_file):
    """
    Retrieve Einstein coefficient data from the HDF5 database for visualization.

    Parameters:
        database_file (str): Path to the HDF5 file.
        set_name (str): Name of the group (table) containing the data.

    Returns:
        dict: Dictionary with the retrieved data structured as:
            'resolution': float
            'B_range': list of floats
            'B_res': float
            'states':
              {
                  'initial_states': list of str,
                  'final_states': list of str,
                  'E12': list of float,
                  'delta_E': list of float,
                  'B': list of float
              }
    """
    with h5py.File(database_file, "r") as db:
        if set_name not in db["simulations"]:
            raise ValueError(f"Set '{set_name}' not found in the database.")

        table_group = db["simulations"][set_name]

        # Load data into a dictionary
        states = {
            "initial_states": [
                state.decode("utf-8") for state in table_group["initial_state"]
            ],
            "final_states": [
                state.decode("utf-8") for state in table_group["resulting_state"]
            ],
            "E12": list(table_group["E12"]),
            "delta_E": list(table_group["delta_E"]),
            "B": list(table_group["B"]),
        }

        data = {
            "resolution": table_group.attrs["resolution"],
            "B_range": table_group.attrs["B_range"],
            "B_res": table_group.attrs["B_res"],
            "states": states,
        }

    return data


def extract_data_from_db():
    """
    Extract and return all simulation and correction data from the database.

    Returns:
        dict: Dictionary with the retrieved data structured as:
              {
                  'dimension': int,
                  'resolution': float,
                  'eigenvalues': list of floats,
                  'wavefunctions': list of arrays of complex floats,
                  'numbers': list of tuples of integers,
                  'spins': list of binary integers / floats
              }
    """
    data = {}
    with h5py.File(database_file, "r") as db:
        if "simulations" not in db:
            print("No simulation data found in the database.")
            return data

        for set_name in db["simulations"]:
            dataset = db["simulations"][set_name]
            print(f"Loading dataset: {set_name}")

            data[set_name] = {
                "dimension": dataset.attrs.get("dimension", None),
                "resolution": dataset.attrs.get("resolution", None),
                "eigenvalues": dataset["eigenvalues"][:],
                "wavefunctions": [np.array(wf) for wf in dataset["wavefunctions"]],
                "numbers": dataset["number"][:],
                "spins": dataset["spins"][:],
            }
    return data


def del_corr_set(database_file, set_name):
    """
    Deletes a selected set in the database

    Parameters:
        database_file (str): Path to the HDF5 file.
        set_name (str): Name of the group (table) containing the data.
    """
    with h5py.File(database_file, "r+") as db:  # Open in read/write mode
        simulations_group = db["simulations"]
        corr_group_name = f"corrected_{set_name}"  # The name of the group to delete

        if corr_group_name in simulations_group:
            del simulations_group[corr_group_name]
            print(f"Group '{corr_group_name}' has been deleted from 'simulations'.")
        else:
            print(f"Group '{corr_group_name}' does not exist in 'simulations'.")


def del_einstein_set(database_file, set_name):
    """
    Deletes a selected set in the database

    Parameters:
        database_file (str): Path to the HDF5 file.
        set_name (str): Name of the group (table) containing the data.
    """
    with h5py.File(database_file, "r+") as db:  # Open in read/write mode
        simulations_group = db
        einstein_group_name = f"einstein_{set_name}"  # The name of the group to delete

        if einstein_group_name in simulations_group:
            del simulations_group[einstein_group_name]
            print(f"Group '{einstein_group_name}' has been deleted from 'simulations'.")
        else:
            print(f"Group '{einstein_group_name}' does not exist in 'simulations'.")


def clear_database(database_file):
    """
    Clears all groups and datasets from the HDF5 database file.

    Parameters:
        database_file (str): Path to the HDF5 database file.
    """
    with h5py.File(database_file, "a") as db:
        # Iterate over all keys and delete each one
        for key in list(db.keys()):  # Use list() to avoid issues during iteration
            del db[key]
        print(f"All data has been cleared from {database_file}.")


if __name__ == "__main__":
    database_file = "/mnt/c/Users/Tiago/Kreuzgasse Onedrive/Desktop/Tiago/Corona Aufgaben/Physik/GYPT Theorie/Database/Simulation_database.h5"
    set_name = "dim_14_962499999999999_res_0_0875_conv_0_0001_Na_6B"
    set_name_e = "dim_14_962499999999999_res_0_0875_conv_0_0001"
    if input("Clear database? ").lower() == "yes":
        if input("Are you sure you want to clear the database? ").lower() == "yes":
            clear_database("path")
    if input(f"Delete {set_name}? Yes / No ").lower() == "yes":
        del_corr_set(database_file, set_name)
    if input(f"Delete Einstein coefficients? ").lower() == "yes":
        del_einstein_set(database_file, set_name_e)
