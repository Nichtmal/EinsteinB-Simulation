import numpy as np
import matplotlib.pyplot as plt


def plot_einstein_coefficients(data, scale="linear"):
    """
    Plots the Einstein coefficients as a function of the magnetic field (B) with interpolation.

    Parameters:
        data (dict): Dictionary containing the data with structure:
            'resolution': float
            'B_range': list of floats
            'B_res': float
            'states': dict containing:
                {
                    'initial_states': list of str (e.g., ['(3, 1, 1, -1, 1.2427907355998478)', ...]),
                    'final_states': list of str (e.g., ['(3, 1, 1, 0, 1.2428079025720868)', ...]),
                    'E12': list of float (e.g., [-0.0007799634751050715, ...]),
                    'B': list of float (e.g., [0.0, 0.1111111111111111, ...]),
                    'delta_E': list of float with unit eV (e.g., [0.001, ...])
                }
        scale (str): Scale for the axes, 'linear' or 'log'.
    """
    # Initialize a list of states with grouped data
    print(f"Magnetic field resolution: {data["B_res"]} T")
    num_entries = len(data["states"]["initial_states"])

    states_data = data["states"]
    num_entries = len(states_data["initial_states"])
    formatted_states = [
        {
            "initial_state": states_data["initial_states"][i],
            "final_state": states_data["final_states"][i],
            "E12": states_data["E12"][i],
            "B": states_data["B"][i],
            "delta_E": states_data["delta_E"][i],
        }
        for i in range(num_entries)
    ]
    # Debugging:
    # print(formatted_states)

    states = []
    debug_list = []
    for entry in formatted_states:
        initial_state = entry["initial_state"]
        final_state = entry["final_state"]
        E12 = entry["E12"]
        
        if E12 > 1:  # Filter valid transitions by defining a threshhold
            existing_state = next(
                (
                    s
                    for s in states
                    if s["initial_state"] == initial_state
                    and s["final_state"] == final_state
                ),
                None,
            )
            if existing_state is None:
                # Add new transition group
                print("New state!")
                states.append(
                    {
                        "initial_state": initial_state,
                        "final_state": final_state,
                        "Bs": [entry["B"]],
                        "E12": [E12],
                        "delta_E": [entry["delta_E"]],
                    }
                )
                debug_list.append((initial_state, final_state))
            else:
                # Append data to the existing group
                if False:
                    pass
                elif entry["B"] not in existing_state["Bs"]:
                    print(entry["B"])
                    existing_state["Bs"].append(entry["B"])
                    existing_state["E12"].append(E12)
                    existing_state["delta_E"].append(entry["delta_E"])
        else:
            print(f"Skip with E12 = {E12}")
    print(debug_list)

    for state in states:
        plt.figure(figsize=(10, 6))

        B_values = np.array(state["Bs"])
        E12_values = np.array(state["E12"])
        for i in E12_values:
            print(i)
        E12_at_B0 = E12_values[0]
        print(E12_at_B0)

        if E12_at_B0 <= 0 or np.isclose(E12_at_B0, 0).any():
            print(
                f"Skipping state {state['initial_state']} → {state['final_state']} due to zero or negative E12 at B = 0"
            )
            continue

        # Normalize E12 values to their value at B = 0
        E12_values_normalized = []
        B_values_new = []
        for i in range(len(E12_values)):
            if (
                E12_values[i] / E12_at_B0 > 0.5 and E12_values[i] / E12_at_B0 < 2
            ):  # Exclude outliers, since a linear trend is expected with at small energy scales compared to the base Hamiltonian
                E12_values_normalized.append(E12_values[i] / E12_at_B0)
                B_values_new.append(B_values[i])

        B_values = np.array(B_values_new)
        E12_values_normalized = np.array(E12_values_normalized)
        # print(E12_values_normalized[1])

        # Perform linear fit
        try:
            slope, intercept = np.polyfit(B_values, E12_values_normalized, 1)
            E12_interp = slope * B_values + intercept
        except np.linalg.LinAlgError:
            print(
                f"Fit failed for state {state['initial_state']} → {state['final_state']}"
            )
            continue

        scale_graph = abs(E12_values_normalized[0] - E12_values_normalized[-1]) * 2

        # Plot normalized data points
        plt.scatter(
            B_values,
            E12_values_normalized,
            label="Data points",
            color="blue",
            alpha=0.7,
        )

        # Plot linear interpolation line
        plt.plot(
            B_values,
            E12_interp,
            label=f"Linear fit: y = {slope:.2e}x + {intercept:.2e}",
            color="red",
        )

        # Configure plot
        label = f"Transition: n, l, j, m_j, E: {state['initial_state']} → {state['final_state']}\nSlope: {slope:.2e}, Initial E12: {E12_at_B0}"
        plt.title(label)
        plt.xlabel("Magnetic Field [T])")
        plt.ylabel("Relative B12 Einstein Coefficient")
        plt.ylim(
            E12_values_normalized.min() * (1 - scale_graph),
            E12_values_normalized.max() * (1 + scale_graph),
        )
        if scale == "log":
            plt.xscale("log")
            plt.yscale("log")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Show the plot for this transition
        plt.show()