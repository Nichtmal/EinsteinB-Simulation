import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def plot_einstein_coefficients(data, scale='linear'):
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
                    'delta_E': list of float (e.g., [0.001, ...])
                }
        scale (str): Scale for the axes, 'linear' or 'log'.
    """
    # Initialize a list of states with grouped data
    states = []
    num_entries = len(data['initial_states'])

    for i in range(num_entries):
        # Parse initial and final states
        initial_state = data['initial_states'][i]
        final_state = data['final_states'][i]
        E12 = data['E12'][i]
        print(E12)
        
        # Check if the transition already exists in the list
        existing_state = next((s for s in states if s['initial_state'] == initial_state and s['final_state'] == final_state), None)
        # print(data['E12'][i])
        if data['E12'][i] > 0:
            if existing_state is None:
                # Add a new transition
                states.append({
                    'initial_state': initial_state,
                    'final_state': final_state,
                    'Bs': [data['B'][i]],
                    'E12': [data['E12'][i]],
                    'delta_E': [data['delta_E'][i]]
                })
            else:
                # Update the existing transition
                existing_state['Bs'].append(data['B'][i])
                existing_state['E12'].append(data['E12'][i])
                existing_state['delta_E'].append(data['delta_E'][i])

    print("Getting to plot")
    # print(states)
    for idx, state in enumerate(states):
        plt.figure(figsize=(10, 6))
        
        B_values = np.array(state['Bs'])
        E12_values = np.array(state['E12'])
        
        # Sort data by B for proper linear fit
        # sorted_indices = np.argsort(B_values)
        # B_values = B_values[sorted_indices]
        # E12_values = E12_values[sorted_indices]
        
        E12_at_B0 = E12_values[0]
        
        if E12_at_B0 <= 0:
            print(f"Skipping state {state['initial_state']} → {state['final_state']} due to zero or negative E12 at B = 0")
            continue
        
        # Normalize E12 values to their value at B = 0
        E12_values_normalized = []
        B_values_new = []
        for i in range(len(E12_values)):
            if E12_values[i] / E12_at_B0 > 0.5 and E12_values[i] / E12_at_B0 < 2:
                E12_values_normalized.append(E12_values[i] / E12_at_B0)
                B_values_new.append(B_values[i])
            
        print(len(E12_values_normalized))
        print(len(B_values))
        B_values = np.array(B_values_new)
        print(B_values)
        E12_values_normalized = np.array(E12_values_normalized)
        print(E12_values_normalized)
        
        # Perform linear fit
        try:
            slope, intercept = np.polyfit(B_values, E12_values_normalized, 1)
            E12_interp = slope * B_values + intercept
        except np.linalg.LinAlgError:
            print(f"Fit failed for state {state['initial_state']} → {state['final_state']}")
            continue
        
        scale_graph = abs(E12_values_normalized[0] - E12_values_normalized[-1])*2
        
        # Plot normalized data points
        plt.scatter(B_values, E12_values_normalized, label="Data points", color="blue", alpha=0.7)
        
        # Plot linear interpolation line
        plt.plot(B_values, E12_interp, label=f"Linear fit: y = {slope:.2e}x + {intercept:.2e}", color="red")
        # print(state['initial_state'])
        # n, l, m_j, s, E = state['initial_state']
        # n2, l2, m_j2, s2, E2 = state['final_state']
        
        
        # Configure plot
        label = f"Transition: n, l, j, m_j, E: {state['initial_state']} → {state['final_state']}\nSlope: {slope:.2e}"
        plt.title(label)
        plt.xlabel("Magnetic Field [T])")
        plt.ylabel("Relative B12 Einstein Coefficient")
        plt.ylim(E12_values_normalized.min() * (1 - scale_graph), E12_values_normalized.max() * (1 + scale_graph))
        if scale == 'log':
            plt.xscale('log')
            plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        # Show the plot for this transition
        plt.show()

