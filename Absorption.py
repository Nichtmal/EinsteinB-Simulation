from scipy.special import wofz
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np

# Voigt profile function
def voigt_profile(nu, nu_0, delta_nu_D, delta_nu_L):
    z = ((nu - nu_0) + 1j * delta_nu_L / 2) / (delta_nu_D * np.sqrt(2))
    return np.real(wofz(z)) / (delta_nu_D * np.sqrt(2 * np.pi))

# Function to compute the integral of the sum of two Voigt profiles
def integral_of_sum(a, nu_0, delta_nu_D, delta_nu_L):
    # Define the sum of two Voigt profiles
    def sum_profiles(nu):
        return voigt_profile(nu, nu_0, delta_nu_D, delta_nu_L) * np.exp(-a* voigt_profile(nu, nu_0, delta_nu_D, delta_nu_L))

    # Integrate the sum over all frequencies (-∞ to ∞)
    integral, _ = quad(sum_profiles, -np.inf, np.inf)
    return integral

# Parameters
nu_0 = 5.1e14                  # Central frequency of the first profile (Hz)
delta_nu_D = 1.7e9             # Doppler FWHM (Hz)
delta_nu_L = 3e9               # Pressure FWHM (Hz)
a_range = np.linspace(1e-5, 1e5, 100)  # Range mass path changes [Arbitrary units]

# Compute the integral for different peak separations
integrals = [integral_of_sum(a, nu_0, delta_nu_D, delta_nu_L) for a in a_range]

# Plot the result
plt.figure(figsize=(8, 6))
plt.plot(delta_range, integrals, label="Convolution of Voigt profiles")
plt.xlabel("Mass path [Arbitrary units]")
plt.ylabel("Convolution [Arbitrary units]")
plt.title("Integral of Voigt Profiles as a Function of Peak Separation")
plt.grid(True)
plt.legend()
plt.show()
