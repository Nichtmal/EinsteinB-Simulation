from scipy.special import wofz
import matplotlib.pyplot as plt
import numpy as np


def voigt_profile(nu, nu_0, delta_nu_D, delta_nu_L):
    z = ((nu - nu_0) + 1j * delta_nu_L / 2) / (delta_nu_D * np.sqrt(2))
    return np.real(wofz(z)) / (delta_nu_D * np.sqrt(2 * np.pi))


# Example parameters
nu_0 = 5.1e14  # Central frequency
nu = np.linspace(nu_0 - 5e10, nu_0 + 5e10, 1000)  # Frequency grid
delta_nu_D = 1.7e9  # Doppler FWHM (Hz)
delta_nu_L = 3e9  # Pressure FWHM (Hz)

profile = voigt_profile(nu, nu_0, delta_nu_D, delta_nu_L) * np.exp(
    -10 * voigt_profile(nu, nu_0, delta_nu_D, delta_nu_L)
)

plt.figure(figsize=(8, 6))
plt.plot(nu, profile, label="Absorption profile")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Intensity [Arbitrary units]")
plt.title("Absorption Profile")
plt.grid(True)
plt.legend()
plt.show()
