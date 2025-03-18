This project is not yet complete and in the testing phase. The most recent version can be found in the Debugging branch

Roadmap:
1. Debug code further and ensure accurate computation of B = 0 values (currently, there seems to be a mismatch of ~3 oders of magnitude)
2. Improve readability
3. Reactivate, optimise and fully implement custom TISE solver to allow for more complicated and accurate potentials

Structure of the project:
1. ASE.py computes base wave functions for Na and Na+
2. Perturbation.py uses perturbation theory to perturb computed wavefunctions to correct for spin-orbit-coupling (SOC) and Zeeman effects
3. Einstein.py performs grouping and computation of all desired transitions and their respective Einstein B12 (E12 in code) coefficients for g_1 = g_2
4. interpretation.py groups and plots E12 coefficients by B

main.py calls all other scripts and is where the user can control all parameters
database_manager.py manages the h5py database which enables all scripts to run seperately without having to compute everything anew
