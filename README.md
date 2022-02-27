# Misinformation can prevent the suppression of epidemics
This code is (c) Andrei Sontag, Tim Rogers, Kit Yates, 2021, and it is made available under the GPL license enclosed with the software.
Data produced for the manuscript entitled "Misinformation can prevent the suppression of epidemics".

Over and above the legal restrictions imposed by this license, if you use this software for an academic publication then you are obliged to provide proper attribution. This can be to this code directly,

A. Sontag, T. Rogers, C. Yates. Misinformation can prevent the suppression of epidemics (2022). github.com/amsontag/mcpse.

or to the paper that describes it

Misinformation can prevent the suppression of epidemics

Andrei Sontag, Tim Rogers, Christian Yates

medRxiv 2021.08.23.21262494; doi: https://doi.org/10.1101/2021.08.23.21262494

or (ideally) both.

Acknowledgements

This research made use of the Balena High Performance Computing (HPC) Service at the University of Bath.

Description of this data set:

This data set aims for the easier reproduction of the figures in our manuscript entitled "Misinformation can prevent the suppression of epidemics". The code solves the system of ODEs (3-10) in the manuscript and reproduces the figures accordingly.

The code 'figure1a.py' reproduces Figure 1 (a) in the manuscript.

The code 'figure1b.py' reproduces Figure 1 (b) in the manuscript.

The code 'figure2ac.py' reproduces Figures 2 (a) and (c) in the manuscript.

The code 'figure2bd.py' reproduces Figures 2 (b) and (d) in the manuscript.

The code 'figure3.py' is set to reproduce Figure 3 (c). Figures 3 (a), (b), and (d) can be obtained easily from this code by changing the parameters at the beginning accordingly to the values specified in Figure 3's caption.

The code 'figure4.py' is set to reproduce Figure 4 (a). Figures 4 (b) can be obtained easily from this code by changing the parameters at the beginning accordingly to the values specified in Figure 4's caption.


The files 'varying_rho_ip.npy', 'varying_rho_tp.npy', and 'varying_rho_fs.npy' correspond to the data produced by figure2ac.py.

'varying_rho_ip.npy' is a 100-by-6 array containing the size of the infection peaks for each of the 100 values of rho in (0,1) and each of the 6 values of alpha_T (0.1, 0.25, 0.5, 1, 2 and 5) for which the ODE system was solved.

'varying_rho_tp.npy' is a 100-by-6 array containing the time when the peak occurred for each of the 100 values of rho in (0,1) and each of the 6 values of alpha_T (0.1, 0.25, 0.5, 1, 2 and 5) for which the ODE system was solved.

'varying_rho_fs.npy' is a 100-by-6 array containing the size of the susceptible population at the end of the outbreak for each of the 100 values of rho in (0,1) and each of the 6 values of alpha_T (0.1, 0.25, 0.5, 1, 2 and 5) for which the ODE system was solved.

The files 'varying_density_ip.npy', 'varying_density_tp.npy', and 'varying_density_fs.npy' correspond to the data produced by figure2bd.py.

'varying_density_ip.npy' is a 100-by-6 array containing the size of the infection peaks for each of the 100 values of distrusting population density in (0,1) and each of the 6 values of alpha_T (0.1, 0.25, 0.5, 1, 2 and 5) for which the ODE system was solved.

'varying_density_tp.npy' is a 100-by-6 array containing the time when the peak occurred for each of the 100 values of distrusting population density in (0,1) and each of the 6 values of alpha_T (0.1, 0.25, 0.5, 1, 2 and 5) for which the ODE system was solved.

'varying_density_fs.npy' is a 100-by-6 array containing the size of the susceptible population at the end of the outbreak for each of the 100 values of distrusting population density in (0,1) and each of the 6 values of alpha_T (0.1, 0.25, 0.5, 1, 2 and 5) for which the ODE system was solved.


The files 'Figure4a.npy' and 'Figure4b.npy' correspond to the output data of 'figure4.py' used to generate Figure 4. Each of these are a 100-by-100 matrix giving the value 'False' if suppression is achieved for the corresponding values of rho and d, and 'True' otherwise.

This data can be used to plot Figure 4 (a) and (b) using the 'figure4.py' code. It must be accompanied by the arrays for the values of rho and density defined within the code.
