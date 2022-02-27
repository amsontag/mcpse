#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
(c) Andrei Sontag, Tim Rogers, Kit Yates, 2021
Code to plot Figures 2b and 2d in our paper "Misinformation can prevent the suppression of epidemics".
Pre-print available at: https://doi.org/10.1101/2021.08.23.21262494

The code solves the system of ODEs Eqs. (3-10) for different values of the parameters alpha_T and density
of distrusting population and then plots the figures. The ODEs are solved in parallel using the solve_ivp 
function with the RK45 method.

The repository contains the data obtained using this code.
"""

import numpy as np
import multiprocessing as mp
import pandas as pd
from numba import njit
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt

#initialise parameter values
a1 = np.array([.1,.25,.5, 1, 2, 5]) # info exchange rate of trusting individuals
a2 = 1 # info exchange rate of distrusting individuals
lamb = 1/5 # info fading rate
om = 1/3 # refresh info rate
be = 5/7.5 # infection rate in the absence of behavioural changes
rr = 1/7.5 # recovery rate

# defines the points in the parameter space for which we solve the ODE system to generate the figures
space1 = np.linspace(0.001,.99,100)

rho = 0.8 # efficiency of the NPIs

# we solve the ODEs for several values of distrusting population density in (0,1)
dens = space1

# gets the size of the arrays
p1 = len(dens)
p2 = len(a1)

# we define the function for the right-hand side of the equations we want to solve
@njit
def rhs(t, n, a1, max_lvl):
    
        # our model considers an infinite number of information levels, however, we need to define a cut-off for the maximum level we can consider when solving the equations in the computer
        # max_lvl is the cut-off level, it aggregates all information levels greater or equal to max_lvl
        # the array n is the vector that contains all subpopulations in the model
        # n = [s_{T,k} for k = {0,1,...,max_lvl}, i_{T,k}, r_{T,k}, s_{D,k}, i_{D,k}, r_{D,k}]
        # n  has size 6*max_lvl        
        
        # we solve the equations for the density of individuals s_{T/D,j} = S_{T/D,j}/N, i_{T/D,j} = I_{T/D,j}/N, etc... 
        N = 1
    
        # separates the classes to facilitate reading the subpopulations and defining the equations
        # current population in each subgroup
        sc = n[0:max_lvl]
        ic = n[max_lvl:(2*max_lvl)]
        rc = n[2*max_lvl:(3*max_lvl)]
        sd = n[3*max_lvl:(4*max_lvl)]
        ifd = n[4*max_lvl:5*max_lvl]
        rd = n[5*max_lvl:]
        
        # initiates arrays to store the value of each subpopulation in the next time step
        sol_sc = np.zeros(max_lvl)
        sol_ic = np.zeros(max_lvl)
        sol_rc = np.zeros(max_lvl)
        sol_sd = np.zeros(max_lvl)
        sol_ifd = np.zeros(max_lvl)
        sol_rd = np.zeros(max_lvl)
        
        # computes the effective transmission rate depending on the awareness of the infected, B
        B = 0 
        for i in range(0,max_lvl):
            B = B + (ic[i] + ifd[i])*(1-rho**i)*be
            
        # defines the eqs ruling the changes in population size according to the model
        # the first two awareness level (0 and 1) are computed separately as they are slightly different
        sol_sc[0] = 0 # susc. don't refresh to zero, there are no dynamics
        sol_sc[1] = -sc[1]*B*(1-rho)/N + a1*(sc[0] + ic[0] + rc[0] + sd[0] + ifd[0] + rd[0])*np.sum(sc[2:])/N - lamb*sc[1]
        sol_ic[0] = om*np.sum(ic[1:]) - lamb*ic[0] - rr*ic[0]
        sol_ic[1] = sc[1]*B*(1-rho)/N - rr*ic[1] + a1*(sc[0] + ic[0] + rc[0] + sd[0] + ifd[0] + rd[0])*np.sum(ic[2:])/N - lamb*ic[1] + lamb*ic[0] - om*ic[1]
        sol_rc[0] = rr*ic[0] - lamb*rc[0] # infected with info 0 can recover
        sol_rc[1] = rr*ic[1] + a1*(sc[0] + ic[0] + rc[0] + sd[0] + ifd[0] + rd[0])*np.sum(rc[2:])/N - lamb*rc[1] + lamb*rc[0]
        sol_sd[0] = 0 # susc. don't refresh to zero, there are no dynamics
        sol_sd[1] = -sd[1]*B*(1-rho)/N - a2*sd[1]*(N - sc[0] - ic[0] - rc[0] - sd[0] - ifd[0] - rd[0] - sc[1] - sd[1] - ic[1] - ifd[1] - rc[1] - rd[1])/N - lamb*sd[1]
        sol_ifd[0] = -rr*ifd[0] + om*np.sum(ifd[1:]) - a2*ifd[0]*(N - ic[0] - ifd[0] - rc[0] - rd[0])/N - lamb*ifd[0]
        sol_ifd[1] = -rr*ifd[1] + sd[1]*B*(1-rho)/N -a2*ifd[1]*(N - ic[0] - rc[0] - ifd[0] - rd[0] - sc[1] - sd[1] - ic[1] - ifd[1] - rc[1] - rd[1])/N - lamb*ifd[1] + lamb*ifd[0] - om*ifd[1]
        sol_rd[0] = rr*ifd[0] - a2*rd[0]*(N - ic[0] - rc[0] - ifd[0] - rd[0])/N - lamb*rd[0] # infected with info 0 can recover
        sol_rd[1] = rr*ifd[1] - a2*rd[1]*(N - ic[0] - rc[0] - ifd[0] - rd[0] - sc[1] - sd[1] - ic[1] - ifd[1] - rc[1] - rd[1])/N - lamb*rd[1] + lamb*rd[0]
        
        # defines some iterative sums of the information levels required for the equations
        # sum of all subpopulations with info quality < k-1. Here, k=1.
        tk2 = 0 
        # sums of trusting susceptibles with info quality > k, k = 1
        sck1 = np.sum(sc) - sc[0] - sc[1] 
        # sums of trusting infectious with info quality > k, k = 1
        ick1 = np.sum(ic) - ic[0] - ic[1]
        # sums of trusting recovered with info quality > k, k = 1
        rck1 = np.sum(rc) - rc[0] - rc[1]
        
        # sums of distrusting susceptibles with info quality < k-1, k = 1
        sdk2 = 0
        # sums of distrusting infectious with info quality < k-1, k = 1
        idk2 = 0
        # sums of distrusting recovered with info quality < k-1, k = 1
        rdk2 = 0
        
        for k in range(2,max_lvl-1): # the last level is defined separately again
            # sum of all subpopulations with info quality < k-1
            tk2 = tk2 + sc[k-2] + ic[k-2] + rc[k-2] + sd[k-2] + ifd[k-2] + rd[k-2]
            # defines the sum of all subpopoulations with info quality > k
            tk1 = N - tk2 - sc[k-1] - sd[k-1] - sc[k] - sd[k] - ic[k-1] - ic[k] - ifd[k-1] - ifd[k] - rc[k-1] - rc[k] - rd[k-1] - rd[k]
            # sums of trusting susceptibles with info quality > k
            sck1 = sck1 - sc[k]
            # sums of trusting infectious with info quality > k
            ick1 = ick1 - ic[k]
            # sums of trusting recovered with info quality > k
            rck1 = rck1 - rc[k]
            
            # sums of distrusting susceptibles with info quality < k-1
            sdk2 = sdk2 + sd[k-2]
            # sums of distrusting infectious with info quality < k-1
            idk2 = idk2 + ifd[k-2]
            # sums of distrusting recovered with info quality < k-1
            rdk2 = rdk2 + rd[k-2]
            
            # defines the eqs ruling the changes in population size according to the model
            sol_sc[k] = -sc[k]*B*(1-rho**k)/N - a1*sc[k]*tk2/N + a1*(sc[k-1] + ic[k-1] + rc[k-1] + sd[k-1] + ifd[k-1] + rd[k-1])*sck1/N - lamb*sc[k] + lamb*sc[k-1]
            sol_ic[k] = sc[k]*B*(1-rho**k)/N - rr*ic[k] - a1*ic[k]*tk2/N + a1*(sc[k-1] + ic[k-1] + rc[k-1] + sd[k-1] + ifd[k-1] + rd[k-1])*ick1/N - lamb*ic[k] + lamb*ic[k-1] - om*ic[k]
            sol_rc[k] = rr*ic[k] - a1*rc[k]*tk2/N + a1*(sc[k-1] + ic[k-1] + rc[k-1] + sd[k-1] + ifd[k-1] + rd[k-1])*rck1/N - lamb*rc[k] + lamb*rc[k-1]
            sol_sd[k] = -sd[k]*B*(1-rho**k)/N - a2*sd[k]*tk1/N + a2*(sc[k-1] + ic[k-1] + rc[k-1] + sd[k-1] + ifd[k-1] + rd[k-1])*sdk2/N - lamb*sd[k] + lamb*sd[k-1]
            sol_ifd[k] = sd[k]*B*(1-rho**k)/N  - rr*ifd[k] - a2*ifd[k]*tk1/N + a2*(sc[k-1] + ic[k-1] + rc[k-1] + sd[k-1] + ifd[k-1] + rd[k-1])*idk2/N - lamb*ifd[k] + lamb*ifd[k-1] - om*ifd[k]
            sol_rd[k] = rr*ifd[k] - a2*rd[k]*tk1/N + a2*(sc[k-1] + ic[k-1] + rc[k-1] + sd[k-1] + ifd[k-1] + rd[k-1])*rdk2/N - lamb*rd[k] + lamb*rd[k-1]
        
        # defines sums for the last information level
        tk2 = tk2 + sc[max_lvl-3] + ic[max_lvl-3] + rc[max_lvl-3] + sd[max_lvl-3] + ifd[max_lvl-3] + rd[max_lvl-3]
        tk1 = N - tk2 - sc[max_lvl-2] - sd[max_lvl-2] - sc[max_lvl-1] - sd[max_lvl-1] - ic[max_lvl-2] - ic[max_lvl-1] - ifd[max_lvl-2] - ifd[max_lvl-1] - rc[max_lvl-2] - rc[max_lvl-1] - rd[max_lvl-2] - rd[max_lvl-1]
        sck1 = sck1 - sc[max_lvl-1]
        ick1 = ick1 - ic[max_lvl-1]
        rck1 = rck1 - rc[max_lvl-1]
        sdk2 = sdk2 + sd[max_lvl-3]
        idk2 = idk2 + ifd[max_lvl-3]
        rdk2 = rdk2 + rd[max_lvl-3]
        k = max_lvl-1
        
        # defines the equations for the sum of the levels max_lvl and above (we must to stop somewhere) 
        sol_sc[k] = -sc[k]*B*(1-rho**k)/N - a1*sc[k]*tk2/N + a1*(sc[k-1] + ic[k-1] + rc[k-1] + sd[k-1] + ifd[k-1] + rd[k-1])*sck1/N + lamb*sc[k-1]
        sol_ic[k] = sc[k]*B*(1-rho**k)/N - rr*ic[k] - a1*ic[k]*tk2/N + a1*(sc[k-1] + ic[k-1] + rc[k-1] + sd[k-1] + ifd[k-1] + rd[k-1])*ick1/N + lamb*ic[k-1] - om*ic[k]
        sol_rc[k] = rr*ic[k] - a1*rc[k]*tk2/N + a1*(sc[k-1] + ic[k-1] + rc[k-1] + sd[k-1] + ifd[k-1] + rd[k-1])*rck1/N + lamb*rc[k-1]
        sol_sd[k] = a2*(sc[k] + sd[k] + ic[k] + ifd[k] + rc[k]+ rd[k])*(sdk2 + sd[k-1])/N - sd[k]*B*(1-rho**k)/N + a2*(sc[k-1] + ic[k-1] + rc[k-1] + sd[k-1] + ifd[k-1] + rd[k-1])*sdk2/N  + lamb*sd[k-1]
        sol_ifd[k] = a2*(sc[k] + sd[k] + ic[k] + ifd[k] + rc[k]+ rd[k])*(idk2 + ifd[k-1])/N + sd[k]*B*(1-rho**k)/N  - rr*ifd[k] + a2*(sc[k-1] + ic[k-1] + rc[k-1] + sd[k-1] + ifd[k-1] + rd[k-1])*idk2/N + lamb*ifd[k-1] - om*ifd[k]
        sol_rd[k] = a2*(sc[k] + sd[k] + ic[k] + ifd[k] + rc[k]+ rd[k])*(rdk2 + rd[k-1])/N + rr*ifd[k] + a2*(sc[k-1] + ic[k-1] + rc[k-1] + sd[k-1] + ifd[k-1] + rd[k-1])*rdk2/N + lamb*rd[k-1]
        
        # appends all arrays again to create a single one for solve_ivp to do its job
        solc = np.append(np.append(sol_sc,sol_ic),sol_rc)
        sold = np.append(np.append(sol_sd,sol_ifd),sol_rd)
        sol = np.append(solc,sold)
        return sol

# defines the task we want to parallelise
def solveq(dens, a1, ml):
   
   tm = 2000 # maximum time for which we solve the ODE system
   nst = 50001 # number of time points where we evaluate the solution

   # defines a tolerance for which we assume the epidemic is over
   tol1 = 1E-5
   
   # initialises the array containing the points where the solution is evaluated
   t_span = np.linspace(0,tm,nst)

   # initialises array for the initial condition
   s0 = np.zeros(6*ml)
   
   # S_{T,inf}
   s0[ml-1] = 1-dens-.5E-3
   # I_{T,inf}
   s0[2*ml-1] = .5E-3
   # S_{D,inf}
   s0[4*ml-1] = dens-.5E-3
   # I_{D,inf}
   s0[5*ml-1] = .5E-3
   
   # solves system of ODEs for the values of alpha_T and ml specified
   solv = solve_ivp(rhs, [0, tm], s0, method='RK45', t_eval=t_span, args=(a1,ml,))
   
   # defines the aggregated infected population (sums I_{T/D,k} over all information indices k)
   tot_i = np.sum(solv.y[ml:2*ml],axis=0) + np.sum(solv.y[4*ml:5*ml],axis=0)

   # finds the value of the peak of infection
   ip = np.max(np.sum(solv.y[ml:2*ml,:nst-1],axis=0) + np.sum(solv.y[4*ml:5*ml,:nst-1],axis=0))

   # saves the time when the peak occurred
   tp = t_span[np.where(tot_i==ip)[0][0]]
   
   # determines the instant when the total infected population size is smaller than the tolerance
   # this determines the instant when the epidemic ended
   try:
      end = np.where(tot_i<=tol1)[0][0]
   except:
      end = nst-1

   # saves the susceptible population size at the end of the epidemic
   fs = np.sum(solv.y[0:ml,end]) + np.sum(solv.y[3*ml:4*ml,end])

   return ip, tp, fs

# parallelises the tasks
def task(k):
    
    # we have p1*p2 tasks, hence, each task correspond to unique a and b integers such that k = p2*a + b
    # the values of alpha_T and density for this task are then dens[a] and a1[b]
    
    # finds the index for the density value corresponding to the task
    a = int(k/p2)
    
    # finds the index of the alpha_T value corresponding to the task
    b = k%p2
    
    # to save time and memory, we want to use the smallest ml such that (1 - rho**ml)^2 is sufficiently close to 1 since at the infinity level this quantity should be 1
    # we tested the dependence of the solution on the value of ml. For most cases, ml = 100 is sufficient to guarantee convergence of the solution
    ml = int(100)
    
    # however, if rho is close to 1, we need more levels to see convergence
    if(rho >= 0.94):
    	ml = int(300)
        
        # if rho is reeeeally close to 1, we need around 500 levels
    	if(rho >= 0.99): ml = int(500)
        
    # defines the task for the value of density and alpha_T specified
    rec = solveq(dens[a], a1[b], ml)
    
    return rec
        

if __name__ == '__main__':

    # initialises pool of tasks to parallelise
    p = mp.Pool()
   
    # defines the list of tasks, we want to solve the ODE system for each value of dens (p1 of them) and of alpha_T (p2 of them)
    lista = np.arange(0,p1*p2)
    
    # parallelises the tasks    
    res = p.map(task, lista)     
    
    # saves the outputs of each task
    df = pd.DataFrame(res)
    
    # initialises the arrays to save the size of the peak, the time when it occurred and the final susceptible population for each task
    ip = np.zeros((p1,p2))
    tp = np.zeros((p1,p2))
    fs = np.zeros((p1,p2))
    
    # save the corresponding values in the arrays
    for i in range(0,p1):
        ip[i,:] = df[0][p2*i:p2*i+p2]
        tp[i,:] = df[1][p2*i:p2*i+p2]
        fs[i,:] = df[2][p2*i:p2*i+p2]

    # saves the data
    np.save('./varying_density_ip.npy', ip)
    np.save('./varying_density_tp.npy', tp)
    np.save('./varying_density_fs.npy', fs)

    # Plots Figure 2
    SMALL_SIZE = 30
    matplotlib.rc('font', size=SMALL_SIZE)
    matplotlib.rc('axes', titlesize=SMALL_SIZE)
    
    fig,ax = plt.subplots(1,1,figsize=(15,12))
    
    plt.setp(ax.spines.values(), linewidth=2)
    plt.plot(dens,ip[:,0], linewidth=3, color='red',ms=15, marker='o',markevery=5,label=r'{0:.2f}'.format(a1[0]))
    plt.plot(dens,ip[:,1].transpose(), linewidth=3, color='darkorange',ms=15, marker='^', markevery=5,label=r'{0:.2f}'.format(a1[1]))
    plt.plot(dens,ip[:,2].transpose(), linewidth=3, color='gold',ms=15, marker='v', markevery=5,label=r'{0:.2f}'.format(a1[2]))
    plt.plot(dens,ip[:,3].transpose(), linewidth=3, color='royalblue',ms=15, marker='D', markevery=5,label=r'{0:.2f}'.format(a1[3]))
    plt.plot(dens,ip[:,4].transpose(), linewidth=3, color='darkblue',ms=15, marker='p', markevery=5,label=r'{0:.2f}'.format(a1[4]))
    plt.plot(dens,ip[:,5].transpose(), linewidth=3, color='purple', ms=12,marker='s', markevery=5,label=r'{0:.2f}'.format(a1[5]))
    ax.xaxis.set_tick_params(width=2)
    plt.xticks(rotation=20)
    ax.yaxis.set_tick_params(width=2)
    plt.xlabel(r'$d$')
    plt.ylabel('Pop. density')
    plt.legend(title=r'$\alpha_T$')
    plt.title('Total infections peak size')
    plt.savefig('Figure2b.pdf', format='pdf')
    plt.show()
    
    fig,ax = plt.subplots(1,1,figsize=(15,12))
    
    plt.setp(ax.spines.values(), linewidth=2)
    plt.plot(dens,fs[:,0], linewidth=3, color='red',ms=15, marker='o',markevery=5,label=r'{0:.2f}'.format(a1[0]))
    plt.plot(dens,fs[:,1].transpose(), linewidth=3, color='darkorange',ms=15, marker='^', markevery=5,label=r'{0:.2f}'.format(a1[1]))
    plt.plot(dens,fs[:,2].transpose(), linewidth=3, color='gold',ms=15, marker='v', markevery=5,label=r'{0:.2f}'.format(a1[2]))
    plt.plot(dens,fs[:,3].transpose(), linewidth=3, color='royalblue',ms=15, marker='D', markevery=5,label=r'{0:.2f}'.format(a1[3]))
    plt.plot(dens,fs[:,4].transpose(), linewidth=3, color='darkblue',ms=15, marker='p', markevery=5,label=r'{0:.2f}'.format(a1[4]))
    plt.plot(dens,fs[:,5].transpose(), linewidth=3, color='purple', ms=12,marker='s', markevery=5,label=r'{0:.2f}'.format(a1[5]))
    ax.xaxis.set_tick_params(width=2)
    plt.xticks(rotation=20)
    ax.yaxis.set_tick_params(width=2)
    plt.xlabel(r'$d$')
    plt.ylabel('Pop. density')
    plt.legend(title=r'$\alpha_T$')
    plt.title('Final susceptible population')
    plt.savefig('Figure2d.pdf', format='pdf')
    plt.show()