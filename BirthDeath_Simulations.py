#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 20:46:48 2026

@author: puneeth
"""

import numpy as np 
import pandas as pd 
import multiprocessing as mp 

def birthdeath(nw0, nm0, bw, bm, dw, dm, mu, K, timeseries = False):
    """
    Gillespie algorithm for two types (nw, nm) with:
        Birthw: nw → nw + 1 at rate bw * (1 - (nw+nm)/K) * nw
        Birthm: nm → nm + 1 at rate bm * (1 - (nw+nm)/K) * nm
        Deathw: nw → nw - 1 at rate dw * nw
        Deathm: nm → nm - 1 at rate dm * nm

    Stops when:
        1. nw + nm = 0   (extinction)
        2. nw + nm ≥ Ncap   (rescued)
    """

    np.random.seed()

    t = 0.0
    nw = nw0
    nm = nm0

    if timeseries :
        times = [t]
        trajw = [nw]
        trajm = [nm]
    
    Ext = 0 
    while True:

        N = nw + nm

        if N == 0 :
            Ext = 1
            break
        
        if nm >= 1000:
            Ext = 0
            break
        
        if K == np.inf : 
            growthfactor = 1 
        else : 
            Kparam = K/(1-dm/bm)
            growthfactor = max(0.0, 1 - N / K)

        a1 = bw * growthfactor * nw        # W birth
        a2 = bm * growthfactor * nm         # M birth
        a3 = dw * nw                    # W death
        a4 = dm * nm                    # M death
        a5 = mu * nw
        
        a = np.array([a1, a2, a3, a4, a5])
        a0 = a.sum()

        if a0 <= 0:
            break

        # Time step
        dt = np.random.exponential(1 / a0)
        t += dt

        # Select event
        r = np.random.rand() * a0

        if r < a1:
            nw += 1
        elif r < a1 + a2:
            nm += 1
        elif r < a1 + a2 + a3:
            nw -= 1
        elif r < a1 + a2 + a3 + a4 :
            nm -= 1
        else : 
            nm += 1 
            nw -= 1 
            
        if timeseries : 
            times.append(t)
            trajw.append(nw)
            trajm.append(nm)

    return Ext 

NoR = 10

Krng = [ 10**4, np.inf ] 
N0rng = [round(10**x) for x in np.arange(1,5.2,0.2)]
bw = 0.5 + np.log(0.95)/2
bm = 0.5 + np.log(1.5)/2
dw = 0.5 - np.log(0.95)/2
dm = 0.5 - np.log(1.5)/2
mu = 10**(-4)


a = mp.cpu_count() - 2
print(a)

outputwriter = pd.ExcelWriter('PR_PopulationSize.xlsx')


# Example usage
ExtProb = [] 
for K in Krng : 
    ExtProbrow = [K]
    for N0 in N0rng: 
        print(K,N0, flush = True)
        
        pool = mp.Pool(a)
        results = pool.starmap(birthdeath, [(N0, 0, bw, bm, dw, dm, mu, K) for rep in range(NoR)])
        pool.close()
        pool.join()
        Ext = np.mean(results)
        ExtProbrow = ExtProbrow + [Ext]
    ExtProb = ExtProb + [ ExtProbrow ]
    
DF = pd.DataFrame(ExtProb,columns = ['CarryingCapacity'] + list(N0rng))
DF.to_excel(outputwriter,sheet_name = 'DensityDependence') 
    
outputwriter.close()
