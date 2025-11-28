#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 14:20:48 2025

@author: puneeth
"""

import numpy as np 
import pandas as pd 
import multiprocessing as mp 



def Hermaphroditic(r,lw,lm,M,b=0,N0=10**4,mu=10**(-4),sgv=0, ID_self=0, ps=0, timeseries=False):
    """
    Simulates the eco-evolutionary dynamics of a population of bisexual (hermaphrodites) until either the population goes
    extinct or reaches twice the starting population size.
    
    Parameters
    ----------
    r : float (positive)
        The mate-searching efficiency
    lw : float (positive)
        The fecundity of the wildtype, i.e. the mean number of offsprings produced conditional on finding a mate.  
    lm : float (positive)
        The fecundity of the mutant, i.e. the mean number of offsprings produced conditional on finding a mate.  
    M : float(postive)
        Constant in the Beverton-Holt model 
    b : float (positive)
        Constant in the Beverton-Holt model. b=0 corresponds to no density dependence
    N0 : int
        The starting total population size. 
    sgv : float (between 0 and 1)
        The percentage of the starting population size that is mutant. 
    sr : float (between 0 and 1)
        The proportion of offsprings that are hermaphrodites. 
    ps : float (between 0 and 1)
        The probability of selfing. 
    ID_self : float (between 0 and 1)
        The coefficient of inbreeding depression, i.e., the fecundity of a selfed individual is reduced by a factor of 1-ID_self
        
    Returns
    -------
    Ext : int (0 or 1)
        Is 0 is the population survives and 1 if the population went extinct
    Xseries : list of lists
        The timeseries data of Xm and Xw. 
    T : int (postive)
        The time to first mutation in generations. 
    """
    
    np.random.seed()
    Xm = round(sgv*N0)
    Xw = N0-Xm
    T = 0 #Time to first Mutation
    DidMutationOccur = False
    Ext = 0 
    NoG = 0
    if timeseries : 
        Xseries = [ [Xw,Xm] ]
    else:
        Xseries = []
    while True :
        NoG += 1 
        if Xm == 0 and DidMutationOccur == False: 
            T += 1
        if Xm > 0 : 
            DidMutationOccur = True
        X = Xm + Xw
        if X == 0 : 
            Ext = 1 
            if DidMutationOccur == False : 
                T = 0
            break
        elif X > max(2*N0,2*10**4) or ( NoG > 1000 and X > 4000 and Xm/X > 0.6 ) : 
            break 
        if r == 'AM' : 
            [Xw_Mate,Xw_Self,Xw_NoMate] = [Xw,0,0]
            [Xm_Mate,Xm_Self,Xm_NoMate] = [Xm,0,0]
        else: 
            [Xw_Mate,Xw_Self,Xw_NoMate] = np.random.multinomial(Xw, [1 - np.exp(-r*(X-1) ), ps*np.exp(-r*(X-1)), (1-ps)*np.exp(-r*(X-1))  ])
            [Xm_Mate,Xm_Self,Xm_NoMate] = np.random.multinomial(Xm, [1 - np.exp(-r*(X-1) ), ps*np.exp(-r*(X-1)), (1-ps)*np.exp(-r*(X-1))  ])
            
        Xww = 0 
        Xwm = 0
        Xmw = 0
        Xmm = 0 
        
        if X - 1 > 0 : 
            if Xw_Mate > 0 :
                [Xww,Xwm] = np.random.multinomial(Xw_Mate, [(Xw - 1)/(X-1),Xm/(X-1)])
            if Xm_Mate > 0 :
                [Xmw,Xmm] = np.random.multinomial(Xm_Mate, [Xw/(X-1),(Xm - 1)/(X-1)])
        
        lamda_w_eff = (Xww + 0.5*Xwm)*lw + 0.5*Xmw*lm + Xw_Self*(1 - ID_self)*lw
        lamda_w_eff = lamda_w_eff/((1 + X/M)**b)
        
        lamda_m_eff = (Xmm + 0.5*Xmw)*lm + 0.5*Xwm*lw + Xm_Self*(1 - ID_self)*lm
        lamda_m_eff = lamda_m_eff/((1 + X/M)**b)
        
        
        Xw = np.random.poisson((1-mu)*lamda_w_eff)
        Xm = np.random.poisson(lamda_m_eff + mu*lamda_w_eff)
        if timeseries: 
            Xseries = Xseries + [ [Xw, Xm] ]
    if timeseries: 
        [Ext, Xseries]
    return Ext

NoR = 10**(4)
M = 10**4
lw = 0.95
lm = 1.5

N0_rng = [10**x for x in np.arange(1,7.2,0.2)]


r_rng = ['AM', 0.01]
b_rng = [0, 1]
output_writer = pd.ExcelWriter('PR_PopulationSize.xlsx')

a = mp.cpu_count() - 2
print(a)


ExtProb = [] 
for r in r_rng :    
    for b in b_rng : 
        ExtProb_row = [r,b]
        for N0 in N0_rng: 
            print(r,b,N0, flush = True)
            
            pool = mp.Pool(a)
            results = pool.starmap(Hermaphroditic, [(r,lw,lm,M,b,N0) for rep in range(NoR)])
            pool.close()
            pool.join()
            Ext = np.mean(results)
            ExtProb_row = ExtProb_row + [Ext]
        ExtProb = ExtProb + [ ExtProb_row ]
        
    DF = pd.DataFrame(ExtProb,columns = ['Mate-finding', 'Competition'] + list(N0_rng))
    DF.to_excel(output_writer,sheet_name = 'Density_Dependence') 
    
output_writer.close()


