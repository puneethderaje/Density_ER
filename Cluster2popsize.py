#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 00:57:11 2026

@author: puneeth
"""

import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize
from scipy.special import hyp2f1
from scipy.integrate import quad
import pickle
from concurrent.futures import ProcessPoolExecutor


def hyp2f1_pq(p,q,x):
    return hyp2f1(q,p,1+p,x)

def pest(x,bw,dw,bm,dm,R,gw,gm,N0): 
    rw = bw-dw 
    rm = bm-dm 
    Jt = (bm+dm)/rm * hyp2f1_pq(-rm/rw, -bm/bw, x) - bm/bw * x/(1-rm/rw) * hyp2f1_pq(1-rm/rw, 1-bm/bw,x) 
    Jt = Jt/  ( 1 -x )**(bm/bw)
    return 2*(1+Jt)**(-1)

def Pres(bw,dw,bm,dm,R,gw,gm,N0,mu) : 
    rw = bw-dw
    def integrand_alt(x): 
        return mu*R*pest(x,bw,dw,bm,dm,R,gw,gm,N0)/(gw*(1-x))    
    
    if R == np.inf: 
        Pres_density = 1-np.exp(mu*N0*(bm-dm)/(bm*(bw-dw)))
    else: 
        n0 = N0/R 
        Pres_density = 1- np.exp( - quad(integrand_alt, 0, n0/(n0-rw/gw))[0] )
    return Pres_density 

def model1(x, theta):
    return 1 - np.exp(-theta[0] * x)
def model2(x, theta):
    return 1 - np.exp(-theta[0] * x/(1+theta[1]*x) )


def neg_log_lik(model, theta, x, y, N):
    p = model(x,theta) 
    p = np.clip(p, 1e-12, 1-1e-12)
    
    ll = y * np.log(p) + (N - y) * np.log(1 - p)
    return -np.sum(ll)


def MLE_fit(Ndata_range, PRdata_range, model, theta0, bounds, NoR): 
    x = np.array(Ndata_range)
    N = np.array([NoR for xi in x])
    y = np.array([PR*NoR for PR in PRdata_range])

    res = minimize(
        lambda t: neg_log_lik(model, t, x, y, N),
        x0=theta0,
        bounds=bounds
    )

    theta = res.x
    
    PRpred_range = np.array([model(N, theta) for N in Ndata_range])
    
    ss_res = np.sum((PRdata_range - PRpred_range)**2)
    ss_tot = np.sum((PRdata_range - np.mean(PRdata_range))**2)
    r2 = 1 - ss_res/ss_tot
    
    
    return theta, r2, neg_log_lik(model, theta, x, y, N)

def alternate_test(N0range, PRdata, t0_init_range, t1_init_range, NoR):
    
    theta0 = [0.001]
    bounds = [(-1e9,None)]
    best_theta, best_nll1, best_r2_1 = None, np.inf, 0
    for t0_init in t0_init_range:
        theta0[0] = t0_init
        theta, r2_1, nll1 = MLE_fit(N0range, PRdata, model1, theta0, bounds, NoR)
        if nll1 < best_nll1:
            best_nll1 = nll1
            best_theta = theta
            best_r2_1 = r2_1
    theta1, r2_1, nll1 = best_theta, best_r2_1, best_nll1
    #theta, r2_1, nll1 = MLE_fit(N0range, PRdata, model1, theta0, bounds, NoR)
    #N_range = [ 10**x for x in np.arange(2,8,0.02) ] 
    #PR_range = [model1(N, theta1) for N in N_range]
    #ax[ind].plot(N_range, PR_range, color="C"+str(ind_R), alpha=0.5)

    theta0 = [0.001, 0]
    bounds = [(-1e9,None),(-1e-9,None)]
    best_theta2, best_nll2, best_r2_2 = None, np.inf, 0
    for t0_init in t0_init_range:
        theta0[0] = t0_init
        for t1_init in t1_init_range:
            theta0[1] = t1_init
            theta2, r2_2, nll2 = MLE_fit(N0range, PRdata, model2, theta0, bounds, NoR)
            if nll2 < best_nll2:
                best_nll2 = nll2
                best_theta2 = theta2
                best_r2_2 = r2_2
    theta2, r2_2, nll2 = best_theta2, best_r2_2, best_nll2

    #N_range = [ 10**x for x in np.arange(2,8,0.02) ] 
    #PR_range = [model2(N, theta1) for N in N_range]
    #ax[ind].plot(N_range, PR_range, color="C"+str(ind_R), linestyle = '--', alpha=0.5)

    LR = 2 * (-nll2 + nll1)
    p_value = chi2.sf(LR, df=1)
    
    return theta1, theta2, p_value, r2_1, r2_2, nll1, nll2

def saturated_test(N0range, PRdata, t0_init_range, NoR):
    theta0 = [0.001]
    bounds = [(-1e9, None)]
    
    best_theta, best_nll1, best_r2_1 = None, np.inf, 0
    
    for t0_init in t0_init_range:
        theta0[0] = t0_init
        theta, r2_1, nll1 = MLE_fit(N0range, PRdata, model1, theta0, bounds, NoR)
        
        if nll1 < best_nll1:
            best_nll1 = nll1
            best_theta = theta
            best_r2_1 = r2_1

    theta1, nll1, r2_1 = best_theta, best_nll1, best_r2_1

    x = np.array(N0range)
    N = np.array([NoR for _ in x])
    y = np.array([PR * NoR for PR in PRdata])

    p_sat = y / N
    p_sat = np.clip(p_sat, 1e-12, 1 - 1e-12)

    ll_sat = y * np.log(p_sat) + (N - y) * np.log(1 - p_sat)
    nll_sat = -np.sum(ll_sat)

    LR = 2 * ( -nll1 + nll_sat )

    df = len(N0range) - 1 
    p_value = chi2.sf(LR, df=df)

    return theta1, p_value, r2_1, nll1, nll_sat
    

def power_analysis_alternate(bw,dw,bm,dm,Rrange,gw,gm,N0range,mu,NoR,NoT): 
    plist = []
    for trial in range(NoT): 
        PRdata = [ np.random.binomial(NoR,Pres(bw,dw,bm,dm,Rrange[ind],gw,gm,N0range[ind],mu))/NoR for ind in range(len(N0range)) ]
        
        t0_init_range = [10**k for k in range(-9,-1,2)]
        t1_init_range = np.arange(0, 1.01, 0.2)
        theta1, theta2, p_value, r2_1, r2_2, nll1, nll2 = alternate_test(N0range, PRdata, t0_init_range, t1_init_range, NoR)
        plist += [p_value]
    
    return plist

def power_analysis_saturated(bw,dw,bm,dm,Rrange,gw,gm,N0range,mu,NoR,NoT): 
    plist = []
    for trial in range(NoT): 
        PRdata = [ np.random.binomial(NoR,Pres(bw,dw,bm,dm,Rrange[ind],gw,gm,N0range[ind],mu))/NoR for ind in range(len(N0range)) ]
        
        t0_init_range = [10**k for k in range(-9,-1,2)]
        t1_init_range = np.arange(0, 1.01, 0.2)
        theta1, p_value, r2_1, nll1, nll_sat = saturated_test(N0range, PRdata, t0_init_range, NoR)
        plist += [p_value]
    
    return plist

def compute_cell1(args):
    N1exp, N2exp, NoR, R, bw, dw, bm, dm, gw, gm, mu, NoT = args
    
    N0range = [round(10**N1exp), round(10**N2exp)]
    Rrange = [R for _ in N0range]
    
    plist = power_analysis_alternate(
        bw, dw, bm, dm, Rrange, gw, gm, N0range, mu, NoR, NoT
    )
    plist = np.array(plist)
    
    return len(plist[plist < 0.05]) / NoT

bw = 0.5 - 0.01 #np.log(0.95)/2
bm = 0.5 + 0.01 #np.log(1 + sm)/2
dw = 0.5 + 0.01 #np.log(0.95)/2
dm = 0.5 - 0.01 #np.log(1+ sm)/2
mu = 10**(-6)
gw = 0.01
gm = 0.01



N1exprange = np.arange(1,8,0.25)
N2exprange = np.arange(1,8,0.25) 
NoT = 200
NoR = 60

data_N1vsN2 = {}
for R in [10**6, 10**5]:
    detection_probability = []
    for N1exp in N1exprange:
        print(N1exp)
        args_list = [
            (N1exp, N2exp, NoR, R, bw, dw, bm, dm, gw, gm, mu, NoT)
            for N2exp in N2exprange
        ]
        with ProcessPoolExecutor(max_workers=None) as executor:
            row = list(executor.map(compute_cell1, args_list))
        detection_probability.append(row)
    data_N1vsN2[R] = detection_probability

with open("data_NoNvsNoR.pkl", "wb") as f:
    pickle.dump(data_N1vsN2, f)