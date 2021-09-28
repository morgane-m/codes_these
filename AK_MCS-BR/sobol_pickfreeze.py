# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:17:47 2020

@author: mmenz
"""

import numpy as np
import scipy as sc
from scipy import stats
import openturns as ot
from openturns.viewer import View
import matplotlib.pyplot as plt
from gaussian_process import GaussianProcess
import os
import subprocess 
import time as t
from numpy import atleast_2d as as2d
from scipy.optimize import rosen
import time 
import scipy.sparse as spa
import scipy.sparse.linalg as spalinalg

DoE=np.load('DoE.pk',allow_pickle=True)
Y=np.load('Y.pk',allow_pickle=True)

stochastic_dim = DoE.shape[1]    
theta0 = np.array([0.1]*stochastic_dim)
thetaL = np.array([1e-6]*stochastic_dim)
thetaU = np.array([100.0]*stochastic_dim)
gp_g = GaussianProcess(corr='matern52',theta0 = theta0,thetaL=thetaL,thetaU=thetaU)   
gp_g.fit(DoE,Y)
Pf=0.00614



k_ni=ot.Normal(75,75*0.02) 
k_cu=ot.Normal(310,310*0.02)
T_hot=ot.Uniform(900*0.9,900*1.1)
h_hot=ot.Uniform(31*0.9,31*1.1)
T_cool=ot.Uniform(40*0.95,40*1.05)
h_cool=ot.Uniform(250*0.9,250*1.1)
T_out=ot.Uniform(293*0.95,293*1.05)
h_out=ot.Uniform(6*0.95,6*1.05)
T_allow=ot.Uniform(730*(1-0.075),730*1.075) 
Lois_Marginales = [T_allow,k_cu,k_ni,h_hot,T_hot,h_out,T_out,h_cool,T_cool]
distribution=ot.ComposedDistribution(Lois_Marginales)


MC_size=800000
MC1 = np.array(distribution.getSample(MC_size))
MC2 = np.array(distribution.getSample(MC_size))
MC_sample_G1= gp_g.predict(MC1)
ind1 = MC_sample_G1 <=0
ind_f1=[j for j, x in enumerate(ind1) if x]
MC_f1=MC1[ind_f1,:]
indicatrice1=np.zeros((MC_size,1))
indicatrice1[ind_f1]=1.
d=9

### indices Sobol ordre 1 
#s_i=list()
#for i in np.arange(d) :
#    MC_new = np.copy(MC2)
#    MC_new [:,i] = MC1 [:,i]
#    MC_sample_G2= gp_g.predict(MC_new)
#    ind2 = MC_sample_G2 <=0
#    ind_f2=[j for j, x in enumerate(ind2) if x]
#    MC_f2=MC2[ind_f2,:]
#    indicatrice2=np.zeros((MC_size,1))
#    indicatrice2[ind_f2]=1.
#    
#    term1 = np.sum(indicatrice1*indicatrice2)/MC_size
#    term2 = np.sum(indicatrice1)*np.sum(indicatrice2)/(MC_size**2)
#    term3 =  np.sum(indicatrice1**2)/MC_size
#    term4 =  (np.sum(indicatrice1)/MC_size)**2
#    s_i.append((term1-term2)/(term3-term4))
    

########## indices Sobol totaux
s_t=list() 
for i in np.arange(d) : 
    MC_new = np.copy(MC1)
    MC_new [:,i] = MC2 [:,i]
    MC_sample_G2= gp_g.predict(MC_new)
    ind2 = MC_sample_G2 <=0
    ind_f2=[j for j, x in enumerate(ind2) if x]
    MC_f2=MC2[ind_f2,:]
    indicatrice2=np.zeros((MC_size,1))
    indicatrice2[ind_f2]=1.
    
    term1 = np.sum(indicatrice1*indicatrice2)/MC_size
    term2 = np.sum(indicatrice1)*np.sum(indicatrice2)/(MC_size**2)
    term3 =  np.sum(indicatrice1**2)/MC_size
    term4 =  (np.sum(indicatrice1)/MC_size)**2
    s_t.append(1.-(term1-term2)/(term3-term4))
    
    
    
    