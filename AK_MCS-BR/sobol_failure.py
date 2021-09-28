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

MC_size=700000
MC = np.array(distribution.getSample(MC_size))
MC_sample_G= gp_g.predict(MC)
ind = MC_sample_G <=0
ind_f=[i for i, x in enumerate(ind) if x]
MC_f=MC[ind_f,:]


d=9 
f_xi_f=list()
f_xi_f_valeur=list()
f_xi_valeur=list()
v=list()
print('ok')



for i in np.arange(d) : 
    values = MC_f[:,i].T
    kernel=stats.gaussian_kde(values)
    f_xi_f.append(kernel)
    f_xi_f_valeur.append(kernel.pdf(MC[:,i].T))
     
    f_xi_valeur_i=np.array(Lois_Marginales[i].computePDF(MC[:,i].reshape(MC_size,1)))
    f_xi_valeur.append(f_xi_valeur_i)
    
    v_i = np.var(f_xi_f_valeur[i].reshape(MC_size,1)/f_xi_valeur[i].reshape(MC_size,1))
    v.append(v_i)
    
c= Pf/(1.-Pf)
s=np.array(v)*c