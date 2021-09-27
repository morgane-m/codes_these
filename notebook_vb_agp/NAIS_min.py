# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:43:08 2020

@author: mchiron
"""


import numpy as np
import scipy as sc
from scipy import stats
#
#d=2 #change la dimension là 
#f=stats.multivariate_normal(mean=np.zeros(d), cov=np.eye(d))
np.random.seed



def Poids(f,g,X,f_type):
    if g==f:
        return np.ones(len(X))
    else :
        if f_type :
            f_valeur=f.pdf(X)
        else : 
            f_valeur=f.pdf(X.T)
        g_valeur=g.pdf(X.T) #attention on doit utiliser la transposée
        return f_valeur/g_valeur

def Indic(Y,gamma): 
    return Y< gamma

def Norme_poids(k,Y,gamma,N,fraction): 
    E_P=Indic(Y,gamma)*fraction
    return E_P


def Densite(X,poids):
    #on fait tout d'un coup
    #X doit etre mi sous une autre forme (transposé)
    values = X.T
    kernel=stats.gaussian_kde(values,'silverman',poids)
    return kernel
        
      
def Boucle_k_d(k,rho,S,f,N,phi,g,population_X,population_Y,fraction,f_type):
    #step 2 : on met tout à jour
    nouvelle_pop=g[k-1].resample(N).T
    population_X=np.concatenate((population_X,nouvelle_pop),axis=0)#kernel resample est un array
    population_Y=np.concatenate((population_Y,phi(nouvelle_pop)),axis=0) #faut que soit un array bon sens
    frac=np.concatenate((fraction,Poids(f,g[k-1],nouvelle_pop,f_type)),axis=0)
    #step 3
    rho_quantil = stats.scoreatpercentile(population_Y[(k-1)*N:k*N],rho)
    gamma = max (S, rho_quantil)
    #step 4 
    if gamma != S:
        poids=Norme_poids(k,population_Y,gamma,N,frac)
        #step 5
        kernel=Densite(population_X,poids)
        g.append(kernel)
    return gamma,population_X,population_Y,frac,g
    
    
def NAIS(rho,S,f,N,phi,iter_max=60,f_type=True): #lancer ça pour l'avoir 1 fois
    #step 1
    k=1
    #step 2a
    if f_type : 
        population_X =f.rvs(N)
    else : 
        population_X=f.resample(N).T
    population_Y = phi(population_X)#il faut donc que ce soit sous forme d'array là
    #step 3a
    rho_quantil = stats.scoreatpercentile(population_Y,rho)
    gamma = max (S, rho_quantil)
    g=[f] #LISTE
    fraction=Poids(f,g[0],population_X,f_type)
    #step 4a
    poids=Norme_poids(k,population_Y,gamma,N,fraction)
    #-step 5a
    kernel=Densite(population_X,poids)
    g.append(kernel)
    alpha=1.
    eps_alpha=0.01
    #Step 6a
    while gamma > S and k < iter_max and alpha > eps_alpha : 
        k=k+1
        alpha=gamma
        Boucle=Boucle_k_d(k,rho,S,f,N,phi,g,population_X,population_Y,fraction,f_type)
        gamma=Boucle[0]
        alpha=(alpha-gamma)/alpha
        population_X=Boucle[1]
        population_Y=Boucle[2]
        fraction=Boucle[3]
        g=Boucle[4]
        print(gamma)
    var_P=np.var(fraction[(k-1)*N:k*N]*Indic(population_Y[(k-1)*N:k*N],S))/N
    P=np.sum(fraction[(k-1)*N:k*N]*Indic(population_Y[(k-1)*N:k*N],S))/N
    

    return P, g ,var_P, population_X

# test NAIS
##
#def f2(X):
#    U1=3+0.1*((X[:,0]-X[:,1])**2)-((X[:,0]+X[:,1])/np.sqrt(2))
#    U2=3+0.1*((X[:,0]-X[:,1])**2)+((X[:,0]+X[:,1])/np.sqrt(2))
#     
#    U3=X[:,0]-X[:,1]+(6/np.sqrt(2))
#    U4=X[:,1]-X[:,0]+(6/np.sqrt(2))
##    U5=X[:,0]-X[:,1]+(7/np.sqrt(2))*np.ones(n).reshape(n,1)
##    U6=X[:,1]-X[:,0]+(7/np.sqrt(2))*np.ones(n).reshape(n,1)
#    Z=np.minimum(U1,U2)
#    Z=np.minimum(Z,np.array(U3))
#    Z=np.minimum(Z,np.array(U4))
##    Z=np.minimum(Z,np.array(U5))
##    Z=np.minimum(Z,np.array(U6))
#    return Z 
#
###
#distribution_nonlin_osci=stats.multivariate_normal(mean=[1.,0.1,1.,0.5,1.,1.], cov=np.diag([0.01,0.0001,0.0025,0.0025,0.04,0.04]))
#distribution_nonlin_osci_verylow_IS=stats.multivariate_normal(mean=[1.,0.1,1.,0.5,1.,0.45], cov=np.diag([0.01,0.0001,0.0025,0.0025,0.04,0.075**2]))
#distribution_nonlin_osci_low_IS=stats.multivariate_normal(mean=[1.,0.1,1.,0.5,1.,0.6], cov=np.diag([0.01,0.0001,0.0025,0.0025,0.04,0.1**2]))   
#def nonlin_osci_mc(X):
#    # X = c1 c2 m r t1 f1 
#    w0=np.sqrt((X[:,0]+X[:,1])/X[:,2])
#    Z=3*X[:,3]-abs(2*X[:,5]*np.sin(0.5*X[:,4]*w0)/(X[:,2]*w0**2))
#    return Z 
#
#stochastic_dim=2
#distribution_f2=stats.multivariate_normal(mean=np.zeros(stochastic_dim), cov=np.eye(stochastic_dim))
#n_test=1
#Pf=np.zeros(n_test)
#var_P=np.zeros(n_test)
#size=np.zeros(n_test)
#for i in np.arange(n_test):
#    Pf, f_aux, var_P, population_X =NAIS(rho=10,S=0.,f=distribution_nonlin_osci_low_IS,N=30000,phi=nonlin_osci_mc,iter_max=50)