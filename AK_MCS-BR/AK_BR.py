# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:04:45 2020

@author: mmenz
"""

import numpy as np
import scipy as sci
from scipy import stats
import openturns as ot
import matplotlib.pyplot as plt
from gaussian_process import GaussianProcess
import time as t
from heat_transfer import Heat_transfer
from numpy import atleast_2d as as2d
from scipy.optimize import rosen
import time 
import scipy.sparse as spa
import scipy.sparse.linalg as spalinalg



class preconditionner():
    def __init__(self,x,prec):
        self.x=list()
        self.mat=list()
        self.x.append(x)
        self.mat.append(prec)
        
    def prec_add(self,x,prec):
        self.x.append(x)
        self.mat.append(prec)
    def ind_nearest(self,x_new):
        norm=list()
        for i in np.arange(len(self.x)):
            norm.append(np.linalg.norm(self.x[i]-x_new))
        norm = np.array(norm)
        ind = np.argmin(norm)
        return ind       





#function that runs AK_MCS algorithm (B. Echard et al Ak-MCS: An active learning reliability method combining Kriging and Monte Carlo Simulation)
#Inputs:
#performance function: callable defines such as g(x)<=0.0 defines the failure region
#rv_distribution: openturns object distribution
#MC_sample_size: size of the MC sample used for the evaluation of the probability of failure
#initial_doe_size: initial doe size for the construction of the kriging surrogate of the performance function 
#learning_function: string, choice betwen 'U' and 'EFF', defines the choice of the learning function
#cov_max: float, admissible coefficient of variation of Pf
def AK_MCS(performance_function,rv_distribution,initial_MC_sample_size = None,initial_doe_size = None, iter_max = 100, learning_function = 'U',cov_max = 0.1, hot_start = False,path='',DoE_init=False,corr='squared_exponential',epsilon_br=1e-3):
    pf_list=list()
    
    msh_file_name= 'mesh_thermique/thermique_mesh_v0.msh'
    my_study=Heat_transfer(msh_file_name,'sparse')
    nodes, elem, gamma_1,gamma_2,gamma_3, Surf1, Surf2 = my_study.read_msh() 
    my_study.assembly_convection()
    my_study.assembly_A_1()
    phi=None
    
    
    
    if hot_start == False:
        #initial MC population
        MC_sample_size = initial_MC_sample_size
        MC_sample = np.array(rv_distribution.getSample(MC_sample_size))
        lhs = ot.LHSExperiment(rv_distribution, initial_doe_size-1)
        lhs.setAlwaysShuffle(True) # randomized
        spaceFilling = ot.SpaceFillingC2()
        N = 50000
        optimalLHSAlgorithm = ot.MonteCarloLHS(lhs, N, spaceFilling)
        DoE=np.array(optimalLHSAlgorithm.generate())
               
        mid_point=np.array([790.,310.,75.,31.,900.,6.,293.,250.,40.]).reshape(1,9) # [T_allow,k_cu,k_ni,h_hot,T_hot,h_out,T_out,h_cool,T_cool]
        DoE_size = initial_doe_size
        DoE=np.array(DoE)
        DoE=np.concatenate((mid_point,DoE),axis=0)
        print "--------------------------------------"
        print "Evalulation of the perfomance function on the initial doe of size ",initial_doe_size
        print "--------------------------------------"
        #Evaluation of the performance function
        Y = np.zeros((initial_doe_size,1)) 

        T_allow= DoE[0,0]
        x1=DoE[0,1:].reshape(8,1)
        
        prec, b ,sol=my_study.solve_heatproblem(x1)
        Y[0,0]=T_allow-max(sol)
        phi=sol/np.linalg.norm(sol,2)
        phi=phi.reshape(phi.shape[0],1)
        my_study.assembly_proj(phi)
        lu_prec = spalinalg.splu(prec)
        if performance_function==temp_br_nearest : 
            lu_prec=preconditionner(DoE[0,:],lu_prec)
            print('ok')

        
        print(Y[0,0])
        i = 1
        for x in DoE[1:,:]:                  
            if performance_function==temp_max:
                Y[i,0]=performance_function(my_study,x)
            else :
                #Y[i,0] = performance_function(my_study,x)
                Y[i,0],phi,my_study = performance_function(my_study,x,phi,lu_prec,epsilon_br)
            print(Y[i,0])
            i = i+1
        print "--------------------------------------"
        print "Saving the initial DoE "
        print "--------------------------------------"
        MC_sample.dump(path+str('MC_sample_init.pk'))
        DoE.dump(path+str('DoE_init.pk'))
        Y.dump(path+str('Y_init.pk'))


        
    elif hot_start == 'init':
        MC_sample = np.load(path+str('MC_sample_init.pk'))
        MC_sample_size = MC_sample.shape[0] 
        initial_MC_sample_size =MC_sample_size 
        DoE = np.load(path+str('DoE_init.pk'))
        DoE_size = DoE.shape[0]
        Y = np.load(path+str('Y_init.pk'))
        print "--------------------------------------"
        print "Reading the initial DoE of size ",DoE_size
        print "--------------------------------------"
 

    elif hot_start == 'init_doe':
        MC_sample = np.load(path+str('MC_sample_init.pk'),allow_pickle=True)
        MC_sample_size = MC_sample.shape[0]   
        initial_MC_sample_size =MC_sample_size 
        DoE = np.load(path+str('DoE_init.pk'),allow_pickle=True)
        DoE_size = DoE.shape[0]
        Y = np.zeros((DoE_size,1))
        i = 0
        for x in DoE:
            Y[i,0] = performance_function(x)
            print(Y[i,0])
            i = i+1
        print "--------------------------------------"
        print "Reading the initial DoE of size ",DoE_size
        print        
        
    elif hot_start == 'doe_given':
        MC_sample = np.load(path+str('MC_sample_init.pk'),allow_pickle=True)
        MC_sample_size = MC_sample.shape[0]   
        initial_MC_sample_size =MC_sample_size 
        DoE = DoE_init
        DoE_size = DoE.shape[0]
        Y = np.zeros((DoE_size,1))
        i = 0
        for x in DoE:
            Y[i,0] = performance_function(x)
            print(Y[i,0])
            i = i+1
        print "--------------------------------------"
        print "Reading the initial DoE of size ",DoE_size
        print 
        
        
    elif hot_start == True:
        MC_sample = np.load(path+str('MC_sample.pk'))
        MC_sample_size = MC_sample.shape[0]  
        initial_MC_sample_size =MC_sample_size 
        DoE = np.load(path+str('DoE.pk'))
        DoE_size = DoE.shape[0]
        Y = np.load(path+str('Y.pk'))        
        print "--------------------------------------"
        print "Hot start with a DoE of size ",DoE_size
        print "--------------------------------------"        
        
    #construction of the kriging surrogate
    stochastic_dim = DoE.shape[1]    
    theta0 = np.array([0.1]*stochastic_dim)
    thetaL = np.array([1e-6]*stochastic_dim)
    thetaU = np.array([100.0]*stochastic_dim)
    gp_g = GaussianProcess(corr=corr,theta0 = theta0,thetaL=thetaL,thetaU=thetaU)     
    #AK-MCS loop
    n_iter = 0
    convergence = False
    while n_iter<iter_max and convergence == False:
        #fitting the kriging surrogate
        gp_g.fit(DoE,Y)
        #estimation of the performance function surrogate model at the MC points
        MC_sample_G,MSE = gp_g.predict(MC_sample,eval_MSE=True)
        
        #estimation of PF
        indicatrice = np.zeros((MC_sample_size,1))
        ind = MC_sample_G<=0.0
        indicatrice[ind] = 1.0
        Pf = indicatrice.sum()/MC_sample_size
        cov = np.sqrt((1.0-Pf)/(Pf*MC_sample_size))
        pf_list.append(Pf)
        filename='pf_algo_akmcs'+corr+'.pk'
        np.array(pf_list).dump(path+filename)   
        print "--------------------------------------"
        print "iteration number ",n_iter
        print "Pf = ",Pf
        print "cov =",cov
        #evaluation of the learning function and selection of the point to be add
        if learning_function == 'U':
            
            print "U learning function:"
            U = abs(MC_sample_G[:,0])/np.sqrt(MSE)
            crit_dist = False
            while crit_dist == False:
                ind_min = np.argmin(U)
                val_star = U[ind_min]
                x_star = MC_sample[ind_min]
                #x_star not in  DoE ?
                if np.sqrt(np.sum((DoE-x_star)**2,axis=1)).min()>=np.sqrt(np.sum((DoE-x_star)**2,axis=1)).max()/10**6:
                    crit_dist = True
                else:
                    crit_dist = False
                    U[ind_min] = 10**6
                
            print "x min =",x_star
            print "val min=", val_star
            if Pf != 0.0:
                if val_star>=2.0:
                    convergence = True
                    print "LEARNING PHASE CONVERGED"
                
        elif learning_function == 'EFF':
            print "EFF learning function:"
            epsilon = 2.0*np.sqrt(MSE)
            X = MC_sample_G[:,0]/np.sqrt(MSE)
            X1 = (-epsilon-MC_sample_G[:,0])/np.sqrt(MSE)
            X2 = (epsilon-MC_sample_G[:,0])/np.sqrt(MSE)
            EFF = MC_sample_G[:,0]*(2.0*sci.stats.norm.cdf(-X)-sci.stats.norm.cdf(X1)-sci.stats.norm.cdf(X2))-\
            np.sqrt(MSE)*(2.0*sci.stats.norm.pdf(-X)-sci.stats.norm.pdf(X1)-sci.stats.norm.pdf(X2))+\
            epsilon*(sci.stats.norm.cdf(X2)-sci.stats.norm.cdf(X1))
            crit_dist = False
            while crit_dist == False:
                ind_max = np.argmax(EFF)
                val_star = EFF[ind_max]
                x_star = MC_sample[ind_max]
                #x_star not in  DoE ?
                if np.sqrt(np.sum((DoE-x_star)**2,axis=1)).min()>=np.sqrt(np.sum((DoE-x_star)**2,axis=1)).max()/10**3:
                    crit_dist = True
                else:
                    crit_dist = False
                    EFF[ind_max] = -10**6
            print "x max =",x_star
            print "val max=", val_star
            if Pf != 0.0:
                if val_star<=0.001:
                    convergence = True
                    print "LEARNING PHASE CONVERGED"
                    
        if convergence == False:
            #evaluation of the point x_star and update the kriging
            
            if performance_function==temp_max:
                y_star=performance_function(my_study,x_star)
            else : 
                y_star, phi,my_study = performance_function(my_study,x_star,phi,lu_prec,epsilon_br)
            new_DoE = np.zeros((DoE_size+1,DoE.shape[1]))
            new_DoE[0:-1,:] = DoE
            new_DoE[-1,:] = x_star
            new_Y = np.zeros((DoE_size+1,1))
            new_Y[0:-1,:] = Y
            new_Y[-1,:] = y_star
            DoE = new_DoE.copy()
            Y = new_Y.copy()
            DoE_size = DoE_size + 1
        elif convergence == True:
            if cov <=cov_max:
                convergence = True
            else:
                convergence = False
                print "NEW MC SAMPLE"
                #add  MC samples
                new_MC_sample = np.zeros((MC_sample.shape[0]+initial_MC_sample_size,MC_sample.shape[1]))
                new_MC_sample[0:MC_sample.shape[0],:] = MC_sample
                new_MC_sample[MC_sample.shape[0]:,:] = np.array(rv_distribution.getSample(initial_MC_sample_size))
                MC_sample = new_MC_sample.copy()
                MC_sample_size = MC_sample_size + initial_MC_sample_size
        n_iter = n_iter + 1
        MC_sample.dump(path+str('MC_sample.pk'))
        DoE.dump(path+str('DoE.pk'))
        Y.dump(path+str('Y.pk'))
        print('MC size',MC_sample_size)
        
    return Pf,cov,DoE,n_iter,gp_g,phi





def temp_max(study,x) :
#    x1=x[1:].reshape(6,1)
    x1=x[1:].reshape(8,1)
    T_allow=x[0]
    A, b , sol= study.solve_heatproblem(x1)
    y=T_allow-max(sol)
    return y

  
        
def temp_br(study,x,phi,prec,epsilon): 
    x1=x[1:].reshape(8,1)
    T_allow=x[0]
    A, b=study.compute_A_b(x1)
    sol_br=study.solve_heatproblemr(x1)
    sol=phi.dot(sol_br)


    #residu sans preconditionnement 
#    res=np.linalg.norm(alpha-b,2)/np.linalg.norm(b,2)
    #residu preconditionne 
    #SPARSE
    alpha=A.dot(sol)
    b_prec = prec.solve(b)
    alpha_prec = prec.solve(alpha)

    res_prec=np.linalg.norm(alpha_prec.reshape(b_prec.shape[0],1)-b_prec.reshape(b_prec.shape[0],1),2)/np.linalg.norm(b_prec,2)
    print(res_prec)
    if res_prec> epsilon :  
         A,b,sol = study.solve_heatproblem(x1)
         phi_i=sol-phi.dot(np.transpose(phi).dot(sol))
         phi_i=phi_i/np.linalg.norm(phi_i,2)
         phi=np.concatenate((phi,phi_i.reshape(phi_i.shape[0],1)),axis=1)
         study.assembly_proj(phi)

    y=T_allow-max(sol)
    return y, phi,study

def temp_br_nearest(study,x,phi,precond,epsilon): 
    k_cu=310.
    h_out=6.
    x1=np.zeros((8,1))
    x1[0]=k_cu 
    x1[1:4]=x[1:4].reshape(3,1)
    x1[4]=h_out
    x1[5:]=x[4:].reshape(3,1)
    T_allow=x[0]
    A, b=study.compute_A_b(x1)

    sol_br=study.solve_heatproblemr(x1)
    sol=phi.dot(sol_br)


    #residu sans preconditionnement 
#    res=np.linalg.norm(alpha-b,2)/np.linalg.norm(b,2)
    #residu preconditionne 
    #SPARSE
    alpha=A.dot(sol)
    #choix du prec
    ind=precond.ind_nearest(x)
    prec=precond.mat[ind]
    b_prec = prec.solve(b)
    alpha_prec = prec.solve(alpha)

    res_prec=np.linalg.norm(alpha_prec.reshape(b_prec.shape[0],1)-b_prec.reshape(b_prec.shape[0],1),2)/np.linalg.norm(b_prec,2)



    if res_prec> epsilon :  
         A,b,sol = study.solve_heatproblem_near(x1)
         precond.prec_add(x,A)

         phi_i=sol-phi.dot(np.transpose(phi).dot(sol))
         phi_i=phi_i/np.linalg.norm(phi_i,2)
         phi=np.concatenate((phi,phi_i.reshape(phi_i.shape[0],1)),axis=1)
         study.assembly_proj(phi)


    y=T_allow-max(sol)
    return y, phi







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
Pf,cov,DoE,Y,n_iter,gp_g = AK_MCS(temp_br,distribution, initial_MC_sample_size = 20000,initial_doe_size = 12, cov_max=0.1,iter_max =200, learning_function = 'EFF',corr='matern52')
