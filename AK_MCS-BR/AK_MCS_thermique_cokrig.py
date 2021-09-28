from heat_transfer import Heat_transfer
import scipy as sci
import openturns as ot
from gaussian_process import GaussianProcess
import os
import subprocess 
from numpy import atleast_2d as as2d
from scipy.optimize import rosen
from smt.extensions.mfk import MFK
from mfk.nested_doe import create_nested_doe
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
params = {'legend.fontsize': 32,
          'legend.handlelength': 2}
plt.rcParams.update(params)
import scipy.sparse as spa
import scipy.sparse.linalg as spalinalg

#function that runs AK_MCS algorithm (B. Echard et al Ak-MCS: An active learning reliability method combining Kriging and Monte Carlo Simulation)
#Inputs:
#performance function: callable defines such as g(x)<=0.0 defines the failure region
#rv_distribution: openturns object distribution
#MC_sample_size: size of the MC sample used for the evaluation of the probability of failure
#initial_doe_size: initial doe size for the construction of the kriging surrogate of the performance function 
#learning_function: string, choice betwen 'U' and 'EFF', defines the choice of the learning function
#cov_max: float, admissible coefficient of variation of Pf
def AK_MCS(performance_function, rv_distribution,initial_MC_sample_size = None,initial_doe_size = None, iter_max = 100, learning_function = 'U',cov_max = 0.1, hot_start = False,path='',epsilon=1e-3):
    # load thermal problem 
    msh_file_name= 'mesh_thermique/thermique_mesh_v0.msh'
    my_study=Heat_transfer(msh_file_name,'sparse')
    nodes, elem, gamma_1,gamma_2,gamma_3, Surf1, Surf2 = my_study.read_msh() 
    my_study.assembly_convection()
    my_study.assembly_A_1()
    
    if hot_start == False:
        #initial MC population
        MC_sample_size = initial_MC_sample_size
        MC_sample = np.array(rv_distribution.getSample(MC_sample_size))
        ind = np.random.randint(0,MC_sample_size,initial_doe_size)
        DoE = MC_sample[ind]
        DoE=ot.LHSExperiment( rv_distribution,initial_doe_size-1).generate()

        mid_point=np.array([790.,310.,75.,31.,900.,6.,293.,250.,40.]).reshape(1,9) # [T_allow,k_cu,k_ni,h_hot,T_hot,h_out,T_out,h_cool,T_cool]
#        MC_sample = np.load(path+str('MC_sample.pk'),allow_pickle =True)
#        MC_sample_size = MC_sample.shape[0]    
#        DoE=np.load(path+str('DoE_i.pk'),allow_pickle =True)
        DoE_size = initial_doe_size
        DoE=np.array(DoE)
        DoE=np.concatenate((mid_point,DoE),axis=0)
        print "--------------------------------------"
        print "Evalulation of the perfomance function on the initial doe of size ",initial_doe_size
        print "--------------------------------------"
        #Evaluation of the performance function
        x1=DoE[0,1:].reshape(8,1)
        T_allow= DoE[0,0]
        prec, b ,sol=my_study.solve_heatproblem(x1)
        prec = spalinalg.splu(prec)
        y=T_allow-max(sol)
        y=np.array([y])
        phi=sol/np.linalg.norm(sol,2)
        phi=phi.reshape(phi.shape[0],1)
        my_study.assembly_proj(phi)
        ye=y
        DoEe=DoE[0,:].reshape(1,DoE.shape[1])
        yc=y
        DoEc=DoE
        y_gp=y
        print(y_gp)
        for x in DoE[1:,:]:
            y,y_br,phi= performance_function(my_study,x,phi,prec,epsilon)
            new_yg = np.zeros((y_gp.shape[0]+1,1))
            new_yg[0:-1,:] = y_gp
                
               
                
            if y == False : 
                new_y = np.zeros((yc.shape[0]+1,1))
                new_y[0:-1,:] = yc
                new_y[-1,:] = y_br
                yc = new_y.copy()
                new_yg[-1,:] = y_br
                
            else : 
                new_y = np.zeros((yc.shape[0]+1,1))
                new_y[0:-1,:] = yc
                new_y[-1,:] = y_br
                yc = new_y.copy()  
                new_y = np.zeros((ye.shape[0]+1,1))
                new_y[0:-1,:] = ye
                new_y[-1,:] = y
                ye = new_y.copy()  
                new_DoE = np.zeros((DoEe.shape[0]+1,DoEe.shape[1]))
                new_DoE[0:-1,:] = DoEe
                new_DoE[-1,:] = x
                DoEe = new_DoE.copy()  
                new_yg[-1,:] = y
                
            y_gp = new_yg.copy()
            
            
            
     
        
    #construction of the kriging surrogate
    stochastic_dim = DoE.shape[1]    
    theta0 = np.array([0.1]*stochastic_dim)
    thetaL = np.array([1e-6]*stochastic_dim)
    thetaU = np.array([100.0]*stochastic_dim)
    gp_g = GaussianProcess(theta0 = theta0,thetaL=thetaL,thetaU=thetaU)     
#    sm = MFK(theta0=np.array(stochastic_dim*[1.]), print_global = False)
    #AK-MCS loop
    n_iter = 0
    convergence = False
    while n_iter<iter_max and convergence == False:
        #fitting the kriging surrogate
        gp_g.fit(DoEc,y_gp)
        MC_sample_G,MSE = gp_g.predict(MC_sample,eval_MSE=True)

    
    
#        sm.set_training_values(DoEc, yc, name = 0)
#        sm.set_training_values(DoEe, ye)
#        sm.train()
#        #estimation of the performance function surrogate model at the MC points
#        
#        MC_sample_G = sm.predict_values(MC_sample)
#        MSE = sm.predict_variances(MC_sample)
#        
        
        
        
        
        
        
        #estimation of PF
        indicatrice = np.zeros((MC_sample_size,1))
        ind = MC_sample_G<=0.0
        indicatrice[ind] = 1.0
        Pf = indicatrice.sum()/MC_sample_size
        cov = np.sqrt((1.0-Pf)/(Pf*MC_sample_size))
        print "--------------------------------------"
        print "iteration number ",n_iter
        print "Pf = ",Pf
        print "cov =",cov
        print "taille de la BR =", phi.shape[1]
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
                if np.sqrt(np.sum((DoEc-x_star)**2,axis=1)).min()>=np.sqrt(np.sum((DoEc-x_star)**2,axis=1)).max()/10**6:
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
            epsilon = 2.0*MSE
            X = MC_sample_G[:,0]/np.sqrt(MSE)
            X1 = (-epsilon-MC_sample_G[:,0])/np.sqrt(MSE)
            X2 = (epsilon-MC_sample_G[:,0])/np.sqrt(MSE)
            EFF = MC_sample_G[:,0]*(2.0*sci.stats.norm.cdf(-X)-sci.stats.norm.cdf(X1)-sci.stats.norm.cdf(X2))-\
            np.sqrt(MSE)*(2.0*sci.stats.norm.pdf(-X)-sci.stats.norm.pdf(X1)-sci.stats.norm.pdf(X2))+\
            sci.stats.norm.cdf(X2)-sci.stats.norm.cdf(X1)
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
            y_star, y_br, phi = performance_function(my_study,x_star,phi,prec,epsilon)
            new_yg = np.zeros((y_gp.shape[0]+1,1))
            new_yg[0:-1,:] = y_gp
                
        
            if y_star == False : 
                new_y = np.zeros((yc.shape[0]+1,1))
                new_y[0:-1,:] = yc
                new_y[-1,:] = y_br
                yc = new_y.copy()
                new_yg[-1,:] = y_br
                
            else : 
                new_y = np.zeros((yc.shape[0]+1,1))
                new_y[0:-1,:] = yc
                new_y[-1,:] = y_br
                yc = new_y.copy()  
                new_y = np.zeros((ye.shape[0]+1,1))
                new_y[0:-1,:] = ye
                new_y[-1,:] = y_star
                ye = new_y.copy()  
                new_DoE = np.zeros((DoEe.shape[0]+1,DoEe.shape[1]))
                new_DoE[0:-1,:] = DoEe
                new_DoE[-1,:] = x_star
                DoEe = new_DoE.copy()  
                new_yg[-1,:] = y_star
            
            new_DoE = np.zeros((DoEc.shape[0]+1,DoEc.shape[1]))
            new_DoE[0:-1,:] = DoEc
            new_DoE[-1,:] = x_star
            DoEc = new_DoE.copy() 
            y_gp = new_yg.copy()
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
        DoEc.dump(path+str('DoE_c.pk'))
        DoEe.dump(path+str('DoE_e.pk'))
        yc.dump(path+str('Y_c.pk'))
        ye.dump(path+str('Y_e.pk'))
        y_gp.dump(path+str('Y_gp.pk'))
        
    yc_2=np.zeros((yc.shape[0],yc.shape[1]))   
    i=0
    for x in DoEc:
        x1=x[1:].reshape(8,1)
        T_allow=x[0]
        sol_br=my_study.solve_heatproblemr(x1)
        sol=phi.dot(sol_br)
        y_br=T_allow-max(sol)
        yc_2[i,:]=y_br
        i=i+1
    yc_2.dump(path+str('Y_c2.pk'))
    return Pf,cov,DoE,n_iter, phi.shape[1],phi


def temp_max(study,x,phi,prec,epsilon) :
    x1=x[1:].reshape(8,1)
    T_allow=x[0]
    A, b , sol= study.solve_heatproblem(x1)
    y=T_allow-max(sol)
    return y,phi 


def temp_br(study,x,phi,prec,epsilon): 
    x1=x[1:].reshape(8,1)
    T_allow=x[0]
   
    A, b=study.compute_A_b(x1)

    sol_br=study.solve_heatproblemr(x1)
    sol=phi.dot(sol_br)
    sol2=sol


    #residu sans preconditionnement 
#    res=np.linalg.norm(alpha-b,2)/np.linalg.norm(b,2)
    #residu preconditionne 
    #SPARSE
    alpha=A.dot(sol)
    b_prec = prec.solve(b)
    alpha_prec = prec.solve(alpha)

    res_prec=np.linalg.norm(alpha_prec.reshape(b_prec.shape[0],1)-b_prec.reshape(b_prec.shape[0],1),2)/np.linalg.norm(b_prec,2)
    y_br=T_allow-max(sol2)
    y=False
    
    if res_prec> epsilon :  

         A,b,sol = study.solve_heatproblem(x1)
         phi_i=sol-phi.dot(np.transpose(phi).dot(sol))
         phi_i=phi_i/np.linalg.norm(phi_i,2)
         phi=np.concatenate((phi,phi_i.reshape(phi_i.shape[0],1)),axis=1)
         study.assembly_proj(phi)

         y=T_allow-max(sol)
    
    return y, y_br, phi 



   
   
    
    
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

pf_list=list()
pf_ck_list=list()
cov_list=list()
cov_ck_list=list()
pf_ck_list2=list()
cov_list2=list()
for k in np.arange(50): 
    Pf,cov,DoE,n_iter,size_br, phi = AK_MCS(temp_br,distribution,initial_MC_sample_size = 100000,initial_doe_size =14, iter_max = 500, learning_function = 'U',epsilon=1e-3)
    
    
    DoEc=np.load('DoE_c.pk',allow_pickle=True)
    DoEe=np.load('DoE_e.pk',allow_pickle=True)
    yc=np.load('Y_c2.pk',allow_pickle=True)
    ye=np.load('Y_e.pk',allow_pickle=True)
    ygp=np.load('Y_gp.pk',allow_pickle=True)
    yc2=np.load('Y_c.pk',allow_pickle=True)
    
    stochastic_dim = DoE.shape[1]    
    theta0 = np.array([0.1]*stochastic_dim)
    thetaL = np.array([1e-6]*stochastic_dim)
    thetaU = np.array([100.0]*stochastic_dim)
    gp_g = GaussianProcess(theta0 = theta0,thetaL=thetaL,thetaU=thetaU)
    sm = MFK(theta0=theta0, print_global = False)
    gp_g.fit(DoEc,ygp)
    sm.set_training_values(DoEc, yc, name = 0)
    sm.set_training_values(DoEe, ye)
    sm.train()
    
    sm2 = MFK(theta0=theta0, print_global = False)
    sm2.set_training_values(DoEc, yc2, name = 0)
    sm2.set_training_values(DoEe, ye)
    sm2.train()
    
    
    MC_sample=np.load('MC_sample.pk',allow_pickle=True)
    pred_gp=gp_g.predict(MC_sample,eval_MSE=False)
    
    
    MC_sample_G = sm.predict_values(MC_sample)
    MSE = sm.predict_variances(MC_sample)
    
    MC_sample_G2 = sm2.predict_values(MC_sample)
    MSE2 = sm2.predict_variances(MC_sample)
    
    MC_sample_size=MC_sample.shape[0]
    # calcul pf cokrig
    indicatrice = np.zeros((MC_sample_size,1))
    ind = MC_sample_G<=0.0
    indicatrice[ind] = 1.0
    Pf_ck = indicatrice.sum()/MC_sample_size
    cov_ck = np.sqrt((1.0-Pf_ck)/(Pf_ck*MC_sample_size))
    
    
    indicatrice = np.zeros((MC_sample_size,1))
    ind = MC_sample_G2<=0.0
    indicatrice[ind] = 1.0
    Pf_ck2 = indicatrice.sum()/MC_sample_size
    cov_ck2 = np.sqrt((1.0-Pf_ck)/(Pf_ck*MC_sample_size))
    
    pf_list.append(Pf)
    pf_ck_list.append(Pf_ck)
    cov_list.append(cov)
    cov_ck_list.append(cov_ck)
    pf_ck_list2.append(Pf_ck2)
    cov_list2.append(cov_ck2)

pf_ref = 0.00622
err_k = abs (np.array(pf_list)-pf_ref)/pf_ref
err_ck3 = abs (np.array(pf_ck_list)-pf_ref)/pf_ref
err_ck2 = abs (np.array(pf_ck_list2)-pf_ref)/pf_ref

#
#
#
#
#
#
###trace predictions 
#fig=plt.plot()
#plt.scatter(Y,pred_gp,color='coral',label='ordinary Kriging')
#plt.scatter(Y,MC_sample_G,color='lightskyblue',label='co-Kriging')
#plt.plot(Y,Y,'r')
#plt.xlabel('Real values',fontsize=40)
#plt.ylabel('Surrogate models predictions',fontsize=40)
#plt.legend()
#left=-20
#right=20
#plt.xlim((left, right))  
#plt.ylim((left, right))  
##
##
#
