
import numpy as np
import scipy as sc
from scipy import stats
import openturns as ot
#
#d=2 #change la dimension là 
#f=stats.multivariate_normal(mean=np.zeros(d), cov=np.eye(d))
np.random.seed

phi_n= stats.norm()

def Poids(f,g,X):
    if g==f:
        return np.ones(len(X))
    else :
        f_valeur=np.array(f.computePDF(X))
        g_valeur=g.pdf(X.T) #attention on doit utiliser la transposée
        frac= f_valeur[:,0]/g_valeur
        return frac

def Indic(Y,gamma): 
    return Y< gamma

def Norme_poids(k,x,gp_g,gamma,N,fraction): 
    E_P=proba_failure(x,gp_g,gamma)*fraction
    return E_P


def proba_failure(x,gp_g,gamma):
    x_size=x.shape[0]
    MC_sample_G,MSE=gp_g.predict(x,eval_MSE=True)
    U=(gamma-MC_sample_G[:,0])/np.sqrt(MSE)
#    U=(gamma-MC_sample_G)/np.sqrt(MSE)

    p=phi_n.cdf(U.reshape((x_size,1)))
    return p[:,0]


def Densite(X,poids):
    #on fait tout d'un coup
    #X doit etre mi sous une autre forme (transposé)
    values = X.T
    kernel=stats.gaussian_kde(values,'silverman',poids)
    return kernel
        
      
def Boucle_k_d(k,rho,S,f,N,phi,g,population_X,population_Y,fraction,gp_g):
    #step 2 : on met tout à jour
    nouvelle_pop=g[k-1].resample(N).T
    population_X=np.concatenate((population_X,nouvelle_pop),axis=0)#kernel resample est un array
    population_Y=np.concatenate((population_Y,phi(nouvelle_pop)),axis=0) #faut que soit un array bon sens
    frac=np.concatenate((fraction,Poids(f,g[k-1],nouvelle_pop)),axis=0)
    #step 3
    rho_quantil = stats.scoreatpercentile(population_Y[(k-1)*N:k*N],rho)
    gamma = max (S, rho_quantil)
    #step 4 
    if gamma != S:
        poids=Norme_poids(k,population_X,gp_g,gamma,N,frac)
        #step 5
        kernel=Densite(population_X,poids)
        g.append(kernel)
    return gamma,population_X,population_Y,frac,g
    
    
def NAIS_m(rho,S,f_ot,N,gp_g,iter_max=60): #lancer ça pour l'avoir 1 fois
    #step 1
    k=1
    stochastic_dim=gp_g.X.shape[1]
    #step 2a
    population_X =np.array(f_ot.getSample(N))


    phi = lambda x : gp_g.predict(x)[:,0]
    population_Y = phi(population_X)#il faut donc que ce soit sous forme d'array là
    #step 3a
    rho_quantil = stats.scoreatpercentile(population_Y,rho)

    gamma = max (S, rho_quantil)
    g=[f_ot] #LISTE
    fraction=Poids(f_ot,g[0],population_X)
    #step 4a
    poids=Norme_poids(k,population_X,gp_g,gamma,N,fraction)

    #-step 5a
    kernel=Densite(population_X,poids)
    g.append(kernel)
    alpha=1.
    eps_alpha=0.1
    #Step 6a

    while gamma > S and k < iter_max and alpha > eps_alpha : 
        k=k+1
        alpha=gamma
        Boucle=Boucle_k_d(k,rho,S,f_ot,N,phi,g,population_X,population_Y,fraction,gp_g)
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

