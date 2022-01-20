import numpy as np

# this file contains functions relevant to the study of the Luminosity function
# in particular the Eufstathiou method, schechter distribution, and multi-level parametrizations.
# it also contain the basis for Bayesian log-likelihood minimization and the use of Information Criteria.


 
#Luminosity function: Eufstathiou method, STWML 
def looplikelyhood(phis, weight, centers, mag_max, mag_min, x):
    weight = weight.reshape((len(weight),1))
    deltaL = abs(centers[1]-centers[0])

    deltaMgal = np.absolute(x.reshape((len(x),1))-centers.reshape((1,len(centers))))/deltaL # Nobs x Nphi
    msk_center_w = (deltaMgal<=.5)
    w = np.zeros(deltaMgal.shape) # Nobs x Nphi
    w[msk_center_w]=1.

    firstpart = np.sum(weight*w,axis = 0) #Nphi

    #faint end
    deltaMgalmax = (mag_max.reshape((len(mag_max),1))-centers.reshape((1,len(centers))))/deltaL # Nobs x Nphi
    msk_max_center_h = (np.absolute(deltaMgalmax)<0.5)
    msk_max_upper_h = (deltaMgalmax>=.5)
    h_max = np.zeros(deltaMgalmax.shape) # Nobs x Nphi
    h_max[msk_max_center_h]=deltaMgalmax[msk_max_center_h]+.5
    h_max[msk_max_upper_h] = 1.

    #bright end
    deltaMgalmin = (centers.reshape((1,len(centers)))-mag_min.reshape((len(mag_min),1)))/deltaL # Nobs x Nphi
    msk_min_center_h = (np.absolute(deltaMgalmin)<0.5)
    msk_min_upper_h = (deltaMgalmin>=.5)
    h_min = np.zeros(deltaMgalmin.shape) # Nobs x Nphi
    h_min[msk_min_center_h]=deltaMgalmin[msk_min_center_h]+.5
    h_min[msk_min_upper_h] = 1.

    histo0 = 0. #Nphi
    histo1 = np.array(phis) #Nphi
    #print(histo1[(histo1>0)])
    while np.nansum(np.absolute(1.-(histo0/histo1)))>=.1**5:
        histo0 = histo1
        thirdpart = np.dot((h_max*h_min),histo1).reshape((len(mag_max),1)) #Nobs
        secondpart = np.sum((h_max*h_min)*weight/thirdpart,axis = 0) #Nphi

        histo1 = firstpart/secondpart #Nphi

    return histo1



#Luminosity function: Schechter model
def schechter(M, phistar,Mstar,alpha):
    #phistar,Mstar,alpha = pars
    return (np.log(10)/2.5)*phistar*(10.**(.4*(Mstar - M)*(1+alpha)))*np.exp(-10.**(.4*(Mstar - M)))



#Bayesian log-likelihood minimization
def log_likelihood(theta, x, y, yerr, fit=schechter):
    model = fit(x,*theta)
    sigma2 = yerr**2
    return -0.5*np.sum((y-model)**2/sigma2 + np.log(sigma2))


def log_prior(theta,parameters):
    #theta=np.array(theta)
    msk_true = (np.absolute(theta-parameters)<0.05*parameters)
    if all(msk_true):
        return 0.0
    else:
        return -10**8#-np.inf

def log_probability(theta, x, y, yerr,parameters, fit=schechter):
    lp = log_prior(theta,parameters)
    if np.isnan(lp):
        return -10**8#-np.inf
    ll = log_likelihood(theta, x, y, yerr, fit)
    
    if np.isnan(ll):
        return -10**8#-np.inf
    else:
        return lp + ll
    

    


    
    
    


def mstar_alphastar(logdens,pars,
                    mstar_tot,alphastar_tot,
                    nmstar = 2,nalphastar = True): #n is the number of free parameters that is n= 2 -> f=a+bx
    if nalphastar == True:
        nalphastar = len(pars)-nmstar
    elif (nalphastar+nmstar)!=len(pars):
        print('check lenghts of parameter vs polynomials')
    par_mstar =pars[:nmstar]
    par_alphastar =pars[nmstar:]
    
    
    pol_logdens_mstar = np.array([logdens**i for i in range(0,nmstar)])
    mstar = np.dot(par_mstar,pol_logdens_mstar)
    mstar += mstar_tot
        
    #polinomial
    pol_logdens_alphastar = np.array([logdens**i for i in range(0,nalphastar)])
    alphastar = np.dot(par_alphastar,pol_logdens_alphastar)
    alphastar += alphastar_tot
    
    #N, mu, sigma,x0 = par_alphastar
    #alphastar = N*(1-np.exp(-((np.log(logdens+x0)-mu)/(sigma))**2)/(logdens+x0))
    
    return (mstar,alphastar)    
    

def schechter_f_logdens(x,pars,
                        mstar_tot,alphastar_tot,
                        nmstar = 2,nalphastar = True): #n is the number of free parameters that is n= 2 -> f=a+bx
    mstar,alphastar = mstar_alphastar(logdens,pars,
                    mstar_tot,alphastar_tot,
                    nmstar = 2,nalphastar = True)
    return schechter(mag,phistar,mstar,alphastar)


    
#Bayesian log-likelihood minimization
def log_likelihood_comp(theta, x, y, yerr):#,function = thflum.schechter_f_logdens):
    model = schechter_f_logdens(x,theta)
    sigma2 = yerr**2
    return -0.5*np.sum((y-model)**2/sigma2 + np.log(sigma2))


def log_prior_comp(theta,parameters):
    #theta=np.array(theta)
    msk_true = (np.absolute(theta-parameters)<0.05*parameters)
    if all(msk_true):
        return 0.0
    else:
        return -10**8#-np.inf


def log_probability_comp(theta, x, y, yerr,parameters):
    lp = log_prior_comp(theta,parameters)
    if np.isnan(lp):
        return -10**8#-np.inf    
    ll = log_likelihood_comp(theta, x, y, yerr)
    if np.isnan(ll):
        return -10**8#-np.inf
    else:
        return lp + ll
    
    
    


def BIC(y, yerr,k,model):
    sigma2 = yerr**2
    n = len(y)
    ll = -0.5*np.sum((y-model)**2/sigma2 + np.log(sigma2))
    return -2*ll + k*np.log(n)

def AIC(y, yerr,k,model):
    sigma2 = yerr**2
    ll = -0.5*np.sum((y-model)**2/sigma2 + np.log(sigma2))
    return -2*ll + k*2

def AICc(y, yerr,k,model):
    sigma2 = yerr**2
    n = len(y)
    ll = -0.5*np.sum((y-model)**2/sigma2 + np.log(sigma2))
    return -2*ll + k*2*(k+1)/(n-k-1)
