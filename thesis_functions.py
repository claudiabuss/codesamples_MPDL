import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

import configparser
config = configparser.ConfigParser()
config.read("parameters_thesis.conf")


env_names = ['knot','fil', 'wall', 'void']
colors = list(map(lambda x: config.get('color_thesis',x), env_names))
colormaps =  list(map(lambda x: config.get('color_maps_thesis',x), env_names))



#CLASSIFICATION FUNCTION
def classif(eigv,thresh, prop = 'ours'):
    """
    Classification function
    Analises the hessian of the gravitational field against a pre-determined threshold
    and returns the corresponding environment.
    Numeric results read as an one-to-one relation to geometrical environments as follows:
    0, knot
    1, filament
    2, wall
    3, void
    
    
    Parameters
    ----------
    eigv:   List with three floats or np.array.size = (3,n).
            Takes in the eigenvalues of the gravitational field hessian.
    thresh: list with three floats
            List of three floats.
            Takes in the thresholds which will limit each of
            - if prop = 'ours', the m-dimensional volums, m = 1,2,3 connected to the hessian eigenvalues;
            - if prop = 'Eardley', 
    Kwarg
    ----------
    prop: takes values {'ours','Eardley'}
        'ours': our classifier
                Notice that the condition Proj_{i}(eigen)>0, i = 0,1,2 from Han's is also enforced in each step 
        'Eardley': Eardley's classifier.
                Eardley reduces to Hahn's proposal when threshold equals to zero.
    
    Returns
    ----------
    integer or n-dimensional list of integers
    
            
    """
        
    a,b,c = eigv
    ta,tb,tc = thresh
    
    if prop == 'ours':
        if (a>0) and (a*b*c>ta): #ta=0.4
            return 0
        elif (b>0) and (b*c>tb): #tb=0.16
            return 1
        elif (c>0) and (c>tc): #tc=0.1 
            return 2
        else:# (c<0)and(a*b>tc):
            return 3
    if prop == 'Eardley':
        if (a>ta):
            return 0
        elif (b>tb):
            return 1
        elif c>tc:
            return 2
        else:
            return 3




# FUNCTIONS RELATED TO THE SAMPLE GEOMETRY
#lightcone boudaries
def fDec(lam,eta,racen=185.0,deccen=32.5):
    d2r = np.pi/180.0
    r2d = 180.0/np.pi
    return r2d*np.arcsin(np.cos(lam*d2r)*np.sin((eta+deccen)*d2r))

def fRa(lam,eta,racen=185.0,deccen=32.5):
    """takes lam,eta SDSS coordinates and returns Ra. """
    d2r = np.pi/180.0
    r2d = 180.0/np.pi
    return r2d*np.arctan(np.tan(lam*d2r)/np.cos((eta+deccen)*d2r))+racen

def fLimN(x,ra_lim_u=-46,ra_lim_d=-53):
    """ Defines the leftmost (most negativ, thefore N) straigth line boundary in the ra-dec, coordinates."""
    return (x-fRa(ra_lim_u,35))*(fDec(ra_lim_d,-33)-fDec(ra_lim_u,35))/(fRa(ra_lim_d,-33)-fRa(ra_lim_u,35))+fDec(ra_lim_u,35)

def fLimP(x,ra_lim=49):
    """ Defines the rightmost (most positive, thefore P) straigth line boundary in the ra-dec, coordinates."""
    return (x-fRa(ra_lim,35))*(fDec(ra_lim,-33)-fDec(ra_lim,35))/(fRa(ra_lim,-33)-fRa(ra_lim,35))+fDec(ra_lim,35)

def f_eta(ra,dec,racen=185.0,deccen=32.5):
    """
    takes RA, Dec SDSS coordinates and returns eta.
    there is a problem with this definition, see comma, Check the right definition.
    """
    d2r = np.pi/180.0
    r2d = 180.0/np.pi
    return r2d*np.arctan(np.sin(dec*d2r),np.cos(dec*d2r)*np.cos((ra-racen)*d2r))-deccen

def f_lambda(ra,dec,racen=185.0,deccen=32.5):
    """takes RA, Dec SDSS coordinates and returns lam. """
    d2r = np.pi/180.0
    r2d = 180.0/np.pi
    return r2d*np.arcsin(np.cos(dec*d2r)*np.sin((ra-racen)*d2r))




#coordinates
def esphToRec(ra,dec,dist):
    """
    Parameters
    ----------
    Spherical coordinates
    
    Returns
    ----------
    Cartesian coordinates with line of sight as z-axis
    
    """
    comoving = dist#cosmo.comoving_distance(z).value 
    rad_ra   = np.radians(ra)
    rad_dec  = np.radians(dec)
    coor_x   = comoving*(np.cos(rad_dec)*np.cos(rad_ra))
    coor_y   = comoving*(np.cos(rad_dec)*np.sin(rad_ra))
    coor_z   = comoving*(np.sin(rad_dec))
    return np.array([coor_x,coor_y,coor_z])*u.Mpc;






def compute_contour_levels(hist_2d,percentage):#[.9973, .954, .68]):
    # see https://corner.readthedocs.io/en/latest/pages/sigmas.html 
    # 
    """
    Function to compute the levels of 1-2-3 sigma (68%, 95%, 99.73%), 1d-distribution sigma
    Parameters
    ----------
     - hist_2d : histogram of type numpy.histogram2d
    Returns
    ----------
     - [float, float, float] : levels to make the confidence contours
    #FIXME - finish documentation
    """
    n_steps = 10000
    total_pts = float( hist_2d.sum() )
    hist_max = hist_2d.max()

    frac = []
    mult = []

    for n_for in np.linspace(0.0, hist_max, n_steps):
        cases = []
        for i in range(hist_2d.shape[0]):
            for j in range(hist_2d.shape[1]):
                if hist_2d[i, j] > n_for:
                    cases.append( hist_2d[i, j])
        frac.append( np.array(cases).sum()/total_pts )
        mult.append( n_for)
        #print np.array(cases).sum()/total_pts, n/hist_max
  

    return_mult = []
    for i in range(len(percentage)):
        #print i
        diff_abs = np.fabs( (np.array(frac) - percentage[i]))
        frac_min = diff_abs.min()
        #print frac_min

        pos = np.where(diff_abs == frac_min)[0][0]
        #print pos[0][0]
        #print mult[pos]
        return_mult.append( mult[pos] )

    return np.array(return_mult)


def func2d(data_x, data_y, nbins,colormap0,
           xlimits,ylimits,
           xname,yname,
           xyrange=True,
           percentage = [.9889,.8647,.3935,0.],
           linewidths=0.5,
           transparency = 1.):
    """
    This function takes in data points and returns their graphical representation
    of their cloud-like distribution
    In particular the cloud plots for the [.9889,.8647,.3935] distribution levels.
    """
    
    histo, xedges, yedges = np.histogram2d(data_x,data_y,nbins)
    
    x = (xedges[1:]+xedges[:-1])/2
    y = (yedges[1:]+yedges[:-1])/2
    X, Y = np.meshgrid(x, y)
    sigmas = compute_contour_levels(histo,percentage)

    CS = plt.contour(X, Y, histo.T, levels=sigmas,
                     colors='black',
                     vmin=min(sigmas), vmax=max(sigmas),
                     alpha=transparency)
    #plt.clabel(CS, inline=1, fontsize=10)

    
    histo[histo<=0]=np.nan
    plt.imshow(histo.T, 
               extent=[xedges.min(),xedges.max(),yedges.min(),yedges.max()],
               origin='lower', 
               cmap=colormap0)
    if xyrange:
        plt.ylim(ylimits)
        plt.xlim(xlimits)
    else:
        plt.ylim(yedges.max(),yedges.min())
        plt.xlim(xedges.min(),xedges.max())
    plt.xlabel(xname)
    plt.ylabel(yname)

    
def func2d_c(data_x, data_y, nbins,
           xlimits,ylimits,
           xname,yname,
           xyrange=True,
           percentage = [.9889,.8647,.3935]):
    """
    This function takes in data points and returns their graphical representation
    of their cloud-like distribution
    In particular the countour plots for the [.9889,.8647,.3935] distribution levels.
    """
    histo, xedges, yedges = np.histogram2d(data_x,data_y,nbins)
    
    x = (xedges[1:]+xedges[:-1])/2
    y = (yedges[1:]+yedges[:-1])/2
    X, Y = np.meshgrid(x, y)
    sigmas = compute_contour_levels(histo,percentage)

    
    CS = plt.contour(X, Y, histo.T, levels=sigmas, colors=config.get('color_thesis','ref1'),linewidths = 0.25)
    #plt.clabel(CS, inline=1, fontsize=10)

    
#    histo[histo<=0]=np.nan
#    plt.imshow(histo.T, 
#               extent=[xedges.min(),xedges.max(),yedges.min(),yedges.max()],
#               origin='lower', 
#               cmap=colormap0)
    if xyrange:
        plt.ylim(ylimits)
        plt.xlim(xlimits)
    else:
        plt.ylim(yedges.max(),yedges.min())
        plt.xlim(xedges.min(),xedges.max())
    plt.xlabel(xname)
    plt.ylabel(yname)


    
def func2d_f(data_x, data_y, nbins,colormap0,
           xlimits,ylimits,
           xname,yname,
           xyrange=True,
           percentage = [.9889,.8647,.3935,0.],
           transparency = 1.):
    
    histo, xedges, yedges = np.histogram2d(data_x,data_y,nbins)
    
    x = (xedges[1:]+xedges[:-1])/2
    y = (yedges[1:]+yedges[:-1])/2
    X, Y = np.meshgrid(x, y)
    sigmas = compute_contour_levels(histo,percentage)

    
    CS = plt.contourf(X, Y, histo.T, levels=sigmas, cmap=colormap0,
                      vmin=min(sigmas), vmax=max(sigmas), alpha=transparency)
    #plt.clabel(CS, inline=1, fontsize=10)

    
#    histo[histo<=0]=np.nan
#    plt.imshow(histo.T, 
#               extent=[xedges.min(),xedges.max(),yedges.min(),yedges.max()],
#               origin='lower', 
#               cmap=colormap0)
    if xyrange:
        plt.ylim(ylimits)
        plt.xlim(xlimits)
    else:
        plt.ylim(yedges.max(),yedges.min())
        plt.xlim(xedges.min(),xedges.max())
    plt.xlabel(xname)
    plt.ylabel(yname)
