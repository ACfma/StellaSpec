# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:02:32 2021

@author: Andrea
"""
import numpy as np
import time
from scipy.optimize import minimize, Bounds, newton
from scipy.special import hyp1f1
import logging

def PCA_denoising(tmp, mode = 'MRS', L = 1, verb = False):
    '''
    PCA_denoising is a function based mainly on the articles by Froeling et al.
    (2020), Veraart et al (2016) and Koay et al. (2006) in wich confusion matrix
    are applied in order to enstablish a treshold between signal and (gaussian)
    noise in a MRS, MRI or dMRI dataset.
    The function returns two different kind of denoised data based on the criteria
    used respectivelly by Veraart and Froeling.
    
    Exceptfor MRS, the modes are still under development!
    
    Parameters
    ----------
    tmp : 3 or 4D numpy array
        The initial dataset containing complex or real values. 3D if you are examining
        an image, 4D if the dataset is made up by a spectrum matrix ore more images
        appended in the fourth dimension.
    mode : str
        Mode of the scansion. Default: 'MRS'.
    L : int
        Number of channels in phased array. Default: 1.
    verb : bool
        Set True to print information other than just warnings. Default: False.

    Returns
    -------
    den : 4D numpy array
        Denoised dataset based on Veraart's criterium.
    den : 4D numpy array
        Denoised dataset based on Froeling's criterium.
    sig_map : 4D numpy array
        Array of sigma from gaussian distribution, corrected for nc-chi squared
        if magnitude image are acquired.

    '''
    def mpPDF(var, q, pts):#taken from: https://medium.com/swlh/an-empirical-view-of-marchenko-pastur-theorem-1f564af5603d
        """
        Creates a Marchenko-Pastur Probability Density Function
        Args:
            var (float): Variance
            q (float): T/N where T is the number of rows and N the number of columns
            pts (int): Number of points used to construct the PDF
        Returns:
            pdf: 2darray: Marchenko-Pastur PDF
            eMax: float: lambda+ in MP distribution
            eMin: float: lambda- in MP distribution
        """
        # Marchenko-Pastur pdf
        # q=T/N
        # Adjusting code to work with 1 dimension arrays
        if isinstance(var, np.ndarray):
            if var.shape == (1,):
                var = var[0]
        eMin, eMax = var*(1 - (1. / q)**.5)**2, var*(1 + (1. / q)**.5)**2
        eVal = np.linspace(eMin, eMax, pts)
        pdf = (q / (2 * np.pi * var * eVal)) * (((eMax - eVal) * (eVal - eMin)) ** .5)
        pdf = np.array([eVal,pdf])
        return pdf, eMax, eMin 
    
    start_time = time.time()
    if verb:
        logging.getLogger().setLevel(logging.INFO)
        
    den = np.zeros(np.shape(tmp), dtype=complex)
    dden = np.zeros(np.shape(tmp), dtype=complex)
    sig_map = np.zeros(np.shape(tmp), dtype=complex)
    if mode == 'MRI' or mode == 'dMRI':
        logging.info('Newton method used. Some values could be different from expectation due to limited iterationa number.')
        logging.warning('Elaborating nc-chi square standard deviation correction. It might take a while...')
        
    #Convolution and denoising by PCA
    '''(From the introduction of"Statistical Noise Analysis in GRAPPA..." about MR image noise modellization)
    As a final remark, note that three requirements are needed for the SoS of Gaussian random
    variables (RV) to be modeled as a stationary nc-χ:
    1. The noise is stationary in each of the Complex Gaussian x–space images. If the k–
    space data is fully sampled, the noise variance will be the same for all the points in
    the image in both the k–space and x–space domains, and the noise can be assumed
    to be stationary.
    2. The variance of noise is the same for each of the coils.
    3. No correlation is assumed between the Gaussian RVs.
    
    CHECK IF THEY ARE TRUE
    '''
    for i in range(3,np.size(tmp,0)-2):
        for j in range(3,np.size(tmp,1)-2):
            for k in range(3,np.size(tmp,2)-2):
                sub_M = tmp[i-2:i+3,j-2:j+3,k-2:k+3,:]
                M = []
                for l in range(0,np.size(sub_M,0)):
                    for m in range(0,np.size(sub_M,1)):
                        for n in range(0,np.size(sub_M,2)):
                            if mode == 'MRS' or mode == 'dMRI':
                                M.append(np.append(np.real(sub_M[l,m,n,:]),np.imag(sub_M[l,m,n,:]),axis=0))
                            elif mode == 'MRI':
                                M.append(sub_M[l,m,n,:])
                            else:
                                ValueError('Mode not compatible with function\'s specifics')
                M = np.array(M)
                #Maybe real and immaginary parts need to be saparated becouse not independent and uncorrelated with themself? NO, becouse the model assumes reeal and immaginary part OF NOISE as independent "Statistical Noise Analysis in GRAPPA..." . 
                U,S,Vs = np.linalg.svd(M, full_matrices=False)
                P_old = 0
                P = np.min(np.shape(S))
                def nb_fun(n):
                    '''
                    Optimal number of bins according to Varaart et al.
    
                    Parameters
                    ----------
                    n : float
                        number of bins.
    
                    Returns
                    -------
                    float
                        Objective function value.
    
                    '''
                    n = int(n)
                    h = np.histogram(M,n)
                    rh = np.mean(h[0])
                    v = np.mean((h[0]-rh)**2)
                    S_p = S[P:]
                    S_nz = np.sort(S_p[S_p!=0])
                    if len(S_nz)!=0:
                        hsq = (np.abs(S_nz[0] - S_nz[-1])/n)**2
                    else:
                        hsq = (np.abs(S[0] - S[-1])/n)**2
                    return (2*rh-v)/hsq
                
                while P != P_old:
                    #Finding best number of bins based on optimal for time series ref: "A Method for Selecting the Bin Size of a Time Histogram" from Shimazaki
                    bounds = Bounds(1,len(S)) # Used a constrained methods with a number of bins between 1 and length of S.
                    nb = minimize(nb_fun,P,method='Nelder-Mead', bounds=bounds)#it should be ok, othervise n could ne a float but inside the function is rounded with int(n).
                    nb = int(nb.x) 
                    h = np.histogram(S,nb)
                    N=(np.size(M,1)-P)
                    gamma = np.size(M,0)
                    #Rough estimate of sigma from lambda minus. This choise was made becouse in a noisy envoironment there musrìt be more count on the infirior extreme while, in this case, lambda plus would be estimated from lambdaP which could belong in a zone with an order of magnitude jump (Read Veraart et al.).
                    sigma_rough = np.sqrt(S[-1])/np.abs(1-np.sqrt(gamma))
                    def opt_sig(sigma):
                        '''
                        Optimal sigma according to Varaart et al.
    
                        Parameters
                        ----------
                        sigma : float
                            Standard deviation used.
    
                        Returns
                        -------
                        float
                            Optimization function value.
    
                        '''
                        S_p = S[P:]
                        S_nz = np.sort(S_p[S_p!=0])
                        if len(S_nz)!=0:
                            zeta = (np.abs(S_nz[0] - S_nz[-1])/nb)*(N)
                        else:
                            zeta = (np.abs(S[0] - S[-1])/nb)*(N)
                        
                        
                        
                        pdf,_,_ = mpPDF((sigma)*2,gamma,len(h[0]))#sigma/xi
                        
                        W = zeta*pdf[1]
                        for i,v in enumerate(zeta*pdf[1]):
                            if v != 0:
                                W[i] = 1/v
                            else:
                                W[i] = np.inf
                        return np.sum(np.multiply(W,((h[0]-zeta*pdf[1])**2)))
                    
                    '''ATTENTION! Used the BIASED sigma, maybe it needs to be corrected for nc-chi, I still 
                    haven't done that, maybe in the future it will be necessary to implement it.
                    In the following lines, the sigma is calculated as reported in the article BEFORE the introduction of nc-chi correction'''
                    sig = minimize(opt_sig, sigma_rough,method='CG') #The article uses a gridsearch, i use a CGD in order to speed up the process (also becouse i don't know the optimal span of the gridsearch)
                    emax = (sig.x*(1+np.sqrt(gamma)))**2
                    Q = np.where(S>emax, S,0)
                    P_old = P
                    P = len(Q[Q!=0])
                    if P == 0: #AC, if P = 0 then no bins exceed MP distribution => all noise, we can exit.
                        P = P_old
                #END WHILE
                #try to mediate voxelwise as done by Dennis
                #adding correction for magnitude sigma
                
                Mone = np.dot(U, np.dot(np.diag(Q), Vs))
                
                if mode == 'MRI' or mode == 'dMRI':
                    def xi(theta,L):
                        '''
                        Calulate xi function from koay article.

                        Parameters
                        ----------
                        theta : float
                            Signal to niose ratio.
                        L : int
                            Number of channels.

                        Returns
                        -------
                        float
                            xi value for given SNR and number of channels.

                        '''
                        def BL(L):
                            '''
                            Moltiplication coefficient.
                            Parameters
                            ----------
                            L : int
                               Number of channels.

                            Returns
                            -------
                            float
                                Moltiplication coefficient.

                            '''
                            return np.sqrt(np.pi/2)*np.math.factorial(np.math.factorial(2*L-1))/(np.math.factorial(2*L-1)*(2**(L-1)))
                        return 2*L + theta**2 - (BL(L)**2)*(hyp1f1(-1/2,L,-(theta**2)/2)**2)
                    def SNR(theta,m,s,L):
                        '''
                        Signal to Noise Ratio optimizator.
                        Ment to be zero when m,s and L combined gives theta.
                        Parameters
                        ----------
                        theta : float
                            SNR.
                        m : real or complex
                            mean of signal.
                        s : real or complex
                            SD of signal.
                        L : int
                            Number of channels.

                        Returns
                        -------
                        float
                            Difference between the estimate and the value of SNR.

                        '''
                        return np.sqrt(xi(theta,L)*(1+(m/s)**2) - 2*L) - theta
                    
                    m = Mone[2*5*5+2*5+2,:]
                    r = m/sig.x
                    low_bound = np.sqrt((2*L/xi(0,L)) - 1)
                    sg_l = np.array([])
                    for c,v in enumerate(m):
                        if r[c]<low_bound:
                            theta = 0
                            sg_l = np.append(sg_l,low_bound)
                        else:
                            theta = newton(SNR,1,args=(v,sig.x,1),maxiter = 100,disp = False) #Still don't know if L=1 or 24 (one if number of channel means parallel imaging), set disp = False becouse otherwise it freezes if one of the data doesn't converge
                            sg_l = np.append(sg_l,sig.x/xi(theta,L))
                    sig_map[i,j,k,:] = sg_l  
                else:
                    sig_map[i,j,k] = sig.x
                if mode == 'MRS':    
                    Cube = np.zeros((5,5,5,540), dtype=complex)
                else:
                    Cube = np.zeros((5,5,5,540), dtype=float)
                for l in range(0,np.size(Cube,0)):
                    for m in range(0,np.size(Cube,1)):
                        for n in range(0,np.size(Cube,2)):
                            if mode == 'MRS':
                                Cube[l,m,n,:] = Mone[l*5*5+m*5+n,0:540] + 1j*Mone[l*5*5 + m*5 + n,540:]
                            else:
                                Cube[l,m,n,:] = Mone[l*5*5+m*5+n,:]
                dden[i-2:i+3,j-2:j+3,k-2:k+3,:] = np.add(dden[i-2:i+3,j-2:j+3,k-2:k+3,:], Cube) #Dennis way (mean of each rapresentation of voxel during convolution)
                den[i,j,k,:] = Cube[2,2,2,:]  #for semplicity, used Veraart method vith central value NOT THE ONE USED IN THE ARTICLE
    
    dden = dden/125 #divided by the time every voxel is calculated (5x5x5)
    
    print("---Denoising done in {} seconds ---".format(time.time() - start_time))
    return den, dden, sig_map


if __name__ == '__main__':
    tmp = np.random.rand(10,10,10,1)
    den,_,_ = PCA_denoising(tmp)