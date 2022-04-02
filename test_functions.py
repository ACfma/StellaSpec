# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:21:27 2022

@author: Andrea
"""
import time
from scipy.stats import ttest_1samp, rayleigh, uniform, kstest, bartlett
from statsmodels.stats.diagnostic import lilliefors #lilliefors is a K-S test with a gaussian with unknown meand and sd.
import numpy as np
import pandas as pd

def Look4Gauss(D, noise_area = 100, pval=0.05):
    '''
    Look4Gauss implement some test of normality for noise in an NMR signal as mentioned in "A statistical analysis of NMR spectrometer noise"(Grage and Akke, 2003).

    Parameters
    ----------
    D : 3Darray
        Array of the given signal with shape (n°signals, n°points, n°channels). The check is done by channel.
    noise_area : int, optional
        Number of point in which is assume only noise presence. The default is 100.
    p : float, optional
        p-value for ALL tests (migth be changed soon). The default is 0.05.
    
    Returns
    -------
    nogood : dict
        Dictionary of position of faulty signal for each channel.

    '''
    
    start = time.time()
    
    def autocorr(x):
        '''
        Calculate autocorrelation of array. Not sure if it works the same with complex data.

        Parameters
        ----------
        x : 1D array
            Data.

        Returns
        -------
        acov : 1D array
            Autocovariance of data.
        acor : 1D array
            Autocorrelation of data (above but normalized for acov[0]).

        '''
        n = len(x)
        m = np.mean(x)
        acov = np.correlate(x-m,x-m, 'full')/n
        acov = acov[len(acov)//2:]
        acor = acov/acov[0]
        return acov, acor
    
    shp = np.shape(D)
    nogood = {}
    for ch in range(0,1):
        tmp = np.array([],dtype = np.uint8)
        for sgn in range(0,shp[0]):
            noise = D[sgn,-noise_area:,ch]
            rnoise,inoise = np.real(noise), np.imag(noise)
            #Let's see if the singular parts are gaussian.
            _, pr = lilliefors(rnoise,dist='norm')
            _, pi = lilliefors(inoise,dist='norm')
            if pr<=pval and pi<=pval:
                tmp = np.append(tmp,sgn)
            else:
                #Check for zero mean
                #We cannot exclude that the distributions are gaussian, let's test if they have zero mean
                _, pr = ttest_1samp(rnoise,0)
                _, pi = ttest_1samp(inoise,0)
                if pr<=pval and pi<=pval:
                    tmp = np.append(tmp,sgn)
                else:
                    #Check for same variances between real and imaginary part
                    _, p = bartlett(rnoise, inoise)#Bartlet test works in asymptotic condition.
                    if p <= pval:
                        tmp = np.append(tmp,sgn)
                    else:
                        #Checking for indipendence between noise's points
                        _,racorr = autocorr(rnoise)
                        _,iacorr = autocorr(inoise)
                        l = len(racorr)
                        if len(racorr[np.abs(racorr)>1.96/np.sqrt(l)])>pval*l +1  or len(iacorr[np.abs(iacorr)>1.96/np.sqrt(l)])>pval*l + 1: #plus one for excluding the autocoralation at zero (only correltion of iid from 1 to n folluwa a distribution N(0,1/n)) Brockwell-Davis pg. 222
                            tmp = np.append(tmp,sgn)
                        else:
                            '''If whe cannot exclude gaussian real and imaginary part with zero mean, 
                            uncorrelation of single components and equal variances,
                            then check for bivariate normal distribution fer asserting 
                            indipendent gaussians distribution between real and imaginary part'''
                            r = np.abs(noise)-np.mean(np.abs(noise))
                            th = np.angle(noise)
                            th = np.where(th>0,th*(360/(2*np.pi)), -th*(360/(2*np.pi))+180)
                            # cdf1 = rayleigh(0,np.sqrt(np.mean((r**2)/2))).cdf(np.linspace(np.min(r),np.max(r),100))#the article states that it could be used for ANY mean while zero mean is checked aside
                            # cdf2 = uniform(0,360).cdf(np.linspace(np.min(th),np.max(th),100))
                            #ATTENTION  this should be lilliefors test (K-S with MLE) but the function "lilliefors" doesn't support reyleigh pdf
                            #Mean was set the aritmetic mean (for CLT it should be the MLE of the gaussian centered on the real value) while the sigma is the MLE of rayleigh as set in https://ocw.mit.edu/ans7870/18/18.443/s15/projects/Rproject3_rmd_rayleigh_theory.html 
                            _, p1 = kstest(r, cdf = rayleigh.cdf, args = (0,np.sqrt(np.mean((r**2)/2))))
                            _, p2 = kstest(th, cdf = uniform.cdf, args = (0,360)) #using degree because angle gives value in (-pi,pi] and ks test between [-arg[0], arg[1]]
                            if p1<=pval or p2<=pval:
                                tmp = np.append(tmp,sgn)
        nogood['{}'.format(ch)] = tmp

    print("---Noise evaluaton completed in {:.2f} seconds.---".format(time.time()-start))
    return nogood

def Slotboom(D,snrt = 2, n_r = 100, nogood=None,):
    '''
    Metods for transient events detection as describen in (Slotboom et al, 2008-2009)

    Parameters
    ----------
    D : 3Darray
        Array of the given signal with shape (n°signals, n°points, n°channels). The check is done by channel.
    snrt : float, optional
        SNR threshold for which under it it is considerated noise. The default is 1.5.
    n_r : int, optional
        Number of last data point to be consider as noise.
        In case threshold doesn't find a good value untill n_r, then n_r is taken as threshold. The default is 100.
    nogood : nogood : dict
        Dictionary of position of faulty signal for each channel.The default is None.
    

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    #creating D matrix and calcoulus of SMs (Reliability testing of in vivo 1H-MRS-signals and elimination of signal artifacts by median filtering Slatboom et al, 2008)
    #The article states that both FIDs or Specroums could be used
    for ch in range(D.shape[2]):
        D_r = np.real(D[:,:,ch])
        print(D.shape)
        if nogood != None:
            D_r = np.delete(D_r,nogood['{}'.format(ch)],axis = 0)
        def moments(D):
            M = np.shape(D)[0]
            um = np.mean(D,axis = 0)
            uvar = np.sum((D-um)**2,axis = 0)/(M-1)
            bsk = (np.sqrt(M*(M-1))/(M-2)) * np.sqrt(M) * np.sum((D-um)**3,axis = 0)/(np.sum((D-um)**2,axis = 0))**(3/2)
            bks = ((M+1)*M/((M-3)*(M-2)*(M-1)))*((np.sum((D-um)**4,axis = 0))/(uvar**2)) - 3*((M-1)**2)/((M-2)*(M-3)) #excess kurtosis
            return um, uvar, bsk, bks
        um, uvar, bsk, bks = moments(D_r)
        #Da anche le espressioni per le incertezze di queste quantità ma forse non sono necessarie per il momento
        def masking(D,snrt, n_r):
            shp = np.shape(D[:,:,ch])
            mask = np.ones(shp, np.dtype('uint8'))
            for sgn in range(0,shp[0]):
                x = D[sgn,:]
                try:
                    indx = np.where(np.abs(x)>snrt*np.std(x[-n_r:]))[0][-1]
                    
                except IndexError as e:
                    print('Error {} has occurred: setting noise to n_r'.format(e))                
                    indx = shp[1]
                if indx>n_r:#at least last 100 position are set to noise (saw from median)
                    indx = shp[1]-n_r
                for k in range(indx,shp[1]):
                    mask[sgn,k] = 0
            return mask
        mask = masking(D, snrt, n_r)
        #@jit(nopython=True)
        def kstat(bs,mask):
            if np.sum(mask) > 0 and np.sum(mask) != D.shape[1]: #if there is a decayed signal  
                k = np.mean(np.abs(bs[mask==1]))
                k_star = np.mean(np.abs(bs[mask==0]))
                vark = np.sum((np.abs(bs[mask==0])-k_star)**2)/(len(bs[mask==0])-1)
            else:
                k,vark = 0, 0
            return k, vark
        rel_ch = pd.DataFrame()
        k_s, k_k, vark_s, vark_k = [],[],[],[]
        for sgn in range(0,np.shape(mask)[0]):
            k_0, vark_0 = kstat(bsk[:], mask[sgn,:])
            k_1, vark_1 = kstat(bks[:], mask[sgn,:])
            k_s.append(k_0), k_k.append(k_1), vark_s.append(vark_0), vark_k.append(vark_1)
        rel_ch['k_s'] = k_s
        rel_ch['vark_s'] = vark_s
        rel_ch['k_k'] = k_k
        rel_ch['vark_k'] = vark_k
        ch_mask = []
        for i in range(D.shape[0]):
             if (rel_ch['k_s'][i]+np.sqrt(rel_ch['vark_s'][i])>=0.3272) and (rel_ch['k_s'][i]-np.sqrt(rel_ch['vark_s'][i])<=0.3272) and (rel_ch['k_k'][i]+np.sqrt(rel_ch['vark_k'][i])>=0.6101)and (rel_ch['k_k'][i]-np.sqrt(rel_ch['vark_k'][i])<=0.6101):
                ch_mask.append(True)
             else:
                ch_mask.append(False)
    return um, uvar, bsk, bks, rel_ch, ch_mask