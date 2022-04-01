# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:21:27 2022

@author: Andrea
"""
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
    import time
    start = time.time()
    from scipy.stats import ttest_1samp, rayleigh, uniform, kstest, bartlett
    from statsmodels.stats.diagnostic import lilliefors #lilliefors is a K-S test with a gaussian with unknown meand and sd.
    
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
    m = 0
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
                    if p < pval:
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
                            r = np.abs(sp-np.mean(sp))
                            th = np.angle(sp)
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
    #As stated in the article, this is a ulterior check for gaussianity.
    # for ch in range(0,shp[2]):
    #     tmp = np.array([],dtype = np.uint8)
    #     for sgn in range(0,shp[0]):
    #         _, p = lilliefors(np.real(D[sgn,-100:,ch]),dist='norm')
    #         if p<0.05:
    #             tmp = np.append(tmp,sgn)
    #     nogoodgauss['{}'.format(ch)] = tmp
    print("---Noise evaluaton completed in {:.2f} seconds.---".format(time.time()-start))
    return nogood