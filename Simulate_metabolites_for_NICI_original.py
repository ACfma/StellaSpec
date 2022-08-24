# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:41:38 2021

@author: Andrea

A ready-to-use MRSI analysis pipeline focused on NICI goals:
    finding the ratio between PME and PDE using 31P-MRSI.
"""
## ---(Wed Sep 22 09:40:11 2021)---
import os
import h5py
import json

import numpy as np
from numba import jit

import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.optimize import minimize, curve_fit
from scipy.io import loadmat
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from skimage import feature, restoration

from fsl_mrs.core import MRS
from fsl_mrs.utils import  plotting, fitting
from fsl_mrs.utils.mrs_io  import lcm_io
from fsl_mrs.utils.report import create_plotly_div
from fsl_mrs.denmatsim import simseq, simulator
from fsl_mrs.utils.misc import checkCFUnits,FIDToSpec, SpecToFID, calculate_lap_cov, calculate_crlb
import fsl_mrs.utils.models as models

import nmrglue as ng
from mrs_denoising import denoising

from PCA_denoising import PCA_denoising
from scan_animation import scan_animation

#Simple routin for iterativelly phasing (from C:\Users\Andrea\Desktop\Spettroscopia\Evaluating tissue bioenergetics byphosphorous-31 magnetic resonancespectroscopy) the spectrum for absorption spectrum
def simple_phasing(spectra, nstep = 10000):
    '''
    Simple phasing made by searching most positive absorption spectra by imposing\
        absorption maximum as positive and dispersion as sum zero.

    Parameters
    ----------
    spectra : complex numpy array
        Array of spectrum
    nstep : int, optional
        number of angles to try between zero and 2*np.pi. The default is 100.

    Returns
    -------
    absorp_spectra : numpy array
        Absorption spectrum extracted from signal
    disp_spectra : numpy array
        Diffusion spectrum extracted from signal.

    '''
    phi = 0
    old_min = np.min(np.real(spectra))#if already phased, the absorption spectra is the real part of the complex one.
    old_disp = 10e8
    perc = np.percentile(np.abs(spectra),99.)
    pos = [i for i,v in enumerate(np.abs(spectra)) if v > perc]
    phi_zero = 0
    phi_best = 0
    while phi < 2*np.pi:
        integral = np.real(spectra)*np.cos(phi) + np.imag(spectra)*np.sin(phi)
        disp_spectra = np.imag(spectra)*np.cos(phi) - np.real(spectra)*np.sin(phi)
        if np.min(integral) > old_min and np.abs(np.sum(disp_spectra))<old_disp:
            phi_zero = phi
            old_min = np.min(integral)
            old_disp = np.abs(np.sum(disp_spectra))
        phi = phi+2*np.pi/nstep
    for phi in np.linspace(0,2*np.pi,nstep):
        phi_one = phi*np.linspace(0,len(spectra)-1,len(spectra))/len(spectra)
        phi_corr = phi_zero + phi_one
        integral = np.real(spectra)*np.cos(phi_corr) + np.imag(spectra)*np.sin(phi_corr)
        disp_spectra = np.imag(spectra)*np.cos(phi_corr) - np.real(spectra)*np.sin(phi_corr)
        if np.min(integral) > old_min and np.abs(np.sum(disp_spectra))<old_disp:
            phi_best = phi_corr
            old_min = np.min(integral)
            old_disp = np.abs(np.sum(disp_spectra))
        
    #phi already found for absorption
    absorp_spectra = np.real(spectra)*np.cos(phi_best)+np.imag(spectra)*np.sin(phi_best)
    disp_spectra = np.imag(spectra)*np.cos(phi_best) - np.real(spectra)*np.sin(phi_best)
    return absorp_spectra, disp_spectra, phi_best
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
def man_phs(data):
    '''
    Return phased spectra

    Parameters
    ----------
    data : array
        Array of spectra to be phased.

    Returns
    -------
    Array
        Output phased spectra.

    '''
    def ps(data, p0=0.0, p1=0.0, pivot = 0.0, inv=False): #AC, added pivot as input to ng.proc_base.ps input so that ot could be compatiblr with manual_ps
        """
        Linear phase correction

        Parameters
        ----------
        data : ndarray
            Array of NMR data.
        p0 : float
            Zero order phase in degrees.
        p1 : float
            First order phase in degrees.
        inv : bool, optional
            True for inverse phase correction

        Returns
        -------
        ndata : ndarray
            Phased NMR data.

        """
        p0 = p0 * np.pi / 180.  # convert to radians
        p1 = p1 * np.pi / 180.
        size = np.shape(data)[0]
        # apod = np.exp(1.0j * (p0 + (p1 * np.arange(size) / size)) #AC, old version
        #               ).astype(data.dtype)
        apod = np.exp(1.0j * (p0+ (p1 * np.arange(-pivot, -pivot + size)  / size)) #AC, new version
                      ).astype(data.dtype)
        if inv:
            apod = 1 / apod
        return apod * data
    a = data
    p0, p1, pivot = ng.proc_autophase.manual_ps(a)
    b = ps(a, p0, p1, pivot)
    return b
def Lor(x,M0,T2_s, omega_0):
    return M0*(T2_s/(1+((x-omega_0)*T2_s)**2))
def Gauss(x, M0,omega_0,sigma):
    return M0*(1/(sigma*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-omega_0)/sigma)**2)))

def Vog(x, M0, T2_s,omega_0,sigma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    from scipy.special import wofz
    return np.real(M0*wofz((x-omega_0 + 1j*T2_s)/sigma/np.sqrt(2))) / sigma\
                                                            /np.sqrt(2*np.pi)
def fit_peak(data, T2_s, kind,  sigma = 1):
    from scipy.optimize import curve_fit
   
    x = np.linspace(-len(data)/2,len(data)/2,len(data))
    if kind == 'Lor':
        fun = Lor
        p0 = [np.max(np.abs(data)), T2_s, len(data)/2]
    elif kind == 'Gauss':
         fun = Gauss
         p0 = [np.max(np.abs(data)), len(data)/2, sigma]
    else:
        fun = Vog
        p0 = [np.max(np.abs(data)), T2_s, len(data)/2, sigma]
    popt,pcov = curve_fit(fun,x,data,p0)
    plt.figure()
    plt.plot(x,data)
    plt.plot(x,fun(x,*popt))
    return popt,pcov

def simsyst(spinsys,B0,LB,points,bw,TR,B1,samp,G,T=[0,1],GB=0.0,gamma=17.235E6,shift=None,shift_cf=None,autophase=False, plot=False):
    '''
    

    Parameters
    ----------
    spinsys : list of dictionary
        List of spin system to simulate.
    B0 : float
        Static field strenght (in T).
    LB : float
        Lorentzian line broadening (in Hz).
    points: int
        number of point of the output fids.
    bw : float
        Acquisition bandwidth (in Hz).
    B1 : 1darray or list
        List of amplitudes for B1 field (in T).
    T : 1darray or list
        Time duration of the B1 pulse (in s). The default is [0,1].
    GB : float
        Gaussian line broadening (in Hz). The default is 0.0.
    gamma : float
        Gyromagnetic ratio of nucleus (in Hz). The default is 17.235E6.
    shift : float
        Shift (in ppm) from the zero. The default is 0.0.
    plot : bool
        If True plot spectra of spinsys. The default is False.


    Returns
    -------
    basis_fids : list
        List of basis fids for the spin system given as input.
    basis_names : list
        List of basis names for the spin system given as input.

    '''
    contcol=0
    cmap = cm.get_cmap('viridis')
    t = np.linspace(0,points/bw,points)
    flip = 2*np.pi*B1*(T[-1]-T[0])*1e-3*gamma #flip angle for hardpulse (soon to be bettered to accomodate all kind of pulses)
    #plt.plot(apo)
    tmp=[]#tmp holds the singular fids for alpha,beta and gamma ATP
    b_f=[]
    b_n=[]
    phase = np.where(samp>0,0.00000, -3.14159)
    for i,sys in enumerate(spinsys):
        Seq = {
            "sequenceName": "fidall",
            "description": "Fidall test",
            "B0": B0, 
            "centralShift": 0.0,
        
            "Rx_Points":points,
            "Rx_SW":bw,
            "Rx_LW": LB[i],#(np.pi*sys['T2']/7),#Rolf usualy uses 1Hz
            "Rx_Phase": np.deg2rad(0),
        
            "x":[-15,15],
            "y":[-15,15],
            "z":[-15,15],
            "resolution":[1,1,1],
        
            "CoherenceFilter": [None],
        
            "RFUnits":"mT",
            "GradUnits":"mT",
            "spaceUnits":"mm",
        
            "RF": [
                {
                    "time": (T[-1]-T[0])*1e-3,
                    "frequencyOffset": 0,
                    "phaseOffset": 0,
                    "amp": np.abs(B1*1000*samp),
                    "phase": [0]*len(samp),
                    "grad": [0,0,G[2]*1000]
                }
            ],
        
            "delays": [0.00046],
            "rephaseAreas": [
                [0,
             0,
             (G[2]*1000)*((T[-1]-T[0])*1e-3)/2]],#np.sum(pulse_seq['gradZ']['Amplitude'][pulse_seq['gradZ']['Time']>20*1e-3]*pulse_seq['gradZ']['Time'][pulse_seq['gradZ']['Time']>20*1e-3].diff())            
            "method": 'full'#'full' method should work also without gradients
        }
        T1 = sys['T1']
        if autophase:
            # Construct a single spin spin system
            apshift = 20.0
            apsystem = {'shifts': np.array([apshift]),
                        'j': np.array([[0.0]]),
                        'scaleFactor': 1.0}
            # Simulate it
            FID, ax, pmat = simseq.simseq(apsystem, Seq)
    
            # Determine phase adj needed
            FID = np.pad(FID, (0, 10 * 8192))
            apspec = FIDToSpec(FID)
            maxindex = np.argmax(np.abs(apspec))
            apPhase = np.angle(apspec[maxindex])
            print(f'Auto-phase adjustment. Phasing peak position = {apshift:0.2f} ppm')
            print(f'Rx_Phase: {Seq["Rx_Phase"]}')
            newRx_Phase = Seq["Rx_Phase"] + apPhase
            Seq["Rx_Phase"] = newRx_Phase
            print(f'Additional phase: {apPhase:0.3f}\nFinal Rx_Phase: {newRx_Phase:0.3f}')

        apo = np.exp(-(1/(4*np.log(2)))*(GB[i]*np.pi*t)**2) #Gaussian broadening for voigt lineshape
        # with open('Seq.json', 'w') as fout:
        #     json.dump(Seq, fout)
        # with open('sys.txt', 'w') as fout:
        #     json.dump(spins[i], fout)
        # os.system(r'python C:\Users\Andrea\Documents\GitHub\fsl_mrs\fsl_mrs\scripts\fsl_mrs_sim -m sys.txt -p -20.0 Seq.json --verbose')
        if np.sum(shift_cf)!=None:
            sys['shifts']=sys['shifts']+shift_cf[i]
            
            
        rel = 1#sys['scaleFactor']#np.sin(flip)*(1-np.exp(-TR/T1))/(1-np.cos(flip)*np.exp(-TR/T1))*sys['scaleFactor']
        
        print(rel)
        # if sys['name']=='aATP' or sys['name']=='bATP' or sys['name']=='gATP':
        #     tmp.append(simseq.simseq(sys, Seq)[0]*apo*rel)
        #     if len(tmp)==3:
        #         if shift==None:
        #             shf = 1
        #         else:
        #             shf=np.exp(2j*np.pi*(shift[i]*B0*gamma*1e-6)*(np.linspace(0,points,points)/bw))
        #         b_f.append(sum(tmp)*shf)
        #         b_n.append('ATP')
        # else:
        if shift==None:
            shf = 1
        else:
            shf=np.exp(2j*np.pi*(shift[i]*B0*gamma*1e-6)*(np.linspace(0,points,points)/bw))
        b_f.append(apo*simseq.simseq(sys, Seq)[0]*rel*shf)
        b_n.append(sys['name'])
    if plot:
        maxplot = np.max(np.abs(np.fft.fftshift(np.fft.fft(np.array(b_f),axis=1),axes=0)))
        maxabs = np.max(np.abs(np.fft.fftshift(np.fft.fft(sum(b_f)),axes=0)))
        if maxabs>maxplot:
            maxplot = maxabs
        plt.figure()
        for fid_b in b_f:
            #plt.plot(np.linspace(-bw/2,bw/2,Seq["Rx_Points"])/(B0*gamma*10**-6),np.abs(np.fft.fftshift(np.fft.fft(fid_b)))/maxplot,color=cmap(contcol))
            contcol+=30
        plt.plot(np.linspace(-bw/2,bw/2,Seq["Rx_Points"])/(B0*gamma*10**-6),np.abs(np.fft.fftshift(np.fft.fft(sum(b_f))))/maxplot,color='black')
        plt.xlim([-20,20])
        plt.xlabel("Chemical Shift [ppm]",fontsize=12)
        plt.ylabel("Amplitude [a.u.]",fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_xaxis()
        plt.ylim([0,1])
        plt.title("Simulated liver spectrum")
        #plt.legend(b_n)
        
    return b_f, b_n

def pyth2LCM(basis_fids,basis_names,n,header = {'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': 7*17.235E6, 'dwelltime': 1./10000,'width':1, 'points':540},\
             info = {'ID': 'Test', 'FMTDAT':'(2E16.6)', 'Volume': 1.0, 'TRAMP': 1.0}):
    # see SEQPAR in LCmodel Manual except for 'width' which is a starting value for the FWHM (used the value found in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4438275/ multiplied by the center frequency in order to obtin the same FWHMBA).
    header['centralFrequency'] = checkCFUnits(header['centralFrequency'], units='MHz')
    cont = 0
    #CAREFULL! LCModel wants all positive amplitude spectra for the metabolites so they need to be phased or reference should be omitted
    #Given that it's just one peak, it should be enaught to use zero phase correction
    ph_fids = []
    for i,fid in enumerate(basis_fids):
        ph_fids.append(fid*np.exp(2j*np.pi*(-4.65*header['centralFrequency'])*(np.linspace(0,header['points'],header['points'])/header['bandwidth'])))#shifting freqiencies with respect to water peaks
    conj = True
    for name in basis_names:
        info = {'ID': '{}_base'.format(name), 'FMTDAT':'(2E16.6)', 'Volume': 1.0, 'TRAMP': 1.0}
        lcm_io.saveRAW('{}.RAW'.format(name),ph_fids[cont], info = info, hdr=header, conj=conj)
        plt.plot(np.abs(np.fft.fft(ph_fids[cont])))
        cont = cont+1
    #Exclude PCr from basis
    # phs = {}
    # for i,n in enumerate(basis_names):
    #     if conj:
    #         phs[n]=[-phase_corrs[i][0],-phase_corrs[i][1]]
    #     else:
    #         phs[n]=[phase_corrs[i][0],phase_corrs[i][1]]
    lcm_io.writeLcmInFile('{}_basis.in'.format(n),basis_names,'','{}'.format(n),header,-4.65,0.0)
    return
#%%
# tmp = man_phs(specs[8,8,8,:])
# popt,pcov=fit_peak(tmp[480:520]+120000,0.7,'Lor',6)
# popt_pi = popt
# x = np.linspace(0,1016,1016)
# popt,pcov=fit_peak(tmp[550:580]+100000,0.7,'Lor',1)
# popt_pcr = popt
# tmp = man_phs(np.fft.fft(basis_fids[6]))
# popt,pcov=fit_peak(np.real(tmp[20:40]),0.7,'Lor',6)
# popt_pi_t = popt
# c = popt_pi_t[1]/popt_pi[1]
# tmp = man_phs(np.fft.fft(basis_fids[0]))
# popt,pcov=fit_peak(tmp[30:50],0.7,'Lor',1)
# popt_pi_pc = popt
# tmp = man_phs(np.fft.fft(basis_fids[3]))
# popt,pcov=fit_peak(tmp[10:40],0.1,'Lor',6)
# popt_pi_gpc = popt
# tmp = man_phs(np.fft.fft(basis_fids[5]))
# popt,pcov=fit_peak(tmp[:50],0.7,'Lor',1)
# popt_pi_ptc = popt
# x = np.linspace(0,1016,1016)
# plt.plot(x,Lor(x,popt_pi[0]/2,popt_pi[1],508.5))#,popt_pi[3]))
# plt.plot(x,Lor(x,popt_pi[0]/8,popt_pi_pc[1],492.5))
# plt.plot(x,Lor(x,-popt_pi[0]/8,popt_pi_gpc[1],524.5))#,popt_pi_t[3]))
# plt.plot(x,Lor(x,-popt_pi[0]/8,popt_pi_ptc[1],542))#,popt_pi_t[3]))
#%%
if __name__ == "__main__":
    #%simulation of 31P NMR of PE, GPE, PC, GPC spynsistem at 7T. It could be imported as file but, for now, this version is more understandable
    #The shifts and coupling constants are taken from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4438275/ and refers to PCr resonance (shift PCr = 0.0 ppm)
    filename = 'SpinSys_NICI_v9_OXSA.txt'
    with open(filename, 'r') as f:
        spinSys = [json.loads(line) for line in f]
    sys_list = []
    gamma=17.235E6
    B0=7
    cf = B0*gamma*1e-6
    for i,_ in enumerate(spinSys):
        sys_list.append(simulator.simulator(spinSys=spinSys[i],B0=B0,gamma=gamma))
    
    #sys.thermalEq()
    #sys.getStaticHamiltonians()
    #%% Importing sequence informations
    pulse_seq = {}
    arg = ['gradX', 'gradY', 'gradZ', 'rho', 'SSFP', 'theta']
    for i,v in enumerate(arg):
        pulse_seq["{}".format(v)] = pd.read_csv(r'./pulse_sequence/{}'.format(v),delimiter=' ', names = ['Time', 'Amplitude'],skiprows=2)
    
    
    
    #%%Simulation of the used sequence
    T = pulse_seq['rho'][pulse_seq['rho']['Amplitude']!=0]['Time'].values
    basis_fids=[]
    basis_names=[]
    flip = 13*(np.pi/180)
    TR = 0.066
    B0=7
    gamma=17.235E6
    bw = 10000
    tbw = 10000
    indx = pulse_seq['rho']['Amplitude'][pulse_seq['rho']['Amplitude']!=0].index
    ampflip = flip/(2*np.pi*(T[-1]-T[0])*1e-3*gamma)#Calculate B1 from nominal flip angle (10-3 for ms to s)
    points = 540
    sthick=0.1
    samp = np.sinc((tbw)*np.linspace(-0.00026,0.00026,30))#Sinc function for hardpulse but, in the end, is the same that giving a constant amplitude for a given period with the exception that is slower
    samp = samp/(np.sum(samp))
    samp=np.array([1])
    Gz = bw/(gamma*sthick)
    G = [0,0,0]
    LB_b = np.array([10, 10, 20, 20, 40, 40, 30, 20, 50, 20, 20, 5])#
    GB_b = [0]*len(spinSys)#[0.1]*len(LB_b)
   
    # zeroRefSS = {"shifts": [0], "j": [[0]], "scaleFactor": 1.0,"T1":4.0 }
    # for iDx, s in enumerate(spinSys):
    #     if isinstance(spinSys[iDx], dict):
    #         spinSys[iDx] = [spinSys[iDx], zeroRefSS]
    #     elif isinstance(spinSys[iDx], list):
    #         spinSys[iDx] = spinSys[iDx] + [zeroRefSS]
    #     else:
    #         raise ValueError('spinsys[iDx] should either be a dict or list of dicts')
    
    # for i,s in enumerate(spinSys):
    #     simsyst([s,spinSys[-2]],B0, LB_b,points,bw,TR,ampflip,T, GB=GB_b, shift = [0]*len(spinSys), shift_cf=[0]*len(spinSys),autophase=False,plot=True)

    basis_fids, basis_names = simsyst(spinSys,B0, LB_b,points,bw,TR,ampflip,samp,G,T, GB=GB_b, shift = None, shift_cf=None,autophase=False,plot=True)#0.049*120.667
    #SIMULATED T1 RELAXATION FACTOR
    sat = np.array([1.59547623, 2.12161307, 2.12828879, 2.29323398, 1.,1.03321265, 1.01011809,1.18823545, 1.29602092, 1.50159809, 1.93042119, 2.15114022])
    relax={}
    for i,name in enumerate(basis_names):
        relax[name]=1./sat[i]
        
    liv_amp = {'PC':1,'PE':1,'GPC':2.5,'GPE':2.5,'aATP':3,'bATP':3,'gATP':3,'PtdC':0.6,'Pi':2,'NAD+':0.5,'UDPG':0.6,'PCr':0.5}
    #liv_conc1 = {'PC':1.06,'PE':0.77,'GPC':2.34,'GPE':1.5,'aATP':2.74,'bATP':2.74,'gATP':2.74,'PtdC':1.38,'Pi':2.23,'NAD+':2.37,'UDPG':2.,'PCr':0.5}
    liv_conc = {'PC':1,'PE':1,'GPC':1,'GPE':1,'aATP':1,'bATP':1,'gATP':1,'PtdC':1,'Pi':1,'NAD+':1,'UDPG':1,'PCr':1}
    #basis_fids = [x for i,x in enumerate(basis_fids)] 
    plt.figure()
    plt.plot(np.linspace(-bw/2,bw/2,points)/(B0*gamma*10**-6),np.real(np.fft.fftshift(np.fft.fft(sum([basis_fids[i]*liv_amp[name] for i,name in enumerate(basis_names)]))))/np.max(np.abs(np.fft.fftshift(np.fft.fft(sum([basis_fids[i]*liv_amp[name] for i,name in enumerate(basis_names)]))))),color='black')
    plt.xlim([-20,10])
   # plt.ylim([0,1.2])
    plt.xlabel("Chemical Shift [ppm]",fontsize=12)
    plt.ylabel("Amplitude [a.u.]",fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_xaxis()
    #plt.ylim([0,1])
    plt.title("Simulated liver spectrum (magnitude)")

    
    #%%Save basis FIDS as rawfile
    
    #base
    name = 'Liver_sim'
    header = {'bandwidth':bw, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./bw,'width':np.min(LB_b), 'points':points}
    info = {'ID': name, 'FMTDAT':'(2E16.6)', 'Volume': 1.0, 'TRAMP': 1.0}
    pyth2LCM(basis_fids, basis_names,name,header,info)
    
    
    
    #%%
    with open('control_voigt_one_pme.txt', 'r') as file:
    # read a list of lines into data
        data = file.readlines()

    for i in range(1500):
        # now change the 2nd line, note that you have to add a newline
        data[20] = " filraw= '/home/andrea/LCMTEST/Test/control{}.RAW'\n".format(i)
        data[21] = " filps= '/home/andrea/LCMTEST/ps{}.PS'\n".format(i) 
        data[22] = " filcsv= '/home/andrea/LCMTEST/csvtest{}.csv'\n".format(i) 

    # and write everything back
        with open(r'C:\Users\Andrea\Documents\GitHub\Spettroscopia\ControlsSNR\control{}.CONTROL'.format(i), 'w') as file:
            file.writelines( data )
            file.close()
    #%%
    plt.figure()
    LB_b1,GB_b1 = 0.049*126.45,0
    basis_fids1, basis_names = simsyst(spinSys,B0, LB_b1,points,10000,TR,ampflip,T, GB=GB_b1, shift = [0]*len(spinSys),plot=False)
    for fid_b in basis_fids1:
        plt.plot(np.linspace(-5000,5000,540),np.abs(np.fft.fftshift(np.fft.fft(fid_b))))
    plt.xlabel("$\Delta$f [Hz]",fontsize=12)
    plt.ylabel("Amplitude [a.u.]",fontsize=12)
    plt.xticks(fontsize=12)
    plt.title("Simulation of possible liver spectrum with lb={:.0f} Hz".format(LB_b1))
    plt.legend(basis_names)
    plt.xlim([-2400,2400])
    plt.figure()
    
    LB_b2,GB_b2 = 20,0
    basis_fids2, basis_names = simsyst(spinSys,B0, LB_b2,points,10000,TR,ampflip,T, GB=GB_b2, shift = [0]*len(spinSys),plot=False)
    for fid_b in basis_fids2:
        plt.plot(np.linspace(-5000,5000,540),np.abs(np.fft.fftshift(np.fft.fft(fid_b))))
    plt.xlabel("$\Delta$f [Hz]",fontsize=12)
    plt.ylabel("Amplitude [a.u.]",fontsize=12)
    plt.xticks(fontsize=12)
    plt.title("Simulation of possible liver spectrum with lb={:.0f} Hz".format(LB_b2))
    plt.legend(basis_names)
    plt.xlim([-2400,2400])
    
    plt.figure()
    plt.plot(np.linspace(-5000,5000,540),np.abs(np.fft.fftshift(np.fft.fft(sum(basis_fids1)))),linestyle='-',color='darkblue')
    plt.plot(np.linspace(-5000,5000,540),np.abs(np.fft.fftshift(np.fft.fft(sum(basis_fids2)))),linestyle='--',color='darkorange')
    plt.xlim([-2400,2400]) 
    plt.xlabel("$\Delta$f [Hz]",fontsize=12)
    plt.ylabel("Amplitude [a.u.]",fontsize=12)
    plt.xticks(fontsize=12)
    plt.title("Confront between the spectras")
    plt.legend(['lb = {:.0f}Hz'.format(LB_b1),'lb = {:.0f}Hz'.format(LB_b2)])

    #%% SIMULATION FOR MODEL FUNCTION
    perc = [0.05]
    B =[[20,0], [20,10],[20,20], [20,30], [20,40]] #[LB,GB]
    model_s =['voigt','lorentzian']
    bmatrix = np.zeros((len(B),len(model_s)))
    vmatrix = np.zeros((len(B),len(model_s)))
    rmsmatrix = np.zeros((len(B),len(model_s)))
    crsdmatrix = np.zeros((len(B),len(model_s)))
    parsdmatrix = np.zeros((len(B),len(model_s)))
    snrmatrix =[]
    indx = np.where(np.array(basis_names)=='PCr')[0][0]
    sysindx = np.where(np.array([x['name'] for x in spinSys]) == 'PCr')[0][0]
    for p in perc:
        for i,lb in enumerate(B):
            sim_fids, _ = simsyst([spinSys[sysindx]],B0,lb[0],2048,10000,ampflip, GB=lb[1])#0.049*120.667
            sig = p*np.max(np.abs(np.fft.fft(sim_fids[0])))
            # a = MRS(FID=basis_fids[indx], header = {'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': 120.664, 'dwelltime': 1./10000},
            #         basis = basis_fids[indx],
            #         names = [basis_names[indx]],
            #         basis_hdr=[{'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': 120.664, 'dwelltime': 1./10000, 'fwhm':20}],
            #         nucleus='31P',bw=10000, cf = 120.664)
            # a.rescaleForFitting()
            # results = fitting.fit_FSLModel(a,method='Newton',ppmlim=(-20.0,10.0),model=model,baseline_order=0, MHSamples=5000,metab_groups=[0])
            tCRLB=[] #theoretical CRLB
            sCRLB=[]
            # init = results.params
            tmpvarconc = []
            tmpbiasconc = []
            for k in range(100):   
                c = 1
                tmp = MRS(FID=np.fft.ifft(np.fft.fft(c*sim_fids[0])+np.random.normal(0,sig,len(sim_fids[0]))+1j*np.random.normal(0,sig,len(sim_fids[0]))), header = {'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': 120.664, 'dwelltime': 1./10000},
                        basis = basis_fids[indx],
                        names = [basis_names[indx]],
                        basis_hdr=[{'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': 120.664, 'dwelltime': 1./10000, 'fwhm':20}],
                        nucleus='31P',bw=10000, cf = 120.664)
                #tmp.rescaleForFitting(ind_scaling=['PCr'])
                for m in model_s:
                    tmpres = fitting.fit_FSLModel(tmp,method='Newton',ppmlim=(-50.0,50.0),model=m,baseline_order=-1, MHSamples=5000,metab_groups=[0])
                    sCRLB.append(tmpres.crlb[0])
                    tmpvarconc.append(tmpres.params[0])
                    tmpbiasconc.append(tmpres.params[0]-c)
            
            n_m = len(model_s)
            bmatrix[i,:], vmatrix[i,:], rmsmatrix[i,:], crsdmatrix[i,:], parsdmatrix[i,:] = [np.mean(tmpbiasconc[0::n_m]),np.mean(tmpbiasconc[1::n_m])], [np.var(tmpvarconc[0::n_m]),np.var(tmpvarconc[1::n_m])],\
                                                                            [np.sqrt(np.mean(np.array(tmpbiasconc[0::n_m]))**2+np.var(tmpvarconc[0::n_m])),np.sqrt(np.array(np.mean(tmpbiasconc[1::n_m])**2)+np.var(tmpvarconc[1::n_m]))],\
                                                                                [np.mean(sCRLB[0::n_m]),np.mean(sCRLB[1::n_m])],\
                                                                                 [np.mean(np.array(tmpbiasconc[0::n_m])**2),np.mean(np.array(tmpbiasconc[1::n_m])**2)]
        ax = [b[1]/(b[0]+b[1])*100 for b in B]
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].plot(ax,bmatrix[:,0]*100/c, marker = 'p')
        axs[0].plot(ax,np.sqrt(vmatrix[:,0])*100/c, marker = 'p')
        axs[0].plot(ax,rmsmatrix[:,0]*100/c, marker = 'p')
        axs[0].plot(ax,np.sqrt(crsdmatrix[:,0])*100/c, marker = 'p')
        axs[0].set_title('Voigt')
        plt.suptitle('SNR = {}'.format(1/p))
        axs[1].plot(ax,bmatrix[:,1]*100/c, marker = 'p')
        axs[1].plot(ax,np.sqrt(vmatrix[:,1])*100/c, marker = 'p')
        axs[1].plot(ax,rmsmatrix[:,1]*100/c, marker = 'p')
        axs[1].plot(ax,np.sqrt(crsdmatrix[:,1])*100/c, marker = 'p')
        axs[0].legend(['bias(%)', 'SD(%)', 'RMSE(%)', 'sCRLB(SD%)'])
        axs[1].set_title('Lorentzian')
        plt.xlabel('GF(%)',fontsize='large')

        #%%SIMULATION FOR SNR BREAKEDOWN
        from FSL_crlb_apofix import LR_apo
        tCRLB=[]
        indx = np.where(np.array(basis_names)=='PCr')[0][0]
        sysindx = np.where(np.array([x['name'] for x in spinSys]) == 'PCr')[0][0]
        
        perc=[0.01,0.05,0.1,0.2,0.5]
        def LR(res, mrs, fitResults, sigma):
            "Load fitting results and calculate some metrics"
            # Populate data frame
            if fitResults.ndim == 1:
                res.fitResults = pd.DataFrame(data=fitResults[np.newaxis, :], columns=res.params_names)
            else:
                res.fitResults = pd.DataFrame(data=fitResults, columns=res.params_names)
            res.params = res.fitResults.mean().values

            # Store prediction, baseline, residual
            res.pred = res.predictedFID(mrs, mode='Full')
            res.baseline = res.predictedFID(mrs, mode='Baseline')
            res.residuals = mrs.FID - res.pred

            # Calculate single point crlb and cov
            first, last = mrs.ppmlim_to_range(res.ppmlim)
            _, _, forward, _, _ = models.getModelFunctions(res.model)

            def forward_lim(p):
                return forward(p, mrs.frequencyAxis,
                               mrs.timeAxis,
                               mrs.basis,
                               res.base_poly,
                               res.metab_groups,
                               res.g)[first:last]
            data = mrs.get_spec(ppmlim=res.ppmlim)
            # self.crlb      = calculate_crlb(self.params,forward_lim,data)
            res.cov = calculate_lap_cov(res.params, forward_lim, data,sigma**2)
            res.crlb = np.diagonal(res.cov)
            std = np.sqrt(res.crlb)
            res.corr = res.cov / (std[:, np.newaxis] * std[np.newaxis, :])
            res.mse = np.mean(np.abs(FIDToSpec(res.residuals)[first:last])**2)

            with np.errstate(divide='ignore', invalid='ignore'):
                res.perc_SD = np.sqrt(res.crlb) / res.params * 100
            res.perc_SD[res.perc_SD > 999] = 999   # Like LCModel :)
            res.perc_SD[np.isnan(res.perc_SD)] = 999
            return res

        bmsnr, vmsnr, rmsmsnr, crsdmsnr, parsdmsnr, crsdmsnrapo = np.zeros((len(perc))),np.zeros((len(perc))),np.zeros((len(perc))),np.zeros((len(perc))),np.zeros((len(perc))),np.zeros((len(perc)))
        a = MRS(FID=basis_fids[indx], header = {'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': 120.664, 'dwelltime': 1./10000},
                basis = basis_fids[indx],
                names = [basis_names[indx]],
                basis_hdr=[{'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000, 'fwhm':20}],
                nucleus='31P',bw=10000, cf = cf)
        #a.rescaleForFitting()
        results = fitting.fit_FSLModel(a,method='Newton',ppmlim=(-50.0,50.0),model='voigt',baseline_order=-1, MHSamples=5000,metab_groups=[0])
        
        for i,p in enumerate(perc):
            sig = p*np.max(np.abs(a.get_spec()))
        #     first, last = a.ppmlim_to_range(results.ppmlim)
        #     _, _, forward, _, _ = models.getModelFunctions('voigt')
        
        #     def f_l(p):
        #         return forward(p, a.frequencyAxis,
        #                        a.timeAxis,
        #                        a.basis,
        #                        results.base_poly,
        #                        results.metab_groups,
        #                        results.g)[first:last]
        #     crlb = calculate_lap_cov([1,0,0,0,0,0,0,0], f_l,[0]*len(basis_fids[0]),sig2=sig**2)
            results = LR(results, a, results.params, sig)
            #apo_res = LR_apo(results, a, results.params, None,apo*0)
            tCRLB.append(results.crlb[0])
            sCRLB=[]
            apocrlb=[]
            tmpvarconc = []
            tmpbiasconc = []
            for k in range(100):   
                c = 1
                tmp = MRS(FID=SpecToFID((FIDToSpec(c*a.FID)+np.random.normal(0,sig,len(a.FID))+1j*np.random.normal(0,sig,len(a.FID)))), header = {'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': 120.664, 'dwelltime': 1./10000},
                        basis = basis_fids[indx],
                        names = [basis_names[indx]],
                        basis_hdr=[{'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000, 'fwhm':20}],
                        nucleus='31P',bw=10000, cf = cf)
                #tmp.rescaleForFitting()
                tmpres = fitting.fit_FSLModel(tmp,method='Newton',ppmlim=(-3.5,3.5),model='voigt',baseline_order=-1, MHSamples=5000,metab_groups=[0])
                sCRLB.append(tmpres.crlb[0])
                apo_res = LR_apo(tmpres, tmp, tmpres.params, None,apo)
                apocrlb.append(apo_res.crlb[0])
                tmpvarconc.append(tmpres.params[0])
                tmpbiasconc.append(tmpres.params[0]-c)
            print("SNR {} done".format(1/p))
            bmsnr[i], vmsnr[i], rmsmsnr[i], crsdmsnr[i], parsdmsnr[i],crsdmsnrapo[i] = np.mean(tmpbiasconc), np.var(tmpvarconc),\
                                                                            np.mean(tmpbiasconc)**2+np.var(tmpvarconc),\
                                                                                    np.mean(sCRLB),\
                                                                                     np.mean(np.array(tmpbiasconc)**2), np.mean(apocrlb)
            plt.figure()
            plt.hist(tmpvarconc,bins=30)
            plt.xlabel('Concentration [a.u]')
            plt.ylabel('Counts')
        plt.figure()
        plt.plot(np.array(perc)**(-1),np.array(vmsnr)*100/c, marker = 'p')
        #plt.plot(np.array(perc)**(-1),np.array(tCRLB)*100/c, marker = 'p')
        #plt.plot(np.array(perc)**(-1),(bmsnr**2)*100/c, marker = 'p')
        #plt.plot(np.array(perc)**(-1),rmsmsnr*100/c, marker = 'p')
        plt.plot(np.array(perc)**(-1),np.array(parsdmsnr*100/c), marker = 'p')
        plt.plot(np.array(perc)**(-1),np.array(crsdmsnr)*100/c, marker = 'p')
        plt.plot(np.array(perc)**(-1),np.array(crsdmsnrapo)*100/c, marker = 'p')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('SNR')
        plt.ylabel('Value [a.u.]')
        plt.suptitle('SNR plot')
        #plt.yscale('log')
        plt.legend(['tCRLB(%)','Var[$C_{SNR}$](%)','sCRLB(%)'])
        print(tCRLB/parsdmsnr)
        #%% Liver MC sim
        perc=[0.001,0.01, 0.1,0.5]
        tcrlbmsnr=np.zeros((len(perc),len(basis_names)))
        fake_liver = np.zeros((len(basis_names),len(basis_fids[0])),dtype = 'complex')
        flconc = [1.06,0.77,2.34,1.50,2.74,1.38,2.23,2.37,2.0,0.5]#From Purvis et al. PCr value is set arbitrarily
        nsim = 100
        parsdmsnr = np.zeros((len(perc),len(basis_names)))
        for n,_ in enumerate(basis_names): #creating fake liver with litterature concentrations 
            fake_liver[n,:] = flconc[n]*basis_fids[n]
        a = MRS(FID=np.sum(fake_liver,axis = 0), header = {'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': 120.664, 'dwelltime': 1./10000},
                basis =  np.array(basis_fids),
                names = basis_names,
                basis_hdr=[{'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': 120.664, 'dwelltime': 1./10000, 'fwhm':20}],
                nucleus='31P',bw=10000, cf = 120.664)
        #a.rescaleForFitting()
        results = fitting.fit_FSLModel(a,method='Newton',ppmlim=(-20.0,10.0),model='voigt',baseline_order=-1, MHSamples=5000,metab_groups=[0,1,2,3,4,5,6,7,8,9])
        
        for i,p in enumerate(perc):
            sig = p*np.max(np.abs(a.get_spec()))
        #     first, last = a.ppmlim_to_range(results.ppmlim)
        #     _, _, forward, _, _ = models.getModelFunctions('voigt')
        
        #     def f_l(p):
        #         return forward(p, a.frequencyAxis,
        #                        a.timeAxis,
        #                        a.basis,
        #                        results.base_poly,
        #                        results.metab_groups,
        #                        results.g)[first:last]
        #     crlb = calculate_lap_cov([1,0,0,0,0,0,0,0], f_l,[0]*len(basis_fids[0]),sig2=sig**2)
            results = LR(results, a, results.params, sig)
            tcrlbmsnr[i,:] = results.crlb[:len(basis_names)]
            tmpbiasconc = np.zeros((nsim,len(basis_names)))
            for k in range(nsim):
                rc = np.zeros(len(basis_fids))
                for n,_ in enumerate(basis_names):
                    #adding 20% variability
                    rc[n] = 1 + np.random.uniform(low=-0.1, high=0.1)
                    fake_liver[n] = flconc[n]*rc[n]*basis_fids[n]
                    
                tmp = MRS(FID=SpecToFID((FIDToSpec(sum(fake_liver))+np.random.normal(0,sig,len(basis_fids[0]))+1j*np.random.normal(0,sig,len(basis_fids[0])))), header = {'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': 120.664, 'dwelltime': 1./10000},
                        basis =  np.array(basis_fids),
                        names = basis_names,
                        basis_hdr=[{'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': 120.664, 'dwelltime': 1./10000, 'fwhm':20}],
                        nucleus='31P',bw=10000, cf = 120.664)
                #tmp.rescaleForFitting(ind_scaling=['PCr'])
                tmpres = fitting.fit_FSLModel(tmp,method='Newton',ppmlim=(-50.0,50.0),model='voigt',baseline_order=-1, MHSamples=5000,metab_groups=[0,1,2,3,4,5,6,7,8,9])
                tmpbiasconc[k,:] = tmpres.params[:len(basis_names)]-rc
            print("SNR {} done".format(1/p))
            parsdmsnr[i,:] = np.mean(np.array(tmpbiasconc)**2, axis=0)
        plt.figure()
        logratio = np.log10(np.abs(parsdmsnr/tcrlbmsnr))
        SNR = []
        for i,p in enumerate(basis_names):
            plt.plot(np.log10(np.array(perc)),logratio) 
        
        plt.legend(basis_names)
        import plotly.graph_objects as go
        import time
        fig = go.Figure(data=[go.Surface(z=logratio)])
        fig.update_layout(title='log10((parsdmsnr-tcrlbmsnr)/parsdmsnr)')
        fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                          highlightcolor="limegreen", project_z=True))
        # fig.update_layout(scene = dict(
        #             xaxis = dict(
        #                 ticktext= basis_names),
        #             yaxis = dict( ticktext= SNR)))
        time.sleep(1)
        fig.show('browser')
   
    #%%
    #from scipy.interpolate import CubicSpline
    import time
    start=time.time()
    cf = B0*gamma*1e-6
    sps = [ {"j": [[0]],"shifts": [7.30],"name": "PE","scaleFactor": 1,"T1":3.9, "T2":0.0556},
        {"j": [[0]],"shifts": [3.19],"name": "GPC","scaleFactor": 1,"T1":3.9, "T2":0.0504},
        {"j": [[0]],"shifts": [3.71],"name": "GPE","scaleFactor": 1,"T1":4.4, "T2":0.044},
        {"j": [[0,16],[0,0]],"shifts": [-7.40,1000000],"name": "aATP","scaleFactor": 1,"T1":0.46, "T2":0.0182},
        {"j": [[0,15,15],[0,0,0],[0,0,0]],"shifts": [-16.0,1000000,1000000],"name": "bATP","scaleFactor": 1,"T1":0.56, "T2":0.0182},
        {"j": [[0,15],[0,0]],"shifts": [-2.21,1000000],"name": "gATP","scaleFactor": 1,"T1":0.50, "T2":0.0182},
        {"j": [[0]],"shifts": [2.2],"name": "PtdC","scaleFactor": 1,"T1":1.05, "T2":0.0},
        {"j": [[0]],"shifts": [-8.25],"name": "NAD+","scaleFactor": 2, "T1":2, "T2":0.0},
        {"j": [[0]],"shifts": [-9.48],"name": "UDPG","scaleFactor": 1, "T1":3.3, "T2":0.0},
        {"j": [[0]],"shifts": [0.0],"name": "PCr","scaleFactor": 1,"T1":4.0, "T2":0.217}]
    
    n=15
    param = 'LB scaling factor'#np.around(Amps,1)#[int(x) for x in GB_p]#np.around(Amps,1)
    if param == 'sample points':
        pts = [540]*n #100,200,500,
        par = pts
        st = 540
    else:
        pts = [540]*n
    if param == 'LB scaling factor':
        LB_p = np.around(np.linspace(0.45,3,n),2) #100,200,500,
        par = LB_p
        st=np.where(LB_p==1)[0][0]
    else:
        LB_p = [1]*n
    if param == 'GF':
        GF = np.around(np.linspace(0,0.5,n),2) #100,200,500,
        par = GF
        st=np.where(GF==0)[0][0]
    else:
        GF=[0]*n
    if param == 'Amplitude':
        Amps = np.around(np.linspace(0,2,n),1) #100,200,500,
        par = Amps
        st = 1
    else:
        Amps=[1]*n
    if param == 'shift offset':
        shfs = np.around(np.linspace(-0.5,0.5,n),2) #100,200,500,
        par = shfs
        sps = [ {"j": [[0]],"shifts": [3.19],"name": "GPC","scaleFactor": 1,"T1":3.9, "T2":0.0504},
        {"j": [[0]],"shifts": [3.71],"name": "GPE","scaleFactor": 1,"T1":4.4, "T2":0.044},
        {"j": [[0,16],[0,0]],"shifts": [-7.40,1000000],"name": "aATP","scaleFactor": 1,"T1":0.46, "T2":0.0182},
        {"j": [[0,15,15],[0,0,0],[0,0,0]],"shifts": [-16.0,1000000,1000000],"name": "bATP","scaleFactor": 1,"T1":0.56, "T2":0.0182},
        {"j": [[0,15],[0,0]],"shifts": [-2.21,1000000],"name": "gATP","scaleFactor": 1,"T1":0.50, "T2":0.0182},
        {"j": [[0]],"shifts": [2.2],"name": "PtdC","scaleFactor": 1,"T1":1.05, "T2":0.0},
        {"j": [[0]],"shifts": [-8.25],"name": "NAD+","scaleFactor": 2, "T1":2, "T2":0.0},
        {"j": [[0]],"shifts": [-9.48],"name": "UDPG","scaleFactor": 1, "T1":3.3, "T2":0.0},
        {"j": [[0]],"shifts": [0.0],"name": "PCr","scaleFactor": 1,"T1":4.0, "T2":0.217}]
        st=np.where(shfs==0)[0][0]
    else:
        shfs = np.array([0]*n)
    if param == 'T1 ratio':
        T1 = np.around(np.linspace(0.8,1.2,n),2) #100,200,500,
        par = T1
        st = np.where(T1==1)[0][0]
    else:
        T1 = [1]*n
    
    #LB_p, GB_p = np.around(np.linspace(20,90,n),0),[3]*n #From best to worse according to OXSA starting values np.around(np.linspace(20,90,n),0), from best shimming to mostly gaussian np.around(np.linspace(3,30,n),0)
    #Amps =  [1]*n#np.around(np.linspace(0,2,15),1)#From very non detected  to a range that could be pathological
    #shfs = [0]*n#np.around(np.linspace(-0.5,0.5,n),2)#,0,0.75,1.5]
    mc=np.zeros((len(basis_names),2,n,n))
    cont=0
    LCMAmps =[]
    fids_mc = np.zeros((225,540),dtype=complex)
    if param!='shift offset':
        fpk, fpk_names = simsyst(sps,B0,  [ 10, 20, 20, 40, 40, 30, 20, 20, 20, 5],540,10000,TR,ampflip,samp,G,T, GB=GB_b, shift = None,plot=False)#0.049*120.667
    else:
        fpk, fpk_names = simsyst(sps,B0,  [ 20, 20, 40, 40, 30, 20, 20, 20, 5],540,10000,TR,ampflip,samp,G,T, GB=GB_b, shift = None,plot=False)#0.049*120.667
        
    for i,name in enumerate(fpk_names):
        fpk[i] = fpk[i]*liv_conc[name]*liv_amp[name]
    for k in range(n):
        for i in range(n):
            p1={"j": [[0]],"shifts": [5.37+shfs[i]],"name": "Pi","scaleFactor": 1,"T1":1.34*T1[i], "T2":0.0464}
            
            p2={"j": [[0]],"shifts": [6.7+shfs[k],1000000,1000000],"name": "PC","scaleFactor": 1,"T1":2.3*T1[k], "T2":0.0512}
            fpk1, _ = simsyst([p1],B0, [LB_p[i]*LB_b[-4]],pts[i],10000,TR,ampflip,samp,G,T, GB=[(GF[i]/(1-GF[i]))*LB_b[-4]], shift = None,plot=False)#0.049*120.667
            if any(shfs!=0):
                p1_1 = {"j": [[0]],"shifts": [7.30+shfs[k],1000000,1000000],"name": "PE","scaleFactor": 1,"T1":3.9, "T2":0.0556}
                fpk2, _ = simsyst([p2,p1_1],B0, [LB_p[k]*LB_b[0],LB_p[k]*LB_b[1]],pts[k],10000,TR,ampflip,samp,G,T, GB=[(GF[k]/(1-GF[k]))*LB_b[0],(GF[k]/(1-GF[k]))*LB_b[1]], shift = None,plot=False)#0.049*120.667
                FID = Amps[i]*fpk1[0]*liv_conc['Pi']*liv_amp['Pi']+Amps[k]*fpk2[0]*liv_conc['PC']*liv_amp['PC']+Amps[k]*fpk2[1]*liv_conc['PE']*liv_amp['PE']+sum(fpk)
            else:
                fpk2, _ = simsyst([p2],B0, [LB_p[k]*LB_b[0]],pts[k],10000,TR,ampflip,samp,G,T, GB=[(GF[k]/(1-GF[k]))*LB_b[0]], shift = None,plot=False)#0.049*120.667
                fpk1,fpk2=fpk1[0],fpk2[0]
                FID = Amps[i]*fpk1*liv_conc['Pi']*liv_amp['Pi']+Amps[k]*fpk2*liv_conc['PC']*liv_amp['PC']+sum(fpk)
            #sig = 0.00*np.max(np.abs(np.fft.fftshift(np.fft.fft(FID))))
            #np.random.seed(42) #Set same random noise between fits (real and imaginary should be dinstinct in each fit but are the same between different fits)
            fids_mc[k*n+i,:]=FID
            tmp_a = MRS(FID=FID, #+np.random.normal(0,sig,len(fpk1))+1j*np.random.normal(0,sig,len(fpk1))
                    header = {'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000},
                    basis =  np.array(basis_fids),
                    names = basis_names,
                    basis_hdr=[{'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000, 'fwhm':5}],
                    nucleus='31P',bw=10000, cf = cf)
            di = liv_conc.copy()
            for key in di:
                di[key]=di[key]*liv_amp[key]*liv_conc[key]
            LCMAmps.append(di)
            #tmp.rescaleForFitting(ind_scaling=['PCr'])
            info = {'ID': '{}_base'.format(cont), 'FMTDAT':'(2E16.6)', 'Volume': 1.0, 'TRAMP': 1.0}
            header = {'bandwidth':10000.0, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000,'width':5, 'points':pts[i]}
            lcm_io.saveRAW(r'.\Test\control{}.RAW'.format(cont),tmp_a.FID, info = info, hdr=header, conj=True)
            cont += 1
            tmp_res = fitting.fit_FSLModel(tmp_a,method='Newton',ppmlim=(-20.0,10.0),model='voigt',baseline_order=-1,
                                           MHSamples=5000,metab_groups=[0,1,2,3,4,5,6,7,8,9,10,11])
            # fig = plotting.plotly_fit(tmp_a,tmp_res,ppmlim=(-20,10), proj='abs')
            # fig.show()
            for l,name in enumerate(basis_names):
                if name!='Pi' and name!='PC':
                    mc[l,0,k,i] = ((tmp_res.params[l]-1*liv_conc[name]*liv_amp[name])/(liv_conc[name]*liv_amp[name]))*100
                elif name=='Pi':
                    if Amps[i]!=0:
                        mc[l,0,k,i] = ((tmp_res.params[l]-Amps[i]*liv_conc[name]*liv_amp[name])/(Amps[i]*liv_conc[name]*liv_amp[name]))*100
                    else:
                        mc[l,0,k,i] =tmp_res.params[l]*100
                elif name=='PC':
                    if Amps[k]!=0:
                        mc[l,0,k,i] = ((tmp_res.params[l]-Amps[k]*liv_conc[name]*liv_amp[name])/(Amps[k]*liv_conc[name]*liv_amp[name]))*100
                    else:
                        mc[l,0,k,i] =tmp_res.params[l]*100
    np.save('conf_fsl.npy',mc)
    np.save('fid {}.npy'.format(param),fids_mc)
    
    with open('LCMAmps.txt', 'w') as fout:
        json.dump(LCMAmps, fout)
    with open('par.txt', 'w') as f:
        for item in par:
            f.write("%s\n" % item)
    #%%
    import matplotlib
    mcLC = np.load(r'conf.npy')
    mc = np.load(r'conf_fsl.npy')
    #par = np.loadtxt(r'par.txt')
    #LCMAmps = np.loadtxt(r'LCMAmps.txt')
    
    pars = [int(x) for x in par]
    # for i,name in enumerate(basis_names):
    #     vmin,vmax = np.min(np.append(mc[i,0,:,:],mcLC[i,0,:,:])),np.max(np.append(mc[i,0,:,:],mcLC[i,0,:,:]))
    #     fig, ax = plt.subplots(1,1)
    #     ax.set_xticks(np.linspace(0,len(par)-1,len(par)))
    #     ax.set_xticklabels(par,fontsize=7)
    #     plt.xlabel('Pi')
    #     ax.set_yticks(np.linspace(0,len(par)-1,len(par)))
    #     ax.set_yticklabels(par[::-1])
    #     plt.ylabel('PC')		
    #     img = ax.imshow(mc[i,0,::-1,:],cmap='plasma',vmin=vmin,vmax=vmax)
    #     fig.colorbar(img,label='Percentage (%)')
    #     plt.title('FSL {} concentration bias at different T1 ratios'.format(name,param))
    #     plt.savefig('FSL {} concentration bias.png'.format(name))
    #     plt.close('all')
    
    #     fig, ax = plt.subplots(1,1)
    #     ax.set_xticks(np.linspace(0,len(par)-1,len(par)))
    #     ax.set_xticklabels(par,fontsize=7)
    #     plt.xlabel('Pi')
    #     ax.set_yticks(np.linspace(0,len(par)-1,len(par))[::-1])
    #     ax.set_yticklabels(par)
    #     plt.ylabel('PC')		
    #     img = ax.imshow(mcLC[i,0,::-1,:],cmap='plasma',vmin=vmin,vmax=vmax)
    #     fig.colorbar(img,label='Percentage (%)')
    #     plt.title('LCM {} concentration bias at different {}/LB'.format(name,param))
    #     plt.savefig('LCM {} concentration bias.png'.format(name))
    #     plt.close('all')
    import matplotlib.patches as patches
    from matplotlib.colors import TwoSlopeNorm
    
    PMEbfsl = (((((mc[0,0,:,:]/100)*liv_amp['PC']+liv_amp['PC'])+((mc[1,0,:,:]/100)*liv_amp['PE']+liv_amp['PE']))-liv_amp['PE']-liv_amp['PC'])/(liv_amp['PE']+liv_amp['PC']))*100
    PMEblcm = (((((mcLC[0,0,:,:]/100)*liv_amp['PC']+liv_amp['PC'])+((mcLC[1,0,:,:]/100)*liv_amp['PE']+liv_amp['PE']))-liv_amp['PE']-liv_amp['PC'])/(liv_amp['PE']+liv_amp['PC']))*100
    if param!='shift offset':
        fpk, fpk_names = simsyst(sps,B0,  [ 10, 20, 20, 40, 40, 30, 20, 20, 20, 5],540,10000,TR,ampflip,samp,G,T, GB=GB_b, shift = None,plot=False)#0.049*120.667
    else:
        fpk, fpk_names = simsyst(sps,B0,  [ 20, 20, 40, 40, 30, 20, 20, 20, 5],540,10000,TR,ampflip,samp,G,T, GB=GB_b, shift = None,plot=False)#0.049*120.667
        
    for i,name in enumerate(fpk_names):
        fpk[i] = fpk[i]*liv_conc[name]*liv_amp[name]
    # vmin,vmax = np.min(np.append(PMEbfsl,PMEblcm)),np.max(np.append(PMEbfsl,PMEblcm))
    # fig, ax = plt.subplots(1,1)
    # ax.set_xticks(np.linspace(0,len(par)-1,len(par)))
    # ax.set_xticklabels(par,fontsize=7)
    # plt.xlabel('Pi')
    # ax.set_yticks(np.linspace(0,len(par)-1,len(par))[::-1])
    # ax.set_yticklabels(par)
    # plt.ylabel('PC')		
    # img = ax.imshow(PMEbfsl,cmap='plasma',vmin=vmin,vmax=vmax)
    # fig.colorbar(img,label='Percentage (%)')
    # plt.title('LCM {} concentration bias at different {}'.format(name,param))
    # plt.savefig('FSL PME concentration bias.png')
    # plt.close('all')
    
    # fig, ax = plt.subplots(1,1)
    # ax.set_xticks(np.linspace(0,len(par)-1,len(par)))
    # ax.set_xticklabels(par,fontsize=7)
    # plt.xlabel('Pi')
    # ax.set_yticks(np.linspace(0,len(par)-1,len(par))[::-1])
    # ax.set_yticklabels(par)
    # plt.ylabel('PC')		
    # img = ax.imshow(PMEblcm,cmap='plasma',vmin=vmin,vmax=vmax)
    # fig.colorbar(img,label='Percentage (%)')
    # plt.title('LCM {} concentration bias at different {}'.format(name,param))
    # plt.savefig('LCM PME concentration bias.png')
    # plt.close('all')
    delta = np.diff(par)[0]
    fig, axs = plt.subplots(2,4,sharey=True)
    plt.figtext(0.435, 0.85, 'Concentration bias', fontsize=18)
    plots = [PMEbfsl[:,:], mc[0,0,:,:], mc[1,0,:,:],mc[-4,0,:,:],PMEblcm[:,:],mcLC[0,0,:,:],mcLC[1,0,:,:],mcLC[-4,0,:,:]]
    plots_name=['PME','PC','PE','Pi','PME','PC','PE','Pi']
    vmin,vmax = np.percentile(np.array(plots),5),np.percentile(np.array(plots),95)
    if np.abs(vmin)<np.abs(vmax):
        vmax=np.abs(vmin)
    else:
        vmin =-vmax      
    if np.abs(vmin)<10:
        vmin,vmax = -10,10
    vmin,vmax=-15,15
    if param=='LB scaling factor':
        vmin,vmax=-30,30
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    fig.subplots_adjust(left=0.25)
    for i, ax in enumerate(axs.flatten()):
        im = ax.pcolormesh(plots[i], norm=norm, cmap='bwr')
        ax.set_aspect('equal')
        ax.set_title('{}'.format(plots_name[i]), fontsize=14)
        ax.set_yticks(np.linspace(0.5,len(par)-0.5,len(par)//5))
        # if i!=0 and i!=4:
        #     ax.set_yticklabels([])
        if i==0:
            if param == 'shift offset':
                ax.set_ylabel('PME {}'.format(param), fontsize=15)
            elif param == 'LB scaling factor':
                ax.set_ylabel('PC s.f.', fontsize=15)
            else:
                ax.set_ylabel('PC {}'.format(param), fontsize=15)
            ax.set_yticks(np.linspace(0.5,len(par)-0.5,len(par)//5))
            ax.set_yticklabels(par[::7],fontsize=14)
            plt.figtext(0.14, 0.675, 'FSL', fontsize=18,fontweight='bold')
        if i==4:
            if param == 'shift offset':
                ax.set_ylabel('PME {}'.format(param), fontsize=15)
            elif param == 'LB scaling factor':
                ax.set_ylabel('PC s.f.', fontsize=15)
            else:
                ax.set_ylabel('PC {}'.format(param), fontsize=15)
            plt.figtext(0.14, 0.29, 'LCM', fontsize=18,fontweight='bold')
            #ax.set_yticks(np.linspace(0.5,len(par)-0.5,len(par)//5))
            ax.set_yticklabels(par[::7],fontsize=14)
        ax.set_xticks(np.linspace(0.5,len(par)-0.5,len(par)//5))
        ax.set_xticklabels(par[::7],fontsize=14)
        rect = patches.Rectangle((st, st), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
            
    plt.subplots_adjust(hspace=0)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.22, 0.02, 0.55])
    cbar = fig.colorbar(im, cax=cbar_ax,  norm=norm)
    
    #Create colorbar showing zero, vmax, vmin and some fairly distanced values
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Bias [%]',size=16)
    lab = np.linspace(int(vmin),int(vmax),5,dtype=int)
    trs = np.min(np.abs(lab))
    if trs == np.min(lab):
        lab= lab[lab!=np.min(np.abs(lab[lab!=np.min(lab)]))]
    elif trs == np.max(lab):
        lab= lab[lab!=np.min(np.abs(lab[lab!=np.max(lab)]))]
    else:
        lab= lab[lab!=trs]
        if len(lab)==5:
            lab= lab[lab!=-trs]
    lab = np.append(lab,0)
    cbar.set_ticks(lab)
    cbar.set_ticklabels(lab)
    fig.text(0.52, 0.11, 'Pi {}'.format(param), fontsize=16, ha='center')
    fig.text(0.52, 0.495, 'Pi {}'.format(param), fontsize=16, ha='center')
    
    
    fig.canvas.manager.full_screen_toggle()
    fig.savefig('FSL_LCM {} concentration bias.png'.format(param))
    plt.close('all')
    
    from matplotlib.colors import LinearSegmentedColormap
    plot_spec = []
    fig, ax = plt.subplots(2,1)
    ax[0].set_title('Effect of {}'.format(param), fontsize=18)
    cdict1 = cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.1),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
        }
    blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)
    plt.register_cmap(cmap=blue_red1)
    cmap = blue_red1
    norm=TwoSlopeNorm(vmin=np.min(par)-0.00001, vcenter=par[st], vmax=np.max(par)+0.00001)
    #norm = matplotlib.colors.Normalize(vmin=np.min(par),vmax=np.max(par))
    for i in range(n):
        p1={"j": [[0]],"shifts": [5.37+shfs[i]],"name": "Pi","scaleFactor": 1,"T1":1.34*T1[i], "T2":0.0464}
        p2={"j": [[0]],"shifts": [6.7],"name": "PC","scaleFactor": 1,"T1":2.3, "T2":0.0512}
        fpk1, _ = simsyst([p1],B0, [LB_p[i]*LB_b[-4]],pts[i],10000,TR,ampflip,samp,G,T, GB=[(GF[i]/(1-GF[i]))*LB_b[-4]], shift = None,plot=False)#0.049*120.667
        if any(shfs!=0):
            p1_1 = {"j": [[0]],"shifts": [7.30],"name": "PE","scaleFactor": 1,"T1":3.9, "T2":0.0556}
            fpk2, _ = simsyst([p2,p1_1],B0, [LB_b[0],LB_b[1]],pts[8],10000,TR,ampflip,samp,G,T, GB=[GB_b[0],GB_b[1]], shift = None,plot=False)#0.049*120.667
            FID = Amps[i]*fpk1[0]*liv_conc['Pi']*liv_amp['Pi']+Amps[8]*fpk2[0]*liv_conc['PC']*liv_amp['PC']+Amps[8]*fpk2[1]*liv_conc['PE']*liv_amp['PE']+sum(fpk)
        else:
            fpk2, _ = simsyst([p2],B0, [LB_b[0]],pts[0],10000,TR,ampflip,samp,G,T, GB=[GB_b[0]], shift = None,plot=False)#0.049*120.667
            fpk1,fpk2=fpk1[0],fpk2[0]
            FID = Amps[i]*fpk1*liv_conc['Pi']*liv_amp['Pi']+Amps[k]*fpk2*liv_conc['PC']*liv_amp['PC']+sum(fpk)
        plot_spec.append(np.fft.fftshift(np.fft.fft(FID)))
    
    plot_spec = np.array(plot_spec)
    
    s_m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    s_m.set_array([])
    for i in range(plot_spec.shape[0]):
        ax[0].plot(np.linspace(-5000,5000,540)/(B0*gamma*10**-6),np.abs(plot_spec[i,:])/np.max(np.abs(plot_spec[i,:]),0),color=s_m.to_rgba(par[i]),linewidth=1)
    bas_fid=sum([basis_fids[i]*liv_amp[name] for i,name in enumerate(basis_names)])
    ax[0].plot(np.linspace(-5000,5000,540)/(B0*gamma*10**-6),np.abs(np.fft.fftshift(np.fft.fft(bas_fid)))/np.max(np.abs(np.fft.fftshift(np.fft.fft(bas_fid)))),color='black')

    cbar = fig.colorbar(s_m,ax = ax[0])
    cbar.ax.tick_params(labelsize=14)
    if param == 'shift offset':
        cbar.set_label('Pi '+param+ ' [ppm]',size=14)
    else:
        cbar.set_label('Pi '+ param, size=14)
    if param=='LB scaling factor':
        cbar.set_ticks([np.min(par),par[st],par[st]+(np.max(par)-par[st])/2,np.max(par)])#par[::7])
        cbar.set_ticklabels([np.min(par),par[st],par[st]+(np.max(par)-par[st])/2,np.max(par)])
    else:  
        cbar.set_ticks([np.min(par),np.min(par)/2,par[st],par[st]+(np.max(par)-par[st])/2,np.max(par)])#par[::7])
        cbar.set_ticklabels([np.min(par),np.min(par)/2,par[st],par[st]+(np.max(par)-par[st])/2,np.max(par)])
    
    ax[0].set_xlim([-20,10])
    ax[0].tick_params(axis='both', labelsize=14)
    
    #ax[0].set_ylabel('Pi', rotation='horizontal',fontsize=16,labelpad=20,fontweight='bold')
    ax[0].invert_xaxis()
    if param == 'shift offset':
        ax[0].text(9.75,0.95,'PME off. = {} ppm'.format(par[st]),fontsize = 12)
    elif param == 'LB scaling factor':
        ax[0].text(9.75,0.95,'PC LB s.f. = {}'.format(par[st]),fontsize = 12)
    else:
        ax[0].text(9.75,0.95,'PC {} = {}'.format(param,par[st]),fontsize = 12)
    plot_spec = []
    for i in range(n):
        p1={"j": [[0]],"shifts": [5.37],"name": "Pi","scaleFactor": 1,"T1":1.34, "T2":0.0464}
        p2={"j": [[0]],"shifts": [6.7+shfs[i]],"name": "PC","scaleFactor": 1,"T1":2.3*T1[i], "T2":0.0512}
        fpk1, _ = simsyst([p1],B0, [LB_b[-4]],pts[8],10000,TR,ampflip,samp,G,T, GB=[GB_b[-4]], shift = None,plot=False)#0.049*120.667
        if any(shfs!=0):
            p1_1 = {"j": [[0]],"shifts": [7.30+shfs[i]],"name": "PE","scaleFactor": 1,"T1":3.9, "T2":0.0556}
            fpk2, _ = simsyst([p2,p1_1],B0, [LB_p[i]*LB_b[0],LB_p[i]*LB_b[1]],pts[i],10000,TR,ampflip,samp,G,T, GB=[(GF[i]/(1-GF[i]))*LB_b[0],(GF[i]/(1-GF[i]))*LB_b[1]], shift = None,plot=False)#0.049*120.667
            FID = fpk1[0]*liv_conc['Pi']*liv_amp['Pi']+Amps[i]*fpk2[0]*liv_conc['PC']*liv_amp['PC']+Amps[i]*fpk2[1]*liv_conc['PE']*liv_amp['PE']+sum(fpk)
        else:
            fpk2, _ = simsyst([p2],B0, [LB_p[i]*LB_b[0]],pts[i],10000,TR,ampflip,samp,G,T, GB=[(GF[i]/(1-GF[i]))*LB_b[0]], shift = None,plot=False)#0.049*120.667
            fpk1,fpk2=fpk1[0],fpk2[0]
            FID = fpk1*liv_conc['Pi']*liv_amp['Pi']+Amps[i]*fpk2*liv_conc['PC']*liv_amp['PC']+sum(fpk)
        plot_spec.append(np.fft.fftshift(np.fft.fft(FID)))
    plot_spec = np.array(plot_spec)
    s_m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    s_m.set_array([])
    for i in range(plot_spec.shape[0]):
        ax[1].plot(np.linspace(-5000,5000,540)/(B0*gamma*10**-6),np.abs(plot_spec[i,:])/np.max(np.abs(plot_spec[i,:]),0),color=s_m.to_rgba(par[i]),linewidth=1)
    ax[1].plot(np.linspace(-5000,5000,540)/(B0*gamma*10**-6),np.abs(np.fft.fftshift(np.fft.fft(bas_fid)))/np.max(np.abs(np.fft.fftshift(np.fft.fft(bas_fid)))),color='black')
    
    cbar = fig.colorbar(s_m,ax = ax[1])
    cbar.ax.tick_params(labelsize=14)
    if param == 'shift offset':
        cbar.set_label('PME '+ param+ ' [ppm]',size=16)
    else:
        cbar.set_label('PC ' + param, size=16)
    # cbar.set_ticks(par[::7])
    # cbar.set_ticklabels(par[::7])
    if param=='LB scaling factor':
        cbar.set_ticks([np.min(par),par[st],par[st]+(np.max(par)-par[st])/2,np.max(par)])#par[::7])
        cbar.set_ticklabels([np.min(par),par[st],par[st]+(np.max(par)-par[st])/2,np.max(par)])
    else:  
        cbar.set_ticks([np.min(par),np.min(par)/2,par[st],par[st]+(np.max(par)-par[st])/2,np.max(par)])#par[::7])
        cbar.set_ticklabels([np.min(par),np.min(par)/2,par[st],par[st]+(np.max(par)-par[st])/2,np.max(par)])
    
    ax[1].set_xlim([-20,10])
    ax[1].invert_xaxis()
    ax[1].tick_params(axis='both', labelsize=14)
    if param == 'LB scaling factor':
        ax[1].text(9.75,0.95,'Pi LB s.f. = {}'.format(par[st]),fontsize = 12)
    elif param == 'shift offset':
        ax[1].text(9.75,0.95,'Pi off. = {} ppm'.format(par[st]),fontsize = 12)
    else:
        ax[1].text(9.75,0.95,'Pi {} = {}'.format(param,par[st]),fontsize = 12)
    # if param == 'shift offset':
    #     ax[1].set_ylabel('PME', rotation='horizontal',fontsize=16,labelpad=20,fontweight='bold')
    # else:
    #     ax[1].set_ylabel('PC', rotation='horizontal',fontsize=16,labelpad=20,fontweight='bold')
    ax[1].set_xlabel('Chemical shift [ppm]',fontsize = 16)
    fig.text(0.06, 0.5, 'Magnitude [a.u.]', va='center', rotation='vertical',fontsize = 16)
    fig.canvas.manager.full_screen_toggle()
    fig.savefig('{} variation.png'.format(param))
    plt.close('all')
    #%%
    mcLC = np.load(r'conf.npy')
    mc = np.load(r'conf_fsl.npy')
    #par = np.loadtxt(r'par.txt')
    #LCMAmps = np.loadtxt(r'LCMAmps.txt')
    
    pars = [int(x) for x in par]
    import matplotlib.patches as patches
    from matplotlib.colors import TwoSlopeNorm
    
    PMEbfsl = (((((mc[0,0,:,:]/100)*liv_amp['PC']+liv_amp['PC'])+((mc[1,0,:,:]/100)*liv_amp['PE']+liv_amp['PE']))-liv_amp['PE']-liv_amp['PC'])/(liv_amp['PE']+liv_amp['PC']))*100
    PMEblcm = (((((mcLC[0,0,:,:]/100)*liv_amp['PC']+liv_amp['PC'])+((mcLC[1,0,:,:]/100)*liv_amp['PE']+liv_amp['PE']))-liv_amp['PE']-liv_amp['PC'])/(liv_amp['PE']+liv_amp['PC']))*100
    
    def lims(arr,b,par):
        '''
        
        Parameters
        ----------
        arr : TYPE
            DESCRIPTION.
        b : TYPE
            DESCRIPTION.
        par : TYPE
            DESCRIPTION.

        Returns
        -------
        rmn : TYPE
            DESCRIPTION.
        rmx : TYPE
            DESCRIPTION.
        cmn : TYPE
            DESCRIPTION.
        cmx : TYPE
            DESCRIPTION.

        '''
        arr = np.squeeze(arr)
        mx_out=np.zeros((2,2))
        rmn,rmx=-1000,1000
        for i in range(arr.shape[0]):
            mask = np.where(np.abs(arr[i,:])<=b,1,0)
            rng = par[mask.astype(bool)]
            if len(rng)!=0:
                mn = np.min(rng)
                mx = np.max(rng)
                if mn>rmn:
                    rmn=mn
                if mx<rmx:
                    rmx=mx
        cmn,cmx=-1000,1000
        for i in range(arr.shape[1]):
            mask = np.where(np.abs(arr[:,i])<=b,1,0)
            rng = par[mask.astype(bool)]
            if len(rng)!=0:
                mn = np.min(rng)
                mx = np.max(rng)
                if mn>cmn:
                    cmn=mn
                if mx<cmx:
                    cmx=mx
        mx_out[0,0],mx_out[0,1],mx_out[1,0],mx_out[1,1] = rmn,rmx,cmn,cmx
        return mx_out
        
        
    mcLC = np.load(r'conf.npy')
    mc = np.load(r'conf_fsl.npy')
    
    PME_lim_fsl= lims(PMEbfsl,15,par)
    PME_lim_lcm= lims(PMEblcm,15,par)
    PC_lim_fsl= lims(mc[0,0,:,:],15,par)
    PC_lim_lcm= lims(mcLC[0,0,:,:],15,par)
    PE_lim_fsl= lims(mc[1,0,:,:],15,par)
    PE_lim_lcm= lims(mcLC[1,0,:,:],15,par)
    Pi_lim_fsl= lims(mc[-4,0,:,:],15,par)
    Pi_lim_lcm= lims(mcLC[-4,0,:,:],15,par)
    print('PME_fsl=[ {} , {} \n {} , {} ]'.format(*PME_lim_fsl.reshape(4)))
    print('PME_lcm=[ {} , {} \n {} , {} ]'.format(*PME_lim_lcm.reshape(4)))
    print('PC_fsl=[ {} , {} \n {} , {} ]'.format(*PC_lim_fsl.reshape(4)))
    print('PC_lcm=[ {} , {} \n {} , {} ]'.format(*PC_lim_lcm.reshape(4)))
    print('PE_fsl=[ {} , {} \n {} , {} ]'.format(*PE_lim_fsl.reshape(4)))
    print('PE_lcm=[ {} , {} \n {} , {} ]'.format(*PE_lim_lcm.reshape(4)))
    print('Pi_fsl=[ {} , {} \n {} , {} ]'.format(*Pi_lim_fsl.reshape(4)))
    print('Pi_lcm=[ {} , {} \n {} , {} ]'.format(*Pi_lim_lcm.reshape(4)))
    
     #%%SNR
    #start=time.time()
    n=15
    n_sim = 100
    mc_SNR=np.zeros((len(basis_names),3,n))
    mc_cr_SNR=np.zeros((len(basis_names),2,n))
    SNR = [int(x) for x in np.around(np.linspace(2,100,15),0)]
    cont=0
    LCMAmps =[]
    fpk, fpk_names = simsyst(spinSys,B0, LB_b,540,bw,TR,ampflip,samp,G,T, GB=GB_b, shift = None,plot=False)#0.049*120.667
    for i,name in enumerate(fpk_names):
        fpk[i] = fpk[i]*liv_amp[name] 
    FID = sum(fpk)
    for k in range(n):
        sig = (1/SNR[k])*np.max([np.max(np.abs(np.fft.fftshift(np.fft.fft(fpk[i])))) for i,name in enumerate(fpk_names)])
        r = np.zeros((n_sim,len(basis_names)))
        cr = np.zeros((n_sim,len(basis_names)))
        for j in range(n_sim):

        #np.random.seed(42) #Set same random noise between fits (real and imaginary should be dinstinct in each fit but are the same between different fits)
            tmp_a = MRS(FID=np.fft.ifft(np.fft.fft(FID)+np.random.normal(0,sig,len(FID))+1j*np.random.normal(0,sig,len(FID))), header = {'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': 120.645, 'dwelltime': 1./10000},
                    basis =  np.array(basis_fids),
                    names = basis_names,
                    basis_hdr=[{'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./bw, 'fwhm':5}],
                    nucleus='31P',bw=bw, cf = cf)
            LCMAmps.append(liv_amp)#Save concentration correction for LCM
            if SNR[k] == 2 and j ==0:
                SNR2 = tmp_a.get_spec()
            if SNR[k] == 30 and j ==0:
                SNR55 = tmp_a.get_spec()
            if SNR[k] == 100 and j ==0:
                SNR150 = tmp_a.get_spec()
            #Save FID for LCM
            info = {'ID': '{}_base'.format(cont), 'FMTDAT':'(2E16.6)', 'Volume': 1.0, 'TRAMP': 1.0}
            header = {'bandwidth':bw, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./bw,'width':10, 'points':len(FID)}
            lcm_io.saveRAW(r'.\TestSNR\control{}.RAW'.format(cont),tmp_a.FID, info = info, hdr=header, conj=True)
            cont += 1
            tmp_res = fitting.fit_FSLModel(tmp_a,method='Newton',ppmlim=(-20.0,10.0),model='voigt',baseline_order=-1,
                                            MHSamples=5000,metab_groups=[0,1,2,3,4,5,6,7,8,9,10,11])
            #Save results of fitting as a matrix for each SNR
            r[j,:]=tmp_res.params[:len(basis_names)]
            cr[j,:]=tmp_res.crlb[:len(basis_names)]
        np.save('r_{}'.format(SNR[k]),r)
        np.save('cr_{}'.format(SNR[k]),cr)
        m = np.mean(r,0)
        sdm = np.std(r,0,ddof=1)/np.sqrt(n_sim)
        mcr = np.mean(cr,0)
        varcr = np.zeros(r.shape)
        for l,name in enumerate(basis_names):
            varcr[l,:] = (r[l,:]-liv_amp[name])**2#zero because it could not be gaussian
        varcr = np.mean(varcr,0)
        for l,name in enumerate(basis_names):
            mc_SNR[l,0,k] = ((m[l]-liv_amp[name])/liv_amp[name])*100
            mc_SNR[l,1,k] = (sdm[l]/liv_amp[name])*100
            mc_cr_SNR[l,0,k] = mcr[l]
            mc_cr_SNR[l,1,k] = varcr[l]
    np.save('mcSNR',mc_SNR)
    np.save('mc_cr_SNR',mc_cr_SNR)
    with open('LCMAmpsSNR.txt', 'w') as fout:
        json.dump(LCMAmps, fout)
    with open('parSNR.txt', 'w') as f:
        for item in SNR:
            f.write("%s\n" % item)
    fig, axs = plt.subplots(3, 1, constrained_layout=True,sharey=True)
    axs[0].set_title('Simulated liver spectrum at different SNR', fontsize=14)
    axs[0].plot(np.linspace(-5000,5000,540)/(B0*gamma*10**-6),np.abs(SNR2),color='black', label='SNR = 2')
    axs[0].legend(loc="upper right")
    axs[0].set_xlim([-20,10])
    axs[0].invert_xaxis()
    axs[1].plot(np.linspace(-5000,5000,540)/(B0*gamma*10**-6),np.abs(SNR55),color='black', label='SNR = 30')
    axs[1].legend(loc="upper right")
    axs[1].set_xlim([-20,10])
    axs[1].invert_xaxis()
    axs[2].plot(np.linspace(-5000,5000,540)/(B0*gamma*10**-6),np.abs(SNR150),color='black', label='SNR = 100')
    axs[2].set_xlim([-20,10])
    axs[2].invert_xaxis()
    axs[2].legend(loc="upper right")
    axs[1].set_ylabel("Amplitude [a.u.]",fontsize=12)
    axs[2].set_xlabel("Chemical Shift [ppm]",fontsize=12)         
            #%%
    
    LCM = {}
    mc_SNR = np.load(r'mcSNR.npy')
    FID = sum(basis_fids)
    mx = np.max(np.abs(np.fft.fftshift(np.fft.fft(FID))))
    for i,name in enumerate(basis_names):
        par = np.array(SNR)
        LCM[name]=np.loadtxt('.\LCM_SNR_{}.txt'.format(name))
   #      fig, ax = plt.subplots(1,1)
   #      snr_scale = 1
   #      par = par*snr_scale
   #      ax.set_xticks(np.linspace(0,len(par)-1,len(par)))
   #      ax.set_xticklabels(np.around(par,0))
   #      plt.xlabel('SNR')
   #      plt.ylabel('Percentage [%]')		
   #      plt.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),mc_SNR[i,0,:],mc_SNR[i,1,:],marker = '.',color = 'darkblue', linestyle = '-.')
   #      plt.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),LCM[name][0,:],LCM[name][1,:],marker = '.',color = 'darkorange', linestyle = '-.')
   #      plt.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),[0]*len(par),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
   #      plt.title('{} concentration bias at different SNR'.format(name))
   #      plt.xlim([-0.5,15.5])
   #      axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
   #      axins.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),mc_SNR[i,0,:],mc_SNR[i,1,:],marker = '.',color = 'darkblue', linestyle = '-.')
   #      axins.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),LCM[name][0,:],LCM[name][1,:],marker = '.',color = 'darkorange', linestyle = '-.')
   #      axins.errorbar(np.around(np.linspace(-10,len(par)-1+10,len(par)),0),[0]*len(par),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
   #      # sub region of the original image
   #      x1, x2, y1, y2 = 1, 14, -15, 15
   #      axins.set_xlim(x1-0.5, x2+0.5)
   #      axins.set_ylim(y1, y2)
   #      axins.set_xticks(np.linspace(x1,x2-1,len(par[x1:-1:2])))    
   #      axins.set_xticklabels(np.around(par[x1:-1:2],0))
   #      axins.set_yticks(np.linspace(y1,y2,4))    
   #      axins.set_yticklabels(np.around(np.linspace(y1,y2,4),0))
   #      plt.legend(['FSL_MRS','LCModel'],loc = 2)
   #      ax.indicate_inset_zoom(axins, edgecolor="black")
   #      plt.savefig('FSL {} concentration bias SNR.png'.format(name))
   #      plt.close('all')
   # # print("---Elaboration completed in {:.2f} seconds.---".format(time.time()-start))
    
    
    PMEbfsl = ((((((mc_SNR[0,0,:]/100)*liv_amp['PC'])+liv_amp['PC'])+(((mc_SNR[1,0,:]/100)*liv_amp['PE'])+liv_amp['PE']))-liv_amp['PE']-liv_amp['PC'])/(liv_amp['PE']+liv_amp['PC']))*100
    PMEblcm = ((((((LCM['PC'][0,:]/100)*liv_amp['PC'])+liv_amp['PC'])+(((LCM['PE'][0,:]/100)*liv_amp['PE'])+liv_amp['PE']))-liv_amp['PE']-liv_amp['PC'])/(liv_amp['PE']+liv_amp['PC']))*100
    dPMEbfsl = (np.sqrt(((mc_SNR[0,1,:]*10/100)*liv_amp['PC'])**2+((mc_SNR[1,1,:]*10/100)*liv_amp['PE'])**2)/(liv_amp['PE']+liv_amp['PC']))*100
    dPMEblcm = (np.sqrt(((LCM['PC'][1,:]*10/100)*liv_amp['PC'])**2+((LCM['PE'][1,:]*10/100)*liv_amp['PE'])**2)/(liv_amp['PE']+liv_amp['PC']))*100
    PDEbfsl = ((((((mc_SNR[2,0,:]/100)*liv_amp['GPC'])+liv_amp['GPC'])+(((mc_SNR[3,0,:]/100)*liv_amp['GPE'])+liv_amp['GPE']))-liv_amp['GPE']-liv_amp['GPC'])/(liv_amp['GPE']+liv_amp['GPC']))*100
    PDEblcm = ((((((LCM['GPC'][0,:]/100)*liv_amp['GPC'])+liv_amp['GPC'])+(((LCM['GPE'][0,:]/100)*liv_amp['GPE'])+liv_amp['GPE']))-liv_amp['GPE']-liv_amp['GPC'])/(liv_amp['GPE']+liv_amp['GPC']))*100
    dPDEbfsl = (np.sqrt(((mc_SNR[0,1,:]*10/100)*liv_amp['GPC'])**2+((mc_SNR[1,1,:]*10/100)*liv_amp['GPE'])**2)/(liv_amp['GPE']+liv_amp['GPC']))*100
    dPDEblcm = (np.sqrt(((LCM['GPC'][1,:]*10/100)*liv_amp['GPC'])**2+((LCM['GPE'][1,:]*10/100)*liv_amp['GPE'])**2)/(liv_amp['GPE']+liv_amp['GPC']))*100
    
    FID = sum(basis_fids)
    mx = np.max(np.abs(np.fft.fftshift(np.fft.fft(FID))))
    par = np.array(SNR)
    snr_scale = 1
    par = par*snr_scale
    fig, ax = plt.subplots(1,1)
    ax.set_xticks(np.linspace(0,len(par)-1,len(par)))
    ax.set_xticklabels(np.around(par,0))
    plt.xlabel('SNR',fontsize=16)
    plt.ylabel('Bias [%]',fontsize=16)		
    plt.errorbar(np.around(np.linspace(-10,len(par)-1+10,len(par)),0),[0]*len(par),linestyle='--',color='gray',alpha = 0.6,label = "_nolegend_") 
    plt.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),PMEbfsl,dPMEbfsl,marker = '.',color = 'darkblue', linestyle = '--',capsize=3)
    plt.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),PMEblcm,dPMEblcm,marker = '.',color = 'darkorange',alpha = 0.8, linestyle = '--',capsize=3)
    plt.title('Mean PME concentration bias at different SNR',fontsize=16)
    plt.xlim([-0.1,14.1])
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins.errorbar(np.around(np.linspace(-10,len(par)-1+10,len(par)),0),[0]*len(par),linestyle='--',color='gray',alpha = 0.6,label = "_nolegend_") 
    axins.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),PMEbfsl,dPMEbfsl,marker = '.',color = 'darkblue', linestyle = '--',capsize=3)
    axins.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),PMEblcm,dPMEblcm,marker = '.',color = 'darkorange',alpha = 0.8, linestyle = '--',capsize=3)
    # sub region of the original image
    x1, x2, y1, y2 = 1, 14, -15, 15
    axins.set_xlim(x1-0.1, x2+0.1)
    axins.set_ylim(y1, y2)
    axins.set_xticks(np.linspace(x1,x2-1,len(par[x1:-1:2])))    
    axins.set_xticklabels(np.around(par[x1:-1:2],0))
    axins.set_yticks(np.linspace(y1,y2,4))    
    axins.set_yticklabels(np.around(np.linspace(y1,y2,4),0))
    plt.legend(['FSL_MRS','LCModel'],loc = 2)
    rectpatch, connects=ax.indicate_inset_zoom(axins,edgecolor="black")
    connects[0].set_visible(False)
    connects[1].set_visible(True)
    connects[2].set_visible(False)
    connects[3].set_visible(True)
    
    plt.savefig('Mean PME concentration bias SNR.png')
    plt.close('all')
    
    FID = sum(basis_fids)
    mx = np.max(np.abs(np.fft.fftshift(np.fft.fft(FID))))
    par = np.array(SNR)
    snr_scale = np.max(np.abs(np.fft.fftshift(np.fft.fft(sum(basis_fids[2:4])))))/mx
    par = par*1
    fig, ax = plt.subplots(1,1)
    ax.set_xticks(np.linspace(0,len(par)-1,len(par)))
    ax.set_xticklabels(np.around(par,0))
    plt.xlabel('SNR',fontsize=16)
    plt.ylabel('Bias [%]',fontsize=16)		
    plt.errorbar(np.around(np.linspace(-10,len(par)-1+10,len(par)),0),[0]*len(par),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
    plt.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),PDEbfsl,dPDEbfsl,marker = '.',color = 'darkblue', linestyle = '--',capsize=3)
    plt.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),PDEblcm,dPDEblcm,marker = '.',color = 'darkorange',alpha = 0.8, linestyle = '--',capsize=3)
    plt.title('Mean PDE concentration bias at different SNR',fontsize=16)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlim([-0.1,14.1])
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins.errorbar(np.around(np.linspace(-10,len(par)-1+10,len(par)),0),[0]*len(par),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
    axins.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),PDEbfsl,dPDEbfsl,marker = '.',color = 'darkblue', linestyle = '--',capsize=3)
    axins.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),PDEblcm,dPDEblcm,marker = '.',color = 'darkorange',alpha = 0.8, linestyle = '--',capsize=3)
    # sub region of the original image
    x1, x2, y1, y2 = 1, 14, -15, 15
    axins.set_xlim(x1-0.1, x2+0.1)
    axins.set_ylim(y1, y2)
    axins.set_xticks(np.linspace(x1,x2-1,len(par[x1:-1:2])))    
    axins.set_xticklabels(np.around(par[x1:-1:2],0))
    axins.set_yticks(np.linspace(y1,y2,4))    
    axins.set_yticklabels(np.around(np.linspace(y1,y2,4),0))
    plt.legend(['FSL_MRS','LCModel'],loc = 2)
    rectpatch, connects=ax.indicate_inset_zoom(axins,edgecolor="black")
    connects[0].set_visible(False)
    connects[1].set_visible(True)
    connects[2].set_visible(False)
    connects[3].set_visible(True)
    
    plt.savefig('Mean PDE concentration bias SNR.png')
    plt.close('all')
    
    fig, ax = plt.subplots(1,1)
    ax.set_xticks(np.linspace(0,len(par)-1,len(par)))
    ax.set_xticklabels(np.around(par,0))
    plt.xlabel('SNR',fontsize=16)
    plt.ylabel('Bias [%]',fontsize=16)
    PME_PDE = (liv_amp['PC']+liv_amp['PE'])/(liv_amp['GPC']+liv_amp['GPE'])	
    # PDE_PME = (liv_amp['GPC']+liv_amp['GPE'])/(liv_amp['PC']+liv_amp['PE'])	
    #plt.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),((((((mc_SNR[0,0,:]/100)*liv_amp['PC']+liv_amp['PC'])+((mc_SNR[1,0,:]/100)*liv_amp['PE']+liv_amp['PE']))/(((mc_SNR[2,0,:]/100)*liv_amp['GPC']+liv_amp['GPC'])+((mc_SNR[3,0,:]/100)*liv_amp['GPE']+liv_amp['GPE'])))-(1/PDE_PME))*PDE_PME)*100,np.sqrt(mc_SNR[2,1,:]**2+mc_SNR[3,1,:]**2),marker = '.',color = 'darkblue', linestyle = '-.')
    PME_val = ((PMEbfsl/100)*(liv_amp['PE']+liv_amp['PC']))+(liv_amp['PE']+liv_amp['PC'])
    dPME_val = (dPMEbfsl/100)*(liv_amp['PE']+liv_amp['PC'])
    PDE_val = ((PDEbfsl/100)*(liv_amp['GPE']+liv_amp['GPC']))+(liv_amp['GPE']+liv_amp['GPC'])
    dPDE_val = (dPDEbfsl/100)*(liv_amp['GPE']+liv_amp['GPC'])
    
    plt.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),((((PME_val)/(PDE_val))-(PME_PDE))/(PME_PDE))*100, (np.sqrt(((dPME_val/PDE_val)**2+(dPDE_val*(PME_val/PDE_val**2))**2))/(PME_PDE))*100,marker = '.',color = 'darkblue', linestyle = '--',capsize=3)
    plt.errorbar(np.around(np.linspace(-10,len(par)-1+10,len(par)),0),[0]*len(par),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
    #plt.xlim([-0.1,15.1])
    
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins.errorbar(np.around(np.linspace(-10,len(par)-1+10,len(par)),0),[0]*len(par),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
    
    axins.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),((((PME_val)/(PDE_val))-(PME_PDE))/(PME_PDE))*100, (np.sqrt(((dPME_val/PDE_val)**2+(dPDE_val*(PME_val/PDE_val**2))**2))/(PME_PDE))*100,marker = '.',color = 'darkblue', linestyle = '--',capsize=3)
    PME_val = ((PMEblcm/100)*(liv_amp['PE']+liv_amp['PC']))+(liv_amp['PE']+liv_amp['PC'])
    PDE_val = ((PDEblcm/100)*(liv_amp['GPE']+liv_amp['GPC']))+(liv_amp['GPE']+liv_amp['GPC'])
    plt.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),((((PME_val)/(PDE_val))-(PME_PDE))/(PME_PDE))*100, (np.sqrt(((dPME_val/PDE_val)**2+(dPDE_val*(PME_val/PDE_val**2))**2))/(PME_PDE))*100,marker = '.',color = 'darkorange',alpha=0.8, linestyle = '--',capsize=3)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    axins.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),((((PME_val)/(PDE_val))-(PME_PDE))/(PME_PDE))*100, (np.sqrt(((dPME_val/PDE_val)**2+(dPDE_val*(PME_val/PDE_val**2))**2))/(PME_PDE))*100,marker = '.',color = 'darkorange',alpha=0.8, linestyle = '--',capsize=3)
    # sub region of the original image
    x1, x2, y1, y2 = 1, 14, -15, 15
    axins.set_xlim(x1-0.1, x2+0.1)
    axins.set_ylim(y1, y2)
    axins.set_xticks(np.linspace(x1,x2-1,len(par[x1:-1:2])))    
    axins.set_xticklabels(np.around(par[x1:-1:2],0))
    axins.set_yticks(np.linspace(y1,y2,4))    
    axins.set_yticklabels(np.around(np.linspace(y1,y2,4),0))
    plt.legend(['FSL_MRS','LCModel'],loc = 2)
    rectpatch, connects=ax.indicate_inset_zoom(axins,edgecolor="black")
    connects[0].set_visible(False)
    connects[1].set_visible(True)
    connects[2].set_visible(False)
    connects[3].set_visible(True)
    # plt.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),PMEblcm/PDEblcm,dPDEblcm,marker = '.',color = 'darkorange', linestyle = '-.')
    # plt.errorbar(np.around(np.linspace(-10,len(par)-1+10,len(par)),0),[0]*len(par),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
    plt.title('Estimated PME/PDE bias at different SNR',fontsize=16)
    plt.xlim([-0.1,14.1])
    plt.savefig('Mean PME_PDE concentration bias SNR.png')
    plt.close('all')
    plt.figure()
    plt.plot(SNR, 20*np.log((np.sqrt(((dPME_val/PDE_val)**2+(dPDE_val*(PME_val/PDE_val**2))**2))/(PME_PDE))*100))
    # axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    # axins.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),PDEbfsl,dPDEbfsl,marker = '.',color = 'darkblue', linestyle = '-.')
    # axins.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),PDEblcm,dPDEblcm,marker = '.',color = 'darkorange', linestyle = '-.')
    # axins.errorbar(np.around(np.linspace(-10,len(par)-1+10,len(par)),0),[0]*len(par),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
    # # sub region of the original image
    # x1, x2, y1, y2 = 1, 14, -15, 15
    # axins.set_xlim(x1-0.5, x2+0.5)
    # axins.set_ylim(y1, y2)
    # axins.set_xticks(np.linspace(x1,x2-1,len(par[x1:-1:2])))    
    # axins.set_xticklabels(np.around(par[x1:-1:2],0))
    # axins.set_yticks(np.linspace(y1,y2,4))    
    # axins.set_yticklabels(np.around(np.linspace(y1,y2,4),0))
    # plt.legend(['FSL_MRS','LCModel'],loc = 2)
    # ax.indicate_inset_zoom(axins, edgecolor="black")
     #%%B1
    start=time.time()
    #liv_conc = {'PC':1,'PE':1,'GPC':1,'GPE':1,'ATP':1,'PtdC':1,'Pi':1,'NAD+':1,'UDPG':1,'PCr':1}
    n=36
    n_sim = 1
    mc_B1=np.zeros((len(basis_names),2,n))
    B1 = np.around(np.linspace(0.25,2,n),2)
    cont=0
    LCMAmps =[]
    #r_B1 = np.zeros((len(SNR),n_sim,len(basis_names)))
    for k in range(n):
        fpk, fpk_names = simsyst(spinSys,B0, LB_b,540,10000,TR,B1[k]*ampflip,samp,G,T, GB=GB_b, shift = None,plot=False)#0.049*120.667
        r = np.zeros((n_sim,len(basis_names)))
        for i,name in enumerate(fpk_names):
            fpk[i] = fpk[i]*liv_amp[name] 
        FID = sum(fpk)
        #np.random.seed(42) #Set same random noise between fits (real and imaginary should be dinstinct in each fit but are the same between different fits)
        tmp_a = MRS(FID=FID, header = {'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': 120.645, 'dwelltime': 1./10000},
                basis =  np.array(basis_fids),
                names = basis_names,
                basis_hdr=[{'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000, 'fwhm':5}],
                nucleus='31P',bw=10000, cf = cf)
        LCMAmps.append(liv_amp)
        #tmp.rescaleForFitting(ind_scaling=['PCr'])
        info = {'ID': '{}_base'.format(cont), 'FMTDAT':'(2E16.6)', 'Volume': 1.0, 'TRAMP': 1.0}
        header = {'bandwidth':10000.0, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000,'width':5, 'points':len(FID)}
        lcm_io.saveRAW(r'.\Test\control{}.RAW'.format(cont),tmp_a.FID, info = info, hdr=header, conj=True)
        cont += 1
        tmp_res = fitting.fit_FSLModel(tmp_a,method='Newton',ppmlim=(-20.0,10.0),model='voigt',baseline_order=-1,
                                       MHSamples=5000,metab_groups=[0,1,2,3,4,5,6,7,8,9,10,11])
        fig = plotting.plotly_fit(tmp_a,tmp_res,ppmlim= (-20.0,10.0))
        fig.show()
        for l,name in enumerate(basis_names):
            mc_B1[l,0,k] = ((tmp_res.params[l]-liv_amp[name])/liv_amp[name])*100
    np.save('mcB1.npy',mc_B1)
    with open('LCMAmpsB1.txt', 'w') as fout:
        json.dump(LCMAmps, fout)
    with open('parB1.txt', 'w') as f:
        for item in par:
            f.write("%s\n" % item)
                
            #%%
    par = B1
    LCM = {}    
    for i,name in enumerate(basis_names):
        LCM[name]=np.loadtxt('LCM_B1_{}.txt'.format(name))
        # fig, ax = plt.subplots(1,1)
        # ax.set_xticks(np.linspace(0,len(par)-2,len(par)//2))
        # ax.set_xticklabels(par[::2],fontsize =15)
        # plt.xlabel('$B_{1}/B_{1,true}$',fontsize =18)
        # plt.ylabel('Percentage [%]',fontsize =18)		
        # plt.errorbar(np.linspace(0,len(par)-1,len(par)),mc_B1[i,0,:],mc_B1[i,1,:],marker = '.',color = 'darkblue', linestyle = '-.')
        # plt.errorbar(np.linspace(0,len(par)-1,len(par)),LCM[name][0,:],LCM[name][1,:],marker = '.',color = 'darkorange', linestyle = '-.')
        # plt.errorbar(np.linspace(0,len(par)-1,len(par)),[0]*len(par),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
        # plt.yticks(fontsize=15)
        # plt.title('{} concentration bias at different $B_1$'.format(name),fontsize =16)
        # plt.legend(['FSL_MRS','LCModel'],fontsize=12)
        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        # plt.xlim([-0.5,len(par)+0.5])
        # plt.savefig('FSL {} concentration bias B1.png'.format(name))
        # plt.close('all')
    print("---Elaboration completed in {:.2f} seconds.---".format(time.time()-start))
    pme_perc = ((((mc_B1[0,0,:]/100)*liv_amp['PC'])+((mc_B1[1,0,:]/100)*liv_amp['PE']))/(liv_amp['PC']+liv_amp['PE']))*100
    pme_percLCM = ((((LCM['PC'][0,:]/100)*liv_amp['PC'])+((LCM['PE'][0,:]/100)*liv_amp['PE']))/(liv_amp['PC']+liv_amp['PE']))*100
    
    fig, ax = plt.subplots(1,1)
    ax.set_xticks(np.linspace(0,len(par)-3,len(par)//3))
    ax.set_xticklabels(par[::3],fontsize =18)
    plt.xlabel('$B_{1}/B_{1,nominal}$',fontsize =24)
    plt.ylabel('Bias [%]',fontsize =24)		
    plt.plot(np.around(np.linspace(-10,len(par)-1+10,len(par)),0),[0]*len(par),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
    plt.plot(np.around(np.linspace(0,len(par)-1,len(par)),0),pme_perc,marker = 'o',color = 'darkblue', linestyle = '',markersize=17)
    plt.plot(np.around(np.linspace(0,len(par)-1,len(par)),0),pme_percLCM,marker = '^',color = 'darkorange', linestyle = '',markersize=10,alpha = 0.9)
    plt.title('PME concentration bias at different $B_1$',fontsize =24)
    plt.xlim([-0.5,len(par)+0.5])
    #plt.ylim([-100,10])
    plt.yticks(fontsize=18)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.legend(['FSL_MRS','LCModel'],fontsize=16)
    plt.savefig('FSL PME concentration bias B1.png')
    plt.close('all')
    pde_perc = ((((mc_B1[2,0,:]/100)*liv_amp['GPC'])+((mc_B1[3,0,:]/100)*liv_amp['GPE']))/(liv_amp['GPC']+liv_amp['GPE']))*100
    pde_percLCM = ((((LCM['GPC'][0,:]/100)*liv_amp['GPC'])+((LCM['GPE'][0,:]/100)*liv_amp['GPE']))/(liv_amp['GPC']+liv_amp['GPE']))*100
    fig, ax = plt.subplots(1,1)
    ax.set_xticks(np.linspace(0,len(par)-3,len(par)//3))
    ax.set_xticklabels(par[::3],fontsize =18)
    plt.xlabel('$B_{1}/B_{1,nominal}$',fontsize =24)
    plt.ylabel('Bias [%]',fontsize =24)		
    plt.plot(np.around(np.linspace(-10,len(par)-1+10,len(par)),0),[0]*len(par),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
    plt.plot(np.around(np.linspace(0,len(par)-1,len(par)),0),pde_perc,marker = 'o',color = 'darkblue', linestyle = '', markersize =17)
    plt.plot(np.around(np.linspace(0,len(par)-1,len(par)),0),pde_percLCM,marker = '^',color = 'darkorange', linestyle = '',markersize=12,alpha = 0.9)
    plt.title('PDE concentration bias at different $B_1$',fontsize =24)
    plt.xlim([-0.5,len(par)+0.5])
    #plt.ylim([-100,10])
    plt.yticks(fontsize=18)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.legend(['FSL_MRS','LCModel'],fontsize=16)
    plt.savefig('FSL PDE concentration bias B1.png')
    plt.close('all')
    #%%
    fig, ax = plt.subplots(1,1)
    ax.set_xticks(np.linspace(0,len(par)-3,len(par)//3))
    ax.set_xticklabels(par[::3],fontsize =18)
    plt.xlabel('$B_{1}/B_{1,nominal}$',fontsize =24)
    plt.ylabel('Bias [%]',fontsize =24)	
    PDE_PME = (liv_amp['GPC']+liv_amp['GPE'])/(liv_amp['PC']+liv_amp['PE'])	
    plt.plot(np.around(np.linspace(-10,len(par)-1+10,len(par)),0),[0]*len(par),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
    plt.plot(np.around(np.linspace(0,len(par)-1,len(par)),0),((((((mc_B1[0,0,:]/100)*liv_amp['PC']+liv_amp['PC'])+((mc_B1[1,0,:]/100)*liv_amp['PE']+liv_amp['PE']))/(((mc_B1[2,0,:]/100)*liv_amp['GPC']+liv_amp['GPC'])+((mc_B1[3,0,:]/100)*liv_amp['GPE']+liv_amp['GPE'])))-(1/PDE_PME))*PDE_PME)*100,marker = 'o',color = 'darkblue', linestyle = '',markersize=16)
    plt.plot(np.around(np.linspace(0,len(par)-1,len(par)),0),((((((LCM['PC'][0,:]/100)*liv_amp['PC']+liv_amp['PC'])+((LCM['PE'][0,:]/100)*liv_amp['PE']+liv_amp['PE']))/(((LCM['GPC'][0,:]/100)*liv_amp['GPC']+liv_amp['GPC'])+((LCM['GPE'][0,:]/100)*liv_amp['GPE']+liv_amp['GPE'])))-(1/PDE_PME))*PDE_PME)*100,marker = '^',color = 'darkorange', linestyle = '',markersize=9,alpha = 0.9)
    plt.title('PME/PDE concentration bias at different $B_1$',fontsize =24)
    plt.xlim([-0.5,len(par)+0.5])
    plt.yticks(fontsize=18)
    plt.ylim([-80,105])
    plt.legend(['FSL_MRS','LCModel'],loc = 2,fontsize=16)
    
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins.errorbar(np.around(np.linspace(-10,len(par)-1+10,len(par)),2),[0]*len(par),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
    axins.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),2),((((((mc_B1[0,0,:]/100)*liv_amp['PC']+liv_amp['PC'])+((mc_B1[1,0,:]/100)*liv_amp['PE']+liv_amp['PE']))/(((mc_B1[2,0,:]/100)*liv_amp['GPC']+liv_amp['GPC'])+((mc_B1[3,0,:]/100)*liv_amp['GPE']+liv_amp['GPE'])))-(1/PDE_PME))*PDE_PME)*100,np.sqrt(mc_B1[2,1,:]**2+mc_B1[3,1,:]**2),marker = 'o',color = 'darkblue', linestyle = '',markersize=8)
    axins.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),2),((((((LCM['PC'][0,:]/100)*liv_amp['PC']+liv_amp['PC'])+((LCM['PE'][0,:]/100)*liv_amp['PE']+liv_amp['PE']))/(((LCM['GPC'][0,:]/100)*liv_amp['GPC']+liv_amp['GPC'])+((LCM['GPE'][0,:]/100)*liv_amp['GPE']+liv_amp['GPE'])))-(1/PDE_PME))*PDE_PME)*100,np.sqrt(LCM['GPC'][1,:]**2+LCM['GPE'][1,:]**2),marker = '^',color = 'darkorange', linestyle = '',alpha = 0.9,markersize=7)
    x1, x2, y1, y2 = 0, 35, -0.5, 1
    axins.set_xlim(x1-0.1, x2+0.1)
    axins.set_ylim(y1, y2)
    axins.set_xticks(np.linspace(x1,x2-3,len(par[x1::4])))    
    axins.set_xticklabels(np.around(par[x1::4],2),fontsize=14)
    axins.set_yticks(np.linspace(y1,y2,4))    
    axins.set_yticklabels(np.around(np.linspace(y1,y2,4),2),fontsize=15)
    rectpatch, connects=ax.indicate_inset_zoom(axins,edgecolor="black",alpha=0)
    connects[0].set_visible(False)
    connects[1].set_visible(False)
    connects[2].set_visible(False)
    connects[3].set_visible(False)
    
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    #plt.ylim([-25,25])
    plt.savefig('PME_PDE concentration bias B1.png')
    plt.close('all')
    
    fig, ax = plt.subplots(1,1)
    ax.set_xticks(np.linspace(0,len(par)-2,len(par)//2))
    ax.set_xticklabels(par[::2],fontsize =12)
    plt.xlabel('$B_{1,true}/B_{1,expected}$',fontsize =18)
    plt.ylabel('Bias [%]',fontsize =18)	
    Pi_PME = (liv_amp['Pi'])/(liv_amp['PC']+liv_amp['PE'])	
    plt.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),((((((mc_B1[0,0,:]/100)*liv_amp['PC']+liv_amp['PC'])+((mc_B1[1,0,:]/100)*liv_amp['PE']+liv_amp['PE']))/((mc_B1[-4,0,:]/100)*liv_amp['Pi']+liv_amp['Pi']))-(1/Pi_PME))*Pi_PME)*100,np.sqrt(mc_B1[2,1,:]**2+mc_B1[3,1,:]**2),marker = '.',color = 'darkblue', linestyle = '-.')
    plt.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),((((((LCM['PC'][0,:]/100)*liv_amp['PC']+liv_amp['PC'])+((LCM['PE'][0,:]/100)*liv_amp['PE']+liv_amp['PE']))/((LCM['Pi'][0,:]/100)*liv_amp['Pi']+liv_amp['Pi']))-(1/Pi_PME))*Pi_PME)*100,np.sqrt(LCM['GPC'][1,:]**2+LCM['GPE'][1,:]**2),marker = '.',color = 'darkorange', linestyle = '-.')
    plt.errorbar(np.around(np.linspace(-10,len(par)-1+10,len(par)),0),[0]*len(par),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
    plt.title('PME/Pi concentration bias at different B1',fontsize =16)
    plt.xlim([-0.5,len(par)+0.5])
    plt.yticks(fontsize=12)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    #plt.ylim([-25,50])
    plt.legend(['FSL_MRS','LCModel'],fontsize=12)
    plt.savefig('Pi_PDE concentration bias B1.png')
    plt.close('all')
    
    fig, ax = plt.subplots(1,1)
    ax.set_xticks(np.linspace(0,len(par)-2,len(par)//2))
    ax.set_xticklabels(par[::2],fontsize =12)
    plt.xlabel('$B_{1,true}/B_{1,expected}$',fontsize =14)
    plt.ylabel('Percentage [%]',fontsize =14)	
    ATP_PME = (liv_amp['gATP'])/(liv_amp['PC']+liv_amp['PE'])	
    plt.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),((((((mc_B1[0,0,:]/100)*liv_amp['PC']+liv_amp['PC'])+((mc_B1[1,0,:]/100)*liv_amp['PE']+liv_amp['PE']))/((mc_B1[4,0,:]/100)*liv_amp['gATP']+liv_amp['gATP']))-(1/ATP_PME))*ATP_PME)*100,np.sqrt(mc_B1[2,1,:]**2+mc_B1[3,1,:]**2),marker = '.',color = 'darkblue', linestyle = '-.')
    plt.errorbar(np.around(np.linspace(0,len(par)-1,len(par)),0),((((((LCM['PC'][0,:]/100)*liv_amp['PC']+liv_amp['PC'])+((LCM['PE'][0,:]/100)*liv_amp['PE']+liv_amp['PE']))/((LCM['gATP'][0,:]/100)*liv_amp['gATP']+liv_amp['gATP']))-(1/ATP_PME))*ATP_PME)*100,np.sqrt(LCM['GPC'][1,:]**2+LCM['GPE'][1,:]**2),marker = '.',color = 'darkorange', linestyle = '-.')
    plt.errorbar(np.around(np.linspace(-10,len(par)-1+10,len(par)),0),[0]*len(par),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
    plt.title('PME/ATP concentration bias at different B1',fontsize =16)
    plt.xlim([-0.5,len(par)+0.5])
    plt.yticks(fontsize=12)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    #plt.ylim([-50,90])
    plt.legend(['FSL_MRS','LCModel'],fontsize=12)
    plt.savefig('ATP_PDE concentration bias B1.png')
    plt.close('all')
    
   
    #%%Visualize spectra
    import matplotlib.gridspec as grid_spec
    gs = grid_spec.GridSpec(len(basis_names),1)
    fig = plt.figure(figsize=(16,9))
    
    i = 0
    cmap = cm.get_cmap('viridis')
    ax_objs = []
    for country in basis_names:
        #country = countries[i]
        x = np.linspace(-5000,5000,len(basis_fids[i]))/(B0*gamma*10**-6)
        # creating new axes object
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
    
        # plotting the distribution
        ax_objs[-1].plot(x, np.abs(np.fft.fftshift(np.fft.fft(basis_fids[i])))/ np.max(np.abs(np.fft.fftshift(np.fft.fft(sum(basis_fids))))),color=cmap(i*20),lw=1.5)
        #ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1,color=colors[i])
        ax_objs[-1].set_ylim([0,1])
    
    
        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)
    
        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticklabels([])
    
        ax_objs[-1].set_xticklabels([])
    
        spines = ["top","right","left","bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)
    
        adj_country = country.replace(" ","\n")
        ax_objs[-1].text(-20,0,adj_country,fontweight="bold",fontsize=14,ha="right",color = cmap(i*20))
        plt.xlim([-20,10])
        ax = plt.gca()
        ax.invert_xaxis()
        if i == len(basis_names)-1:
            plt.xticks([-20,-15,-10,-5,0,5,10],[-20,-15,-10,-5,0,5,10],fontsize = 14)
            plt.xlabel('Chemical Shift [ppm]',fontsize = 16)
            plt.yticks([])
        else:
            plt.xticks([]),plt.yticks([])
            
            
        i += 1
    
    gs.update(hspace=-0.7)
    
    plt.tight_layout()
    plt.show()
    #%%Import data from real acquisition
    file_name = r'D:\Some_data\SoggettiNICI\Rach.mat'#Luca_14_03_22, SoggettiNICI\Rach
    def read_mat(filename):
        '''
        read_mat return the matlab workspace as dictionary.
    
        Parameters
        ----------
        filename : str
            Full path of .mat file.
    
        Returns
        -------
        exam : python dictionary
            Imported workspace.
    
        '''
        mat = h5py.File(filename, 'r')
        exam={}
        for k, v in mat.items():
            exam['{}'.format(k)] = np.array(v)
        mat.close()
        return exam
    #Complex values are imported in a strange format
    #FOR SOME REASONS DIMENSIONS ARE INVERTED WITH RESPECT TO MATLAB:(S,X,Y,Z)=>(Z,Y,X,S)
    #Remember indicization in matlab starts form 1 while in python from zero.
    try:
        exam = loadmat(file_name)
        FIDS = exam['FIDS']
        specs = exam['spec']
    except:
        exam = read_mat(file_name)
        tmp = exam['FID'].view(np.double).reshape(16,16,16,540,2)
        FIDS = tmp[:,:,:,:,0] + 1j*tmp[:,:,:,:,1]
        tmp = exam['spec'].view(np.double).reshape(16,16,16,540,2)
        specs = tmp[:,:,:,:,0] + 1j*tmp[:,:,:,:,1]
        del tmp
    psf = loadmat(r'PSF.mat')['psf']
    #d = loadmat(r'dmat.mat')['d']
    # xx = loadmat(r'xxmat.mat')['xx']
    # kk = loadmat(r'kkmat.mat')['kk']
    
    # mat = loadmat(r'D:\SoggettiNICI\Raw_Rach.mat')
    # xx = mat['xx']
    # kk = mat['kk']
    # d = mat['d']
    #%apodization
    #FID = np.fft.ifft(np.fft.ifftshift(specs[8,8,8,:]))[:508]
    t = exam['t'][0,:]
    t0=0
    lb=20
    apo = np.exp(-(((np.pi*lb)*(t-t0))**2)/(4*np.log(2)))
    apoFIDS = FIDS*apo
    # #%%
    # rand_basis_fids = []
    # name = 'Simulated_randomized_basis'
    # n=10
    # rc=np.ones(len(basis_names))
    # for i,fid in enumerate(ph_fids[-n:]):
    #     rc[i] = (1 + np.random.uniform(low=-0.1, high=0.1))
    #     rand_basis_fids.append(fid*rc[i])
    # signal = sum(rand_basis_fids)
    # info = {'ID': name, 'FMTDAT':'(2E16.6)', 'Volume': 1.0, 'TRAMP': 1.0}
    # header = {'bandwidth':10000.0, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000,
    #           'width':10, 'points':len(signal)}
    # lcm_io.saveRAW('{}.RAW'.format(name),signal, info = info, hdr=header, conj=True)#create raw basis file 
    # #%% Zero-fill
    # def zf(a):
    #     a = np.append(a,np.zeros(np.shape(a)))
    #     return np.fft.fftshift(np.fft.fft(a))
    # zf_specs = np.apply_along_axis(zf, 3, FIDS[:,:,:,25:])
    # zf_basis = np.apply_along_axis(zf, 1, ph_fids)
    #%% Try restoration by Richardson-Lucy
    
    #r = restoration.richardson_lucy(np.abs(specs[:,:,9,270]), np.abs(psf[:,:,9]))
    
    #absorp_spectra, disp_spectra, ph = simple_phasing(specs[8,8,7,:])
    #%% Denoising using Dennis article
    
    den, dden, sig_map = PCA_denoising(zf_specs, 'MRS', verb = True)
    #%% Deconvolutuion
    # #zero padding
    # kmat = np.zeros((32,32,32,540), dtype=complex)
    # specs_zf = np.zeros((32,32,32,540), dtype=complex)
    # for i in range(0,540):
    #     kmat[8:24,8:24,8:24,i] = np.fft.ifftn((specs[:,:,:,i]),axes=(0,1,2))
    #     specs_zf[:,:,:,i] = np.fft.fftn(kmat[:,:,:,i],axes=(0,1,2))
    # #ASSUMING THAT THEPHASE OF A SINGLE SPECTRAL POINT IS THE SAME ALONG ALL THE IMAGE
    # # phase = np.angle(dden_zf)
    # # np.nan_to_num(phase,False,0)
    # #%%Phasing all spectra
    # plt.figure()
    # ans = []
    # for i in range(1,np.size(specs,0)):
    #     for j in range(1,np.size(specs,1)):
    #         for k in range(1,np.size(specs,2)):
    #             #specs[i,j,k,:] = ng.proc_autophase.autops(specs[i,j,k,:],'acme',disp = False)
    #             if any(np.real(specs[i,j,k,:])>10**5):
    #                 ans.append((specs[i,j,k,:]))
    #                 plt.plot(np.imag(FIDS[i,j,k,:]))

  
    #%%Metodo di Stoch et al. per eliminare la baseline
    # base = lambda x: np.ones(len(b))*x[0]+np.sum(np.array([k*np.exp(-1j*2*(i+1)*np.pi*np.linspace(0,len(b)-1,len(b))/len(b)) for i,k in enumerate(x[1::2])]),axis = 0)+np.sum(np.array([k*np.exp(1j*2*(i+1)*np.pi*np.linspace(0,len(b)-1,len(b))/len(b)) for i,k in enumerate(x[2::2])]),axis = 0)
    # fun = lambda x: np.sum(np.square(np.real(b-base(x)))[0:100])+ np.sum(np.square(np.real(b-base(x)))[330:])
    # res = minimize(fun, np.array([0]*9), method='CG', tol=1e-6, options = {'maxiter': 10000000})
    # if res.success:
    #     sign = b - base(res.x)
    # #%% define aposization function (already implrmented in fidall so not needed)
    # ea = lambda x: np.exp(-((np.linspace(0,540,540)-150)/(2*x))**2)
    # apo = ea(50)
    #%% grid display
    fig, axes = plt.subplots(nrows=16, ncols=16, sharex=True, sharey=True)
    slab = 8
    conty = 0
    contx = 0
    for _, ax in enumerate(axes.flatten()):
        if conty == 16:
            conty = 0
            contx = contx +1
        #ax.set_facecolor('black')
        ax.plot(np.abs(specs[-slab-1,-conty-1,-contx-1,:]),color='black',linewidth=0.5)
        ax.set_ylim(0,150000)
        ax.set_xlim([200,400])
        ax.invert_xaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        conty = conty + 1
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # #%% Manual FT for single voxel
    # v = [8,10,7]
    # vox = v[2]*16*16+v[1]*16+v[0]#2504 #(8,12,9)
    # s = np.shape(d)
    # D = np.zeros((s[0],s[1],s[-1]),dtype = np.complex64)
    # ones=np.ones((540,24))
    # ones[:10,0]=np.linspace(0.9,1,10)
    # for i in range(0,np.shape(d)[0]):
    #     D[i,:,:] = d[i,:,0,0,0,:]*ones*np.exp(np.sum(kk*xx[vox,:],1))[i]
    # #del d,s  #delete d and s for memory space
    # #%% Find proper interval for noise evaluation as in "A statistical analysis of NMR spectrometer noise"
    # from test_functions import Look4Gauss,Slotboom
    
    # nogood = Look4Gauss(D, noise_area = 100, pval=0.05)
    
    # #%% Slotboom et al validation method
    # uml, uvarl, bskl, bksl, _=[],[],[],[],[] 
    # #k = np.random.normal(0,1000,(50,540)) #testing Slootboom with gaussian noise
    # um, uvar, bsk, bks, ch_mask = Slotboom(np.real(D[:,:,0]),snrt = 2, n_r = 100, nogood=None)

    
    # #%% Interpolazione per trovare i punti iniziali del fid mancanti
    
    # def sin_fid(x,amp,om,phi,dump):
    #     return np.array(amp*np.cos(om*x+phi)*np.exp(-x/dump))
    # def all_fid(x, *argv):
    #     return np.sum(np.array([sin_fid(x,argv[i],argv[i+1],argv[i+2],argv[i+3]) for i,_ in enumerate(argv[0::4])]),axis = 0)
    
    # pa = [1000,1,1,1,1000,1,1,1,1000,1,1,1,1000,1,1,1,1000,1,1,1,1000,1,1,1]    
    # param,_ = curve_fit(all_fid,np.linspace(0,100,100),np.real(FIDS[7,9,9,0:100]), p0=pa,bounds=(0,np.inf),method = 'dogbox')

    #%%Part created in order to understand how phases changes between peaks (MUST CONVERT TO IMPORTABLE FUNCTION)

    # auto =0
    # ma = np.array([],dtype = int)
    # tmp = 0
    # if auto == 1:
    #     ma = feature.peak_local_max(savgol_filter(np.max(np.abs(specs), axis=(0,1,2)),5,3),num_peaks=5)
    #     am = ma[:,0]
    # elif auto == 0:
    #     temp = 0
    #     plt.draw()
    #     plt.plot(np.max(np.abs(specs), axis=(0,1,2)))
    #     plt.pause(0.5)
    #     while tmp != 'end':
    #         tmp = input('Insert peak location(insert \'end\' to end) then press enter:')
    #         if tmp != 'end': ma = np.append(ma, int(tmp))
    #     plt.close('all')
    #     am = ma
    
    
    # am = np.sort(am)
    # dam = np.diff(am)
    # dam = np.diff(am)/2
    # areas =np.floor(dam)
    # areas = np.array(areas)
    # areas = np.append(areas[0],areas)
    # areas = np.append(areas,areas[-1])
    # hp = np.array([])
    # whp = np.array([])
    # m = np.zeros(len(sp))
    
    # new = sp.copy()
    # ph0 = 0
    # for i,v in enumerate(am):  
    #     tsp = ng.proc_autophase.ps(sp[int(v-areas[i]):int(v+areas[i+1])],ph0,0) #Uses old ph as starting point for next peak
    #     ph,_ = ng.proc_autophase.manual_ps(tsp)
    #     new[int(v-areas[i]):int(v+areas[i+1])],whp = np.append(hp,ng.proc_autophase.ps(sp[int(v-areas[i]):int(v+areas[i+1])],ph+ph0,0)), np.append(whp,ph+ph0)
    #     ph0 = ph
    #     #m[int(v-areas[i]):int(v+areas[i+1])] = np.mean(phase)*(areas[i]+areas[i+1])*np.mean(np.abs(sp[int(v-areas[i]):int(v+areas[i+1])]))
    # inam = np.append(0,am)
    # inam = np.append(inam,539)
    # inwhp = np.append(whp[0],whp)
    # inwhp = np.append(inwhp,whp[-1])
    # model =  interp1d(inam, inwhp, kind='linear')(np.linspace(0,539,540))
    # ans = ng.proc_autophase.ps(sp,model)
    # phi = np.sum((hp*whp))/np.sum(whp)
    # model = scipy.interpolate.CubicSpline(np.linspace(0,539,540),m/np.sum(whp))
    # #phi = model(m/np.sum(whp))
    # absorp_spectra = np.real(sp)*np.cos(phi)+np.imag(sp)*np.sin(phi)
    # disp_spectra = np.imag(sp)*np.cos(phi) - np.real(sp)*np.sin(phi)
        
    #%%Try fitting

    
    #They want FIDs of both: basis and signal to analize. The Basis class seems not to exist, the specificated headers are the minimal requrements needed in order to fit the data.
    #for some reason this plots only the real part of the spectrum so, i suppose, phasing must be done in time frequency (this could lead at at different fit paraeter, EXTEND THIS ASPECT BEFORE CONTINUING!)
    # #Pay attention to the fact that names and basis_hdr must be a list becouse there could be more than one basis
    # #Attention water shift is calculater relative to TMS while, for 31P, is set to zero.
    # '''COULD BE 31P BUT, BY DEFAULT, SHIFT IS SET TO 0.0 FOR THIS NUCLEUS'''
    # nucleus = '31P' #Ref. nucleus
    # nucleus_freq = B0*gamma #Resonance frequency expected at this B0
    # #Change default shift for display, the default ones are set so i could only cange them grafically after calcoulation.
    # #An alternative ts to change directly the constants.py file
    # changed_shift = {'1H':2.91, '2H':2.91, '13C':0.0, '31P':0.0}#<---------CANNOT MANAGE TO CHANGE IT WHITOUT CREATING A NEW FILE
    # #%%Put togheter metabolites
    # groups = np.array([['PC','PE'],['GPC','GPE']])
    # tmp_basis_names = []
    # tmp_basis_fids = []
    
    # for i in range(groups.shape[0]):
    #     tmp_n = np.array(basis_names)
    #     tmp_basis_names.append('{}+{}'.format(groups[i,0],groups[i,1]))
    #     tmp_basis_fids.append(np.array(basis_fids)[tmp_n==groups[i,0]][0]+np.array(basis_fids)[tmp_n==groups[i,1]][0])
    # for n in basis_names:
    #     if (groups!=n).all():
    #         tmp_basis_names.append(n)
    #         tmp_basis_fids.append(np.array(basis_fids)[tmp_n == n][0])
    
    #%%Denoising
    ddata = denoising.sure_svt(specs, np.var(specs[0,0,0,100:500])/2,patch=(2,3,3))
    
    one,done=0.55,0.2
    two,dtwo = 0.49,0.14
    #%%Using synthetic FID
    m=[0.48,0.26,0.54]
    re = np.array([0.78,0.48])#[8,10,7,:],[8,10,8,:]
    are = np.array([0.76,0.26])
    denre = np.array([0.67,0.23,0.85])
    dre = np.array([0.09,0.11,0.13])
    dare = np.array([0.0,0.14])
    # dre = np.min(dre)/dre
    # dare =np.min(dare)/dare
    # from statsmodels.stats.weightstats import DescrStatsW
    # m=np.sum(re*dre)/np.sum(dre)
    # dm= DescrStatsW(re, weights=dre, ddof=1).std
    # m=np.sum(re*dre)/np.sum(dre)
    #from fsl_mrs.utils.preproc import phasing as phs
    #a = ng.proc_autophase.autops(zf_specs[8,8,8,:],'acme')
    #Creating fids from absorption spectra of basis set for basis fids
    #p0,p1 = ng.proc_autophase.manual_ps(a)
    #a = ng.proc_base.ps(a, p0, p1)
    #dFIDS=np.fft.ifft(np.fft.ifftshift(ddata,axes=3),axis=3)[:] #np.fft.ifft(np.fft.ifftshift((specs[8,9,9,:])))[7:] with basis LB=10,GB=3 works perfectly and has gaussian residuals
    #FID = FIDS[9,11,6,7:]#LUCA 9,11,7,7:/9,11,6,7: 
    #FID=D1
    #plt.plot(np.fft.fft(FID))
    import json
    from test_functions import Look4Gauss
    file1 = open('.\LCMfits\Luca_6.txt', 'r')
    Lines = file1.readlines()
    plindx=int(json.loads(Lines[0]))
    shift=-np.array(json.loads(Lines[1]))*(1/1000)
    data_shift = np.array(json.loads(Lines[2]))
    shift += data_shift
    dlb = np.array(json.loads(Lines[3]))
    [p0,p1]=np.array(json.loads(Lines[4]))
    lineshape=np.array(json.loads(Lines[5]))
    c=np.array(json.loads(Lines[6]))
    def ps(data, p0=0.0, p1=0.0, pivot = 0.0, inv=False): #AC, added pivot as input to ng.proc_base.ps input so that ot could be compatiblr with manual_ps
        """
        Linear phase correction

        Parameters
        ----------
        data : ndarray
            Array of NMR data.
        p0 : float
            Zero order phase in degrees.
        p1 : float
            First order phase in degrees.
        inv : bool, optional
            True for inverse phase correction

        Returns
        -------
        ndata : ndarray
            Phased NMR data.

        """
        p0 = p0 * np.pi / 180.  # convert to radians
        p1 = p1 * np.pi / 180.
        size = np.shape(data)[0]
        # apod = np.exp(1.0j * (p0 + (p1 * np.arange(size) / size)) #AC, old version
        #               ).astype(data.dtype)
        apod = np.exp(1.0j * (p0+ (p1 * np.arange(-pivot, -pivot + size)  / size)) #AC, new version
                      ).astype(data.dtype)
        if inv:
            apod = 1 / apod
        return apod * data
    def LCMCH5plot(D0):
        def ppm2points(ppm,cf,bw,ln):
            Hz = cf*ppm*1E-6
            points = Hz/(bw/ln)
            return points
        
        bw,ln=10000,len(D0)*2
        def LMplot(basis,c,lineshape,shift,dlb,p0,p1):
            # def convS(spec,shape):
            #     l = len(shape)
            #     newspec=spec.copy()
            #     for j in range(l,len(spec)-l):#tolgo i primi l punti tanto non mi interessano
            #         newspec[j]=np.sum(spec[j-l//2:j+l//2 +1]*shape)
            #     return newspec
                    
            new_spec=[]
            for i,fid in enumerate(basis):
                fidcor = fid*np.exp(-(dlb[i]+1j*shift[i]*cf*1E-6*2*np.pi)*(np.linspace(0,len(fid)-1,len(fid))/bw),dtype=complex)
                pred = np.fft.fftshift(np.fft.fft(np.append(fidcor,np.zeros(len(fidcor)))))
                #inx = np.where(np.abs(pred)==np.max(np.abs(pred)))[0][0]
                pred = np.convolve(pred,lineshape[::-1],'same')
                new_spec.append(c[i]*pred)
            fit_spec=ps(sum(new_spec),0,0*len(new_spec)/ppm2points(1,cf,bw,ln),len(new_spec)/2)#-ppm2points(4.65,cf,bw,ln))
            return fit_spec, new_spec
        #plt.plot(fit)
        fit, sing = LMplot(basis_fids,c,lineshape,shift,dlb,p0,p1)
        if any(fit<0):
            pf0,pf1,pivf = ng.proc_autophase.manual_ps(fit)
            data_spec = ps(np.fft.fftshift(np.fft.fft(np.append(D0,np.zeros(len(D0))))),-p0,-p1*len(fit)/ppm2points(1,cf,bw,ln),len(fit)/2)
            plt.plot(np.linspace(-bw/2,bw/2,D0.shape[0]*2)/(cf*10**-6),np.real(ps(data_spec,pf0,pf1,pivf)),color='black',linewidth=1) 
            plt.plot(np.linspace(-bw/2,bw/2,D0.shape[0]*2)/(cf*10**-6),np.real(ps(fit,pf0,pf1,pivf)),color='darkred',alpha =0.8,linewidth=2)
            plt.title('Absorption spectum of acquired liver voxel',fontsize=20)
            print('PME SNR={}'.format(np.max(np.abs(data_spec)[620:640])/np.std(np.real(data_spec)[-100:])))
        else:
            plt.title('Magnitude spectum of acquired liver voxel',fontsize=20)
            plt.plot(np.linspace(-bw/2,bw/2,D0.shape[0]*2)/(cf*10**-6),np.abs(ps(np.fft.fftshift(np.fft.fft(np.append(D0,np.zeros(len(D0))))))),color='black',linewidth=1)
            plt.plot(np.linspace(-bw/2,bw/2,D0.shape[0]*2)/(cf*10**-6),np.abs(fit),color='darkred',alpha =0.8,linewidth=2)
            
        plt.xlim([10,-20])
        plt.xlabel('Chemical Shift [ppm]',fontsize=18)
        plt.ylabel('Amplitude [a.u.]',fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(3,4))
        #plt.legend(['data','LCM fit'],fontsize=16)
        plt.figure()
        res = ps(data_spec,pf0,pf1,pivf)-ps(fit,pf0,pf1,pivf)
        chi = np.sum((np.abs(res)**2)/(4*np.var(np.real(res))))
        
        print('chinorm={}'.format(chi/((len(basis_names)*3)+2-1)))
        plt.plot(np.linspace(-bw/2,bw/2,D0.shape[0]*2)/(cf*10**-6),res)
        plt.xlim([-20,10])
        
        plt.figure()
        col = plt.cm.hot(np.linspace(0,1,24))   
        if len(Look4Gauss(ps(data_spec,pf0,pf1,pivf)-ps(fit,pf0,pf1,pivf),len(data_spec),0.05)['0'])==0:
             print('ok')
        for i,sp in enumerate(sing): 
            plt.plot(np.linspace(-bw/2,bw/2,D0.shape[0]*2)/(cf*10**-6),np.real(ps(sp,pf0,pf1,pivf)),color=col[i],alpha =0.8,linewidth=2)
        # plt.plot(np.linspace(-bw/2,bw/2,D0.shape[0]*2)/(cf*10**-6),np.real(ps(data_spec,pf0,pf1,pivf)),color='black',linewidth=1) 
        # plt.xlim([10,-20])
        # plt.xlabel('Chemical Shift [ppm]',fontsize=18)
        # plt.ylabel('Amplitude [a.u.]',fontsize=18)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(3,4))
        # plt.legend(basis_names,fontsize=16)
        return
    cf = 120663345
    #p = phs.phaseCorrect(FID,10000,B0*gamma,'31P',(20,-10),False)
    cont=0
    #RAC[(6,11)(10,12)(6,9)]  120663345,120663324
    #Tiz[(6,10)(8,12)(6,10)] 120663290
    #Luc[(7,10)(10,12)(6,9)] 120663399
    
    for i in range(6,11):
        for j in range(8,12):
            for k in range(6,10): 
                FID = FIDS[i,j,-k,:]#Don't know why but - sign neede to match specs with FIDS locations
                spec=np.fft.fftshift(np.fft.fft(FID))
                if np.max(np.abs(spec))>=15*np.std(np.real(spec[-100:])):
                    info = {'ID': 'Test_subject', 'FMTDAT':'(2E16.6)', 'Volume': 1.0, 'TRAMP': 1.0}
                    header = {'bandwidth':10000.0, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000,
                              'width':10, 'points':len(FID)}
                    lcm_io.saveRAW('FID{}.RAW'.format(cont),FID*np.exp(-2j*np.pi*(0*header['centralFrequency'])*(np.linspace(0,header['points'],header['points'])/header['bandwidth'])), info = info, hdr=header, conj=True)#create raw basis file *np.exp(2j*np.pi*(-4.65*header['centralFrequency'])*(np.linspace(0,header['points'],header['points'])/header['bandwidth']))
                    with open('Liver_control.txt', 'r') as file:
                        data = file.readlines()
                    data[6] = " filraw= '/home/andrea/CH5/FID{}.RAW'\n".format(cont)
                    data[7] = " filps= '/home/andrea/CH5/FID{}.PS'\n".format(cont) 
                    data[8] = " filcsv= '/home/andrea/CH5/FID{}.csv'\n".format(cont) 
                    data[9] = " filpri= '/home/andrea/CH5/FID{}_det'\n".format(cont)
                    data[17] = " nunfil= {}".format(len(FID))
                    data[11] = "hzpppm= {}".format(cf*1E-6)
                    print('{},{},{}'.format(i,j,k))
                    if cont == 9:
                        LCMCH5plot(FID)
                        # plt.figure()
                        # pf0,pf1,pivf = ng.proc_autophase.manual_ps(np.fft.fftshift(np.fft.fft(np.append(FID,np.zeros(len(FID))))))
                        # plt.plot(np.linspace(-bw/2,bw/2,FID.shape[0]*2)/(cf*10**-6),ps(np.fft.fftshift(np.fft.fft(np.append(FID,np.zeros(len(FID))))),pf0,pf1,pivf),color='black',linewidth=1)
                        
                        # plt.xlim([10,-20])
                        # plt.xlabel('Chemical Shift [ppm]',fontsize=18)
                        # plt.ylabel('Amplitude [a.u.]',fontsize=18)
                        # plt.ticklabel_format(style='sci', axis='y', scilimits=(3,4))
                        # plt.xticks(fontsize=16)
                        # plt.yticks(fontsize=16)
                    # and write everything back
                    with open(r'C:\Users\Andrea\Documents\GitHub\Spettroscopia\control_liver{}.CONTROL'.format(cont), 'w') as file:
                        file.writelines( data )
                        file.close()
                    
                    
                    cont+=1
    # if len(Look4Gauss(FID,100,0.05)['0'])==0:
    #     print('ok')
        # fig, ax = plt.subplots(1,1)
    # plt.xlabel('t [0.1 ms]',fontsize=14)
    # plt.ylabel('Amplitude [a.u.]',fontsize=14)		
    # plt.plot(np.linspace(0,539,540),np.real(FID),color = 'darkblue', linestyle = '-')
    # plt.plot(np.linspace(0,539,540),np.imag(FID),color = 'darkorange', linestyle = '-',alpha=0.6)
    # plt.errorbar(np.linspace(0,539,540),[0]*len(np.linspace(0,539,540)),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_") 
    # plt.legend(['Real','Imag'])
    # plt.xlim([0,540])
    # plt.ylim([-2000,5000])
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    # axins.plot(np.linspace(0,539,540),np.real(FID),color = 'darkblue', linestyle = '-')
    # axins.plot(np.linspace(0,539,540),np.imag(FID),color = 'darkorange', linestyle = '-',alpha=0.6)
    # axins.errorbar(np.linspace(0,539,540),[0]*len(np.linspace(0,539,540)),linestyle='--',color='black',alpha = 0.6,label = "_nolegend_")  
    # # sub region of the original image
    # x1, x2, y1, y2 = 440, 540, -700, 700
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # #axins.set_xticks([])    
    # axins.set_xticklabels([])
    # #axins.set_yticks(np.linspace(y1,y2,4))    
    # axins.set_yticklabels([])
    
    # ax.indicate_inset_zoom(axins, edgecolor="black")
        #%%
    #noise = np.random.normal(0,0.0000001,len(sum(ph_fids)))+1j*np.random.normal(0,0.0000001,len(sum(ph_fids)))
    #MRS object wants just FIDs as input for it's calculations
    '''IT USES WATER SIGNAL AS REFERENCE SO MAYBE IF ANOTHER REFERENCE IS USED IN FID AND TISSUE CONCENTRATION IT COILD BE USED FOR 31P'''
    a = MRS(FID=FID, header = {'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000},
            basis = np.array([x for x in basis_fids]),
            names = basis_names,
            basis_hdr=[{'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': 120.664, 'dwelltime': 1./10000, 'fwhm':10}],
            nucleus='31P',bw=10000, cf = 120.664)
            #H2O = np.fft.ifft(basis_sp[basis_names == 'PCr']),#H20 reference FID needed. Can it be turned in some 31P usefull reference?
            #bw=10000, nucleus = nucleus, cf = 120.645)
    #%Process and fitting
    # freq = (changed_shift[nucleus]*10E-6*nucleus_freq) + nucleus_freq
    # sft = (a.frequencyAxis-freq)*10E6/freq
    ppmlim = (-20.,10.)
    a.rescaleForFitting(scale=100) #Process data in a suitable design for fitting
    results = fitting.fit_FSLModel(a,method='Newton',ppmlim=ppmlim,model='voigt',
                                   baseline_order=-1, MHSamples=500,metab_groups=[1,1,4,3,5,10,10,6,7,8,9,0])
    inf, sup = a.ppmlim_to_range(ppmlim)
    rres = np.fft.fftshift(np.fft.fft(results.residuals))
    FQN = np.var(rres[inf:sup])/np.var(np.append(rres[:inf],rres[sup:])) #NOTE: ddof is zero in the variance because it the noise it's supposed to be gaussian

    print('FQN:{}'.format(FQN))#FIT quality number, see Near et al.
    #ATTENTION: SNR has been calculated as maximum over SD of noise OUTSIDE the spectrum of interest as sqrt(2)*np.real(noise) SO IT'S ASSUMED GAUSSIAN NOISE!
    print('FQN/SNR:{}'.format(FQN/results.SNR.spectrum)) #results.SNR.spectrum shoould be the SNR calculated like max value of the spectrum divided by noise
    #Assign an internal reference to the object result
    #<-------Da capire come modificare questi T1 e T2---------->
    #quant = quantify.QuantificationInfo(TE,TR,basis_names,nucleus_freq, water_ref_metab='PC',water_ref_metab_protons=1,water_ref_metab_limits=(2,5))
    #quant.set_fractions({'GM':1,'WM':0.,'CSF':0.})#Set information about tissue volume fraction. It must contain GM, WM, CSF
    #quant.set_densitites({'GM':1,'WM':0.,'CSF':0.})#Tissue water densities (g/cm^3)
    #results.calculateConcScaling(a,quant_info=quant,internal_reference='PC')
    # molar = results.getConc(scaling='molarity')#Molarity concentration of metabolites
    # print('Molarity concentration: {} mM/dm^3'.format(molar))
    # molal = results.getConc(scaling='molality')#Molality concentration of metabolites
    # print('Molality concentration: {} mM/kg'.format(molal))
    #results.calculateConcScaling(a,internal_reference=['GPC', 'GPE'])
    dpc = (results.params[0]*results.perc_SD[0]/100)
    dpe = (results.params[1]*results.perc_SD[1]/100)
    dgpc = (results.params[2]*results.perc_SD[2]/100)
    dgpe = (results.params[3]*results.perc_SD[3]/100) 
    dpme = np.sqrt(dpc**2+dpe**2)
    dpde = np.sqrt(dgpc**2+dgpe**2)
    pme = results.params[0]+results.params[1]
    pde = results.params[2]+results.params[3]
    e = np.sqrt((dpme/pde)**2 + ((pme/pde)**2)*dpde**2)
    print('PME/PDE={}+-{}'.format(pme/pde,e))
    
    lb_tot=np.array(LB_b)+results.params[12:24]
    #%%Plotting fitting results
    
    fig = plotting.plotly_fit(a,results,ppmlim=(ppmlim[0]+0.5,ppmlim[1]-0.5),phs=results.getPhaseParams(), proj='real')
    fig.show('browser')
    if a.names[0]=='PC+PE':
        pme = results.params[0]
        pde = results.params[1]
    else:
        pme = results.params[0]+results.params[1]
        pde = results.params[2]+results.params[3]
    print('PME/PDE={}'.format(pme/pde))
    print('PME/ATP={}'.format(pme/results.params[4]))
    print('Pi/ATP={}'.format(results.params[6]/results.params[4]))
    # print('NAD+/ATP={}'.format(results.params[-3]/results.params[4]))
    # for s in results.SNR[1].columns:
    #     print('{}={}'.format(s,results.SNR[1][s][0]))
    #%%plot individual
    fig = plotting.plot_indiv_stacked(a,results,ppmlim,kind = 'imag')
    fig.show('browser')
    #%%Testing residuals
    import test_functions
    p = 0.05
    res_spec = a.get_spec(ppmlim=ppmlim)-results.predictedSpec(a,ppmlim=ppmlim)
    ng = test_functions.Look4Gauss(res_spec,noise_area=500,pval=p)
    if ng['0'].size == 0:#if array of signals (just one) is empty than the signal is probably gaussian
        print('!!!!!!!!! O M G !!!!!!!!! ')
        print('The model is reasonably (p = {}) correct. Hurray!'.format(p))
        plt.figure()
        plt.title('Residual plot')
        plt.scatter(np.real(res_spec),np.imag(res_spec), color = 'green')
        plt.xlabel('Real part of residuals', fontsize=12)
        plt.ylabel('Imaginay part of residuals', fontsize=12)
    else:
        print('The model is probably (p = {}) NOT correct. This is so sad...'.format(p))
        plt.figure()
        plt.title('Residual plot')
        plt.scatter(np.real(res_spec),np.imag(res_spec),color='red')
    #%%Save plot
    fig = plotting.plot_real_imag(a, results, ppmlim=(-20,10))
    fig.write_image("fig1.png")
    #%%Plot estimated correlation matrix for concentrations
    fig, ax = plt.subplots(1,1)

    img = ax.imshow(results.corr[:len(basis_names),:len(basis_names)],cmap = 'plasma', 
                             vmin=-1,vmax=1)
    fig.colorbar(img)
    plt.title('Estimated concentrations correlation matrix')
    ax.set_xticks(np.linspace(0,len(basis_names)-1,len(basis_names)))
    ax.set_xticklabels(basis_names)
    ax.set_yticks(np.linspace(0,len(basis_names)-1,len(basis_names)))
    ax.set_yticklabels(basis_names)
    #%%pH evaluation
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    pi_idx = np.where(np.array(basis_names)=='Pi')[0][0]
    pcr_idx = np.where(np.array(basis_names)=='PCr')[0][0]
    atp_idx = np.where(np.array(basis_names)=='ATP')[0][0]
    if results.getConc()[pi_idx] != 0 and results.getUncertainties(metab='Pi') < 100: #50 of %SD fixed by me as a reasonable limit
        pi_idx = np.where(np.array(basis_names)=='Pi')[0][0]
        pi_group = results.metab_groups[pi_idx]
        pi_shift = spinSys[pi_idx+2]['shifts'][0]+results.getShiftParams()[pi_group] #Add to prior chemical shift the fitted eps
        if results.getConc()[pcr_idx] and results.getUncertainties(metab='PCr') < 10:
            pcr_group = results.metab_groups[pcr_idx]
            ref_shift = spinSys[pcr_idx+2]['shifts'][0]+results.getShiftParams()[pcr_group] #Add to prior chemical shift the fitted eps
            pH = 6.77 +np.log10(((pi_shift-ref_shift)-3.23)/(5.7-(pi_shift-ref_shift)))
            print('pH:{}'.format(pH))
        elif results.getConc()[atp_idx] and results.getUncertainties(metab='ATP') < 30:#Using alpha-ATP as backup as seen in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4290015/
            atp_group = results.metab_groups[atp_idx]
            ref_shift = -(spinSys[atp_idx+2]['shifts'][2]+results.getShiftParams()[atp_group]) #minus is needed because ATP becomes the reference (so ATPshift is zero and Pi shift is Pishift+ATPshift)
            pH = 6.77 +np.log10(((pi_shift-ref_shift)-(3.23-ref_shift))/((5.7-ref_shift)-(pi_shift-ref_shift)))
            print('pH:{}'.format(pH))
        else:
            ref_shift = 'No_relaible_PCr_or_ATP'
            logging.warning("No reference peak available for pH calculation!")
    else:
        logging.warning("No reliable Pi peak available for pH calculation!")
    #%%Save report in different html files
    #|<----Magari, in futuro, mettere un argparse per il path----->|

    path = input('Insert full path (ending with folder to create if not already existing) for saving reports:')
    try:
        os.mkdir(path)
    except FileExistsError:
        print('Directory already exists. The file will be saved in this directory and overwritten if same name occours.')
    rep = create_plotly_div(a,results)
    for key in rep.keys():
        html_file= open(path + "\Report_{}.html".format(key),"w")
        html_file.write(rep[key])
    html_file.close()
#%%
    apo_res=LR_apo(results, a, results.params, None,apo)
    #%%
    import seaborn as sns
    s1=np.array([0.6403018096,	0.5835644558,	0.7841135586,	0.6849112426,	0.7655564655,	0.7888529032,	0.663334202,	0.6857243506])
    s2=np.array([0.6781414912,	0.6869469178,	0.6105527638,	0.6680584551])
    S = pd.DataFrame([s1,s2])
    S.index=['Subject #1','Subject #2']
    
    eb1=plt.bar([1,2],[np.mean(s1),np.mean(s2)], yerr=[np.std(s1),np.std(s2)], align='center', alpha=1,width=0.8, ecolor='black', capsize=10,edgecolor='black',color='lightgray')
    
    plt.xticks([1,2],['Subject #1','Subject #2'])#ax = sns.boxplot(data=S.T, palette="Set3",linewidth=2)
    plt.plot([1]*len(s1),s1,marker='o',markersize=7,alpha=0.8,linestyle='',color='green',dash_capstyle='round')
    plt.plot([2]*len(s2),s2,marker='o',markersize=7,alpha=0.8,linestyle='',color='orange')
    #ax.set_xlabel("X Label",fontsize=18)
    plt.xlim([0.5,2.5])
    plt.ylabel("PME/PDE",fontsize=18)
    plt.tick_params(labelsize=16)
    #%%
    pc,pe,gpc,gpe=0.22,0.24,0.55,0.52
    dpc,dpe,dgpc,dgpe=0.04,0.06,0.14,0.15
    pme,dpme=1.83,0.27#pe+pc,np.sqrt(dgpe**2+dgpc**2)
    pde,dpde=3.84,0.68#gpe+gpc,np.sqrt(dgpe**2+dgpc**2)
    print('{}+-{}'.format(pme/pde,np.sqrt((dpme/pde)**2+(dpde*pme/(pde**2))**2)))
    #%%
    filename=r'D:\Some_data\ncov_Luca.mat'
    exam = loadmat(filename)
    if 'ncorr' in filename:
        ncorr = exam['noisecor']
    elif 'ncov' in filename:
        ncorr = exam['noisecov']
    
    fig, ax = plt.subplots(1,1)
    img=plt.imshow(ncorr)
    plt.xticks(fontsize=18)
    ax.set_xticks([0,5,11,17,23])

    ax.set_xticklabels([1,6,12,18,24])
    ax.set_yticks([0,5,11,17,23])

    ax.set_yticklabels([1,6,12,18,24])
    plt.yticks(fontsize=18)
    cbar=fig.colorbar(img)
    cbar.ax.tick_params(labelsize=16) 
    plt.xlabel('Channel',fontsize=22)
    plt.ylabel('Channel',fontsize=22)
    if 'ncorr' in filename:
        plt.title('Correlation matrix (abs)', fontsize=20)
    elif 'ncov' in filename:
        plt.title('Covariance matrix (abs)', fontsize=20)
     #%%
    from matplotlib.colors import Normalize
    filename=r'D:\Some_data\mrsi3D_31P_scheme4_fov350_mtx16_bw10000_npts540_nexc16206_gmax19_smax185.mat'
    exam = loadmat(filename)
    K=exam['k']
    dcf=exam['dcf']
    c_m=np.in1d(dcf, np.unique(dcf)[::3]).reshape(dcf.shape)
    fig=plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    norm = Normalize(vmin=np.min(1/dcf)/np.sum(1/dcf), vmax=1)
    p=ax.scatter3D(K[0,c_m[0,:],0], K[0,c_m[0,:],1], K[0,c_m[0,:],2],c=(1/dcf[0,c_m[0,:]])/np.max(1/dcf), cmap='jet',alpha=0.3,marker='.')
    # plt.tick_params(left = False, right = False, labelleft = False ,
    #             labelbottom = False, bottom = False)
    ax.set_xlabel(r'$k_x$ [a.u]',fontsize=18)
    ax.set_ylabel(r'$k_y$ [a.u]',fontsize=18)
    ax.set_zlabel(r'$k_z$ [a.u]',fontsize=18)
    cbar=fig.colorbar(p,norm=norm,ticks=np.around(np.linspace(np.min(1/dcf)/np.max(1/dcf),1,len(np.unique(dcf)[::3])+1),2))
    cbar.set_label('Relative sampling density', rotation=90,fontsize=16)
    plt.title('3D view of k-space sampling scheme', fontsize=18)
    plt.figure()
    plt.plot(np.linspace(0,np.max(np.abs(K)),len(np.unique(dcf))),(1/np.unique(dcf))/np.sum((1/np.unique(dcf))),color='black')
    plt.xlim([0,np.max(np.abs(K))])
    plt.ylabel('Radial sampling density',fontsize=18)
    plt.xlabel('Distance from center',fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #%%
    plt.plot(np.linspace(-10,100,1000)*0,color='gray',linewidth=1,label = "_nolegend_")
    plt.plot(np.zeros(2),[0,0.8],color='gray',linewidth=1,label = "_nolegend_")
    plt.plot(0.6*np.exp((-0.6)*np.linspace(0,100,1000)),color='darkgreen',alpha=0.8,linestyle='--')
    plt.plot(0.7*np.exp((-0.06)*np.linspace(0,100,1000)),color='darkgoldenrod',alpha=0.8,linestyle='--')
    plt.plot(0.5*np.exp((-0.02)*np.linspace(0,100,1000)),color='peru',alpha=0.8,linestyle='--')
    # plt.figure()
    # plt.plot(np.linspace(-10,100,1000)*0,color='gray',linewidth=1,label = "_nolegend_")
    # plt.plot(np.zeros(2),[0,0.8],color='gray',linewidth=1,label = "_nolegend_")
    # plt.plot(0.6*(1-np.exp((-0.04)*np.linspace(0,100,1000))),color='darkgreen',alpha=1)
    # plt.plot(0.7*(1-np.exp((-0.14)*np.linspace(0,100,1000))),color='darkgoldenrod',alpha=1)
    # plt.plot(0.5*(1-np.exp((-0.04)*np.linspace(0,100,1000))),color='peru',alpha=1)
    #%%
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.pyplot as plt
    from math import exp,sin,cos
    from pylab import *
    
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    a=0.05
    b=0.01
    # took the liberty of reducing the max value for th 
    # as it was giving you values of the order of e42
    th=np.linspace(8, 12.6, 10000)  
    # x=a*sin(a*z)*cos(th)
    # y=a*sin(a*z)*sin(th)
    # z=np.linspace(0,2, 10000)  # creating the z array with the same length as th
    a = np.arctan(b*th)
    x=b*cos(th)*cos(a*th)
    y=b*sin(th)*cos(a*th)
    z=sin(a*th)
    ax.plot(x, y, z, color='red',linestyle='--')  # adding z as an argument for the plot
    ax.set_axis_off()
    
    plt.show()