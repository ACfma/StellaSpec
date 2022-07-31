# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 10:21:04 2022

A ready-to-use MRSI analysis pipeline focused on NICI goals:
    finding the ratio between PME and PDE using 31P-MRSI.

@author: Andrea
"""
import h5py
import json

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.io import loadmat

from fsl_mrs.utils.misc import FIDToSpec,checkCFUnits

#Modified packages
from fsl_mrs.utils.mrs_io  import lcm_io
from fsl_mrs.denmatsim import simseq, simulator
import nmrglue as ng



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
    TR : float
        Repetition time (in s).
    B1 : 1darray or list
        List of amplitudes for B1 field (in T).
    samp: 1darray
        Shape of the RF pulse (area needs to be normalized to one).
    G : 1daarray or list
        Iterable of three gradient areas:[Gx,Gy,Gz]
    T : 1darray or list
        Time duration of the B1 pulse (in s). The default is [0,1].
    GB : float
        Gaussian line broadening (in Hz). The default is 0.0.
    gamma : float
        Gyromagnetic ratio of nucleus (in Hz). The default is 17.235E6.
    shift : float
        Shift (in ppm) from the zero. The default is 0.0.
    shift_cf : float
        Simulate shift of transmission offset (not used or tested). The default is None.
    autophase : bool
        If autophase is True, phase the spectra with a reference peak in zero.
        if all the peaks are simulated singularly it's not necessary (not used).
        The default is None.
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
            "Rx_LW": LB[i],
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
                    "phaseOffset": phase,
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
    #Save as RAW file
    lcm_io.writeLcmInFile('{}_basis.in'.format(n),basis_names,'','{}'.format(n),header,-4.65,0.0)
    return

#%%STEP 0: reconstruct spectral disposition by using recon_mrsi.m
if __name__ == "__main__":
    #%%STEP 1: Simulation of the used sequence for basis set simulation
    
    #Importing informations about 31P liver metabolites at 7T.
    filename = 'SpinSys_NICI_v9_OXSA.txt'
    with open(filename, 'r') as f:
        spinSys = [json.loads(line) for line in f]
    sys_list = []
    gamma=17.235E6
    B0=7 #you can also derive it from the cf of the acquisition but it will not affect fitting very much. Also because LCModel should correct for it.
    cf = B0*gamma*1e-6
    for i,_ in enumerate(spinSys):
        sys_list.append(simulator.simulator(spinSys=spinSys[i],B0=B0,gamma=gamma))
    
    #Importing sequence informations taken from plotter
    pulse_seq = {}
    arg = ['gradX', 'gradY', 'gradZ', 'rho', 'SSFP', 'theta']
    for i,v in enumerate(arg):
        pulse_seq["{}".format(v)] = pd.read_csv(r'./pulse_sequence/{}'.format(v),delimiter=' ', names = ['Time', 'Amplitude'],skiprows=2)
    
    
    #Generation of the simulated signals
    T = pulse_seq['rho'][pulse_seq['rho']['Amplitude']!=0]['Time'].values #Time of RF
    basis_fids=[]
    basis_names=[]
    flip = 13*(np.pi/180) #flip angle in rad
    TR = 0.066
    bw = 10000 #Acquisition bandwidth
    tbw = 0 #Transmission bandwidth
    indx = pulse_seq['rho']['Amplitude'][pulse_seq['rho']['Amplitude']!=0].index #indexing of RF values different from zero
    ampflip = flip/(2*np.pi*(T[-1]-T[0])*1e-3*gamma) #Calculate B1 from nominal flip angle (10^-3 for ms to s)
    points = 540 #number of acqusition points
    sthick=0.1 #slice thickness (not used but could be implemented)
    #samp = np.sinc((tbw)*np.linspace(-0.00026,0.00026,30))#Sinc function for frequency transmission 
    #samp = samp/(np.sum(samp))
    samp=np.array([1]) #Rect pulse (the simulator routne will automatically set the sampling and duration)
    Gz = bw/(gamma*sthick) #Gradient for slice selection (not used)
    G = [0,0,0] #Gradient areas
    LB_b = np.array([10, 10, 20, 20, 40, 40, 30, 20, 50, 20, 20, 5])# LB factors for metabolites (must be given in order with the input file)
    GB_b = [0]*len(spinSys) # GB factors for metabolites (must be given in order with the input file). Set to zero because it will not affect significantly the fit with LCM
   
    basis_fids, basis_names = simsyst(spinSys,B0, LB_b,points,bw,TR,ampflip,samp,G,T, GB=GB_b, shift = None, shift_cf=None,autophase=False,plot=True) #List of simulated FID and correspective names
    #simulated T1 relaxation factor (not used)
    sat = np.array([1.59547623, 2.12161307, 2.12828879, 2.29323398, 1.,1.03321265, 1.01011809,1.18823545, 1.29602092, 1.50159809, 1.93042119, 2.15114022])
    relax={}
    for i,name in enumerate(basis_names):
        relax[name]=1./sat[i]
    #%%STEP 2: Save basis FIDS as rawfile
    cf = 120663950 #CENTRAL FREQUENCY OF ACQUISITION
    name = 'Liver_sim'
    header = {'bandwidth':bw, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./bw,'width':np.min(LB_b), 'points':points}
    info = {'ID': name, 'FMTDAT':'(2E16.6)', 'Volume': 1.0, 'TRAMP': 1.0}
    pyth2LCM(basis_fids, basis_names,name,header,info)
    ##!!!##
    #This base must be moved to the environment of LCModel together with all the RAW file for the basis
    #Then, you have to run the command .lcmodel/bin/makebasis < Liver_sim.in to create te .BASIS file
    
    #%%STEP 3: Import the FIDs from the output of recon_mrsi.mat and save it as a RAW file
    file_name = r'C:\Users\Andrea\Downloads\Luca_14_03_22.mat'
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
    try:
        exam = loadmat(file_name)
        specs = exam['spec']
    except:
        exam = read_mat(file_name)
        tmp = exam['spec'].view(np.double).reshape(16,16,16,540,2)
        specs = tmp[:,:,:,:,0] + 1j*tmp[:,:,:,:,1]
        del tmp
    FIDS = np.fft.ifft(np.fft.ifftshift(specs,axes=3),axis=3)
    # Gaussian apodization, if you want to include it.
    t = exam['t'][0,:]
    t0=0
    gb=0 #gaussian broadening parameter
    apo = np.exp(-(((np.pi*gb)*(t-t0))**2)/(4*np.log(2)))
    apoFIDS = FIDS*apo
    
    #%%STEP 4: Save data as RAW file and create the control files (one for each spectra which needs to be analyzed)
    #This implementation will include just one FID but can be extended to a volume by including a for cycle in each spatial direction
    FID = FIDS[8,10,8,:] #example of FID selection
    info = {'ID': 'Test_subject', 'FMTDAT':'(2E16.6)', 'Volume': 1.0, 'TRAMP': 1.0}
    header = {'bandwidth':bw, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000,
              'width':np.mean(LB_b), 'points':len(FID)}#width do not affect the fit significantly, it is the starting value for processing singlets but not the actual fit. I used a reasonable value (like 10 Hz) but with the mean is more general
    lcm_io.saveRAW('FID0.RAW',FID*np.exp(-2j*np.pi*(0*header['centralFrequency'])*(np.linspace(0,header['points'],header['points'])/header['bandwidth'])), info = info, hdr=header, conj=True)#create raw file for 31P-LCModel
    with open('control_voigt_one_pme.txt', 'r') as file: #open a tamplate file with all the modification for phosphorous fit at 7T
    # read a list of lines into data
        data = file.readlines()
    for i in range(1):
        data[5] = " filbas= '/home/andrea/{}.BASIS'\n".format(name)#path to BASIS file
        data[6] = " filraw= '/home/andrea/CH5/FID{}.RAW'\n".format(i)#path to FID to be analyzed
        data[7] = " filps= '/home/andrea/CH5/FID{}.PS'\n".format(i) #path of the one-page output from LCM
        data[8] = " filcsv= '/home/andrea/CH5/FID{}.csv'\n".format(i) #path of the csv file containing concentrations output from LCM
        data[9] = " filpri= '/home/andrea/CH5/FID{}_det'\n".format(i) #path of the cv file containing fit informations from LCM
        data[17] = " nunfil= {}".format(len(FID)) #number of sampled points
        data[11] = "hzpppm= {}".format(cf*1E-6) #Hz per ppm
        
        with open(r'C:\Users\Andrea\Documents\GitHub\Spettroscopia\ControlsSNR\control{}.CONTROL'.format(i), 'w') as file: #save control file for each FID
            file.writelines(data)
            file.close()
    
    #To fit with LCModel you can use the GUI or the command line "$HOME/.lcmodel/bin/lcmodel < $path_to_control_file/control_file_name.CONTROL" since the control file contains all the other necessary it is sufficient to obtain the fit of the signal/s
    #%%STEP 5 (optonal): plot with Python by using LCM outputs
    #Since the plot of the signal is not phased, you can fix it manually in this step. Since it is a mess to do it by phase moltiplications or by authomatic phasing (which do not perform well when a large ppm range is considered)
    #I adjusted a pre-existing tool for graphic phasing so that it can be used for 31P signals
    #Follow the instruction and click on them when you are done phasing otherwise it will freeze the program (still don't know how to fix this bug).
    #Phasing is done when all the peaks in the spectrum are "up".
    
    import json
    from test_functions import Look4Gauss
    file1 = open('.\LCMfits\file_name.txt', 'r')#Name of the file containing the information from LCModel 
    Lines = file1.readlines()
    #translate the information into different variavbles
    plindx=int(json.loads(Lines[0]))
    shift=-np.array(json.loads(Lines[1]))*(1/1000)
    data_shift = np.array(json.loads(Lines[2]))
    shift += data_shift
    dlb = np.array(json.loads(Lines[3]))
    [p0,p1]=np.array(json.loads(Lines[4]))
    lineshape=np.array(json.loads(Lines[5]))
    c=np.array(json.loads(Lines[6]))
    
    def ps(data, p0=0.0, p1=0.0, pivot = 0.0, inv=False):
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
    def LCMCH5plot(D0, pra='real'):
        '''
        Generate a fitted plot of data.

        Parameters
        ----------
        D0 : ndarray
            FID signal.
        pra : string
            Type of plot, set to "real" for plotting the real part otherwise it
            plot the absolute value. Default is "real".

        Returns
        -------
        None.

        '''
        def ppm2points(ppm,cf,bw,ln):
            Hz = cf*ppm*1E-6
            points = Hz/(bw/ln)
            return points
        
        ln=len(D0)*2 #zero filled plot
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
        if pra=='real':
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
        plt.legend(['data','LCM fit'],fontsize=16)
        #res = ps(data_spec,pf0,pf1,pivf)-ps(fit,pf0,pf1,pivf)
         
        return
    
    LCMCH5plot(FID,'real')