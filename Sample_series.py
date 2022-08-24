# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:36:19 2022

@author: Andrea
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 10:29:15 2022

@author: Andrea
"""
import h5py
import test_functions
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import bartlett
from Simulate_metabolites_for_NICI_v2 import simsyst,man_phs,pyth2LCM
import pandas as pd
import json
from fsl_mrs.core import MRS
from fsl_mrs.utils import  plotting, fitting
from fsl_mrs.utils.mrs_io  import lcm_io
import nmrglue as ng
from scipy.optimize import minimize

colors = ['darkblue','darkgreen','darkolivegreen','darkgoldenrod','darkorange','peru']

#%% Creating Golden standard
fn = [r'D:\Series24-20220402T201020Z-001\Series24\20220328_132008_P26112.mat',r'D:\TR15_FA90.mat',r'D:\data_for_thesis_v2\data_for_thesis_v2\TR15_FA13.mat']
nex = [r'D:\Series23-20220402T082815Z-001\Series23\20220328_131102_P25600.mat',r'D:\Datas_for_thesis\SNR_FA13.mat',r'D:\data_for_thesis_v2\data_for_thesis_v2\off0_FA13.mat']
f=2
filename = fn[f]

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
    exam = loadmat(filename)
    #FIDS = exam['FID']
    fid = exam['fid']
except:
    exam = read_mat(filename)
    #tmp = exam['FID'].view(np.double).reshape(16,16,16,540,2)
    #FIDS = tmp[:,:,:,:,0] + 1j*tmp[:,:,:,:,1]
    tmp = exam['spec'].view(np.double).reshape(16,16,16,1016,2)
    specs = tmp[:,:,:,:,0] + 1j*tmp[:,:,:,:,1]
    del tmp


def cutnphs(D,n,bw):
    D = D[n:]
    D=np.fft.ifft(np.fft.fft(D)*np.exp(-1j*(2*np.pi*n/bw)*np.arange(D.shape[0])/D.shape[0]))
    return D
n=5
D0 = fid.reshape(2048)[3:]

pulse_seq = {}
arg = ['gradX', 'gradY', 'gradZ', 'rho', 'SSFP', 'theta']
for _,v in enumerate(arg):
    pulse_seq["{}".format(v)] = pd.read_csv(r'./pulse_sequence/{}'.format(v),delimiter=' ', names = ['Time', 'Amplitude'],skiprows=2)



##%%Simulation of the used sequence
T = pulse_seq['rho'][pulse_seq['rho']['Amplitude']!=0]['Time'].values
flip = 90*(np.pi/180)
TR = 15

spinSys=[{"j": [[0]],"shifts": [-4.80],"name": "PCr","scaleFactor": 1,"T1":7, "T2":0.217},#6.6
         {"j": [[0]],"shifts": [-0.00],"name": "Pi","scaleFactor": 1,"T1":3.7, "T2":0.0464}]
LB_b = [4,2]#, 10, 20, 20, 40, 40, 30, 20, 50, 20, 20, 5]#
GB_b = [0,0]#[0.1]*len(LB_b)
cf = 120663825
gamma=17.235E6
B0=cf/gamma

bw = 10000
ampflip = flip/(2*np.pi*(T[-1]-T[0])*1e-3*gamma)#Calculate B1 from nominal flip angle (10-3 for ms to s)
points = 7000
sthick=10
samp = np.sinc(bw*np.linspace(-0.00026,0.00026,21))#Sinc function for hardpulse but, in the end, is the same that giving a constant amplitude for a given period with the exception that is slower
samp = samp/(np.sum(samp))
#samp=[1]
samp=np.array([1])
Gz = bw/(gamma*sthick)
G=[0,0,0]
apo= np.exp(-(((np.pi*20)*(np.linspace(0,539,539)/bw))**2)/(4*np.log(2)))
# basis_fids, basis_names = simsyst(spinSys,B0, LB_b,points,bw,TR,ampflip,samp,G,T, GB=GB_b, shift = None, shift_cf=None,autophase=False,plot=False)#0.049*120.667
basis_header = [{'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000, 'fwhm':x} for x in LB_b]
def phs(sp1,phi0):
    size = sp1.shape[0]
    sp1 = sp1*np.exp(-1j*(phi0))
    return sp1

def create_n_fit(D0,TR,flip,shf,pl=True):
    
    def ang(par):
        sp = np.fft.fft(D0)
        sp = phs(sp,par[0])
        return np.sum((sp-np.fft.fftshift(np.fft.fft(sum(basis_fids540))))**2)
    # sp = np.fft.fftshift(np.fft.fft(D0))
    # res = minimize(ang, x0=np.array([75]),method='CG', options={'maxiter':1500})
    # D0 = cutnphs(D0,n,bw)
    # print(res.x)
    # D0 = np.fft.ifft(np.fft.ifftshift(phs(np.fft.fftshift(np.fft.fft(D0)),*res.x)))
    
    # D0 = np.fft.ifft(np.fft.ifftshift(ng.proc_autophase.autops(sp,'acme',disp=False))*np.exp(-1j*np.pi/2))
    # phases = ng.proc_autophase.autops(sp,'acme',return_phases=True,disp=False)[1]
    # print(phases[0]-90,phases[1])
    # D0 = np.fft.ifft(np.fft.ifftshift(sp*np.exp(-1j*np.pi/2)))
    #plt.plot(np.fft.fftshift(np.fft.fft(D0)))
    a = MRS(FID=D0, header = {'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000},
            basis = np.array([x for x in basis_fids]),
            names = basis_names,
            basis_hdr=basis_header,
            nucleus='31P',bw=10000, cf = cf)
            #H2O = np.fft.ifft(basis_sp[basis_names == 'PCr']),#H20 reference FID needed. Can it be turned in some 31P usefull reference?
        #bw=10000, nucleus = nucleus, cf = 120.645)
    info = {'ID': '{}_base'.format(cont), 'FMTDAT':'(2E16.6)', 'Volume': 1.0, 'TRAMP': 1.0}
    header = {'bandwidth':10000.0, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000,'width':2, 'points':len(a.FID)}
    lcm_io.saveRAW(r'.\Testch4\control{}.RAW'.format(cont),a.FID, info = info, hdr=header, conj=True)
    
    #%Process and fitting
    # freq = (changed_shift[nucleus]*10E-6*nucleus_freq) + nucleus_freq
    # sft = (a.frequencyAxis-freq)*10E6/freq
    ppmlim = (-10,5.)
    a.rescaleForFitting(scale=100) #Process data in a suitable design for fitting
    results = fitting.fit_FSLModel(a,method='Newton',ppmlim=ppmlim,model='voigt',
                                   baseline_order=-1, MHSamples=500,metab_groups=[0,1])
    GS = results.params[1]/results.params[0]#GS is the ratio between Pi and PCr. Should be 8
    if pl==True:
        fig=plotting.plotly_fit(a,results,ppmlim=(ppmlim[0]+0.5,ppmlim[1]-0.5),phs=results.getPhaseParams(), proj='abs')
        fig.show()
    dpcr = (results.params[0]*results.perc_SD[0]/100)
    dpi = (results.params[1]*results.perc_SD[1]/100)
    pi = results.params[1]
    pcr = results.params[0]
    dGS = np.sqrt((dpi/pcr)**2 + ((pi/pcr**2)**2)*dpcr**2)
    print('GS={}+-{}'.format(GS,dGS))
    return GS,dGS, pi,dpi,pcr,dpcr
# GS,dGS, pi,dpi,pcr,dpcr = create_n_fit(D0,TR,flip,None)
##%%

colors = ['darkblue','darkgreen','darkolivegreen','darkgoldenrod','darkorange','peru']

mat = loadmat(nex[f])
D = mat['fid']
D = D[:,:]
if len(D.shape)==2:
    D=D[...,np.newaxis]
def lin(x,a,b):
    return a*x+b
##%%Preliminary analysis: Gaussian test for noise + transitory event test
#um, uvar, bsk, bks, mask = test_functions.Slotboom(D[:,:,:],2,100)
# nogood = test_functions.Look4Gauss(D, noise_area = 100, pval=0.05)
##%%SNR
ss=300#number of first repetition to exclude from the mean
fid = D[ss:,n:,0]
if fid.shape[0]>1000:#To first watch how the program fits 1000 mean, it's useless to go higher if this is already enough
    l = 150 #number of fit
    step = fid.shape[0]//l #Step for mean
    fid = fid[:int(l*step),:]
#um, uvar, bsk, bks, mask = test_functions.Slotboom(fid,2,100)


s = np.shape(fid)
sig = np.zeros(s[1])
S = np.zeros(s[0])
N = np.zeros(s[0])
bart = []
R=[]
dR=[]
pi=[]
dpi=[]
pcr=[]
dpcr=[]
flip = 13*(np.pi/180)
TR = 0.066
cf = 120663825
gamma=17.235E6
B0=cf/gamma

bw = 10000
ampflip = flip/(2*np.pi*(T[-1]-T[0])*1e-3*gamma)#Calculate B1 from nominal flip angle (10-3 for ms to s)
points = fid.shape[1]
sthick=10
samp = np.sinc(10000*np.linspace(-0.00026,0.00026,21))#Sinc function for hardpulse but, in the end, is the same that giving a constant amplitude for a given period with the exception that is slower
samp = samp/(np.sum(samp))
samp=np.array([1])
Gz = bw/(gamma*sthick)
G=[0,0,0]

basis_fids, basis_names = simsyst(spinSys,B0, LB_b,points,bw,TR,ampflip,samp,G,T, GB=GB_b, shift = None, shift_cf=None,autophase=False,plot=False)#0.049*120.667
basis_fids540, basis_names = simsyst(spinSys,B0, LB_b,fid.shape[1],bw,TR,ampflip,samp,G,T, GB=GB_b, shift = None, shift_cf=None,autophase=False,plot=False)#0.049*120.667

name = 'Vitro_sim_one'
header = {'bandwidth':bw, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./bw,'width':np.min(LB_b), 'points':points}
info = {'ID': name, 'FMTDAT':'(2E16.6)', 'Volume': 1.0, 'TRAMP': 1.0}
pyth2LCM(basis_fids, basis_names,name,header,info)

cont=0
for k in range(0,s[0],step):
    if f == 2:#for some reason, these spectra are flipped
        mea = np.conj(np.mean(fid[:k+1,:],0))
    #plt.plot(np.abs(np.fft.fftshift(np.fft.fft(mea))))
    sig = np.fft.fftshift(np.fft.fft(mea))
    # S[k]= np.max(np.abs(sig))
    # bart.append(np.append(np.real(sig[:75]), np.real(sig[-75:])))
    # N[k]= np.std(bart[-1])
    if k==step*10 or k == step*20 or k==step*40 or k==step*60 or k==step*80:
        GS,dGS, Pi,dPi,PCr,dPCr = create_n_fit(mea,TR,flip,None,pl=True)
    else:
        GS,dGS, Pi,dPi,PCr,dPCr = create_n_fit(mea,TR,flip,None,pl=False)
    
    cont+=1
    R.append(GS)
    dR.append(dGS)
    pi.append(Pi)
    dpi.append(dPi)
    pcr.append(PCr)
    dpcr.append(dPCr)
SNR = S/N

line =np.linspace(1,l*step,l)
# axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
# axins.plot(np.max(np.abs(np.fft.fftshift(np.fft.fft(np.conj(fid[1:,:]),axis = 1),axes=1)),axis=1),color='darkblue')
 
# # sub region of the original image
# x1, x2, y1, y2 = 0, 500, np.max(np.max(np.abs(np.fft.fftshift(np.fft.fft(np.conj(fid[1:,:]),axis = 1),axes=1)),axis=1)), np.min(np.max(np.abs(np.fft.fftshift(np.fft.fft(np.conj(fid[1:,:]),axis = 1),axes=1)),axis=1))
# axins.set_xlim(x1-10, x2+10)
# axins.set_ylim(y1, y2)

# ax.indicate_inset_zoom(axins, edgecolor="black")
plt.figure()
plt.errorbar(line,(np.array(R)-8)*(100/8),np.array(dR)*(100/8),linestyle='-', marker='.',linewidth=0.5,markersize=4,capsize=3,color='darkblue')
plt.plot(line*2-20,np.zeros(len(line)),linestyle='--',color='darkgray')
plt.xlim([1,fid.shape[0]])
#plt.ylim([-100,50])
plt.xlabel('Repetitions')
plt.ylabel(r'$\frac{Pi}{PCr}$ concentration bias[%]')
plt.title('Concentration ratio over mean signal')
sv = np.zeros((2,len(R)))
sv[0,:]=R
sv[1,:]=dR
np.save('conf_vitro_fsl.npy',sv)
#%%plot fids
import matplotlib
plt.figure()
# cmap = blue_red1
# norm=TwoSlopeNorm(vmin=np.min(par)-0.00001, vcenter=par[st], vmax=np.max(par)+0.00001)
# s_m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
# s_m.set_array([])
plt.plot(np.linspace(0,D0.shape[1],D0.shape[1])/10,np.abs(np.mean(D0[300:,:],0)),color='black')#,color=s_m.to_rgba(i),linewidth=1)
plt.xlim(np.array([-5,D0.shape[1]+1])/10)
plt.ylim([0,500000])
plt.ticklabel_format(style='sci', axis='y', scilimits=(3,5))
ax=plt.gca()
axins = ax.inset_axes([0.48, 0.48, 0.47, 0.47])

ax.tick_params(axis='both', which='major', labelsize=16)
x1, x2, y1, y2 = -0.05, 5, 175000, 400000
axins.set_xlim(x1-0.5, x2+1)
axins.plot(np.linspace(0,D0.shape[1],D0.shape[1])/10,np.abs(np.mean(D0[:,:],0)),color='black')#,color=s_m.to_rgba(i),linewidth=1)

axins.set_ylim(y1, y2)
axins.set_yticks(np.linspace(y1,y2,4,dtype=int)) 
axins.ticklabel_format(style='sci', axis='y', scilimits=(3,5))
axins.tick_params(axis='both', labelsize=12)
rectpatch, connects=ax.indicate_inset_zoom(axins,edgecolor="black")
connects[0].set_visible(False)
connects[1].set_visible(True)
connects[2].set_visible(True)
connects[3].set_visible(False)   
plt.xlabel('t [ms]',fontsize=18)
plt.ylabel(r'Amplitude [a.u.]',fontsize=18)
plt.title('Mean of acquired FIDs',fontsize=20)
#%%plot fids
import matplotlib
plt.figure()
# cmap = blue_red1
# norm=TwoSlopeNorm(vmin=np.min(par)-0.00001, vcenter=par[st], vmax=np.max(par)+0.00001)
# s_m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
# s_m.set_array([])
thir = [r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR15_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR10_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR5_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR1_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR05_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR025_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR0066_FA13.mat']
liz = thir

R=[]
dR=[]
flip = 13*(np.pi/180)
n=5
mat = loadmat(liz[0])
D0 = mat['fid']
ss=0
D0 = D0[:,n:]
mat = loadmat(liz[1])
D1 = mat['fid']
D1 = D1[:,n:]
mat = loadmat(liz[2])
D2 = mat['fid']
D2 = D2[:,n:]
mat = loadmat(liz[3])
D3 = mat['fid']
D3 = D3[:,n:]
mat = loadmat(liz[4])
D4 = mat['fid']
D4 = D4[:,n:]
mat = loadmat(liz[5])
D5 = mat['fid']
D5 = D5[:,n:]
mat = loadmat(liz[6])
D6 = mat['fid']
D6 = D6[:,n:]
plt.figure()
# plt.plot(np.max(np.abs(D0[:,:]),1),color='black')#,color=s_m.to_rgba(i),linewidth=1)#
# plt.figure()
# plt.plot(np.max(np.abs(D1[:,:]),1),color='black')#,color=s_m.to_rgba(i),linewidth=1)#
# plt.figure()
# plt.plot(np.max(np.abs(D2[:,:]),1),color='black')#,color=s_m.to_rgba(i),linewidth=1)#
# plt.figure()
# plt.plot(np.max(np.abs(D3[:,:]),1),color='black')#,color=s_m.to_rgba(i),linewidth=1)#
# plt.figure()
# plt.plot(np.max(np.abs(D4[:,:]),1),color='black')#,color=s_m.to_rgba(i),linewidth=1)#
# plt.figure()
# plt.plot(np.max(np.abs(D5[:,:]),1),color='black')#,color=s_m.to_rgba(i),linewidth=1)#
# plt.figure()
#plt.plot(np.max(np.abs(D6[:,:]),1),color='black')#,color=s_m.to_rgba(i),linewidth=1)#
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
# cdict1 = cdict1 = {'red':   ((0.0, 0.0, 0.0),
#                (0.5, 0.0, 0.1),
#                (1.0, 1.0, 1.0)),

#      'green': ((0.0, 0.0, 0.0),
#                (1.0, 0.0, 0.0)),

#      'blue':  ((0.0, 0.0, 1.0),
#                (0.5, 0.1, 0.0),
#                (1.0, 0.0, 0.0))
#     }
# blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)
# plt.register_cmap(cmap=blue_red1)
# cmap = blue_red1
# norm=TwoSlopeNorm(vmin=-50, vcenter=0, vmax=0+0.000001)
# s_m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
# s_m.set_array([])
# fig=plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel("Chemical Shift [ppm]",fontsize=14)
# ax.set_ylabel('Repetition',fontsize=14)
# ax.set_zlabel("Magnitude [a.u.]",fontsize=14)
# ax.set_yticks(np.linspace(1,50,5,dtype=int))

# for i in range(70):
#     if i<50:
#         if i%3==0:
#             ax.plot((np.linspace(-5000,5000,539)/(cf*10**-6))[247:294], np.abs(np.fft.fftshift(np.fft.fft(D6[i,:])))[247:294], zs=i+1, zdir='y', 
#                     color=s_m.to_rgba(-i), alpha=0.8,linewidth=1)
#     else:
#         if i%5==0:
#             ax.plot([-2,0,2], [0,0,0], zs=i+1, zdir='y', 
#                     color='black', alpha=1,marker='.',linestyle='',markersize=3) 
# ax.set_xlim(-4,4)
# fig.set_size_inches(15, 15)
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.xlim([-5,D.shape[0]+5])
# #plt.ylim([0,160000])
# 
# 
plt.plot(np.max(np.abs(D6[:,:]),1),color='black')
ax=plt.gca()
axins = ax.inset_axes([0.49, 0.49, 0.47, 0.47])
ax.tick_params(axis='both', which='major', labelsize=16)
x1, x2, y1, y2 = 0, 400, 160000, 450000
axins.set_xlim(x1-0.1, x2+0.1)
axins.plot(np.max(np.abs(D6[:,:]),1),color='black')#,color=s_m.to_rgba(i),linewidth=1)
plt.xlim([-5,D6.shape[0]+5])
axins.set_ylim(y1, y2)
axins.set_yticks([160000, 250000, 350000, 450000]) 
plt.ticklabel_format(style='sci', axis='y', scilimits=(5,5))
axins.ticklabel_format(style='sci', axis='y', scilimits=(5,5))
axins.tick_params(axis='both', labelsize=14)
rectpatch, connects=ax.indicate_inset_zoom(axins,edgecolor="black")
connects[0].set_visible(False)
connects[1].set_visible(True)
connects[2].set_visible(True)
connects[3].set_visible(False)   
plt.xlabel('Acquisition',fontsize=22)
plt.ylabel(r'Max. amplitude of Pi [a.u.]',fontsize=22)
#plt.title('Repetitions',fontsize=20)
#%%Different TR
nine = [r'D:\Datas_for_thesis\TR15_FA90.mat',r'D:\Datas_for_thesis\TR10_FA90.mat',r'D:\Datas_for_thesis\TR5_FA90.mat',r'D:\Datas_for_thesis\TR1_FA90.mat',r'D:\Datas_for_thesis\TR05_FA90.mat',r'D:\Datas_for_thesis\TR025_FA90.mat'] 
thir = [r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR15_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR10_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR5_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR1_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR05_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR025_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR0066_FA13.mat']
liz = thir

R=[]
dR=[]
flip = 13*(np.pi/180)
n=5
mat = loadmat(liz[0])
D0 = mat['fid']
ss=0
D0 = np.mean(D0[ss:,:],axis=0)[n:]
mat = loadmat(liz[1])
D1 = mat['fid']
D1 = np.mean(D1[ss:,:],axis=0)[n:]
mat = loadmat(liz[2])
D2 = mat['fid']
D2 = np.mean(D2[ss:,:],axis=0)[n:]
mat = loadmat(liz[3])
D3 = mat['fid']
D3 = np.mean(D3[ss:,:],axis=0)[n:]
mat = loadmat(liz[4])
D4 = mat['fid']
D4 = np.mean(D4[ss:,:],axis=0)[n:]
mat = loadmat(liz[5])
D5 = mat['fid']
D5 = np.mean(D5[ss:,:],axis=0)[n:]
mat = loadmat(liz[6])
D6 = mat['fid']
D6 = np.mean(D6[ss:,:],axis=0)[n:]
# plt.plot(np.linspace(-5000,5000,len(D1)),np.abs(np.fft.fftshift(np.fft.fft(D0))),color = colors[0])
# plt.plot(np.linspace(-5000,5000,len(D1)),np.abs(np.fft.fftshift(np.fft.fft(D1))),color = colors[1])
# plt.plot(np.linspace(-5000,5000,len(D1)),np.abs(np.fft.fftshift(np.fft.fft(D2))),color = colors[2])
# plt.plot(np.linspace(-5000,5000,len(D1)),np.abs(np.fft.fftshift(np.fft.fft(D3))),color = colors[3])
# plt.plot(np.linspace(-5000,5000,len(D1)),np.abs(np.fft.fftshift(np.fft.fft(D4))),color = colors[4])
# plt.plot(np.linspace(-5000,5000,len(D1)),np.abs(np.fft.fftshift(np.fft.fft(D5))),color = colors[5])
# plt.plot(np.linspace(-5000,5000,len(D6)),np.abs(np.fft.fftshift(np.fft.fft(D6))),color = 'black')

# plt.legend(['TR=15s','TR=10s','TR=1s','TR=0.5s','TR=0.25s','TR=0.066s'],fontsize=12)
# plt.xlabel("$\Delta$f [Hz]",fontsize=12)
# plt.ylabel("Amplitude [a.u.]",fontsize=12)

# plt.xlim([-750,500])
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.figure()
# plt.plot(np.abs(D0),color=colors[0])
# plt.plot(np.abs(D1),color=colors[1])
# plt.plot(np.abs(D2),color=colors[2])
# plt.plot(np.abs(D3),color=colors[3])
# plt.plot(np.abs(D4),color=colors[4])
# plt.plot(np.abs(D5),color=colors[5])
# plt.xlabel('t [0.1ms]',fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlim([0,len(D3)])
# plt.ylabel("Amplitude [a.u.]",fontsize=12)
# plt.legend(['TR=15s','TR=10s','TR=1s','TR=0.5s','TR=0.25s'],fontsize=12)

dic = {'D0':D0,'D1':D1,'D2':D2,'D3':D3,'D4':D4,'D5':D5,'D6':D6}
LB_b = [4,2]#, 10, 20, 20, 40, 40, 30, 20, 50, 20, 20, 5]#
GB_b = [0,0]#[0.1]*len(LB_b)
cf = 120663794
gamma=17.235E6
B0=cf/gamma


bw = 10000
ampflip = flip/(2*np.pi*(T[-1]-T[0])*1e-3*gamma)#Calculate B1 from nominal
sthick=10
samp = np.abs(np.sinc(bw*np.linspace(-0.00026,0.00026,21)))#Sinc function for hardpulse but, in the end, is the same that giving a constant amplitude for a given period with the exception that is slower
samp = samp/(np.sum(samp))

samp=np.array([1])
Gz = bw/(gamma*sthick)
G=[0,0,0]


init = [76,0,0] 

TRs = np.array([15,10,5,1,0.5,0.25,0.066])
cont=0
for i,TR in enumerate(TRs):
    basis_fids, basis_names = simsyst(spinSys,B0, LB_b,dic['D{}'.format(i)].shape[0],bw,TR,ampflip,samp,G,T, GB=GB_b, shift = None, shift_cf=None,autophase=False,plot=False)#0.049*120.667
    #basis_fids, basis_names = simsyst(spinSys,B0, LB_b,points,bw,TR,ampflip,samp,G,T, GB=GB_b, shift = None, shift_cf=None,autophase=False,plot=False)#0.049*120.667
    basis_names = [(x+'_{}'.format(TR)).replace('.','') for x in basis_names]
    name = ('Vitro_sim_one_{}'.format(TR)).replace('.','')
    header = {'bandwidth':bw, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./bw,'width':np.min(LB_b), 'points':dic['D{}'.format(i)].shape[0]}
    info = {'ID': name, 'FMTDAT':'(2E16.6)', 'Volume': 1.0, 'TRAMP': 1.0}
    pyth2LCM(basis_fids, basis_names,name,header,info)
    #basis_header = [{'bandwidth':10000, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000, 'fwhm':x} for x in LB_b]
    # name = 'Vitro_sim_0066'
    # header = {'bandwidth':bw, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./bw,'width':np.min(LB_b), 'points':points}
    # info = {'ID': name, 'FMTDAT':'(2E16.6)', 'Volume': 1.0, 'TRAMP': 1.0}
    # pyth2LCM(basis_fids, basis_names,name,header,info)
    # plt.plot(np.fft.fftshift(np.fft.fft(5*(np.array(basis_fids540[0])+8*np.array(basis_fids540[1]))))*1e5)
    # dic['D{}'.format(i)]=np.fft.ifft(np.fft.ifftshift(man_phs(np.fft.fftshift(np.fft.fft(dic['D{}'.format(i)])))))
    GS,dGS,_,_,_,_ = create_n_fit(dic['D{}'.format(i)],TR,flip,None,pl=True)
    R.append(GS)
    dR.append(dGS)
    cont+=1
sv=np.zeros((2,7))
sv[0,:]=R
sv[1,:]=dR
np.save('conf_vitro_fsl_TR.npy',sv)
#%%
thir = [r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR15_FA13.mat']
liz = thir
cf = 120663794
R=[]
dR=[]
flip = 13*(np.pi/180)
n=5
mat = loadmat(liz[0])
D0 = mat['fid']
g=5
sl=400
for i in range(g):
    mea = np.mean(D0[i*sl:(i+1)*sl,n:],0)
    info = {'ID': '{}_base'.format(i), 'FMTDAT':'(2E16.6)', 'Volume': 1.0, 'TRAMP': 1.0}
    header = {'bandwidth':10000.0, 'nucleus': '31P', 'centralFrequency': cf, 'dwelltime': 1./10000,'width':2, 'points':D0.shape[1]}
    lcm_io.saveRAW(r'.\Testch4_TR15\control{}.RAW'.format(i),mea, info = info, hdr=header, conj=True)
with open('control_GS.txt', 'r') as file:
# read a list of lines into data
    data = file.readlines()

for i in range(g):
    # now change the 2nd line, note that you have to add a newline
    data[20] = " filraw= '/home/andrea/LCMTEST/Test_vitro/control{}.RAW'\n".format(i)
    data[21] = " filps= '/home/andrea/LCMTEST/ps_vitro/ps{}.PS'\n".format(i) 
    data[22] = " filcsv= '/home/andrea/LCMTEST/ps_vitro/csvtest{}.csv'\n".format(i) 

# and write everything back
    with open(r'C:\Users\Andrea\Documents\GitHub\Spettroscopia\Control_vitro_GS\control{}.CONTROL'.format(i), 'w') as file:
        file.writelines( data )
        file.close()
    #%%
plt.figure()
Gold=np.load('./GS_and_semidisp/Gold_stand.npy')
lcm=np.load('./CH4/conf_vitro_TR.npy')
line=np.array(TRs)

plt.errorbar(line*10**3,(lcm[0,:]-Gold[0])*100/Gold[0],np.sqrt((lcm[1,:]/Gold[0])**2+(Gold[1]*lcm[0,:]/Gold[0]**2)**2)*100,linestyle='', marker='.',linewidth=0.5,markersize=10,capsize=3,color='black',elinewidth=1.4)

plt.plot(np.linspace(-1000,100000,7),np.zeros(len(line)),linestyle='--',color='darkgray',label = "_nolegend_")
#plt.legend(['FSL_MRS','LCModel'])
plt.title('TR effect over bias',fontsize = 20)
plt.xlim([(TRs[-1]-0.5)*10**3,(TRs[0]+0.5)*10**3])
plt.ticklabel_format(style='sci', axis='x', scilimits=(3,3))
plt.xlabel('TR [ms]', fontsize = 18)
plt.ylabel('bias [%]', fontsize = 18)
plt.title(r'Effect of TR', fontsize = 20)
plt.gca().xaxis.get_offset_text().set_fontsize(12)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#%%
plt.figure()
line =np.linspace(1,l*step,l)
lcm=np.load('./CH4/conf_vitro.npy')
acq=1600+50
k = acq//step
line =np.linspace(1,acq,k)
plt.errorbar(line,(lcm[0,:k]-Gold[0])*100/Gold[0],np.sqrt((lcm[1,:k]/Gold[0])**2+(Gold[1]*lcm[0,:k]/Gold[0]**2)**2)*100,linestyle='', marker='.',linewidth=0.5,markersize=10,capsize=3,color='darkred')
line =np.linspace(acq+step,l*step,l-k)
plt.errorbar(line,(lcm[0,k:]-Gold[0])*100/Gold[0],np.sqrt((lcm[1,k:]/Gold[0])**2+(Gold[1]*lcm[0,k:]/Gold[0]**2)**2)*100,linestyle='', marker='.',linewidth=0.5,markersize=10,capsize=3,color='darkgreen')

plt.plot(np.linspace(-2000,l*step+2000,l),np.zeros(l),linestyle='--',color='darkgray',label = "_nolegend_")
#plt.legend(['FSL_MRS','LCM'])
plt.ylim([-60,5])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Number of averages',fontsize=20)
plt.ylabel(r'bias [%]',fontsize=20)
plt.title('Effect of the number of averages',fontsize=22)
plt.xlim([1,fid.shape[0]])
plt.tight_layout()
#%%
with open('control_voigt_one_ch4.txt', 'r') as file:
# read a list of lines into data
    data = file.readlines()

for i in range(l):
    # now change the 2nd line, note that you have to add a newline
    data[20] = " filraw= '/home/andrea/LCMTEST/Test_vitro/control{}.RAW'\n".format(i)
    data[21] = " filps= '/home/andrea/LCMTEST/ps_vitro/ps{}.PS'\n".format(i) 
    data[22] = " filcsv= '/home/andrea/LCMTEST/ps_vitro/csvtest{}.csv'\n".format(i) 

# and write everything back
    with open(r'C:\Users\Andrea\Documents\GitHub\Spettroscopia\Controls_vitro\control{}.CONTROL'.format(i), 'w') as file:
        file.writelines( data )
        file.close()
#%% Shimming

mat = loadmat(r'D:\data_for_thesis_v2\data_for_thesis_v2\off0_FA13.mat')

D0 = mat['fid']
ss = 300
g =4

D0 = D0[ss:,n:]
sl = D0.shape[0]//g
flip = 13*(np.pi/180)
TR = 0.066
cf = 120663825
gamma=17.235E6
B0=cf/gamma

bw = 10000
ampflip = flip/(2*np.pi*(T[-1]-T[0])*1e-3*gamma)

basis_fids, basis_names = simsyst(spinSys,B0, LB_b,points,bw,0.066,ampflip,samp,G,T, GB=GB_b, shift = None, shift_cf=None,autophase=False,plot=False)#0.049*120.667
R=[]
dR=[]
cont=0
for i in range(g):
    
    mea = np.conj(np.mean(D0[i*sl:(i+1)*sl,:],0))
    GS,dGS,_,_,_,_ = create_n_fit(mea,TR,flip,None,pl=True)
    R.append(GS)
    dR.append(dGS)
    cont+=1
b = (np.array(R)-8)*(100/8)
print('OK shimm:{}+-{}%'.format(np.mean(b),np.std(b)))

mat = loadmat(r'D:\data_for_thesis_v2\data_for_thesis_v2\shimm300_FA13.mat')

Dm = mat['fid']
Dm = Dm[ss:,n:]
R=[]
dR=[]


for i in range(g):
    mea = np.conj(np.mean(Dm[i*sl:(i+1)*sl,:],0))
    GS,dGS,_,_,_,_ = create_n_fit(mea,TR,flip,None,pl=True)
    R.append(GS)
    dR.append(dGS)
    cont+=1
b = (np.array(R)-8)*(100/8)
mask = np.isfinite(b)
print('-300 shimm:{}+-{}%'.format(np.nanmean(b[mask]),np.nanstd(b[mask])))

mat = loadmat(r'D:\data_for_thesis_v2\data_for_thesis_v2\shimp300_FA13.mat')

Dp = mat['fid']
Dp = Dp[ss:,n:]
R=[]
dR=[]
for i in range(g):
    mea = np.conj(np.mean(Dp[i*sl:(i+1)*sl,:],0))
    GS,dGS,_,_,_,_ = create_n_fit(mea,TR,flip,None,pl=True)
    R.append(GS)
    dR.append(dGS)
    cont+=1
b = (np.array(R)-8)*(100/8)
mask = np.isfinite(b)
print('+300 shimm:{}+-{}%'.format(np.nanmean(b[mask]),np.nanstd(b[mask])))


# flip = 90*(np.pi/180)
# TR = 15
# GS1,dGS1 = create_n_fit(D1,TR,flip,None,pl=True)
# mat = loadmat(r'D:\Datas_for_thesis\TR15_FA90_shimmoff.mat')
# D2 = mat['fid'][0]
# D2 = D2[5:]
# flip = 90*(np.pi/180)
# TR = 15
# GS2,dGS2 = create_n_fit(D2,TR,flip,None,pl=True)
plt.figure()
plt.plot(np.linspace(-bw/2,bw/2,D0.shape[1])/(B0*gamma*10**-6),np.abs(np.fft.fftshift(np.fft.fft(np.conj(np.mean(D0,0))))),color=colors[0])

plt.xlim([-10,5])

plt.plot(np.linspace(-bw/2,bw/2,D0.shape[1])/(B0*gamma*10**-6),np.abs(np.fft.fftshift(np.fft.fft(np.conj(np.mean(Dm,0))))),color=colors[5])
plt.plot(np.linspace(-bw/2,bw/2,D0.shape[1])/(B0*gamma*10**-6),np.abs(np.fft.fftshift(np.fft.fft(np.conj(np.mean(Dp,0))))),color=colors[2])

plt.legend([r'$G_z$: -3 Hz/cm',r'$G_z$: -300 Hz/cm',r'$G_z$: +300 Hz/cm'],fontsize=18)
plt.xlabel("Chemical Shift [ppm]",fontsize=20)
plt.ylabel("Magnitude [a.u.]",fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.invert_xaxis()
plt.tight_layout()
# plt.plot(np.abs(D2),color=colors[-1],alpha=0.8)
# plt.legend(['Shimming: ON','Shimming: OFF'],fontsize=12)
# plt.xlabel("t [0.1ms]",fontsize=12)
# plt.ylabel("Amplitude [a.u.]",fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# print('GS:{}; {}'.format(GS1,GS2))
#%%Gold=np.load('./CH4/Gold_stand.npy')
Gold=np.load('./GS_and_semidisp/Gold_stand.npy')
lcm=np.load('./CH4_nomean/conf_vitro_shimm.npy')
bias=(lcm-Gold[0])*100/Gold[0]

# for i in range(lcm.shape[1]):
#     print('{}+-{}'.format((lcm[0,i]-Gold[0])*100/Gold[0],np.sqrt((lcm[1,i]/Gold[0])**2+(Gold[1]*lcm[0,i]/Gold[0]**2)**2)*100))
    
fig, ax = plt.subplots()
m = np.mean(bias,1)
err = np.abs(np.max(bias,1)-np.min(bias,1))/2
tmp=m[0]
tmp_err=err[0]
m[0]=m[1]
err[0]=err[1]
m[1]=tmp
err[1]=tmp_err
ax.bar([1,2,3], m, yerr=err, align='center', alpha=1,width=0.8, ecolor='black', capsize=10,edgecolor='black',color='lightgray')
ax.set_ylabel('bias[%]',fontsize=18)
ax.set_xlabel(r'$G_z$ [Hz/cm]',fontsize=18)
ax.set_xticks([1,2,3])
ax.set_xticklabels(['-300','-3','300'])
ax.set_title('Effect of field inhomogeneity',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
#%% Test for Flip angle
liz = [r'D:\data_for_thesis_v2\data_for_thesis_v2\off0_FA7.mat',r'D:\data_for_thesis_v2\data_for_thesis_v2\off0_FA10.mat',r'D:\data_for_thesis_v2\data_for_thesis_v2\off0_FA13.mat',r'D:\data_for_thesis_v2\data_for_thesis_v2\off0_FA16.mat',r'D:\data_for_thesis_v2\data_for_thesis_v2\off0_FA19.mat']
FA = [7,10,13,16,19]
FAm = []
FAdm = []
cont=0
for i,f in enumerate(liz):
    mat = loadmat(f)
    
    D0 = mat['fid']
    ss = 300
    g =4
    
    D0 = D0[ss:,n:]
    sl = D0.shape[0]//g
    flip = 13*(np.pi/180)
    TR = 0.066
    cf = 120663825
    gamma=17.235E6
    B0=cf/gamma
    bw = 10000
    ampflip = flip/(2*np.pi*(T[-1]-T[0])*1e-3*gamma)
    
    basis_fids, basis_names = simsyst(spinSys,B0, LB_b,points,bw,TR,ampflip,samp,G,T, GB=GB_b, shift = None, shift_cf=None,autophase=False,plot=False)#0.049*120.667
    R=[]
    dR=[]
    for i in range(g):
        mea = np.conj(np.mean(D0[i*sl:(i+1)*sl,:],0))
        GS,dGS,_,_,_,_ = create_n_fit(mea,TR,flip,None,pl=True)
        R.append(GS)
        dR.append(dGS)
        cont+=1
    b = (np.array(R)-8)*(100/8)
    mask = np.isfinite(b)
    FAm.append(np.nanmean(b[mask]))
    FAdm.append(np.nanstd(b[mask]))
    
plt.figure()
plt.errorbar(FA,FAm,FAdm,linestyle='--',linewidth=0.5,color='darkblue', marker ='.')
plt.plot([FA[0]-1,*FA,FA[-1]+1],np.zeros(len(FA)+2),linestyle='--',color='darkgray')
plt.xlim([FA[0]-1,FA[-1]+1])
plt.xlabel('Nominal Flip Angle [deg]')
plt.ylabel(r'$\frac{Pi}{PCr}$ concentration bias [%]')
plt.title('Bias over Flip Angle changes')

plt.figure()
for i,f in enumerate(liz):
 mat = loadmat(f)
 
 D0 = mat['fid']
 ss = 300
 g =5
 
 D0 = D0[ss:,n:]
 mea = np.mean(D0,0)
 plt.plot(np.linspace(-bw/2,bw/2,mea.shape[0])/(B0*gamma*10**-6),np.abs(np.fft.fftshift(np.fft.fft(np.conj(mea)))),color = colors[i])

plt.xlabel('Chemical Shift [ppm]')
plt.ylabel('Magnitude [a.u.]')
plt.title('Mean spectra')
plt.legend(['FA:7','FA:10','FA:13','FA:16','FA:19'])
plt.xlim([-6,2])

plt.gca().invert_xaxis()
sv=np.zeros((2,len(FAm)))
sv[0,:]=FAm
sv[1,:]=FAdm
np.save('conf_vitro_fsl_flip.npy',sv)
#%%
plt.figure()
Gold=np.load('./GS_and_semidisp/Gold_stand.npy')
lcm=np.load('./CH4_nomean/conf_vitro_FA.npy')
line=np.array(FA)
bias=(lcm-Gold[0])*100/Gold[0]
plt.errorbar(line,np.mean(bias,1),np.abs(np.max(bias,1)-np.min(bias,1))/2,linestyle='', marker='.',linewidth=0.5,markersize=10,capsize=3,color='black')
#plt.plot(line*5-30,np.zeros(len(line)),linestyle='--',color='darkgray',label = "_nolegend_")

plt.xlabel('Flip Angle [deg]', fontsize=20)
plt.ylabel(r'bias [%]', fontsize=20)
plt.title('Effect of FA', fontsize=22)
plt.xticks(line,fontsize=18)
plt.yticks(fontsize=18)
plt.xlim([FA[0]-1,FA[-1]+1])
#%% Frequency offset #HERE ACME WORKS BETTER
liz = [r'D:\data_for_thesis_v2\data_for_thesis_v2\offm10_FA13.mat',r'D:\data_for_thesis_v2\data_for_thesis_v2\offm5_FA13.mat',r'D:\data_for_thesis_v2\data_for_thesis_v2\off0_FA13.mat',r'D:\data_for_thesis_v2\data_for_thesis_v2\offp5_FA13.mat',r'D:\data_for_thesis_v2\data_for_thesis_v2\offp10_FA13.mat']
off = [-10,-5,0,5,10]
offm = []
offdm = []
cont=0
for i,f in enumerate(liz):
    mat = loadmat(f)
    
    D0 = mat['fid']
    ss = 300
    g =4
    
    D0 = D0[ss:,n:]
    sl = D0.shape[0]//g
    flip = 13*(np.pi/180)
    TR = 0.066
    cf = 120663825
    gamma=17.235E6
    B0=cf/gamma
    
    bw = 10000
    ampflip = flip/(2*np.pi*(T[-1]-T[0])*1e-3*gamma)
    
    basis_fids, basis_names = simsyst(spinSys,B0, LB_b,points,bw,0.066,ampflip,samp,G,T, GB=GB_b, shift = None, shift_cf=None,autophase=False,plot=False)#0.049*120.667
    R=[]
    dR=[]
    for i in range(g):
        mea = np.conj(np.mean(D0[i*sl:(i+1)*sl,:],0))
        GS,dGS,_,_,_,_ = create_n_fit(mea,TR,flip,None,pl=True)
        R.append(GS)
        dR.append(dGS)
        cont+=1
    b = (np.array(R)-8)*(100/8)
    mask = np.isfinite(b)
    offm.append(np.nanmean(b[mask]))
    offdm.append(np.nanstd(b[mask]))
plt.figure()
plt.errorbar(off,offm,offdm,linestyle='--',linewidth=0.5,color='darkblue', marker ='.')
plt.plot([off[0]-1,*off,off[-1]+1],np.zeros(len(off)+2),linestyle='--',color='darkgray')
plt.xlim([off[0]-1,off[-1]+1])
plt.xlabel('Offset [Hz]')
plt.ylabel(r'$\frac{Pi}{PCr}$ concentration bias[%]')
plt.title('Bias over central frequency offset changes')

plt.figure()
for i,f in enumerate(liz):
 mat = loadmat(f)
 
 D0 = mat['fid']
 ss = 300
 g =5
 
 D0 = D0[ss:,n:]
 mea = np.mean(D0,0)
 plt.plot(np.linspace(-bw/2,bw/2,mea.shape[0])/(B0*gamma*10**-6),np.abs(np.fft.fftshift(np.fft.fft(np.conj(mea)))),color = colors[i])

plt.xlabel('Chemical Shift [ppm]')
plt.ylabel('Magnitude [a.u.]')
plt.title('Mean spectra')
plt.legend(['off.:-10','off.:-5','off.:0','off.:+5','off.:+10'])
plt.xlim([-6,2])

plt.gca().invert_xaxis()
sv[0,:]=offm
sv[1,:]=offdm
#np.save('conf_vitro_fsl_off.npy',sv)
#%%
plt.figure()
Gold=np.load('./GS_and_semidisp/Gold_stand.npy')
lcm=np.load('./CH4_nomean/conf_vitro_off.npy')
line=np.array(off)
bias=(lcm-Gold[0])*100/Gold[0]
plt.errorbar(line,np.mean(bias,1),np.abs(np.max(bias,1)-np.min(bias,1))/2,linestyle='', marker='.',linewidth=0.5,markersize=10,capsize=3,color='black')
#plt.plot(line*5-30,np.zeros(len(line)),linestyle='--',color='darkgray',label = "_nolegend_")

plt.xlabel('Offset [Hz]', fontsize=18)
plt.ylabel(r'bias [%]', fontsize=18)

plt.title('Effect of transmission offset', fontsize=20)
plt.xlim([off[0]-1,off[-1]+1])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#%%Plot LCM fit
plt.figure()

# Using readlines()
file1 = open('ppmax.txt', 'r')
Lines = file1.readlines()

spec=[]
fit=[]
def flatten(A,ty):
    rt=[]
    for i,line in enumerate(Lines):
        if ty == 'ppm':
            lin = line.split('    ')
        elif ty== 'spec':
            lin = line.split(' ')
        for L in lin:
            if L != '': 
                rt.append(float(L.replace('\n','')))
    return rt


ppmax = flatten(Lines,'ppm')

file1 = open('spec.txt', 'r')
Lines = file1.readlines()
spec=np.array(flatten(Lines,'spec'))

file1 = open('fit.txt', 'r')
Lines = file1.readlines()
fit=np.array(flatten(Lines,'spec'))
    # if line == ' NY points of the fit to the data follow\n':
    #     while any(line)!='following':
    #         fit.append(line.split(' '))

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
a = spec
p0, p1, pivot = ng.proc_autophase.manual_ps(a)
spec = ps(spec, p0, p1, pivot)
fit = ps(fit, p0, p1, pivot)
plt.plot(ppmax,spec,color='black',linewidth=0.5)
plt.plot(ppmax,fit,color='darkred',alpha =0.6,linewidth=2)
ax = plt.gca()
ax.invert_xaxis()
ax.set_xlim([5,-6])
plt.xlabel('Chemical Shift [ppm]',fontsize=18)
plt.ylabel('Amplitude [a.u.]',fontsize=18)
plt.title('Results at TR = 15s',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(['data','LCM fit'],fontsize=16)
#%%plot LCM
thir = [r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR15_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR10_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR5_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR1_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR05_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR025_FA13.mat',\
        r'D:\data_for_thesis_TR-20220525T083221Z-001\data_for_thesis_TR\TR0066_FA13.mat']
liz = thir

R=[]
dR=[]
flip = 13*(np.pi/180)
n=5
mat = loadmat(liz[0])
D0 = mat['fid']
D0 = np.mean(D0[:,:],axis=0)[n:]
basis_fids, basis_names = simsyst(spinSys,B0, LB_b,D0.shape[0],bw,TR,ampflip,samp,G,T, GB=GB_b, shift = None, shift_cf=None,autophase=False,plot=False)#0.049*120.667
# b1,_=lcm_io.readLCModelRaw('PCr_150.RAW', unpack_header=False, conjugate=True)
# b2,_=lcm_io.readLCModelRaw('Pi_150.RAW', unpack_header=False, conjugate=True)
# basis_fids = [b1*np.exp(2j*np.pi*(4.65*cf*1E-6)*(np.linspace(0,2043,2043)/10000)),b2*np.exp(2j*np.pi*(4.65*cf*1E-6)*(np.linspace(0,2043,2043)/10000))]

def ppm2points(ppm,cf,bw,ln):
    Hz = cf*ppm*1E-6
    points = Hz/(bw/ln)
    return points
shift=-np.array([-19.5,      19.1])*(1/1000)
data_shift = -0.041 #10 volte il valore dato?
shift += data_shift
dlb = np.array([2.40,       4.37])
p0,p1=-67,-4.6
lineshape=np.array([6.176E-02,1.944E-02,2.396E-02,2.995E-02,-1.279E-02,9.231E-02,5.145E-01,1.314E-01,8.208E-02,1.629E-02,3.984E-02])
c=np.array([5.00E+04, 9.68E+05])
cf,bw,ln=120663794,10000,len(D0)
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
    new_spec=ps(sum(new_spec),0,0*len(new_spec)/ppm2points(1,cf,bw,ln),len(new_spec)/2)#-ppm2points(4.65,cf,bw,ln))
    return new_spec
#plt.plot(fit)
fit =np.real(LMplot(basis_fids,c,lineshape,shift,dlb,p0,p1))
if any(fit<0):
    pf0,pf1,pivf = ng.proc_autophase.manual_ps(LMplot(basis_fids,c,lineshape,shift,dlb,p0,p1))
    data_spec = ps(np.fft.fftshift(np.fft.fft(np.append(D0,np.zeros(len(D0))))),-p0,-p1*len(fit)/ppm2points(1,cf,bw,ln),len(fit)/2)
    plt.plot(np.linspace(-bw/2,bw/2,D0.shape[0]*2)/(cf*10**-6),np.real(ps(data_spec,pf0,pf1,pivf)),color='black',linewidth=1) 
    plt.plot(np.linspace(-bw/2,bw/2,D0.shape[0]*2)/(cf*10**-6),np.real(ps(LMplot(basis_fids,c,lineshape,shift,dlb,p0,p1),pf0,pf1,pivf)),color='darkred',alpha =0.8,linewidth=2)
else:
    plt.plot(np.linspace(-bw/2,bw/2,D0.shape[0]*2)/(cf*10**-6),np.abs(ps(np.fft.fftshift(np.fft.fft(np.append(D0,np.zeros(2043)))))),color='black',linewidth=1)
    plt.plot(np.linspace(-bw/2,bw/2,D0.shape[0]*2)/(cf*10**-6),np.abs(LMplot(basis_fids,c,lineshape,shift,dlb,p0,p1)),color='darkred',alpha =0.8,linewidth=2)

plt.xlim([5,-10])
plt.xlabel('Chemical Shift [ppm]',fontsize=18)
plt.ylabel('Amplitude [a.u.]',fontsize=18)
plt.title('Results at TR = 15s',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(['data','LCM fit'],fontsize=16)

#plt.plot(np.abs(np.fft.fftshift(np.fft.fft(sum([x*c[i] for i,x in enumerate(basis_fids)])))))
#%%
def convertSeconds(seconds):
    h = seconds//(60*60)
    m = (seconds-h*60*60)//60
    s = seconds-(h*60*60)-(m*60)
    return [h, m, s]