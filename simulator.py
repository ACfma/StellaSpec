from . import operators as op
import numpy as np
from copy import deepcopy


class simulator:
    """ Density matrix simulator class"""

    def __init__(self, spinSys, B0, gamma=17.235E6, centralShift=0.0): #AC, modified gamma from 42.5774E6
        """ simulator class constructor

        Args:
            spinSys (dict): Dict containing description of spin system.
            B0 (float): Main static field strength.
            gamma (float, optional): Gyromagnetic ratio in Hz/T. Defaults to 1H.
            centralShift (float, optional): Offset of central reciever frequency in ppm.
        """
        self.loadSpinSys(spinSys)
        self.spinSys['shifts'] -= centralShift

        self.B0 = B0
        self.GAMMA = gamma

        self.nSpins = self.spinSys['shifts'].shape[0]

        # initialize operators
        self.Izs = op.createIz(self.nSpins)
        self.Ixs = op.createIx(self.nSpins)
        self.Iys = op.createIy(self.nSpins)

        # initialize
        self.Hcs, self.Hj = self.getStaticHamiltonians()
        self.p = self.thermalEq()
        self.orders = simulator.getorders(self.nSpins)

    def loadSpinSys(self, insys):
        """ Convert spin system to numpy arrays if required."""
        self.spinSys = insys
        if isinstance(self.spinSys['shifts'], list):
            self.spinSys['shifts'] = np.asarray(self.spinSys['shifts'])
        if isinstance(self.spinSys['j'], list):
            self.spinSys['j'] = np.asarray(self.spinSys['j'])

    # Methods primarily for initilisation

    def getStaticHamiltonians(self):
        """Return the static hamiltonians for chemical shift and J-coupling."""
        Omega = self.GAMMA * self.B0 * 1E-6 * self.spinSys['shifts']
        Hcs = Omega[0] * self.Izs[0]
        for om, iz in zip(Omega[1:], self.Izs[1:]):
            Hcs += om * iz

        jcoupling = self.spinSys['j']
        Hj = np.zeros(self.Izs[0].shape, dtype=np.complex128)
        for iDx in range(0, jcoupling.shape[0]):
            for jDx in range(iDx + 1, jcoupling.shape[1]):
                if jcoupling[iDx, jDx] != 0:
                    # Calculate the j-coupling terms I[iDx].I[jDx] = IixIjx + IiyIjy + IizIjz
                    # This is the scalar product, so normal matrix multiplication not direct (kroneker) product
                    # print(f'{iDx},{jDx} = {jcoupling[iDx,jDx]}')
                    Hj += jcoupling[iDx, jDx] * (self.Ixs[iDx] @ self.Ixs[jDx] + self.Iys[iDx]
                                                 @ self.Iys[jDx] + self.Izs[iDx] @ self.Izs[jDx])
                    # Hj += jcoupling[iDx,jDx]*(self.Izs[iDx]@self.Izs[jDx]) # Secular approximation

        return Hcs, Hj

    def thermalEq(self):
        """Return the thermal equilibrium state of the system."""
        out = np.sum(self.Izs, 0)
        return out

    # Hamiltonians
    def getHGrad(self, G, r):
        """
        Return the gradient hamiltonian

        Args:
            G (ndarray): 3x1 or 3xN vector of gradient amplitudes in T/m
            r (ndarray): 3x1 vector of spatial position in m

        Returns:
            H (ndarray): Gradient hamiltonian
        """
        # Ensure that a single gradient value is treated the same as a vector
        if G.ndim == 1:
            G = G[:, np.newaxis]

        Hg = []
        for g in G.T:
            H = self.GAMMA * g @ r * self.Izs[0]
            for iz in self.Izs[1:]:
                H += self.GAMMA * g @ r * iz
            Hg.append(H)

        return np.asarray(Hg)

    def getHRF(self, pulse, offset):
        """
        Calculate and return the RF hamiltonian

        Args:
            pulse (ndarray): Array of complex RF values in units of Hz (gamma B1+) for each time point
            offset (float): Offest frequency in Hz.

        Returns:
            H (list of ndarray): RF hamiltonian for each pulse time point.
        """
        if pulse is None:
            return 0
        Hrf = []
        pulseAmp = np.abs(pulse)
        pulsePhs = np.angle(pulse)

        # Note the negative sign before the offset here. I'm not sure why it's needed.
        theta = np.arctan2(pulseAmp, -offset)
        weff = np.sqrt(pulseAmp**2 + offset**2)
        for w, t, phi in np.nditer((weff, theta, pulsePhs)):

            Hrf.append(w * np.sum(self.Ixs * np.array(np.sin(t) * np.cos(phi)) + self.Iys * np.array(np.sin(t)
                       * np.sin(phi)) + self.Izs * np.array(np.cos(t)), 0))

        # for amp,phs in np.nditer((pulseAmp,pulsePhs)):
        #     Hrf.append(amp*np.sum(self.Ixs*np.array(np.cos(phs)) + self.Iys*np.array(np.sin(phs)),0))
        return np.asarray(Hrf)

    # Methods to propagate
    def applyPropagator(self, prop, p=None):
        """
        Apply propagator to density matrix

        Args:
            prop (ndarray): Pre calculated propagator
            p (ndarray, optional): Density matrix, if not passed class property density matrix is used.

        Returns:
            p (ndarray): Updated density matrix.
        """
        if p is None:
            p = self.p
        p = prop @ p @ prop.conj().T
        self.p = p
        return p

    @staticmethod
    def createPropagator(H, timeStep):
        """
        Create propagator from list of hamiltonians and step duration

        Args:
            H (ndarray or list of ndarray): Hamiltonian or list of hamiltonians.
            timeStep (float): Duration of each time step in seconds.

        Returns:
            prop (ndarray): Propagator.
        """

        if H.ndim > 2:  # If list of H or just one time step to apply
            prop = op.opExp(-1j * timeStep * 2 * np.pi * H[0])
            for h in H[1:]:
                prop = op.opExp(-1j * timeStep * 2 * np.pi * h) @ prop
        else:
            prop = op.opExp(-1j * timeStep * 2 * np.pi * H)
        return prop

    # Main propagator, all others just call this with appropriate inputs
    # Though it in turn calls the one above so that propagators can be assembled outside this mechanism
    def propagate(self, H, timeStep, p=None):
        """ Propagate density matrix (p) using Hamiltonian H for time timeStep"""
        if p is None:
            p = self.p

        prop = simulator.createPropagator(H, timeStep)

        p = self.applyPropagator(prop, p=p)
        return p

    def freeEvolution(self, time, p=None):
        """ Free evolution of density matrix (p) for time."""
        if p is None:
            p = self.p
        H = self.Hcs + self.Hj
        p = self.propagate(H, time, p=p)
        return p

    def applyGrad(self, G, r, time, p=None):
        """
        Construct gradient hamiltonian and propagate density matrix.

        Args:
            G (ndarray): 3x1 vector of gradient amplitudes in T/m
            r (ndarray): 3x1 vector of spatial position in m
            time (float): Duration of each time step in seconds.
            p (ndarray, optional): Density matrix, if not passed class property density matrix is used.

        Returns:
            p (ndarray): density matrix.
        """
        if p is None:
            p = self.p
        Hg = self.getHGrad(G, r)
        H = self.Hcs + self.Hj + Hg
        p = self.propagate(H, time, p=p)
        return p

    def applyRF(self, pulse, time, offset=0, G=None, r=None, p=None):
        """
        Construct RF hamiltonian and propagate density matrix.

        Args:
            pulse (ndarray): Array of complex RF values in units of Hz (gamma B1+) for each time point
            time (float): Duration of each time step in seconds.
            offset (float, optional): Offest frequency in Hz. Default = 0
            G (ndarray, optional): 3x1 or 3xN vector of gradient amplitudes in T/m
            r (ndarray, optional): 3x1 vector of spatial position in m
            p (ndarray, optional): Density matrix, if not passed class property density matrix is used.

        Returns:
            p (ndarray): density matrix.
        """
        if p is None:
            p = self.p
        H = self.Hcs + self.Hj + self.getHRF(pulse, offset)
        if G is not None\
                and r is not None:
            if G.shape != (3,)\
                    and G.shape != (3, pulse.size):
                raise ValueError(f'Gradient must be 3x1 or 3xpulse-points (3, {pulse.size}), currently {G.shape}')

            Hg = self.getHGrad(G, r)
            H += Hg

        p = self.propagate(H, time, p=p)
        return p

    # Detection operators
    def detect(self, p=None):
        "Detect -1 coherences in density matrix (p)"
        if p is None:
            p = self.p
        S = np.trace((self.Ixs[0] + 1j * self.Iys[0]) @ p)
        for ix, iy in zip(self.Ixs[1:], self.Iys[1:]):
            S += np.trace((ix + 1j * iy) @ p)
        N = self.nSpins
        S *= 1 / (2**(N - 2))
        return S

    def readout(self, steps, dwellTime, lw, p=None):
        """
        Readout FID from density matrix.

        Args:
            steps (int): Number of readout steps
            dwellTime (float): Readout step duration in seconds (1/bandwidth)
            lw (float): Linewidth of exponential dampin applied to FID signal (in Hz).
            p (ndarray, optional): Density matrix, if not passed class property density matrix is used.

        Returns:
            St (ndarray): Complex FID signal
            ax (dict): dict containing time, frequency and chemical shift axes.
        """
        if p is None:
            p = self.p
        Hfe = self.Hcs + self.Hj
        St = []
        pcurr = deepcopy(p)
        prop = op.opExp(-1j * dwellTime * 2 * np.pi * Hfe)
        for idx in range(0, steps):
            St.append(self.detect(p=pcurr))
            pcurr = prop @ pcurr @ prop.conj().T

        # Construct associated time and frequency axis
        ax = {}
        ax.update({'time': dwellTime * np.arange(0, steps)})
        bw = 1 / dwellTime
        ax.update({'freq': np.linspace(-bw / 2, bw / 2, steps)})
        ax.update({'ppm': np.linspace(-bw / 2, bw / 2, steps) / (self.B0 * self.GAMMA * 1E-6)})
        ax.update({'centreFreq': (self.B0 * self.GAMMA * 1E-6)})

        # Apply damping
        if lw!=0:#AC, added if for enabling deltas
            tconst = 1 / (np.pi * lw)
            St *= np.exp(-ax['time'] / tconst)

        return np.array(St), ax

    # Filtering functions
    def selectCoherence(self, order, p=None):
        "Filter desnity matrix (p) to contain only coherences of specified order"
        if p is None:
            p = self.p
        if order is None:
            return p
        # Construct filter matrix
        pOrders = self.orders
        F = np.zeros(pOrders.shape)
        F[pOrders == order] = 1

        # Apply filter
        p = F * p  # Element wise multiplication
        self.p = p
        return p

    @staticmethod
    def getorders(nspins):
        "Calculate coherence orders for sytem of nspins"
        matList = []
        for idx in range(0, nspins):
            matList.append(np.array([[0, 1], [-1, 0]]))
        mat1 = matList[0]
        out = mat1
        for sDx in range(1, len(matList)):
            mat2 = matList[sDx]
            out = np.tile(mat2, mat1.shape)
            toBlock = []
            for iDx in range(0, mat1.shape[1]):
                currLine = []
                for jDx in range(0, mat1.shape[0]):
                    currLine.append(np.full(mat2.shape, mat1[iDx, jDx]))
                toBlock.append(currLine)

            blockMat = np.block(toBlock)
            out += blockMat
            mat1 = out
        return out
