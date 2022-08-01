from . import simulator as sim
import numpy as np
from copy import deepcopy


# Define a class to represent a single block of the simulation (one RF period + one delay/gradient period + one filter)
class simBlock:
    """Class to store information about one sequence segement or block """

    def __init__(self, RF, delay, rephase, cfilter, RFUnits='Hz', GUnits='T'):
        """ Constructor of simBlock class

        Args:
            RF (dict): dict containing ndarrays of 'cmplx' or 'amp' and 'phase' pulse information.
                Must also contain: 'time' key with total pulse duration (seconds);
                'grad' key with 3x1 or 3xN array of gradient amplitudes for slice selection;
                'frequencyOffset' key with pulse offset (Hz);
                 and 'phaseOffset' key with global phase offset. Phases in radians.
            delay (float): delay period after rf pulse to next sequence block.
            rephase (ndarray): 3x1 vector of gradient areas on spatial axes to apply during rephase
            cfilter (int): Coherence filter to apply at end of block. Can be None.
            RFUnits (str, optional): Amplitude units 'Hz', 'T', 'mT' or 'uT'. default ='Hz'
            GUnits (str, optional): 'mT', 'T' or 'Hz' (per meter). default = 'T'

        """
        if GUnits.lower() == 'mt':
            GMultiplier = 1E-3
        elif GUnits.lower() == 't':
            GMultiplier = 1
        elif GUnits.lower() == 'hz':
            GMultiplier = 1 / 17.235E3#AC,modified42.5774
        else:
            raise ValueError(f'Unknown GradUnits {GUnits}.')

        # RF parameters
        if RF is None:  # Create mostly empty class
            self.cmplxPulse = None
            self.ampPulse = None
            self.phsPulse = None
            self.tStep = 0.0
            self.offset = 0.0
            self.phase = 0.0
            self.sliceGrad = np.array([0., 0., 0.])
            self.axis = None
        else:
            if RFUnits.lower() == 'hz':
                rfMultiplier = 1
            elif RFUnits.lower() == 't':
                rfMultiplier = 17.235E6#AC,modified42.5774
            elif RFUnits.lower() == 'mt':
                rfMultiplier = 17.235E3#AC,modified42.5774
            elif RFUnits.lower() == 'ut':
                rfMultiplier = 17.235#AC,modified42.5774
            else:
                raise ValueError(f'Unknown RFUnits {RFUnits}.')

            if 'ampScale' in RF.keys():
                rfMultiplier = rfMultiplier * RF['ampScale']

            if 'cmplx' in RF.keys():
                self.cmplxPulse = rfMultiplier * np.array(RF['cmplx'])
                npoints = len(self.cmplxPulse)
                self.ampPulse = None
                self.phsPulse = None
            elif 'amp' in RF.keys():
                self.ampPulse = rfMultiplier * np.array(RF['amp'])
                npoints = len(self.ampPulse)
                if 'phase' in RF.keys():
                    self.phsPulse = np.array(RF['phase'])
                else:
                    self.phsPulse = 0.0
                self.cmplxPulse = None

            self.tStep = RF['time'] / npoints
            self.offset = float(RF['frequencyOffset'])
            self.phase = float(RF['phaseOffset'])

            self.sliceGrad = GMultiplier * np.array(RF['grad'])
            if self.sliceGrad.ndim == 1:
                self.sum_grads = self.sliceGrad
                self.axis = np.nonzero(self.sliceGrad)[0]
            else:
                self.sum_grads = np.sum(np.abs(self.sliceGrad), axis=1)
                self.axis = np.nonzero(self.sum_grads)[0]
            if self.axis.size == 0:
                self.axis = np.asarray([None])

        # Other parameters
        self.rephase = GMultiplier * np.array(rephase)
        self.delay = delay
        self.cfilter = cfilter

    def getPulse(self):
        """ Return complex rf contained in block."""
        if (self.cmplxPulse is None) and (self.ampPulse is None):
            return None
        elif self.cmplxPulse is None:
            rf = self.ampPulse * np.exp(1j * self.phsPulse)
        else:
            rf = self.cmplxPulse

        rf = rf * np.exp(1j * self.phase)

        return rf


# Convert (rephase) gradient areas to amplitudes for a given duration
def areaToAmp(areas, durations):
    """ Convert gradient area to amplitude given duration."""
    return areas / durations


# Grumble over any inconsistencies in the passed parameters
def grumble(parameterdict):
    """ Function to handle and clean sequence parameters passed to seqsim."""
    pulses = parameterdict['RF']
    R = np.array(parameterdict['rephaseAreas'])
    delays = np.array(parameterdict['delays'])
    coherenceFilter = parameterdict['CoherenceFilter']

    # Check if number of RF, delays, gradients,rephase areas etc are consistent.
    # And otherwise have consistent shapes (3 axes)
    checkList = [R, delays, coherenceFilter]
    checkListNames = ['rephaseAreas', 'delays', 'CoherenceFilter']
    simSteps = len(pulses)
    for param, pName in zip(checkList, checkListNames):
        currentLength = len(param)
        if currentLength != simSteps:
            raise ValueError(f'{simSteps} RF pulses were specified but {pName} has {currentLength} entries ({param}).'
                             ' Aborting.')

        if any(x in pName for x in ['rephaseAreas']):
            if param.shape[1] != 3:
                raise ValueError(f'{pName} must be a NRFx3 ({simSteps}x3) matrix. It has shape {param.shape}.'
                                 ' Aborting.')

    # Check RF has required fields if not then add those that are possible
    for iDx, rf in enumerate(pulses):
        # Throw errors
        if 'time' not in rf.keys():
            raise ValueError(f'RF {iDx} does not contain a time field. Aborting.')
        if 'grad' not in rf.keys():
            raise ValueError(f'RF {iDx} does not contain a grad field. Aborting.')

        # Add defualt values
        if 'frequencyOffset' not in rf.keys():
            parameterdict['RF'][iDx].update({'frequencyOffset': 0.0})
        if 'phaseOffset' not in rf.keys():
            parameterdict['RF'][iDx].update({'phaseOffset': 0.0})

        # Check possible times
        if (np.array(rf['time']) < 0).any():
            raise ValueError(f'pulseTime in RF {iDx} must have values >= 0.0.')

    # If some of the optional parameters aren't passed then generate them here
    # Optional parameters with defaults are:
    optParams = ['spatiallyResolve', 'centralShift', 'RFUnits', 'GradUnits', 'spaceUnits']
    defaults = [False, 0.0, 'Hz', 'T', 'm']
    for optP, val in zip(optParams, defaults):
        if optP not in parameterdict.keys():
            parameterdict.update({optP: val})

    # Check time parameters are not less than zero
    if (np.array(delays) < 0).any():
        raise ValueError(f'Delays must have values >= 0.0. Delays = {delays}.')

    # Check delays don't have 0 time if there is a rephase gradient
    for r, d in zip(R, delays):
        if (r != 0).any() and (d == 0):
            raise ValueError(f'Delay has a value of 0 when there is a rephase gradient in same block (rephase = {r}).')

    return parameterdict


def grumblespins(spinsys):
    """ Function to handle and clean spin system parameters passed to seqsim."""
    # Check spinsys input and values are numpy arrays
    if 'shifts' in spinsys:
        if isinstance(spinsys['shifts'], list):
            spinsys['shifts'] = np.array(spinsys['shifts'], dtype=float)
        elif isinstance(spinsys['shifts'], (float, int)):
            spinsys['shifts'] = np.array([spinsys['shifts']], dtype=float)
    else:
        raise ValueError('spinsys must have a field ''shifts''.')

    if 'J' in spinsys:  # Normalise case
        spinsys['j'] = spinsys.pop('J')
    if 'j' in spinsys:
        if isinstance(spinsys['j'], list):
            spinsys['j'] = np.array(spinsys['j'], dtype=float)
        elif isinstance(spinsys['j'], (float, int)):
            spinsys['j'] = np.array([[spinsys['j']]], dtype=float)
    else:
        raise ValueError('spinsys must have a field ''j''.')

    return spinsys


# This function checks the order of the entries of the slice-select gradient variable and determines
# what mode the simulator runs in.
# If the gradients occur on axes in an order without reoccurance e.g 2,1,1,3 then 1D projection may be used
# If they occur in an interleaved order e.g. 1,2,1,3 then use an intermediate 'interleaved' mode.
# If gradients occur simulataneously across 2 or more axes then use the full simulation method. Slow!
def gradOrderChecker(seqBlocks):
    """Determine simulation mode from gradient order in sequence blocks."""
    fullSim = False
    disable1D = False
    axisUsed = []
    for block in seqBlocks:

        if block.axis.size > 1:
            print('More than one gradient axis on per pulse (G).\n '
                  'Setting to full spatial simulation mode.\n '
                  'This might take a while!')
            fullSim = True
            break

        currentAxis = block.axis

        if currentAxis[0] is None:
            axisUsed.append(currentAxis[0])
            continue

        if (axisUsed == currentAxis).any():  # Same axis used again, not a problem yet
            if (axisUsed[-1] != currentAxis).any():  # Not the last used axis, now it's a problem
                disable1D = True
                print('Slice select gradients not in linear order, 1D projection method disabled.')

        axisUsed.append(currentAxis[0])

    axisUsed = np.array(axisUsed)
    if fullSim:
        return 'full', axisUsed
    elif disable1D:
        return 'interleaved', axisUsed
    elif ~fullSim & ~disable1D:
        # 1D is fine
        return '1d', axisUsed


# In the 1D projection method the simulatator can't cope with rephase gradients applied on another axis.
# If this is the case (e.g. 2nd rephase in STEAM) then insert a new entry into all of the looping variables.
# This new entry has zeros for most entries, has an entry for the rephase in the different axis
# and shares the delay time with the previous axis.
def splitConcurrentRephase(seqBlock):
    """Split sequence blocks with multiple gradients directions occuring in the rephase section"""
    seqBlockOut = []
    axisOrderOut = []
    for iDx, block in enumerate(seqBlock):  # Loop over blocks
        numAxisReph = np.sum(block.rephase != 0)  # Number of elements in row of R that aren't 0
        dims = np.arange(0, 3)
        dims = dims[dims != block.axis]
        insertNeeded = (block.rephase[dims] != 0).any()
        # if numAxisReph>1: # If more than 1
        if insertNeeded:
            currentLength = len(seqBlockOut)  # Store current length
            stableBlock = deepcopy(block)
            for axis, rephG in enumerate(stableBlock.rephase):  # Loop over elements in row
                if block.sum_grads[axis] != 0:  # If on the same axis as slice select then insert at currentLength
                    # Modify values in block
                    toInsert = np.zeros(3)
                    toInsert[axis] = rephG
                    block.rephase = toInsert
                    block.delay = stableBlock.delay / numAxisReph
                    # Insert
                    seqBlockOut.insert(currentLength, block)
                    axisOrderOut.insert(currentLength, block.axis[0])

                elif rephG == 0:  # If zero and not on slice select axis just skip
                    continue
                else:  # Otherwise make new line and append
                    # import pdb; pdb.set_trace()
                    toInsert = np.zeros(3)
                    toInsert[axis] = rephG
                    tmpBlock = simBlock(None, stableBlock.delay / numAxisReph, toInsert, None)
                    tmpBlock.axis = axis
                    seqBlockOut.append(tmpBlock)
                    axisOrderOut.append(axis)

        else:  # If just one value of rephase keep as is.
            seqBlockOut.append(block)
            if block.axis[0] is None:
                axisOrderOut.append(None)
            else:
                axisOrderOut.append(int(block.axis[0]))

    return seqBlockOut, np.asarray(axisOrderOut)


# Main sequence simulation function
def simseq(spinsys, parameterdict):
    """Sequence density matrix simulations.

    Args:
        spinsys (dict): Dict describing spin system to simulate.
        parameterdict (dict): Dict describing sequence and simulation parameters.

    Returns:
        FID (ndarray): Complex FID signal from readout.
        ax (dict): Dict of time, frequency and ppm axes.
        finalP (ndarray): Final state of the density matrix before readout.
    """
    # make copies of inputs as they are small and it avoids fun with unexpected modifications
    spinsys = deepcopy(spinsys)
    parameterdict = deepcopy(parameterdict)

    # Check inputs
    parameterdict = grumble(parameterdict)
    spinsys = grumblespins(spinsys)

    # Interpret parameters in the dict:
    # General
    centralShift = parameterdict['centralShift']
    B0 = parameterdict['B0']
    # Readout
    lw = parameterdict['Rx_LW']
    dwellTime = 1 / parameterdict['Rx_SW']
    points = parameterdict['Rx_Points']
    recieverPhs = parameterdict['Rx_Phase']

    # Spatial dimension
    x = np.linspace(parameterdict['x'][0], parameterdict['x'][1], parameterdict['resolution'][0])
    y = np.linspace(parameterdict['y'][0], parameterdict['y'][1], parameterdict['resolution'][1])
    z = np.linspace(parameterdict['z'][0], parameterdict['z'][1], parameterdict['resolution'][2])

    if 'spaceUnits' in parameterdict.keys():
        if parameterdict['spaceUnits'].lower() == 'mm':
            spaceScale = 1000
        elif parameterdict['spaceUnits'].lower() == 'cm':
            spaceScale = 100
        elif parameterdict['spaceUnits'].lower() == 'm':
            spaceScale = 1
        else:
            spcunits = parameterdict['spaceUnits']
            raise ValueError(f'Unknown spatial units {spcunits}. Use mm, cm or m.')
    else:
        spaceScale = 1
    x /= spaceScale  # mm to m
    y /= spaceScale  # mm to m
    z /= spaceScale  # mm to m
    spatialDims = [x, y, z]

    # Main sequence blocks comprising the RF, delays, rephase areas and coherence filters
    seqBlocks = []
    for rf, delays, reph, cf in zip(
            parameterdict['RF'],
            parameterdict['delays'],
            parameterdict['rephaseAreas'],
            parameterdict['CoherenceFilter']):
        seqBlocks.append(simBlock(rf, delays, reph, cf, parameterdict['RFUnits'], parameterdict['GradUnits']))

    # Determine simulation method
    # Options are 1D projection, out of order or full simulation.
    if 'method' in parameterdict.keys():
        methodPoss, axisOrder = gradOrderChecker(seqBlocks)
        method = parameterdict['method'].lower()
        if (method == '1d') & ((methodPoss == 'interleaved') | (methodPoss == 'full')):
            print(f'1D method requested but not possible, running with {methodPoss}')
            method = methodPoss
        elif (method == 'interleaved') & (methodPoss == 'full'):
            print(f'interleaved method requested but not possible, running with {methodPoss}')
            method = methodPoss
    else:  # Determine automatically
        method, axisOrder = gradOrderChecker(seqBlocks)

    # Handle option for resolving spatial response
    if ('spatiallyResolve' in parameterdict.keys()) and (parameterdict['spatiallyResolve']):
        resolve = True
        if method == '1d':  # 1d projection method incompatible with spatially resolving output
            method == 'interleaved'
    else:
        resolve = False

    print(f'Simulation running using mode {method}. Axis order = {axisOrder}.')

    # Deal with rephase gradients on different axes
    # Split the delay into more blocks, inserting zeros into pulses etc
    if method == '1d':
        seqBlocks, axisOrder = splitConcurrentRephase(seqBlocks)

    # Initilise the class object
    obj = sim.simulator(spinsys, B0=B0, centralShift=centralShift)

    # Get thermal equilibrium
    pEq = obj.thermalEq()

    # Loop over the pulses
    currentP = pEq
    previousAx = None
    for ax, block in zip(axisOrder, seqBlocks):
        if ax is None:
            ax = 0
        if method == '1d':
            # Loop over spatial dimension relavent to this pulse
            if currentP.ndim == 2:  # Single density mat
                currentP = np.full(spatialDims[ax].shape + currentP.shape, currentP, dtype=np.complex128)
            # If this axis is not the same as the last then take the mean and redistribute over new axis.
            elif previousAx != ax:
                currentP = np.full(spatialDims[ax].shape + currentP.shape[1:],
                                   np.mean(currentP, axis=0), dtype=np.complex128)

            # Precalculate the RF Hamiltonian so that it isn't repeated uncescerrily for each point
            # import pdb; pdb.set_trace()
            Hrf = obj.getHRF(block.getPulse(), block.offset)
            Hconst = Hrf + obj.Hcs + obj.Hj
            for iDx, r in enumerate(spatialDims[ax]):
                # Position vector
                currR = np.zeros(3)
                currR[ax] = r

                # Pulse
                H = Hconst + obj.getHGrad(block.sliceGrad, currR)
                currentP[iDx] = obj.propagate(H, block.tStep, p=currentP[iDx])

                # Delay with rephase
                rph = areaToAmp(block.rephase, block.delay)
                currentP[iDx] = obj.applyGrad(rph, currR, block.delay, p=currentP[iDx])

                # Coherence selection
                if block.cfilter is not None:
                    currentP[iDx] = obj.selectCoherence(block.cfilter, p=currentP[iDx])
            previousAx = ax

        elif method == 'interleaved':
            # Hybrid method. We can't use 1d projection method but we know there aren't
            # any concurrent slice select gradients.
            # We can therefore precalculate the pulse propagators for a column of isochromats.
            # Then repeatedly apply this column to all values of the second and third dimensions.

            # Make an array of density matricies of the correct shape
            if currentP.ndim == 2:
                fullshape = (x.size, y.size, z.size)
                fullSize = np.prod(fullshape)
                currentP = np.repeat(currentP[np.newaxis, :, :], fullSize, axis=0).astype(np.complex128)
                currentP = currentP.reshape(fullshape + currentP.shape[1:])

            # Make propagators for rf pulse + slice select gradient over current axis
            currProp = []
            for r in spatialDims[ax]:
                currR = np.zeros(3)
                currR[ax] = r
                H = obj.getHRF(block.getPulse(), block.offset) + obj.Hcs + obj.Hj + obj.getHGrad(block.sliceGrad, currR)
                currProp.append(sim.simulator.createPropagator(H, block.tStep))

            # This next step isn't pretty! But I couldn't figure out how to vectorise the step in the numpy framework
            innerDim = ax
            outerdims = np.arange(0, 3)
            outerdims = outerdims[outerdims != innerDim]
            for iDx, r1 in enumerate(spatialDims[outerdims[0]]):
                for jDx, r2 in enumerate(spatialDims[outerdims[1]]):
                    for kDx, r3 in enumerate(spatialDims[innerDim]):
                        pIndices = np.zeros(3, dtype=int)
                        pIndices[outerdims[0]] = iDx
                        pIndices[outerdims[1]] = jDx
                        pIndices[innerDim] = kDx

                        currR = np.zeros(3)
                        currR[outerdims[0]] = r1
                        currR[outerdims[1]] = r2
                        currR[innerDim] = r3

                        tmpP = currentP[pIndices[0], pIndices[1], pIndices[2], :, :]
                        # Pulse
                        tmpP = obj.applyPropagator(currProp[kDx], p=tmpP)

                        # Rephase+delay
                        rph = areaToAmp(block.rephase, block.delay)

                        tmpP = obj.applyGrad(rph, currR, block.delay, p=tmpP)

                        # Coherence selection
                        if block.cfilter is not None:
                            tmpP = obj.selectCoherence(block.cfilter, p=tmpP)

                        currentP[pIndices[0], pIndices[1], pIndices[2], :, :] = tmpP

        elif method == 'full':
            # Loop over mesh grid of all spatial points
            # This could take a very long time!
            X, Y, Z = np.meshgrid(x, y, z)
            if currentP.ndim == 2:
                currentP = np.repeat(currentP[np.newaxis, :, :], X.size, axis=0).astype(np.complex128)

            Hrf = obj.getHRF(block.getPulse(), block.offset)
            Hconst = Hrf + obj.Hcs + obj.Hj
            for iDx, (curx, cury, curz) in enumerate(zip(X.ravel(), Y.ravel(), Z.ravel())):
                currR = np.array([curx, cury, curz])
                # Pulse
                H = Hconst + obj.getHGrad(block.sliceGrad, currR)
                currentP[iDx] = obj.propagate(H, block.tStep, p=currentP[iDx])

                # Delay with rephase
                rph = areaToAmp(block.rephase, block.delay)
                currentP[iDx] = obj.applyGrad(rph, currR, block.delay, p=currentP[iDx])

                # Coherence selection
                if block.cfilter is not None:
                    currentP[iDx] = obj.selectCoherence(block.cfilter, p=currentP[iDx])

    # Generate final result
    if method == '1d':
        finalP = np.mean(currentP, axis=0)
        FID, ax = obj.readout(points, dwellTime, lw, p=finalP)
    elif method == 'interleaved':
        if resolve:
            outShape = (points, x.size, y.size, z.size)
            currentP = currentP.reshape((-1,) + currentP.shape[-2:])
            FID = np.zeros(outShape, dtype=np.complex128)
            FID = FID.reshape((points, currentP.shape[0]))
            for iDx, p in enumerate(currentP):
                FID[:, iDx], ax = obj.readout(points, dwellTime, lw, p=p)
            FID = FID.reshape(outShape)
            finalP = currentP  # Just for returning
        else:
            finalP = np.mean(currentP.reshape((-1,) + currentP.shape[-2:]), axis=0)
            FID, ax = obj.readout(points, dwellTime, lw, p=finalP)
    elif method == 'full':
        if resolve:
            outShape = (points, x.size, y.size, z.size)
            FID = np.zeros(outShape, dtype=np.complex128)
            FID = FID.reshape((points, currentP.shape[0]))
            for iDx, p in enumerate(currentP):
                FID[:, iDx], ax = obj.readout(points, dwellTime, lw, p=p)
            FID = FID.reshape(outShape)
            finalP = currentP  # Just for returning
        else:
            finalP = np.mean(currentP, axis=0)
            FID, ax = obj.readout(points, dwellTime, lw, p=finalP)

    FID = FID * np.exp(-1j * recieverPhs)  # *= doesn't work for some reason.

    return FID, ax, finalP
