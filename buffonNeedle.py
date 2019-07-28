import random
import numpy as np
import matplotlib.pyplot as plt

class Needle:
    def __init__(self, length=1):
        ''' Constructor '''
        self._length = length
        self._position = 0
        self._angle = 0 # in radian
    
    @property
    def length(self):
        '''  Return the length of the needle '''
        return self._length
    
    @property
    def angle(self):
        ''' Return the angle in radian of the needle'''
        return self._angle
    
    @angle.setter
    def angle(self, value):
        ''' Set the needle angle in radian '''
        self._angle = value
    
    @property
    def position(self):
        ''' Return the position where the needle lands, closest to the the tile crack/line'''
        return self._position
    
    @position.setter
    def position(self, value):
        ''' Set the position of the needle relatively to the closest tile crack/line '''
        self._position = value
    
    def __str__(self):
        ''' '''
        return "angle={0}, distance={1}".format(self._angle, self._distance)

class Floor:
    def __init__(self, tileWidth=1):
        ''' Constructor '''
        if (tileWidth <= 0):
            raise ValueError("Tile width must be greater than 0")

        self._tileWidth = tileWidth
        self._totalNeedleCount = 0
        self._crossedNeedleCount = 0
        
        self._possibleAngles = np.linspace(0, np.pi / 2, 1000)
        self._possiblePositions = np.linspace(0, tileWidth / 2, 1000)
        
    @property
    def totalNeedleCount(self):
        ''' Return the total number of needles on the floor '''
        return self._totalNeedleCount
        
    @property
    def crossedNeedleCount(self):
        ''' Return the number of needles that cross any tile crack/line '''
        return self._crossedNeedleCount
    
    @property
    def tileWidth(self):
        ''' Return the width of each tile on the floor '''
        return self._tileWidth
    
    @tileWidth.setter
    def tileWidth(self, value):
        ''' Set the the width of each tile '''
        if (value <= 0):
            raise ValueError("Tile width must be greater than 0")

        self._tileWidth = value
        self._possiblePositions = np.linspace(0, value / 2, 1000)
    
    @property
    def isEmpty(self):
        ''' Check if the floor is empty '''
        return self._totalNeedleCount == 0 and self._crossedNeedleCount == 0

    def clear(self):
        ''' Remove all needles on floor '''
        self._totalNeedleCount = 0
        self._crossedNeedleCount = 0
    
    def dropNeedle(self, needle):
        ''' Randomly drop the given needle on the floor '''

        # Randomly "drop" the needle
        needle.angle = random.choice(self._possibleAngles)
        needle.position = random.choice(self._possiblePositions)
         
        # l/2 * sin(theta) >= t
        if ((needle.length / 2) * np.sin(needle.angle) >= needle.position):
            self._crossedNeedleCount += 1
        
        self._totalNeedleCount += 1
        
    def __str__(self):
        ''' '''
        return "total={0}, crossed={1}".format(self._totalNeedleCount, self._crossedNeedleCount)

class BuffonNeedleSimulation:
    ''' Perform the Buffon Needle problem simulation '''

    class SimulationDatum:
        ''' Helper class to store simulation result '''
        def __init__(self, tileWidth, needleLength, estimatedProb, estimatedPi):
            ''' '''
            self.tileWidth = tileWidth
            self.needleLength = needleLength
            self.estimatedProb = estimatedProb
            self.estimatedPi = estimatedPi
    
    def __init__(self):
        ''' Constructor '''        
        # Key is each total number of needles and value is a SimulationDatum object of each simulation
        self._simulationDataDict = {}
        self._floor = Floor()
        
    @property
    def floor(self):
        ''' Get the floor instance '''
        return self._floor

    @property
    def simulationCount(self):
        ''' Get the number of simulations after method simulate() was invoked '''
        return len(self._simulationDataDict)

    def clear(self):
        ''' Clear the floor and all simulation data '''
        self._floor.clear()
        self._simulationDataDict.clear()
        
    @staticmethod
    def estimatePi(prob, tileWidth, needleLength):
        ''' Estimate Pi given the probability, floor tile width and needle length 

            @prob: probability value
            @tileWidth: width between two lines/cracks
            @needleLength: length of the needle
            @return: the estimated pi value
        '''
        if (prob == 0):
            return 0
        elif (tileWidth < needleLength):
            a = 2 * (needleLength - np.sqrt(needleLength**2 - tileWidth**2))
            b = 2 * (tileWidth) * (np.arcsin(tileWidth / needleLength))
            c = tileWidth * (prob -  1)
            return (a - b) / float(c)
        return (2 * needleLength) / (prob * tileWidth)
    
    @staticmethod
    def computeProbability(tileWidth, needleLength):
        ''' Compute the Buffon Needle probability using formulas '''
        if (tileWidth < needleLength):
            a = (needleLength / tileWidth) * (1 - np.sqrt(1 - (tileWidth**2 / needleLength**2)))
            b = np.arcsin(tileWidth / needleLength)
            return 1 + ( (2 / np.pi) * (a - b) )

        return (2 * needleLength) / (tileWidth * np.pi)
        
    def plot(self, isNumericalSim=True, indexes=None, axis=None):
        ''' Plot the simulation data using the probabilties from simulations
        
            @isNumericalSim: plot using numerical simulation or analytic formula
            @indexes: list of indexes use for specifying which simulation runs to plot.
                      None means plot all.
            @axis: the axis instance of matplotlib library. Use to place the plot into a
                 subplot outside of this method if desired. None means simply show this plot.

            @return: a newly created figure object if no axis is given. Else the figure object 
                     that contains the given axis. Use fig.show() to prevent Jupyter Notebook 
                     from displaying the plot twice.
        '''
        # Provide the correct plot data if we plot numerical simulation or analytic formula
        probFunc = None
        plotTitle = "Buffon Needle"
        yLabel = ""
        if (isNumericalSim):
            probFunc = lambda x: x.estimatedProb
            plotTitle += " - Numerical Simulation"
            yLabel = "Estimated Probability"
        else:
            probFunc = lambda x: BuffonNeedleSimulation.computeProbability(x.tileWidth, x.needleLength)
            plotTitle += " - Analytic Formula"
            yLabel = "Probability"

        # Initialize the axis object if there is no given axis
        fig = None
        if (axis is None):
            fig = plt.figure(plotTitle, figsize=(10, 8))
            plt.plot() # plot nothing so an axis object can be added to the figure object
            axis = fig.axes[0]
        else:
            fig = axis.get_figure()

        axis.set_title(plotTitle)
        axis.set_xlabel("l/t ratio")
        axis.set_ylabel(yLabel)

        # Sort the keys because each key is the total number of needles that was dropped
        # in a simulation and we want to display each number in the plot legend in ascending order
        sortedNeedleCounts = sorted(self._simulationDataDict.keys())
        if (indexes is not None):
            sortedNeedleCounts = sorted([sortedNeedleCounts[i] for i in indexes])
        
        for needleCount in sortedNeedleCounts:
            # Assume each simulation data array is already sorted by l/t ratio
            simulationData = self._simulationDataDict[needleCount]
            ratios = list(map(lambda x: x.needleLength / x.tileWidth, simulationData))
            probs = list(map(probFunc, simulationData))
            axis.plot(ratios, probs, "o-", label=str(needleCount) + " needles")

        axis.legend(loc="best")
        return fig
        
    def simulate(self, tileWidths=[2], needleLengths=[1], needleCountList=[1000], verbose=True):
        ''' Simulate the Buffon Needle problem 
        
            @tileWidths: a list of tile widths
            @needleLengths: a list of needle lengths
            @needleCountList: a list of number of needes to simulate
            @verbose: set to True to turn on print output
            
            @tileWidths and @needleLengths must have the same size. Each tile width & needle length
            correspond to its respective position in the lists.

            Note this method clears all data before it does the simulation.
        '''
        
        if (len(tileWidths) != len(needleLengths)):
            raise ValueError("Number of tile widths must match with number of needle lengths")
        
        # Make sure our simulation state is cleaned
        self.clear()
        
        for needleCount in needleCountList:
            self._simulationDataDict[needleCount] = []
            
            for tileWidth, needleLength in zip(tileWidths, needleLengths):
                # Clean the floor before using each tile width & needle length pair
                self._floor.clear()
                self._floor.tileWidth = tileWidth
                
                # Drop x needles
                for i in range(needleCount):
                    n = Needle(needleLength)
                    self._floor.dropNeedle(n)
                
                # Estimate probability and pi
                estimatedProb = self._floor.crossedNeedleCount / self._floor.totalNeedleCount
                estimatedPi = BuffonNeedleSimulation.estimatePi(estimatedProb, tileWidth, needleLength)
                
                # Store each simulation result
                simulationDatum = self.SimulationDatum(tileWidth, needleLength, estimatedProb, estimatedPi)
                self._simulationDataDict[needleCount].append(simulationDatum)
                
                if (verbose):
                    msgFormat = "Floor's tile width        = {0}    \nNeedle length             = {1}\n" + \
                                "Number of crossed needles = {2}    \nTotal number of needles   = {3}\n" + \
                                "Estimated probability     = {4:.5f}\nActual probability        = {5:.5f}\n" + \
                                "Estimated Pi              = {6:.5f}\n"
                    actualProb = BuffonNeedleSimulation.computeProbability(self._floor.tileWidth, needleLength)

                    print(msgFormat.format(self._floor.tileWidth, needleLength, \
                                           self._floor.crossedNeedleCount, self._floor.totalNeedleCount, \
                                           estimatedProb, actualProb, estimatedPi))
        
            # Sort each simulation data array by the l/t ratio. This is for plotting the simulation data, see @plot()
            self._simulationDataDict[needleCount].sort(key=lambda x: x.needleLength / x.tileWidth)      