# -*- coding: utf-8 -*-

##########################################################################################
##                                                                                      ##
##  This script analyzes the energy consumption of appliances recorded with an          ##
##  PicoScope 2204A. The recorded data needs to be written to .csv files.               ##
##                                                                                      ##
##  Before it starts processing all csv files (which may take a while)                  ##
##  it outputs how many valid files were found for analysis.                            ##
##                                                                                      ##
##  When processing is done, it will plot some characteristics of the recorded signal.  ##
##                                                                                      ##
##                                                                                      ##
##  This script is compatible with python 2.7 and 3.5.                                  ##
##  Run script from command line like:                                                  ##
##      python plot.py /path/to/folder/containing/csv/files/                            ##
##                                                                                      ##
##                                                                                      ##
##  This script was written by Felix Maaß, an IT-Engineering student at the             ##
##  University of Applied Sciences Wedel in June 2018.                                  ##
##                                                                                      ##
##########################################################################################


from __future__ import print_function

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

import sys
from os import listdir
import os.path
from os.path import isdir, isfile, join

from math import pi, sin, cos, sqrt
from time import sleep

# Reads in files optionally multiplying values by a
# factor, e.g. 10^-3 if amperage is given in mA
#
# @param filepath       string representation of filepath (absolute or relative)
# @param timeFactor     factor to be applied elementwise
# @param voltageFactor  factor to be applied to time values elementwise
# @param currentFactor  factor to be applied to current values elementwise
# @param voltageFactor  factor to be applied to voltage values elementwise
#
# @return               the sanitized raw 2D numpy data object
def readFile(filepath, timeFactor=1, voltageFactor=1, currentFactor=1, applyCurrentSmoothing=False):
    #print("Reading %s" % (filepath), end="")
    data = np.genfromtxt(filepath, delimiter=',', skip_header=3, names=['t', 'u', 'i'])
    #print(" - Done.")

    data['t'] *= timeFactor
    data['u'] *= voltageFactor
    data['i'] *= currentFactor

    if applyCurrentSmoothing:
        data['i'] = smoothValues(data['i'], radius=20)

    return data

# Smoothes the values using the average of values within the specified
# radius to the left and the right.
# The data is copied and therefore original values aren't touched.
#
# @param data       should be a 1-dimensional array of numerical values
# @param radius     a way to define, how many values should be averaged
#
# @return           the smoothed value array
def smoothValues(data, radius=5):

    temp = data.copy()

    for i in range(0, 2*radius+1):
        start = i
        end = -(radius * 2 - i)

        print("Smoothing with radius %d: [%d:%d]" %(radius, start, end))
        data[radius:-radius] += temp[start:end] if end != 0 else temp[start:]

    data[radius:-radius] /= 2*radius+2

    return data


# Gets valid chunks from a raw data sample.
# A valid chunk consists of exactly one period (evaluation based on voltage).
#
# @param data   common raw 2D numpy data
#
# @return       the valid chunks contained in the sample.
def getChunks(data):

    chunks = []

    crossings = getIndicesOfZeroCrossings(data['u'])
    if len(crossings) < 2:
        print(crossings)
        return None


    for i in range(1, len(crossings)):

        start = crossings[i - 1]
        end = crossings[i]

        chunks.append(data[start:end])

    return chunks


# Filters given Chunks.
#
# A valid chunk contains at least one value which is larger
# than minimumAbsoluteAmperage. If minimumAbsoluteAmperage <= 0
# all chunks are valid and therfore returned.
#
# @param an array of chunks
# @param minimumAbsoluteAmperage the minimum amperage
#
# @return the valid chunks contained in the sample.
def filterChunks(chunks, minimumAbsoluteAmperage=0):
    if minimumAbsoluteAmperage <= 0:
        return chunks

    filteredChunks = []

    for chunk in chunks:
        if np.max(np.abs(chunk['i'])) > minimumAbsoluteAmperage:
            filteredChunks.append(chunk)

    return np.array(filteredChunks)

# Gets the indices of zero crossing of some 1-dimensional data.
#
# @param 1D array of numerical values.
# @param rising flag, specifying whether rising or falling crossings should be reconized.
#
# @return an array of indices.
def getIndicesOfZeroCrossings(data, rising=True):
    indices = []

    for i in range(1, data.size):
        lastItem = data[i - 1]
        item = data[i]
        #print(i, item)

        if (rising and lastItem < 0 and item >= 0)\
          or (not rising and lastItem > 0 and item <= 0):
            indices.append(i)

    return indices

# Normalizes Chunks for fourier transformation to match exactly a thousand values.
#
# @param data           common raw 2D numpy data containing time, voltage and current values.
# @param normalizeTime  flag, specifying whether the recorded times should
#                       get normalized to have a spacing of 20µs each
#
# @return an array of normalized chunks contained in that data sample
def getNormalizedChunksForFourier(data, normalizeTime=True):
    # 50kHz / each 20µs -> 1000 values per period

    crossings = getIndicesOfZeroCrossings(data['u'])

    adjustedChunks = []

    for crossing in crossings:
        start = crossing
        
        # lets have a look at the last value
        # and decide wether it's closer to the zero crossing
        if crossing > 0 and abs(data['u'][crossing]) > abs(data['u'][crossing - 1]):
            start = crossing - 1

        # now try to collect the 1000 corresponding values
        end = start + 1000

        if end >= data['u'].size:
            continue

        copied = data[start:end].copy()

        if normalizeTime:
            # produce evenly spaced timestamps
            copied['t'] = np.arange(start=0, step=20e-3, stop=20)

        adjustedChunks.append(copied)

    return adjustedChunks

# Aggregates normalized chunks from multiple common raw
# 2D numpy data objects into a single array.
#
# @param files  array of common raw 2D numpy data
#
# @return an array of normalized chunks.
def gatherNormalizedChunksFromFiledata(files):
    gatheredChunks = []

    for data in files:
        chunks = getNormalizedChunksForFourier(data)

        gatheredChunks += chunks

    return gatheredChunks

# Applies the fast fourier transfomation to the given data.
# This data does not need to be sanitized to chunks beforehand.
#
# @param data   some 2D array containing time, voltage and current values
#
# @return       tuple of frequencies and their corresponding magnitudes
def applyFourierTransform(data):

    # apply fourier transform
    spectrum = np.fft.fft(data['i'])

    # 'fftfreq' generates an omega that goes from -0.5 to 0.5
    # (actually rises from 0 to 0.5 than swaps to -0.5 and rises again to 0)
    omega = np.fft.fftfreq(data['i'].size)

    # normalize omega to our signal
    actualDuration = abs(data['t'][-1:] - data['t'][0]) / 1000.0
    numSamples = data['t'].size
    actualSamplingFrequency = actualDuration / numSamples

    # print("Sampling frequency = %f Hz | Samples = %d | Time = %.4f ms" % (
    #     actualSamplingFrequency,
    #     numSamples,
    #     actualDuration * 1000
    # ))

    omega /= actualSamplingFrequency

    # only show the positive side (its mirrored)
    freq = omega[:int(omega.size/2)]
    mag = np.absolute(spectrum)[:int(omega.size/2)]

    return freq, mag

# Calculates the phase shift of voltage to current of the given signal.
# The signal needs to be exactly one full period long.
#
# @param chunk  exactly one chunk of data
#
# @return       the phase shift in radians.
def calculatePhaseShift(chunk):

    maxDiff = np.argmax(chunk['u']) - np.argmax(chunk['i'])
    minDiff = np.argmin(chunk['u']) - np.argmin(chunk['i'])

    diff = (maxDiff - minDiff) / 2.0

    #print("phaseShift: %f, %f, %f" % (maxDiff, minDiff, diff))

    # chunk is expected to be exactly one full period
    return diff / len(chunk) * 2 * pi


#MARK: Plotting
def plotForDataChunk(currentDatasetIndex, totalNumDatasets, data, fourierData=None):

    fig = plt.figure(42)
    fig.clf()
    fig.canvas.set_window_title("Non-Intrusive Load Monitoring - Chunk %d/%d - Felix Maass" % (currentDatasetIndex + 1, totalNumDatasets))

    #######

    rows = 2
    columns = 2

    #######
    ax1 = fig.add_subplot(str(rows) + str(columns) + str(1))
    plotRawSignals(data, axis=ax1)

    #######
    ax3 = fig.add_subplot(str(rows) + str(columns) + str(columns + 1))
    plotPowerCurve(data, axis=ax3)

    #######
    ax4 = fig.add_subplot(str(rows) + str(columns) + str(columns))
    plotIVCurve(data, axis=ax4)

    ###########################

    ax5 = fig.add_subplot(str(rows) + str(columns) + str(columns * 2))

    freq, mag = fourierData if fourierData is not None else applyFourierTransform(data)
    plotFourierData(freq, mag, axis=ax5)


    #######

    fig.tight_layout()
    plt.show()

def plotRawSignals(chunk, axis=None):
    ax1 = axis

    if axis is None:
        fig = plt.figure(101)
        fig.clf()
        fig.canvas.set_window_title("Non-Intrusive Load Monitoring - Raw Signals - Felix Maass")

        ax1 = fig.add_subplot(111)


    ax1.set_title("Raw Signals")
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Voltage [V]')

    voltageline = ax1.plot(chunk['t'], chunk['u'], color='r', label='Voltage')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Amperage [A]')
    ampline = ax2.plot(chunk['t'], chunk['i'], color='b', label='Amperage')

    ax2.get_xaxis().set_major_formatter(ticker.EngFormatter())
    ax2.get_xaxis().set_minor_formatter(ticker.NullFormatter())


    if np.max(np.abs(chunk['i'])) < 0.3:
        ax2.set_ylim([-500,500])
        ax2.set_autoscaley_on(False)
    else:
        ax2.set_autoscaley_on(True)

    ax1Lines = voltageline + ampline
    ax1Labels = [l.get_label() for l in ax1Lines]
    ax1.legend(ax1Lines, ax1Labels)

    #######

    if axis is None:
        fig.tight_layout()
        plt.show()

def plotPowerCurve(chunk, axis=None):
    ax = axis

    if axis is None:
        fig = plt.figure(271)
        fig.clf()
        fig.canvas.set_window_title("Non-Intrusive Load Monitoring - Power Characteristics - Felix Maass")

        ax = fig.add_subplot(111)


    ax.set_title("Power")
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Power [VA / W / VAr]')

    ax.get_xaxis().set_major_formatter(ticker.EngFormatter())
    ax.get_xaxis().set_minor_formatter(ticker.NullFormatter())

    apparentPower   = chunk['u'] * chunk['i']

    phaseShift      = calculatePhaseShift(chunk)
    realPower       = apparentPower * cos(phaseShift)
    reactivePower   = apparentPower * sin(phaseShift)

    ax.plot(chunk['t'], apparentPower, color='r', label='Apparent')
    ax.plot(chunk['t'], realPower, color='g', label='Real')
    ax.plot(chunk['t'], reactivePower, color='b', label='Reactive')

    ax.legend()

    #######

    if axis is None:
        fig.tight_layout()
        plt.show()

def plotIVCurve(chunk, axis=None):
    ax = axis

    if axis is None:
        fig = plt.figure(314)
        fig.clf()
        fig.canvas.set_window_title("Non-Intrusive Load Monitoring - Current-Voltage Characteristics - Felix Maass")

        ax = fig.add_subplot(111)

    ax.set_title("IV-Curve")
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('Amperage [A]')

    ax.get_xaxis().set_major_formatter(ticker.EngFormatter())
    ax.get_xaxis().set_minor_formatter(ticker.NullFormatter())

    if np.max(np.abs(chunk['i'])) < 0.3:
        ax.set_ylim([-500,500])
        ax.set_autoscaley_on(False)
    else:
        ax.set_autoscaley_on(True)

    ax.plot(chunk['u'], chunk['i'], color='b')


    #######

    if axis is None:
        fig.tight_layout()
        plt.show()

def plotFourierData(freq, mag, axis=None, color='b', alpha=1.0):
    ax = axis

    if axis is None:
        fig = plt.figure(1337)
        fig.clf()
        fig.canvas.set_window_title("Non-Intrusive Load Monitoring - Frequency Spectrum - Felix Maass")

        ax = fig.add_subplot(111)

    ax.plot(freq, mag, color=color, alpha=alpha)

    ax.set_title("Frequency Spectrum")

    ax.set_xlabel('Frequency [Hz]')
    ax.get_xaxis().set_major_formatter(ticker.EngFormatter())
    ax.get_xaxis().set_minor_formatter(ticker.NullFormatter())

    ax.set_ylabel('Magnitude (log10)')
    ax.set_yscale('log')

    #######

    if axis is None:
        fig.tight_layout()
        plt.show()



if __name__ == '__main__':

    #############################################
    #MARK: - Read Files

    #SOURCE_DIR = "/Users/Felix/DefaultDesktop/csv/20180516-0001/"
    #SOURCE_DIR = "/Users/Felix/DefaultDesktop/nilm-samples/csv/20180516-felix-phone-47pro-iclever-2_4amps-charger"
    #SOURCE_DIR = "/Users/Felix/Documents/FH/NILM/Seminar/plotting/data/laptop"
    SOURCE_DIR = "/home/morschel/Documents/Studium/2018SS/DataScienceProjekt/DataScienceProjekt/data/laptom_single"

    if len(sys.argv) >= 2:
    	SOURCE_DIR = sys.argv[1]

    SOURCE_DIR = os.path.abspath(SOURCE_DIR)
    if not isdir(SOURCE_DIR):
    	exit("That's not a directory!")


    filepaths = []
    for f in listdir(SOURCE_DIR):
        path = join(SOURCE_DIR, f)
        if isfile(path) and f.endswith(".csv"):
            filepaths.append(path)

    if len(filepaths) == 0:
    	exit("Directory does not contain any csv files: %s" % SOURCE_DIR)

    filepaths = np.sort(filepaths)
    print("Found %s files." % len(filepaths))



    #############################################
    #MARK: - Read Chunks

    filedata = []
    for filepath in filepaths:
        # the files contain values in [V] and [A]
        # use a current Factor of 10e-3 if given in [mA]
        filedata.append(readFile(filepath, voltageFactor=1, currentFactor=1))


    chunks = gatherNormalizedChunksFromFiledata(filedata)
    chunks = filterChunks(chunks, minimumAbsoluteAmperage=0.05)
    print("Found %d valid chunks." % len(chunks['i']))


    #############################################
    #MARK: - Plot Fourier Transform

    frequencies = None
    magnitudes = []

    fourierChunks = filterChunks(chunks, minimumAbsoluteAmperage=0.5)
    for chunk in fourierChunks:
        freq, mag = applyFourierTransform(chunk)
        frequencies = freq if frequencies is None else frequencies
        magnitudes.append(mag)

    magnitudes = np.array(magnitudes)
    avg_magnitudes = np.mean(magnitudes, axis=0)

    # plot mean frequency spectrum
    plt.ion()
    plotFourierData(frequencies, avg_magnitudes, color='b')
    plt.ioff()

    #############################################
    #MARK: - Plot data

    chunks = filterChunks(chunks, minimumAbsoluteAmperage=0.05)
    minBound = 0
    maxBound = len(chunks)
    totalNumChunks = maxBound - minBound

    # plot characteristics of first chunk
    firstChunk = chunks[minBound]
    plotRawSignals(firstChunk)
    plotPowerCurve(firstChunk)
    plotIVCurve(firstChunk)

    # plot mean frequency overlay
    fig = plt.figure(24)
    ax = fig.add_subplot(111)
    freq, mag = applyFourierTransform(firstChunk)
    plotFourierData(freq, mag, axis=ax, color='b', alpha=0.3)
    plotFourierData(frequencies, avg_magnitudes, axis=ax, color='r')
    fig.canvas.set_window_title("Non-Intrusive Load Monitoring - Mean Frequency Spectrum Overlay - Felix Maass")
    plt.show()

    # uncomment for continuous plotting of whole sample
    plt.ion()
    for i in range(minBound, maxBound):
        chunk = chunks[i]
        plotForDataChunk(i - minBound, totalNumChunks, chunk)
        plt.pause(0.1)
