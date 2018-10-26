import os
import subprocess
import numpy as np
import pylab

# TO DO:
# Plot timings vs N for various tpb - OK
# Incorporate different iterations (maybe leave this)
# Write a new file which contains only the average times

##################### INPUTS TO SCRIPT!! ###################################################################
# Directory to place results in
resultsFolder = 'Results/New_Trial_1'

# Define file numbers for timing txt file, and output png file names
speedUpPlot = 'speedupFactor.png'
timings = 'timings'
average = 'average-timings.txt'
trialParameters = 'parameters.txt'

# Define parameters for computation such as the number of DOFs (size of system), threads Per Block, and number of iterations to perform
numberOfPoints = np.array([2**10, 2**11, 2**12])
threadsPerBlock = np.array([32, 64, 128, 256, 512])
iterations = 1024
averageTrials = 10

###################### RUNNING THE TESTS ####################################################################

# Create folder where results will be placed
os.system('mkdir ' + resultsFolder)

# Create txt file containing numberOfPoints, threadsPerBlock, and iterations parameters
parameterFile = open(trialParameters,'wb')
parameterFile.write('Swept Rule Parameters (nDofs, tpb, iterations): \n')
parameterFile.write('numberOfPoints = ' + str(numberOfPoints) + '\n')
parameterFile.write('threadsPerBlock = ' + str(threadsPerBlock) + '\n')
parameterFile.write('iterations = ' + str(iterations) + '\n')
os.system('mv ' + trialParameters + ' ' + resultsFolder)

# Indicate to user that the calculations are beginning
print ("Starting the computations!\n")

# Remove any former time.txt files (if they exist)
os.system('rm time.txt')

# Compile the jacobi.cu code
os.system('nvcc jacobi.cu -o jacobi')

# Perform all permutations of the cases based on above inputs
for N in numberOfPoints:
    for tpb in threadsPerBlock:
	for t in range(1,averageTrials+1):
	    os.system('./jacobi ' + str(N) + ' ' + str(tpb) + ' ' + str(iterations))
	    print('Trial ' + str(t) + '/' + str(averageTrials) + ' Case N = ' + str(N) + ' , tpb = ' + str(tpb) + ' IS DONE!!')


###################### PLOT THE RESULTS ######################################################################

# Indicate to user that we are starting to print results
print ("Starting the plotting - Plots will be constructed shortly!\n")

# Obtain data from text file time.txt
data = pylab.loadtxt(open("time.txt"), delimiter='\t',usecols=(3,4,5,6))
cpuTime = data[:,0]
classicTime = data[:,1]
sweptTime = data[:,2]
speedUp = data[:,3]

# Write average times for each case to a txt file
parameterFile = open(average,'wb')
for i in range(0,len(numberOfPoints)):
    for j in range(0,len(threadsPerBlock)):
    	index = range(i*(len(threadsPerBlock)*averageTrials)+j*averageTrials,i*(len(threadsPerBlock)*averageTrials)+(j+1)*averageTrials)
	meanCpu = np.mean(cpuTime[index])
	meanClassic = np.mean(classicTime[index])
	meanSwept = np.mean(sweptTime[index])
	meanspeedUp = np.mean(speedUp[index])
	parameterFile.write(str(numberOfPoints[i]) + '\t' + str(threadsPerBlock[j]) + '\t' + str(iterations) + '\t' + str(meanCpu) + '\t' + str(meanClassic) + '\t' + str(meanSwept) + '\t' + str(meanspeedUp) + '\n')
parameterFile.close()

# Obtain data from text file time.txt
dataAverage = pylab.loadtxt(open(average), delimiter='\t',usecols=(3,4,5,6))
print(dataAverage[:,0])
averageCpuTime = dataAverage[:,0]
averageClassicTime = dataAverage[:,1]
averageSweptTime = dataAverage[:,2]
averageSpeedUp = dataAverage[:,3]

# Plot the speedup factor
pylab.figure()
for i in range(0,len(numberOfPoints)):
    index = range(i*len(threadsPerBlock), (i+1)*len(threadsPerBlock))
    pylab.loglog(threadsPerBlock, averageSpeedUp[index], '-o',linewidth=2.0, label='$N = $' + str(numberOfPoints[i]))
    print(averageSpeedUp[index])
pylab.xlabel(r'Threads Per Block', fontsize = 20)
pylab.ylabel(r'Speedup Factor', fontsize = 20)
pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)
pylab.legend()
pylab.grid()
pylab.savefig(speedUpPlot)
os.system('mv ' + speedUpPlot + ' ' + resultsFolder)


# Plot the cpu, classic and swept timings
for i in range(0,len(numberOfPoints)):
    pylab.figure()
    index = range(i*len(threadsPerBlock), (i+1)*len(threadsPerBlock))
    pylab.loglog(threadsPerBlock, averageCpuTime[index],'-o', label='CPU ' + str(numberOfPoints[i]), linewidth=2.0)
    pylab.loglog(threadsPerBlock, averageClassicTime[index],'-o', label='GPU Classic ' + str(numberOfPoints[i]), linewidth=2.0)
    pylab.loglog(threadsPerBlock, averageSweptTime[index],'-o', label='GPU Swept ' + str(numberOfPoints[i]), linewidth=2.0)
    pylab.xlabel(r'Threads Per Block', fontsize = 20)
    pylab.ylabel(r'Timing Per Jacobi Step [ms]', fontsize = 20)
    pylab.xticks(fontsize = 16)
    pylab.yticks(fontsize = 16)
    pylab.legend()
    pylab.grid()
    pylab.savefig(timings + '-N-' + str(numberOfPoints[i]) + '.png')
    os.system('mv ' + timings + '-N-' + str(numberOfPoints[i]) + '.png' + ' ' + resultsFolder)

# Plot the variability of timings with N for each tpb
for i in range(0,len(threadsPerBlock)):
    pylab.figure()
    index = range(i, len(numberOfPoints)*len(threadsPerBlock), len(threadsPerBlock))
    pylab.loglog(numberOfPoints, averageCpuTime[index],'-o', label='CPU ' + str(threadsPerBlock[i]), linewidth=2.0)
    pylab.loglog(numberOfPoints, averageClassicTime[index],'-o', label='GPU Classic ' + str(threadsPerBlock[i]), linewidth=2.0)
    pylab.loglog(numberOfPoints, averageSweptTime[index],'-o', label='GPU Swept ' + str(threadsPerBlock[i]), linewidth=2.0)
    pylab.xlabel(r'Number of Points', fontsize = 20)
    pylab.ylabel(r'Timing Per Jacobi Step [ms]', fontsize = 20)
    pylab.xticks(fontsize = 16)
    pylab.yticks(fontsize = 16)
    pylab.legend()
    pylab.grid()
    pylab.savefig(timings + '-tpb-' + str(threadsPerBlock[i]) + '.png')
    os.system('mv ' + timings + '-tpb-' + str(threadsPerBlock[i]) + '.png' + ' ' + resultsFolder)

# Move txt file with timings in same directory as plots
os.system('mv time.txt ' + resultsFolder)
os.system('mv ' + average + ' ' + resultsFolder)
