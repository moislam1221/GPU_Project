#!/usr/bin/env python

import os
import subprocess
import matplotlib
import numpy as np
import pylab

os.system('nvcc main_rectangular.cu -o main')

# os.system('./main ' + nGrids + threadsPerBlock + nInnerUpdates + TOL);

N_array = [68, 130, 258, 514, 1026]
tpb_array = [32]
trials = 1
TOL = 1

for N in N_array:
    for tpb in tpb_array:
        for t in range(1,trials+1):
            nInnerUpdates = tpb/2
            print('Trial ' + str(t) + ':' + ' N = ' + str(N) + ', tpb = ' + str(tpb) + ', TOL = ' + str(TOL) + ' IN PROGRESS!!')
            os.system('./main ' + str(N) + ' ' + str(tpb) + ' ' + str(nInnerUpdates) + ' ' + str(TOL))
            print('Trial ' + str(t) + ':' + ' N = ' + str(N) + ', tpb = ' + str(tpb) + ', TOL = ' + str(TOL) + ' IS DONE!!')



