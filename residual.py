import os
import subprocess
import numpy as np
import pylab

N = 1024;
tpb = 32;
maxcycles = 500;

os.system('nvcc iterative.cu -o iterative')

for i in range(1,maxcycles+1):
    iterations = i*tpb
    os.system('./iterative ' + str(N) + ' ' + str(tpb) + ' ' + str(iterations))

# ===========================================================================
