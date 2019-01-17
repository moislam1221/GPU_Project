#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab

jacobiData = pylab.loadtxt(open("residual-jacobi.txt"), delimiter='\t', usecols=(2,3))
gsData = pylab.loadtxt(open("residual-gs.txt"), delimiter='\t', usecols=(2,3))

iterationJacobi = jacobiData[:,0]
residualJacobi = jacobiData[:,1]

iterationGS = gsData[:,0]
residualGS = gsData[:,1]

pylab.figure()
pylab.semilogy(iterationJacobi, residualJacobi, '-o',linewidth=2.0,label='Jacobi')
pylab.semilogy(iterationGS, residualGS, '-o', linewidth = 2.0, label='Gauss-Seidel')

pylab.xlabel(r'Iterations', fontsize = 20)
pylab.ylabel(r'Residual', fontsize = 20)

pylab.xticks(fontsize = 16)
pylab.yticks(fontsize = 16)

pylab.legend()
pylab.grid()

pylab.savefig('residual-Comparison-plot.png')
