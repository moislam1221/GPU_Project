#include<utility>
#include<stdio.h>
#include<assert.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <ostream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <utility>

enum method_type { JACOBI, GS, SOR };

template <typename method_type>
__host__ __device__
double iterativeOperation(const double leftMatrix, const double centerMatrix, const double rightMatrix, double leftX, double centerX, double rightX, const double centerRhs, int gridPoint, method_type method)
{
    double gridValue = centerX;
    switch(method)
    {
        case JACOBI:
	    return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix;
	case GS:
	    if (gridPoint % 2 == 1) {
	        return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix;
	    }
	case SOR:
	    double relaxation = 1.2596;
	    if (gridPoint % 2 == 1) {
	        return gridValue = relaxation*((centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix) + (1.0-relaxation)*centerX;
	    }
    }
    return gridValue;
}


template <typename method_type>
__host__ __device__
double iterativeOperation2(const double leftMatrix, const double centerMatrix, const double rightMatrix, double leftX, double centerX, double rightX, const double centerRhs, int gridPoint, method_type method)
{
    double gridValue = centerX;
    switch(method)
    {
	case JACOBI:	
	    return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix;
	case GS:
	    if (gridPoint % 2 == 0) {
	        return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix;
	    }
	case SOR:
	    double relaxation = 1.2596;
	    if (gridPoint % 2 == 0) {
	        return gridValue = relaxation*((centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix) + (1.0-relaxation)*centerX;
	    }
    }
    return gridValue;
}

__device__ __host__
double jacobiGrid(const double leftMatrix, const double centerMatrix, const double rightMatrix,
                 const double leftX, double centerX, const double rightX, const double centerRhs)
{
    return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX))
         / centerMatrix;
}

__device__ __host__
double RBGSGrid(const double leftMatrix, const double centerMatrix, const double rightMatrix,
		      const double leftX, double centerX, const double rightX, const double centerRhs,
		      const int gridPoint)
{  
    
    // Update all points of a certain parity (i.e. update red, keep black the same)
    if (gridPoint % 2 == 1)
    {
    	return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX))
	 / centerMatrix;
    }
    else
    {
	return centerX;
    }
}

__device__ __host__
double SORGrid(const double leftMatrix, const double centerMatrix, const double rightMatrix,
		      const double leftX, double centerX, const double rightX, const double centerRhs,
		      const int gridPoint)
{  
    // Similar to red-black gauss-seidel, but take weighted average of rbgs 
    // value and current centerX value based on relaxation parameter
    // printf("Relaxation is %f\n", relaxation);
    double relaxation = 1.0;
    if (gridPoint % 2 == 1)
    {
    	return relaxation*((centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix) + (1.0-relaxation)*centerX;
    }
    else
    {
	return centerX;
    }
}

double normFromRow(double leftMatrix, double centerMatrix, double rightMatrix, double leftX, double centerX, double rightX,  double centerRhs) 
{
    return centerRhs - (leftMatrix*leftX + centerMatrix*centerX + rightMatrix*rightX);
}

double Residual(const double * solution, const double * rhs, const double * leftMatrix, const double * centerMatrix, const double * rightMatrix, int nGrids)
{
    int nDofs = nGrids;
    double residual = 0.0;
    for (int iGrid = 0; iGrid < nDofs; iGrid++) {
        double leftX = (iGrid > 0) ? solution[iGrid - 1] : 0.0f; 
        double centerX = solution[iGrid];
        double rightX = (iGrid < nGrids - 1) ? solution[iGrid + 1] : 0.0f;
        double residualContributionFromRow = normFromRow(leftMatrix[iGrid], centerMatrix[iGrid], rightMatrix[iGrid], leftX, centerX, rightX, rhs[iGrid]);
	residual = residual + residualContributionFromRow * residualContributionFromRow;
	// printf("For gridpoint %d, residual contribution is %f\n", iGrid, residualContributionFromRow);
    }
    residual = sqrt(residual);
    return residual;
}

/*double * readExactSolution(int nGrids)
{
    double exactSolution[nGrids];
    std::ifstream input("exactSolution.txt");
    for (int i = 0; i < nGrids; i++)
    {
        input >> exactSolution[i];
        // printf("Data is %f\n", exactSolution[i]);
    }
    return exactSolution;
}*/

double solutionError(double * solution, double * exactSolution, int nGrids)
{
    double error = 0.0;
    double difference; 
    for (int iGrid = 0; iGrid < nGrids; iGrid++) {
         difference = solution[iGrid] - exactSolution[iGrid];
	 error = error + difference*difference;
    }
    error = sqrt(error);
    return error;
}


double * iterativeCpu(const double * initX, const double * rhs,
                  const double * leftMatrix, const double * centerMatrix,
                  const double * rightMatrix, int nGrids, int nIters, int method)
{
    double * x0 = new double[nGrids];
    double * x1 = new double[nGrids];
    memcpy(x0, initX, sizeof(double) * nGrids);
    memcpy(x1, initX, sizeof(double)*nGrids);
    for (int iIter = 0; iIter < nIters; ++ iIter) {
        for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
            double leftX = (iGrid > 0) ? x0[iGrid - 1] : 0.0f;
            double centerX = x0[iGrid];
            double rightX = (iGrid < nGrids - 1) ? x0[iGrid + 1] : 0.0f;
	    if (iIter % 2 == 0) {
                x1[iGrid] = iterativeOperation(leftMatrix[iGrid], centerMatrix[iGrid],
                                    rightMatrix[iGrid], leftX, centerX, rightX,
                                    rhs[iGrid], iGrid, method);
	    }
	    else { 
                x1[iGrid] = iterativeOperation2(leftMatrix[iGrid], centerMatrix[iGrid],
                                    rightMatrix[iGrid], leftX, centerX, rightX,
                                    rhs[iGrid], iGrid, method);
            }
        }
        double * tmp = x0; x0 = x1; x1 = tmp;
    }
    delete[] x1;
    return x0;
}


__global__
void _iterativeGpuClassicIteration(double * x1,
                         const double * x0, const double * rhs,
                         const double * leftMatrix, const double * centerMatrix,
                         const double * rightMatrix, int nGrids, int iteration, int method)
{
    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (iGrid < nGrids) {
        double leftX = (iGrid > 0) ? x0[iGrid - 1] : 0.0f;
        double centerX = x0[iGrid];
        double rightX = (iGrid < nGrids - 1) ? x0[iGrid + 1] : 0.0f;
	if (iteration % 2 == 0) {
            x1[iGrid] = iterativeOperation(leftMatrix[iGrid], centerMatrix[iGrid],
                                    rightMatrix[iGrid], leftX, centerX, rightX,
                                    rhs[iGrid], iGrid, method);
	}
	else { 
            x1[iGrid] = iterativeOperation2(leftMatrix[iGrid], centerMatrix[iGrid],
                                    rightMatrix[iGrid], leftX, centerX, rightX,
                                    rhs[iGrid], iGrid, method);
	}
    }
    __syncthreads();
}

double * iterativeGpuClassic(const double * initX, const double * rhs,
                         const double * leftMatrix, const double * centerMatrix,
                         const double * rightMatrix, int nGrids, int nIters,
                         const int threadsPerBlock, int method)
{
    // Allocate memory in the CPU for all inputs and solutions
    double * x0Gpu, * x1Gpu;
    cudaMalloc(&x0Gpu, sizeof(double) * nGrids);
    cudaMalloc(&x1Gpu, sizeof(double) * nGrids);
    double * rhsGpu, * leftMatrixGpu, * rightMatrixGpu, * centerMatrixGpu;
    cudaMalloc(&rhsGpu, sizeof(double) * nGrids);
    cudaMalloc(&leftMatrixGpu, sizeof(double) * nGrids);
    cudaMalloc(&centerMatrixGpu, sizeof(double) * nGrids);
    cudaMalloc(&rightMatrixGpu, sizeof(double) * nGrids);
    
    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu, leftMatrix, sizeof(double) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu, centerMatrix, sizeof(double) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu, rightMatrix, sizeof(double) * nGrids,
            cudaMemcpyHostToDevice);

    // Run the classic iteration for prescribed number of iterations
    // int threadsPerBlock = 16;
    int nBlocks = (int)ceil(nGrids / (double)threadsPerBlock);
    for (int iIter = 0; iIter < nIters; ++iIter) {
	// Jacobi iteration on the CPU
        _iterativeGpuClassicIteration<<<nBlocks, threadsPerBlock>>>(
                x1Gpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu,
                rightMatrixGpu, nGrids, iIter, method); 
        double * tmp = x1Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
    }

    // Write solution from GPU to CPU variable
    double * solution = new double[nGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(double) * nGrids,
            cudaMemcpyDeviceToHost);

    // Free all memory
    cudaFree(x0Gpu);
    cudaFree(x1Gpu);
    cudaFree(rhsGpu);
    cudaFree(leftMatrixGpu);
    cudaFree(centerMatrixGpu);
    cudaFree(rightMatrixGpu);

    return solution;
}


__device__ 
void __iterativeBlockUpperTriangleFromShared(
		double * xLeftBlock, double *xRightBlock, const double *rhsBlock,
		const double * leftMatrixBlock, const double * centerMatrixBlock,
                const double * rightMatrixBlock, int nGrids, int iGrid, int method)
{
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x; 

    for (int k = 1; k < blockDim.x/2; ++k) {
        if (threadIdx.x >= k && threadIdx.x <= blockDim.x-k-1) {
            double leftX = x0[threadIdx.x - 1];
            double centerX = x0[threadIdx.x];
            double rightX = x0[threadIdx.x + 1];
	    if (iGrid == 0) {
		leftX = 0.0f;
	    }
	    if (iGrid == nGrids-1) {
		rightX = 0.0f;
	    }
            double leftMat = leftMatrixBlock[threadIdx.x];
            double centerMat = centerMatrixBlock[threadIdx.x];
            double rightMat = rightMatrixBlock[threadIdx.x];
            double rhs = rhsBlock[threadIdx.x];
	    if (k % 2 == 1) {
	        x1[threadIdx.x] = iterativeOperation(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs, iGrid, method);
	    }
	    else {
	        x1[threadIdx.x] = iterativeOperation2(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs, iGrid, method);
	    }
        }
        __syncthreads();	
	double * tmp = x1; x1 = x0; x0 = tmp;
    } 
    
    double * tmp = x1; x1 = x0; x0 = tmp;

    int remainder = threadIdx.x % 4;
    xLeftBlock[threadIdx.x] = x0[(threadIdx.x+1)/2 + blockDim.x*(remainder > 1)];
    xRightBlock[threadIdx.x] = x0[blockDim.x-1-(threadIdx.x+1)/2 + blockDim.x*(remainder > 1)];

}

__global__
void _iterativeGpuUpperTriangle(double * xLeftGpu, double *xRightGpu,
                             const double * x0Gpu, const double *rhsGpu, 
                             const double * leftMatrixGpu, const double *centerMatrixGpu,
                             const double * rightMatrixGpu, int nGrids, int method)
{
    int blockShift = blockDim.x * blockIdx.x;
    double * xLeftBlock = xLeftGpu + blockShift;
    double * xRightBlock = xRightGpu + blockShift;
    const double * x0Block = x0Gpu + blockShift;
    const double * rhsBlock = rhsGpu + blockShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double sharedMemory[];
    sharedMemory[threadIdx.x] = x0Block[threadIdx.x];
    sharedMemory[threadIdx.x + blockDim.x] = x0Block[threadIdx.x];

    __iterativeBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
    		                       leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);
}

__device__ 
void __iterativeBlockLowerTriangleFromShared(
		const double * xLeftBlock, const double *xRightBlock, const double *rhsBlock,
		const double * leftMatrixBlock, const double * centerMatrixBlock,
                const double * rightMatrixBlock, int nGrids, int iGrid, int method)
{
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x;

    int remainder = threadIdx.x % 4;

    if (threadIdx.x != blockDim.x-1) {
        x0[blockDim.x-1-((blockDim.x+threadIdx.x+1)/2) + blockDim.x*(remainder>1)] = xLeftBlock[threadIdx.x];
	x0[(blockDim.x+threadIdx.x+1)/2 + blockDim.x*(remainder>1)] = xRightBlock[threadIdx.x];
    } 

    # pragma unroll
    for (int k = blockDim.x/2; k > 0; --k) {
	if (k < blockDim.x/2) {
	    if (threadIdx.x >= k && threadIdx.x <= blockDim.x-k-1) {
                double leftX = x0[threadIdx.x - 1];
                double centerX = x0[threadIdx.x];
                double rightX = x0[threadIdx.x + 1];
		if (iGrid == 0) {
		    leftX = 0.0f;
		}
		if (iGrid == nGrids-1) {
		    rightX = 0.0f;
		}
		double leftMat = leftMatrixBlock[threadIdx.x];
		double centerMat = centerMatrixBlock[threadIdx.x];
 		double rightMat = rightMatrixBlock[threadIdx.x];
		double rhs = rhsBlock[threadIdx.x];
	        if (k % 2 == 1) {	
	            x1[threadIdx.x] = iterativeOperation(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs, iGrid, method);
		}
		else {
		    x1[threadIdx.x] = iterativeOperation2(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs, iGrid, method);
		}
	    }
 	    double * tmp = x1; x1 = x0; x0 = tmp;
        }
	__syncthreads();
    }

    double leftX = (threadIdx.x == 0) ? xLeftBlock[blockDim.x - 1] : x0[threadIdx.x - 1];
    double centerX = x0[threadIdx.x];
    double rightX = (threadIdx.x == blockDim.x-1) ? xRightBlock[blockDim.x - 1] : x0[threadIdx.x + 1];
    if (iGrid == 0) {
       leftX = 0.0;    
    }
    if (iGrid == nGrids-1) {
        rightX = 0.0;
    }
    // The last step! - Should i just perform one of the grid operations
    // The last step of the for loop above uses k = 1 where gridOperation is used, so I'll use gridOperation2 here
    x1[threadIdx.x] = iterativeOperation2(leftMatrixBlock[threadIdx.x],
                                centerMatrixBlock[threadIdx.x],
                                rightMatrixBlock[threadIdx.x],
                                leftX, centerX, rightX, rhsBlock[threadIdx.x], iGrid, method);
    double * tmp = x1; x1 = x0; x0 = tmp; 

}

__global__
void _iterativeGpuLowerTriangle(double * x0Gpu, double *xLeftGpu,
                             double * xRightGpu, double *rhsGpu, 
                             double * leftMatrixGpu, double *centerMatrixGpu,
                             double * rightMatrixGpu, int nGrids, int method)
{
    int blockShift = blockDim.x * blockIdx.x;
    double * xLeftBlock = xLeftGpu + blockShift;
    double * xRightBlock = xRightGpu + blockShift;
    double * x0Block = x0Gpu + blockShift;
    double * rhsBlock = rhsGpu + blockShift;
    double * leftMatrixBlock = leftMatrixGpu + blockShift;
    double * centerMatrixBlock = centerMatrixGpu + blockShift;
    double * rightMatrixBlock = rightMatrixGpu + blockShift;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ double sharedMemory[];
    
    __iterativeBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                         leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);

    x0Block[threadIdx.x] = sharedMemory[threadIdx.x];

}

__global__       
void _iterativeGpuShiftedDiamond(double * xLeftGpu, double * xRightGpu,
                              double * rhsGpu, 
			      double * leftMatrixGpu, double * centerMatrixGpu,
                              double * rightMatrixGpu, int nGrids, int method)
{

    int blockShift = blockDim.x * blockIdx.x;
    double * xLeftBlock = xRightGpu + blockShift;
    double * xRightBlock = (blockIdx.x == (gridDim.x-1)) ?
                          xLeftGpu : 
                          xLeftGpu + blockShift + blockDim.x;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x/2;
    iGrid = (iGrid < nGrids) ? iGrid : threadIdx.x - blockDim.x/2;

    int indexShift = blockDim.x/2;
    double * rhsBlock = rhsGpu + blockShift + indexShift;
    double * leftMatrixBlock = leftMatrixGpu + blockShift + indexShift;
    double * centerMatrixBlock = centerMatrixGpu + blockShift + indexShift;
    double * rightMatrixBlock = rightMatrixGpu + blockShift + indexShift;
    
    extern __shared__ double sharedMemory[];
    
    __iterativeBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                         leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);  

    __iterativeBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                                       leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);

}

__global__
void _iterativeGpuDiamond(double * xLeftGpu, double * xRightGpu,
                       const double * rhsGpu,
		       const double * leftMatrixGpu, const double * centerMatrixGpu,
                       const double * rightMatrixGpu, int nGrids, int method)
{
    int blockShift = blockDim.x * blockIdx.x;
    double * xLeftBlock = xLeftGpu + blockShift;
    double * xRightBlock = xRightGpu + blockShift;

    const double * rhsBlock = rhsGpu + blockShift;
    const double * leftMatrixBlock = leftMatrixGpu;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift;

    int iGrid = blockDim.x * blockIdx.x + threadIdx.x;
    
    extern __shared__ double sharedMemory[];

    __iterativeBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                        leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);
    
    __iterativeBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                                      leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);
}
double * iterativeGpuSwept(const double * initX, const double * rhs,
        const double * leftMatrix, const double * centerMatrix,
        const double * rightMatrix, int nGrids, int nIters,
        const int threadsPerBlock, const int method) { 
    
    // Determine number of threads and blocks 
    const int nBlocks = (int)ceil(nGrids / (double)threadsPerBlock);

    // Allocate memory for solution and inputs
    double *xLeftGpu, *xRightGpu;
    cudaMalloc(&xLeftGpu, sizeof(double) * threadsPerBlock * nBlocks);
    cudaMalloc(&xRightGpu, sizeof(double) * threadsPerBlock * nBlocks);
    double * x0Gpu, * rhsGpu, * leftMatrixGpu, * rightMatrixGpu, * centerMatrixGpu;
    cudaMalloc(&x0Gpu, sizeof(double) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&rhsGpu, sizeof(double) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&leftMatrixGpu, sizeof(double) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&centerMatrixGpu, sizeof(double) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&rightMatrixGpu, sizeof(double) * (nGrids + threadsPerBlock/2));

    // Allocate memory in the GPU
    cudaMemcpy(x0Gpu, initX, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(double) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu, leftMatrix, sizeof(double) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu, centerMatrix, sizeof(double) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu, rightMatrix, sizeof(double) * nGrids,
            cudaMemcpyHostToDevice);

    // Allocate a bit more memory to avoid memcpy within shifted kernels
    cudaMemcpy(x0Gpu + nGrids, initX, sizeof(double) * threadsPerBlock/2, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu + nGrids, rhs, sizeof(double) * threadsPerBlock/2, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu + nGrids, leftMatrix, sizeof(double) * threadsPerBlock/2,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu + nGrids, centerMatrix, sizeof(double) * threadsPerBlock/2,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu + nGrids, rightMatrix, sizeof(double) * threadsPerBlock/2,
            cudaMemcpyHostToDevice);

    int sharedFloatsPerBlock = threadsPerBlock * 2;

    double residualSwept;
    double nCycles = nIters / threadsPerBlock;
    double * currentSolution = new double[nGrids];
    std::ofstream residuals;
    residuals.open("dummy.txt",std::ios_base::app);
    
    for (int i = 0; i < nCycles; i++) {
        _iterativeGpuUpperTriangle <<<nBlocks, threadsPerBlock, sizeof(double) * sharedFloatsPerBlock>>> (xLeftGpu, xRightGpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, method);
	_iterativeGpuShiftedDiamond <<<nBlocks, threadsPerBlock, sizeof(double) * sharedFloatsPerBlock>>> (xLeftGpu, xRightGpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, method);
	_iterativeGpuLowerTriangle <<<nBlocks, threadsPerBlock, sizeof(double) * sharedFloatsPerBlock>>> (x0Gpu, xLeftGpu, xRightGpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, method);
        cudaMemcpy(currentSolution, x0Gpu, sizeof(double) * nGrids,
            cudaMemcpyDeviceToHost);
        residualSwept = Residual(currentSolution, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
        residuals << nGrids << "\t" << threadsPerBlock << "\t" << i*threadsPerBlock << "\t" << residualSwept << "\n";
    }
   
    residuals.close();
/*
    _iterativeGpuUpperTriangle <<<nBlocks, threadsPerBlock,
        sizeof(double) * sharedFloatsPerBlock>>>(
                xLeftGpu, xRightGpu,
                x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu,
                rightMatrixGpu, nGrids, method);
    _iterativeGpuShiftedDiamond <<<nBlocks, threadsPerBlock,
            sizeof(double) * sharedFloatsPerBlock>>>(
                    xLeftGpu, xRightGpu,
                    rhsGpu, leftMatrixGpu, centerMatrixGpu,
                    rightMatrixGpu, nGrids, method);

    for (int i = 0; i < nIters/threadsPerBlock-1; i++) {
    _iterativeGpuDiamond <<<nBlocks, threadsPerBlock,
                sizeof(double) * sharedFloatsPerBlock>>>(
                        xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu, nGrids, method); 
    _iterativeGpuShiftedDiamond <<<nBlocks, threadsPerBlock,
            sizeof(double) * sharedFloatsPerBlock>>>(
                    xLeftGpu, xRightGpu,
                    rhsGpu, leftMatrixGpu, centerMatrixGpu,
                    rightMatrixGpu, nGrids, method); 
    }

    _iterativeGpuLowerTriangle <<<nBlocks, threadsPerBlock,
                sizeof(double) * sharedFloatsPerBlock>>>(
                        x0Gpu, xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu, nGrids, method); */

    double * solution = new double[nGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(double) * nGrids,
            cudaMemcpyDeviceToHost);

    cudaFree(x0Gpu);
    cudaFree(xLeftGpu);
    cudaFree(xRightGpu);
    cudaFree(rhsGpu);
    cudaFree(leftMatrixGpu);
    cudaFree(centerMatrixGpu);
    cudaFree(rightMatrixGpu);

    return solution;
}

int main(int argc, char *argv[])
{
    // Ask user for inputs
    const int nGrids = atoi(argv[1]); 
    const int threadsPerBlock = atoi(argv[2]); 
    const int nIters = atoi(argv[3]);

    method_type method = JACOBI;

    // Declare arrays and population with values for Poisson equation
    double * initX = new double[nGrids];
    double * rhs = new double[nGrids];
    double * leftMatrix = new double[nGrids];
    double * centerMatrix = new double[nGrids];
    double * rightMatrix = new double[nGrids];
    double dx = 1.0f / (nGrids + 1);
    for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
        initX[iGrid] = 1.0f; 
        rhs[iGrid] = 1.0f;
        leftMatrix[iGrid] = -1.0f / (dx * dx);
        centerMatrix[iGrid] = 2.0f / (dx * dx);
        rightMatrix[iGrid] = -1.0f / (dx * dx);
    }

    // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    // Run the CPU Implementation and measure the time required
    clock_t cpuStartTime = clock();
    double * solutionCpu = iterativeCpu(initX, rhs, leftMatrix, centerMatrix,
                                    rightMatrix, nGrids, nIters, method);
    clock_t cpuEndTime = clock();
    double cpuTime = (cpuEndTime - cpuStartTime) / (double) CLOCKS_PER_SEC;

    // Run the Classic GPU Implementation and measure the time required
    cudaEvent_t startClassic, stopClassic;
    float timeClassic;
    cudaEventCreate( &startClassic );
    cudaEventCreate( &stopClassic );
    cudaEventRecord(startClassic, 0);
    double * solutionGpuClassic = iterativeGpuClassic(initX, rhs, leftMatrix,
            centerMatrix, rightMatrix, nGrids, nIters, threadsPerBlock, method);
    cudaEventRecord(stopClassic, 0);
    cudaEventSynchronize(stopClassic);
    cudaEventElapsedTime(&timeClassic, startClassic, stopClassic);

    // Run the Swept GPU Implementation and measure the time required
    cudaEvent_t startSwept, stopSwept;
    float timeSwept;
    cudaEventCreate( &startSwept );
    cudaEventCreate( &stopSwept );
    cudaEventRecord( startSwept, 0);
    // TODO: change the name of jacobiXXX since they are not just doing jacobi
    double * solutionGpuSwept = iterativeGpuSwept(initX, rhs, leftMatrix,
            centerMatrix, rightMatrix, nGrids, nIters, threadsPerBlock, method);
    cudaEventRecord(stopSwept, 0);
    cudaEventSynchronize(stopSwept);
    cudaEventElapsedTime(&timeSwept, startSwept, stopSwept);

    // Print parameters of the problem to screen
    printf("===============INFORMATION============================\n");
    printf("Number of grid points: %d\n", nGrids);
    printf("Threads Per Block: %d\n", threadsPerBlock);
    printf("Method used: %d\n", method);
    printf("Number of Iterations performed: %d\n", nIters);
    printf("\n");

    // Print out results to the screen, notify if any GPU Classic or Swept values differ significantly
    for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
        printf("%d %f %f %f \n",iGrid, solutionCpu[iGrid],
                             solutionGpuClassic[iGrid],
                             solutionGpuSwept[iGrid]); 
	//assert(solutionGpuClassic[iGrid] == solutionGpuSwept[iGrid]);
	if (abs(solutionGpuClassic[iGrid] - solutionGpuSwept[iGrid]) > 1e-2) {
	    printf("For grid point %d, Classic and Swept give %f and %f respectively\n", iGrid, solutionGpuClassic[iGrid], solutionGpuSwept[iGrid]);
	}
    }

    // Print out time for cpu, classic gpu, and swept gpu approaches
    double cpuTimePerIteration = (cpuTime / nIters) * 1e3;
    double classicTimePerIteration = timeClassic / nIters;
    double sweptTimePerIteration = timeSwept / nIters;
    double timeMultiplier = classicTimePerIteration / sweptTimePerIteration;
    printf("Time needed for the CPU (per iteration): %f ms\n", cpuTimePerIteration);
    printf("Time needed for the Classic GPU (per iteration) is %f ms\n", classicTimePerIteration);
    printf("Time needed for the Swept GPU (per iteration): %f ms\n", sweptTimePerIteration); 

    // Compute the residual of the resulting solution (|b-Ax|)
    double residualClassic = Residual(solutionGpuClassic, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    double residualSwept = Residual(solutionGpuSwept, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    printf("Residual of the converged solution is %f\n", residualSwept);
    printf("Residual of Classic result is %f\n", residualClassic); 
  
    // Save residual to a file
    /* std::ofstream residuals;
    residuals.open("residual-gs.txt",std::ios_base::app);
    residuals << nGrids << "\t" << threadsPerBlock << "\t" << nIters << "\t" << residualSwept << "\n";
    residuals.close(); */

    // Save Results to a file "N tpb Iterations CPUTime/perstep ClassicTime/perstep SweptTime/perStep ClassicTime/SweptTime"
    std::ofstream timings;
    timings.open("time.txt",std::ios_base::app);
    timings << nGrids << "\t" << threadsPerBlock << "\t" << nIters << "\t" << cpuTimePerIteration << "\t" << classicTimePerIteration << "\t" << sweptTimePerIteration << "\t" << timeMultiplier << "\n";
    timings.close();

    // Free memory
    cudaEventDestroy(startClassic);
    cudaEventDestroy(startSwept);
    delete[] initX;
    delete[] rhs;
    delete[] leftMatrix;
    delete[] centerMatrix;
    delete[] rightMatrix;
    delete[] solutionCpu;
    delete[] solutionGpuClassic;
}
