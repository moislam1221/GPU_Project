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
float iterativeOperation(const float leftMatrix, const float centerMatrix, const float rightMatrix, float leftX, float centerX, float rightX, const float centerRhs, int gridPoint, method_type method)
{
    float gridValue = centerX;
    switch(method)
    {
        case JACOBI:
	    return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix;
	case GS:
	    if (gridPoint % 2 == 1) {
	        return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix;
	    }
	case SOR:
	    float relaxation = 1.9939;
	    if (gridPoint % 2 == 1) {
	        return gridValue = relaxation*((centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix) + (1.0-relaxation)*centerX;
	    }
    }
    return gridValue;
}


template <typename method_type>
__host__ __device__
float iterativeOperation2(const float leftMatrix, const float centerMatrix, const float rightMatrix, float leftX, float centerX, float rightX, const float centerRhs, int gridPoint, method_type method)
{
    float gridValue = centerX;
    switch(method)
    {
	case JACOBI:	
	    return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix;
	case GS:
	    if (gridPoint % 2 == 0) {
	        return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix;
	    }
	case SOR:
	    float relaxation = 1.9939;
	    if (gridPoint % 2 == 0) {
	        return gridValue = relaxation*((centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix) + (1.0-relaxation)*centerX;
	    }
    }
    return gridValue;
}

__device__ __host__
float jacobiGrid(const float leftMatrix, const float centerMatrix, const float rightMatrix,
                 const float leftX, float centerX, const float rightX, const float centerRhs)
{
    return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX))
         / centerMatrix;
}

__device__ __host__
float RBGSGrid(const float leftMatrix, const float centerMatrix, const float rightMatrix,
		      const float leftX, float centerX, const float rightX, const float centerRhs,
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
float SORGrid(const float leftMatrix, const float centerMatrix, const float rightMatrix,
		      const float leftX, float centerX, const float rightX, const float centerRhs,
		      const int gridPoint)
{  
    // Similar to red-black gauss-seidel, but take weighted average of rbgs 
    // value and current centerX value based on relaxation parameter
    // printf("Relaxation is %f\n", relaxation);
    float relaxation = 1.0;
    if (gridPoint % 2 == 1)
    {
    	return relaxation*((centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix) + (1.0-relaxation)*centerX;
    }
    else
    {
	return centerX;
    }
}

float normFromRow(float leftMatrix, float centerMatrix, float rightMatrix, float leftX, float centerX, float rightX,  float centerRhs) 
{
    return centerRhs - (leftMatrix*leftX + centerMatrix*centerX + rightMatrix*rightX);
}

float Residual(const float * solution, const float * rhs, const float * leftMatrix, const float * centerMatrix, const float * rightMatrix, int nGrids)
{
    int nDofs = nGrids;
    float residual = 0.0;
    for (int iGrid = 0; iGrid < nDofs; iGrid++) {
        float leftX = (iGrid > 0) ? solution[iGrid - 1] : 0.0f; 
        float centerX = solution[iGrid];
        float rightX = (iGrid < nGrids - 1) ? solution[iGrid + 1] : 0.0f;
        float residualContributionFromRow = normFromRow(leftMatrix[iGrid], centerMatrix[iGrid], rightMatrix[iGrid], leftX, centerX, rightX, rhs[iGrid]);
	residual = residual + residualContributionFromRow * residualContributionFromRow;
	// printf("For gridpoint %d, residual contribution is %f\n", iGrid, residualContributionFromRow);
    }
    residual = sqrt(residual);
    return residual;
}

/*float * readExactSolution(int nGrids)
{
    float exactSolution[nGrids];
    std::ifstream input("exactSolution.txt");
    for (int i = 0; i < nGrids; i++)
    {
        input >> exactSolution[i];
        // printf("Data is %f\n", exactSolution[i]);
    }
    return exactSolution;
}*/

float solutionError(float * solution, float * exactSolution, int nGrids)
{
    float error = 0.0;
    float difference; 
    for (int iGrid = 0; iGrid < nGrids; iGrid++) {
         difference = solution[iGrid] - exactSolution[iGrid];
	 error = error + difference*difference;
    }
    error = sqrt(error);
    return error;
}


float * iterativeCpu(const float * initX, const float * rhs,
                  const float * leftMatrix, const float * centerMatrix,
                  const float * rightMatrix, int nGrids, int nIters, int method)
{
    float * x0 = new float[nGrids];
    float * x1 = new float[nGrids];
    memcpy(x0, initX, sizeof(float) * nGrids);
    memcpy(x1, initX, sizeof(float)*nGrids);
    for (int iIter = 0; iIter < nIters; ++ iIter) {
        for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
            float leftX = (iGrid > 0) ? x0[iGrid - 1] : 0.0f;
            float centerX = x0[iGrid];
            float rightX = (iGrid < nGrids - 1) ? x0[iGrid + 1] : 0.0f;
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
        float * tmp = x0; x0 = x1; x1 = tmp;
    }
    delete[] x1;
    return x0;
}


__global__
void _iterativeGpuClassicIteration(float * x1,
                         const float * x0, const float * rhs,
                         const float * leftMatrix, const float * centerMatrix,
                         const float * rightMatrix, int nGrids, int iteration, int method)
{
    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (iGrid < nGrids) {
        float leftX = (iGrid > 0) ? x0[iGrid - 1] : 0.0f;
        float centerX = x0[iGrid];
        float rightX = (iGrid < nGrids - 1) ? x0[iGrid + 1] : 0.0f;
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

float * iterativeGpuClassic(const float * initX, const float * rhs,
                         const float * leftMatrix, const float * centerMatrix,
                         const float * rightMatrix, int nGrids, int nIters,
                         const int threadsPerBlock, int method)
{
    // Allocate memory in the CPU for all inputs and solutions
    float * x0Gpu, * x1Gpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nGrids);
    cudaMalloc(&x1Gpu, sizeof(float) * nGrids);
    float * rhsGpu, * leftMatrixGpu, * rightMatrixGpu, * centerMatrixGpu;
    cudaMalloc(&rhsGpu, sizeof(float) * nGrids);
    cudaMalloc(&leftMatrixGpu, sizeof(float) * nGrids);
    cudaMalloc(&centerMatrixGpu, sizeof(float) * nGrids);
    cudaMalloc(&rightMatrixGpu, sizeof(float) * nGrids);
    
    // Allocate GPU memory
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu, leftMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu, centerMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu, rightMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);

    // Run the classic iteration for prescribed number of iterations
    // int threadsPerBlock = 16;
    int nBlocks = (int)ceil(nGrids / (float)threadsPerBlock);
    for (int iIter = 0; iIter < nIters; ++iIter) {
	// Jacobi iteration on the CPU
        _iterativeGpuClassicIteration<<<nBlocks, threadsPerBlock>>>(
                x1Gpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu,
                rightMatrixGpu, nGrids, iIter, method); 
        float * tmp = x1Gpu; x0Gpu = x1Gpu; x1Gpu = tmp;
    }

    // Write solution from GPU to CPU variable
    float * solution = new float[nGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids,
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
		float * xLeftBlock, float *xRightBlock, const float *rhsBlock,
		const float * leftMatrixBlock, const float * centerMatrixBlock,
                const float * rightMatrixBlock, int nGrids, int iGrid, int method)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x; 

    for (int k = 1; k < blockDim.x/2; ++k) {
        if (threadIdx.x >= 1 && threadIdx.x <= blockDim.x-2) {
            float leftX = x0[threadIdx.x - 1];
            float centerX = x0[threadIdx.x];
            float rightX = x0[threadIdx.x + 1];
	    if (iGrid == 0) {
		leftX = 0.0f;
	    }
	    if (iGrid == nGrids-1) {
		rightX = 0.0f;
	    }
            float leftMat = leftMatrixBlock[threadIdx.x];
            float centerMat = centerMatrixBlock[threadIdx.x];
            float rightMat = rightMatrixBlock[threadIdx.x];
            float rhs = rhsBlock[threadIdx.x];
	    if (k % 2 == 1) {
	        x1[threadIdx.x] = iterativeOperation(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs, iGrid, method);
	    }
	    else {
	        x1[threadIdx.x] = iterativeOperation2(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs, iGrid, method);
	    }
        }
        __syncthreads();	
	float * tmp = x1; x1 = x0; x0 = tmp;
    } 
    
    float * tmp = x1; x1 = x0; x0 = tmp;

    int remainder = threadIdx.x % 4;
    xLeftBlock[threadIdx.x] = x0[(threadIdx.x+1)/2 + blockDim.x*(remainder > 1)];
    xRightBlock[threadIdx.x] = x0[blockDim.x-1-(threadIdx.x+1)/2 + blockDim.x*(remainder > 1)];

}

__global__
void _iterativeGpuUpperTriangle(float * xLeftGpu, float *xRightGpu,
                             const float * x0Gpu, const float *rhsGpu, 
                             const float * leftMatrixGpu, const float *centerMatrixGpu,
                             const float * rightMatrixGpu, int nGrids, int method)
{
    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = xLeftGpu + blockShift;
    float * xRightBlock = xRightGpu + blockShift;
    const float * x0Block = x0Gpu + blockShift;
    const float * rhsBlock = rhsGpu + blockShift;
    const float * leftMatrixBlock = leftMatrixGpu + blockShift;
    const float * centerMatrixBlock = centerMatrixGpu + blockShift;
    const float * rightMatrixBlock = rightMatrixGpu + blockShift;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float sharedMemory[];
    sharedMemory[threadIdx.x] = x0Block[threadIdx.x];
    sharedMemory[threadIdx.x + blockDim.x] = x0Block[threadIdx.x];

    __iterativeBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
    		                       leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);
}

__device__ 
void __iterativeBlockLowerTriangleFromShared(
		const float * xLeftBlock, const float *xRightBlock, const float *rhsBlock,
		const float * leftMatrixBlock, const float * centerMatrixBlock,
                const float * rightMatrixBlock, int nGrids, int iGrid, int method)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x;

    int remainder = threadIdx.x % 4;

    if (threadIdx.x != blockDim.x-1) {
        x0[blockDim.x-1-((blockDim.x+threadIdx.x+1)/2) + blockDim.x*(remainder>1)] = xLeftBlock[threadIdx.x];
	x0[(blockDim.x+threadIdx.x+1)/2 + blockDim.x*(remainder>1)] = xRightBlock[threadIdx.x];
    } 

    # pragma unroll
    for (int k = blockDim.x/2; k > 0; --k) {
	if (k < blockDim.x/2) {
	    if (threadIdx.x >= 1 && threadIdx.x <= blockDim.x-2) {
                float leftX = x0[threadIdx.x - 1];
                float centerX = x0[threadIdx.x];
                float rightX = x0[threadIdx.x + 1];
		if (iGrid == 0) {
		    leftX = 0.0f;
		}
		if (iGrid == nGrids-1) {
		    rightX = 0.0f;
		}
		float leftMat = leftMatrixBlock[threadIdx.x];
		float centerMat = centerMatrixBlock[threadIdx.x];
 		float rightMat = rightMatrixBlock[threadIdx.x];
		float rhs = rhsBlock[threadIdx.x];
	        if (k % 2 == 1) {	
	            x1[threadIdx.x] = iterativeOperation(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs, iGrid, method);
		}
		else {
		    x1[threadIdx.x] = iterativeOperation2(leftMat, centerMat, rightMat, leftX, centerX, rightX, rhs, iGrid, method);
		}
	    }
 	    float * tmp = x1; x1 = x0; x0 = tmp;
        }
	__syncthreads();
    }

    float leftX = (threadIdx.x == 0) ? xLeftBlock[blockDim.x - 1] : x0[threadIdx.x - 1];
    float centerX = x0[threadIdx.x];
    float rightX = (threadIdx.x == blockDim.x-1) ? xRightBlock[blockDim.x - 1] : x0[threadIdx.x + 1];
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
    float * tmp = x1; x1 = x0; x0 = tmp; 

}

__global__
void _iterativeGpuLowerTriangle(float * x0Gpu, float *xLeftGpu,
                             float * xRightGpu, float *rhsGpu, 
                             float * leftMatrixGpu, float *centerMatrixGpu,
                             float * rightMatrixGpu, int nGrids, int method)
{
    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = xLeftGpu + blockShift;
    float * xRightBlock = xRightGpu + blockShift;
    float * x0Block = x0Gpu + blockShift;
    float * rhsBlock = rhsGpu + blockShift;
    float * leftMatrixBlock = leftMatrixGpu + blockShift;
    float * centerMatrixBlock = centerMatrixGpu + blockShift;
    float * rightMatrixBlock = rightMatrixGpu + blockShift;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float sharedMemory[];
    
    __iterativeBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                         leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);

    x0Block[threadIdx.x] = sharedMemory[threadIdx.x];

}

__global__       
void _iterativeGpuShiftedDiamond(float * xLeftGpu, float * xRightGpu,
                              float * rhsGpu, 
			      float * leftMatrixGpu, float * centerMatrixGpu,
                              float * rightMatrixGpu, int nGrids, int method)
{

    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = xRightGpu + blockShift;
    float * xRightBlock = (blockIdx.x == (gridDim.x-1)) ?
                          xLeftGpu : 
                          xLeftGpu + blockShift + blockDim.x;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x/2;
    iGrid = (iGrid < nGrids) ? iGrid : threadIdx.x - blockDim.x/2;

    int indexShift = blockDim.x/2;
    float * rhsBlock = rhsGpu + blockShift + indexShift;
    float * leftMatrixBlock = leftMatrixGpu + blockShift + indexShift;
    float * centerMatrixBlock = centerMatrixGpu + blockShift + indexShift;
    float * rightMatrixBlock = rightMatrixGpu + blockShift + indexShift;
    
    extern __shared__ float sharedMemory[];
    
    __iterativeBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                         leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);  

    __iterativeBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                                       leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);

}

__global__
void _iterativeGpuDiamond(float * xLeftGpu, float * xRightGpu,
                       const float * rhsGpu,
		       const float * leftMatrixGpu, const float * centerMatrixGpu,
                       const float * rightMatrixGpu, int nGrids, int method)
{
    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = xLeftGpu + blockShift;
    float * xRightBlock = xRightGpu + blockShift;

    const float * rhsBlock = rhsGpu + blockShift;
    const float * leftMatrixBlock = leftMatrixGpu;
    const float * centerMatrixBlock = centerMatrixGpu + blockShift;
    const float * rightMatrixBlock = rightMatrixGpu + blockShift;

    int iGrid = blockDim.x * blockIdx.x + threadIdx.x;
    
    extern __shared__ float sharedMemory[];

    __iterativeBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                        leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);
    
    __iterativeBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                                      leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid, method);
}
float * iterativeGpuSwept(const float * initX, const float * rhs,
        const float * leftMatrix, const float * centerMatrix,
        const float * rightMatrix, int nGrids, int nIters,
        const int threadsPerBlock, const int method) { 
    
    // Determine number of threads and blocks 
    const int nBlocks = (int)ceil(nGrids / (float)threadsPerBlock);

    // Allocate memory for solution and inputs
    float *xLeftGpu, *xRightGpu;
    cudaMalloc(&xLeftGpu, sizeof(float) * threadsPerBlock * nBlocks);
    cudaMalloc(&xRightGpu, sizeof(float) * threadsPerBlock * nBlocks);
    float * x0Gpu, * rhsGpu, * leftMatrixGpu, * rightMatrixGpu, * centerMatrixGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&rhsGpu, sizeof(float) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&leftMatrixGpu, sizeof(float) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&centerMatrixGpu, sizeof(float) * (nGrids + threadsPerBlock/2));
    cudaMalloc(&rightMatrixGpu, sizeof(float) * (nGrids + threadsPerBlock/2));

    // Allocate memory in the GPU
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu, leftMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu, centerMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu, rightMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);

    // Allocate a bit more memory to avoid memcpy within shifted kernels
    cudaMemcpy(x0Gpu + nGrids, initX, sizeof(float) * threadsPerBlock/2, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu + nGrids, rhs, sizeof(float) * threadsPerBlock/2, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu + nGrids, leftMatrix, sizeof(float) * threadsPerBlock/2,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu + nGrids, centerMatrix, sizeof(float) * threadsPerBlock/2,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu + nGrids, rightMatrix, sizeof(float) * threadsPerBlock/2,
            cudaMemcpyHostToDevice);

    int sharedFloatsPerBlock = threadsPerBlock * 2;

    float residualSwept;
    float nCycles = nIters / threadsPerBlock;
    float * currentSolution = new float[nGrids];
    std::ofstream residuals;
    residuals.open("dummy.txt",std::ios_base::app);
    
    for (int i = 0; i < nCycles; i++) {
        _iterativeGpuUpperTriangle <<<nBlocks, threadsPerBlock, sizeof(float) * sharedFloatsPerBlock>>> (xLeftGpu, xRightGpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, method);
	_iterativeGpuShiftedDiamond <<<nBlocks, threadsPerBlock, sizeof(float) * sharedFloatsPerBlock>>> (xLeftGpu, xRightGpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, method);
	_iterativeGpuLowerTriangle <<<nBlocks, threadsPerBlock, sizeof(float) * sharedFloatsPerBlock>>> (x0Gpu, xLeftGpu, xRightGpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, method);
        cudaMemcpy(currentSolution, x0Gpu, sizeof(float) * nGrids,
            cudaMemcpyDeviceToHost);
        residualSwept = Residual(currentSolution, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
        residuals << nGrids << "\t" << threadsPerBlock << "\t" << i*threadsPerBlock << "\t" << residualSwept << "\n";
    }
   
    residuals.close();
/*
    _iterativeGpuUpperTriangle <<<nBlocks, threadsPerBlock,
        sizeof(float) * sharedFloatsPerBlock>>>(
                xLeftGpu, xRightGpu,
                x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu,
                rightMatrixGpu, nGrids, method);
    _iterativeGpuShiftedDiamond <<<nBlocks, threadsPerBlock,
            sizeof(float) * sharedFloatsPerBlock>>>(
                    xLeftGpu, xRightGpu,
                    rhsGpu, leftMatrixGpu, centerMatrixGpu,
                    rightMatrixGpu, nGrids, method);

    for (int i = 0; i < nIters/threadsPerBlock-1; i++) {
    _iterativeGpuDiamond <<<nBlocks, threadsPerBlock,
                sizeof(float) * sharedFloatsPerBlock>>>(
                        xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu, nGrids, method); 
    _iterativeGpuShiftedDiamond <<<nBlocks, threadsPerBlock,
            sizeof(float) * sharedFloatsPerBlock>>>(
                    xLeftGpu, xRightGpu,
                    rhsGpu, leftMatrixGpu, centerMatrixGpu,
                    rightMatrixGpu, nGrids, method); 
    }

    _iterativeGpuLowerTriangle <<<nBlocks, threadsPerBlock,
                sizeof(float) * sharedFloatsPerBlock>>>(
                        x0Gpu, xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu, nGrids, method); 
*/
    float * solution = new float[nGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids,
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
    float * initX = new float[nGrids];
    float * rhs = new float[nGrids];
    float * leftMatrix = new float[nGrids];
    float * centerMatrix = new float[nGrids];
    float * rightMatrix = new float[nGrids];
    float dx = 1.0f / (nGrids + 1);
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
    float * solutionCpu = iterativeCpu(initX, rhs, leftMatrix, centerMatrix,
                                    rightMatrix, nGrids, nIters, method);
    clock_t cpuEndTime = clock();
    float cpuTime = (cpuEndTime - cpuStartTime) / (float) CLOCKS_PER_SEC;

    // Run the Classic GPU Implementation and measure the time required
    cudaEvent_t startClassic, stopClassic;
    float timeClassic;
    cudaEventCreate( &startClassic );
    cudaEventCreate( &stopClassic );
    cudaEventRecord(startClassic, 0);
    float * solutionGpuClassic = iterativeGpuClassic(initX, rhs, leftMatrix,
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
    float * solutionGpuSwept = iterativeGpuSwept(initX, rhs, leftMatrix,
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
/*    for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
        printf("%d %f %f %f \n",iGrid, solutionCpu[iGrid],
                             solutionGpuClassic[iGrid],
                             solutionGpuSwept[iGrid]); 
	//assert(solutionGpuClassic[iGrid] == solutionGpuSwept[iGrid]);
	if (abs(solutionGpuClassic[iGrid] - solutionGpuSwept[iGrid]) > 1e-2) {
	    printf("For grid point %d, Classic and Swept give %f and %f respectively\n", iGrid, solutionGpuClassic[iGrid], solutionGpuSwept[iGrid]);
	}
    }
*/
    // Print out time for cpu, classic gpu, and swept gpu approaches
    float cpuTimePerIteration = (cpuTime / nIters) * 1e3;
    float classicTimePerIteration = timeClassic / nIters;
    float sweptTimePerIteration = timeSwept / nIters;
    float timeMultiplier = classicTimePerIteration / sweptTimePerIteration;
    printf("Time needed for the CPU (per iteration): %f ms\n", cpuTimePerIteration);
    printf("Time needed for the Classic GPU (per iteration) is %f ms\n", classicTimePerIteration);
    printf("Time needed for the Swept GPU (per iteration): %f ms\n", sweptTimePerIteration); 

    // Compute the residual of the resulting solution (|b-Ax|)
    float residualCpu = Residual(solutionCpu, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    float residualClassic = Residual(solutionGpuClassic, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    float residualSwept = Residual(solutionGpuSwept, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    printf("Residual of the CPU solution is %f\n", residualCpu);
    printf("Residual of the converged solution is %f\n", residualSwept);
    printf("Residual of GPU Classic result is %f\n", residualClassic); 
  
    // Save residual to a file
    std::ofstream residuals;
    residuals.open("residual-gs.txt",std::ios_base::app);
    residuals << nGrids << "\t" << threadsPerBlock << "\t" << nIters << "\t" << residualSwept << "\n";
    residuals.close(); 

    // Save Results to a file "N tpb Iterations CPUTime/perstep ClassicTime/perstep SweptTime/perStep ClassicTime/SweptTime"
  //  std::ofstream timings;
  //  timings.open("time.txt",std::ios_base::app);
  //  timings << nGrids << "\t" << threadsPerBlock << "\t" << nIters << "\t" << cpuTimePerIteration << "\t" << classicTimePerIteration << "\t" << sweptTimePerIteration << "\t" << timeMultiplier << "\n";
  //  timings.close();

    // Free memory
 //   cudaEventDestroy(startClassic);
 //   cudaEventDestroy(startSwept);
    delete[] initX;
    delete[] rhs;
    delete[] leftMatrix;
    delete[] centerMatrix;
    delete[] rightMatrix;
    delete[] solutionCpu;
    delete[] solutionGpuClassic;
}
