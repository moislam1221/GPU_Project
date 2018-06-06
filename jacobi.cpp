#include<stdio.h>

#include"UpperTriangular.h"

__device__ __host__
float jacobiGrid(float leftMatrix, float centerMatrix, float rightMatrix,
                 float leftX, float centerX, float rightX, float centerRhs)
{
    return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX))
         / centerMatrix;
}

float * jacobiCpu(const float * initX, const float * rhs,
                  const float * leftMatrix, const float * centerMatrix,
                  const float * rightMatrix, int nGrids, int nIters)
{
    float * x0 = new float[nGrids];
    float * x1 = new float[nGrids];
    memcpy(x0, initX, sizeof(float) * nGrids);
    for (int iIter = 0; iIter < nIters; ++ iIter) {
        for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
            float leftX = (iGrid > 0) ? x0[iGrid - 1] : 0.0f;
            float centerX = x0[iGrid];
            float rightX = (iGrid < nGrids - 1) ? x0[iGrid + 1] : 0.0f;
            x1[iGrid] = jacobiGrid(leftMatrix[iGrid], centerMatrix[iGrid],
                                   rightMatrix[iGrid], leftX, centerX, rightX,
                                   rhs[iGrid]);
        }
        float * tmp = x0; x0 = x1; x1 = tmp;
    }
    delete[] x1;
    return x0;
}

__global__
void _jacobiGpuClassicIteration(float * x1,
                         const float * x0, const float * rhs,
                         const float * leftMatrix, const float * centerMatrix,
                         const float * rightMatrix, int nGrids)
{
    int iGrid = blockIdx.x * blockDim.x + threadIdx.x;
    if (iGrid < nGrids) {
        float leftX = (iGrid > 0) ? x0[iGrid - 1] : 0.0f;
        float centerX = x0[iGrid];
        float rightX = (iGrid < nGrids - 1) ? x0[iGrid + 1] : 0.0f;
        x1[iGrid] = jacobiGrid(leftMatrix[iGrid], centerMatrix[iGrid],
                               rightMatrix[iGrid], leftX, centerX, rightX,
                               rhs[iGrid]);
    }
}

float * jacobiGpuClassic(const float * initX, const float * rhs,
                         const float * leftMatrix, const float * centerMatrix,
                         const float * rightMatrix, int nGrids, int nIters)
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
    int threadsPerBlock = 256;
    int nBlocks = (int)ceil(nGrids / (float)threadsPerBlock);
    for (int iIter = 0; iIter < nIters; ++iIter) {
        _jacobiGpuClassicIteration<<<nBlocks, threadsPerBlock>>>(
                x1Gpu, x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu,
                rightMatrixGpu, nGrids);
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
void __jacobiBlockLowerTriangleFromShared(
		const float * xLeftBlock, const float *xRightBlock, const float *rhsBlock,
		const float * leftMatrixBlock, const float * centerMatrixBlock,
                const float * rightMatrixBlock)
{
    // printf("I have entered block %d and thread %d\n", blockIdx.x, threadIdx.x);

    printf("The value in xLeft in thread %d is %f\n", threadIdx.x, xLeftBlock[threadIdx.x]);
    
    printf("The value in xRight in thread %d is %f\n", threadIdx.x, xRightBlock[threadIdx.x]);

    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x;
    float * tmp; // helpful to perform swap
    
    // printf("My x0 in thread %d is now %f\n", threadIdx.x, x0[threadIdx.x]);

    for (int k = blockDim.x/2; k > 0; --k) {
	if (k < blockDim.x/2) {
	    if (threadIdx.x >= k && threadIdx.x <= blockDim.x-k-1) {
                float leftX = x0[threadIdx.x - 1];
                float centerX = x0[threadIdx.x];
                float rightX = x0[threadIdx.x + 1];
                x1[threadIdx.x] = jacobiGrid(leftMatrixBlock[threadIdx.x],
                                centerMatrixBlock[threadIdx.x],
                                rightMatrixBlock[threadIdx.x],
                                leftX, centerX, rightX, rhsBlock[threadIdx.x]);
		//printf("I am on step %d of LowerTriangle, on thread %d, and the solution is %f\n", blockDim.x/2-k, threadIdx.x, x1[threadIdx.x]);
            }
            // Error: calling a host function from a device function
	    // std::swap(x0, x1);
 	    tmp = x1; 
            x1 = x0;
            x0 = tmp;
            __syncthreads();
        }
	//printf("My x0 in thread %d is now %f\n", threadIdx.x, x0[threadIdx.x]);
        if (threadIdx.x == k-1 or threadIdx.x == k-2) { // k-1 k-2
	    x0[threadIdx.x] = xLeftBlock[blockDim.x-k-threadIdx.x-1];
        }
	//printf("My x0 in thread %d is now %f\n", threadIdx.x, x0[threadIdx.x]);
	int reversedIdx = blockDim.x -1 - threadIdx.x;
        if (reversedIdx == k-1 or reversedIdx == k-2) {
            x0[threadIdx.x] = xRightBlock[blockDim.x-k-reversedIdx-1];
        }
	printf("My x0 in thread %d is now %f\n", threadIdx.x, x0[threadIdx.x]);
	//printf("In shared Memory, thread %d we have %f\n", threadIdx.x, sharedMemory[threadIdx.x]);
	//printf("In shared Memory, thread %d we have %f\n", threadIdx.x, sharedMemory[threadIdx.x+blockDim.x]);
    }

    // Make sure first half of shared memory has updated solution (odd number of iterations are performed)
    sharedMemory[threadIdx.x] = sharedMemory[threadIdx.x + blockDim.x];

}


__global__
void _jacobiGpuLowerTriangle(const float * xLeftGpu, const float *xRightGpu,
                             float * x0Gpu, float *rhsGpu, 
                             float * leftMatrixGpu, float *centerMatrixGpu,
                             float * rightMatrixGpu)
{
    /*printf("Upon entering Lower Triangle In GpuDiamond, in thread %d my xLeftBlock and xRightBlock values are %f and %f\n", threadIdx.x, xLeftBlock[threadIdx.x], xRightBlock[threadIdx.x]);*/

    int blockShift = blockDim.x * blockIdx.x;
    const float * xLeftBlock = xRightGpu + blockShift;
    // The xRight of block n is the xLeft of block n+1, unless we are at the last block, then take the xLeft of block 1
    const float * xRightBlock = (blockIdx.x == (gridDim.x-1)) ?
                          xLeftGpu : 
                          xLeftGpu + blockShift + blockDim.x;

    int indexShift = blockDim.x/2;
    float * rhsBlock = rhsGpu + blockShift + indexShift;
    float * leftMatrixBlock = leftMatrixGpu + blockShift + indexShift;
    float * centerMatrixBlock = centerMatrixGpu + blockShift + indexShift;
    float * rightMatrixBlock = rightMatrixGpu + blockShift + indexShift;

    // Need to make second part of rhs, matrix blocks point to the beginning of these arrays
    if (blockIdx.x == gridDim.x - 1) {
	rhsBlock[8] = rhsGpu[0];
 	rhsBlock[9] = rhsGpu[1];
	rhsBlock[10] = rhsGpu[2];
	rhsBlock[11] = rhsGpu[3];
	rhsBlock[12] = rhsGpu[4];
	rhsBlock[13] = rhsGpu[5];
	rhsBlock[14] = rhsGpu[6];
	rhsBlock[15] = rhsGpu[7];
	leftMatrixBlock[8] = leftMatrixGpu[0];
	leftMatrixBlock[9] = leftMatrixGpu[1];
	leftMatrixBlock[10] = leftMatrixGpu[2];
	leftMatrixBlock[11] = leftMatrixGpu[3];
	leftMatrixBlock[12] = leftMatrixGpu[4];
	leftMatrixBlock[13] = leftMatrixGpu[5];
	leftMatrixBlock[14] = leftMatrixGpu[6];
	leftMatrixBlock[15] = leftMatrixGpu[7];
	centerMatrixBlock[8] = centerMatrixGpu[0];
	centerMatrixBlock[9] = centerMatrixGpu[1];
	centerMatrixBlock[10] = centerMatrixGpu[2];
	centerMatrixBlock[11] = centerMatrixGpu[3];
	centerMatrixBlock[12] = centerMatrixGpu[4];
	centerMatrixBlock[13] = centerMatrixGpu[5];
	centerMatrixBlock[14] = centerMatrixGpu[6];
	centerMatrixBlock[15] = centerMatrixGpu[7];
	rightMatrixBlock[8] = rightMatrixGpu[0];
	rightMatrixBlock[9] = rightMatrixGpu[1];
	rightMatrixBlock[10] = rightMatrixGpu[2];
	rightMatrixBlock[11] = rightMatrixGpu[3];
	rightMatrixBlock[12] = rightMatrixGpu[4];
	rightMatrixBlock[13] = rightMatrixGpu[5];
	rightMatrixBlock[14] = rightMatrixGpu[6];
	rightMatrixBlock[15] = rightMatrixGpu[7];
	// rhsBlock[indexShift] = *rhsGpu;
        //leftMatrixBlock+indexShift = leftMatrixGpu;
        //centerMatrixBlock+indexShift = centerMatrixGpu;
	//rightMatrixBlock+indexShift = rightMatrixGpu;
    }

    /*int blockShift = blockDim.x * blockIdx.x;
    const float * xLeftBlock = xLeftGpu + blockShift;
    const float * xRightBlock = xRightGpu + blockShift;

    printf("Upon entering Lower Triangle In GpuDiamond, in thread %d my xLeftBlock and xRightBlock values are %f and %f\n", threadIdx.x, xLeftBlock[threadIdx.x], xRightBlock[threadIdx.x]);

    float * x0Block = x0Gpu + blockShift;
    float * rhsBlock = rhsGpu + blockShift;
    float * leftMatrixBlock = leftMatrixGpu + blockShift;
    float * centerMatrixBlock = centerMatrixGpu + blockShift;
    float * rightMatrixBlock = rightMatrixGpu + blockShift;*/

    printf("Upon entering Lower Triangle In GpuDiamond, in thread %d my xLeftBlock and xRightBlock values are %f and %f\n", threadIdx.x, xLeftBlock[threadIdx.x], xRightBlock[threadIdx.x]);

    // 1 - Allocate shared memory and move x0Gpu to appropriate spot in shared memory
    extern __shared__ float sharedMemory[];
    // sharedMemory[threadIdx.x] = x0Block[threadIdx.x];
    // __syncthreads();
    // sharedMemory[threadIdx.x] = 1.024221;
    // sharedMemory[threadIdx.x+blockDim.x] = 1.024221;

    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
		                       leftMatrixBlock, centerMatrixBlock, rightMatrixBlock);

    printf("After finishing Lower Triangle (the last leg), in thread %d my value in Shared Memory are %f and %f\n", threadIdx.x, sharedMemory[threadIdx.x], sharedMemory[threadIdx.x+blockDim.x]);


    __syncthreads();
    // x0Block[threadIdx.x] = sharedMemory[threadIdx.x];

}

__global__       
void _jacobiGpuShiftedDiamond(float * xLeftGpu, float * xRightGpu,
                              float * rhsGpu, 
			      float * leftMatrixGpu, float * centerMatrixGpu,
                              float * rightMatrixGpu)
{
    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = xRightGpu + blockShift;
    // The xRight of block n is the xLeft of block n+1, unless we are at the last block, then take the xLeft of block 1
    float * xRightBlock = (blockIdx.x == (gridDim.x-1)) ?
                          xLeftGpu : 
                          xLeftGpu + blockShift + blockDim.x;

    int indexShift = blockDim.x/2;
    float * rhsBlock = rhsGpu + blockShift + indexShift;
    float * leftMatrixBlock = leftMatrixGpu + blockShift + indexShift;
    float * centerMatrixBlock = centerMatrixGpu + blockShift + indexShift;
    float * rightMatrixBlock = rightMatrixGpu + blockShift + indexShift;

    // Need to make second part of rhs, matrix blocks point to the beginning of these arrays
    if (blockIdx.x == gridDim.x - 1) {
	rhsBlock[8] = rhsGpu[0];
 	rhsBlock[9] = rhsGpu[1];
	rhsBlock[10] = rhsGpu[2];
	rhsBlock[11] = rhsGpu[3];
	rhsBlock[12] = rhsGpu[4];
	rhsBlock[13] = rhsGpu[5];
	rhsBlock[14] = rhsGpu[6];
	rhsBlock[15] = rhsGpu[7];
	leftMatrixBlock[8] = leftMatrixGpu[0];
	leftMatrixBlock[9] = leftMatrixGpu[1];
	leftMatrixBlock[10] = leftMatrixGpu[2];
	leftMatrixBlock[11] = leftMatrixGpu[3];
	leftMatrixBlock[12] = leftMatrixGpu[4];
	leftMatrixBlock[13] = leftMatrixGpu[5];
	leftMatrixBlock[14] = leftMatrixGpu[6];
	leftMatrixBlock[15] = leftMatrixGpu[7];
	centerMatrixBlock[8] = centerMatrixGpu[0];
	centerMatrixBlock[9] = centerMatrixGpu[1];
	centerMatrixBlock[10] = centerMatrixGpu[2];
	centerMatrixBlock[11] = centerMatrixGpu[3];
	centerMatrixBlock[12] = centerMatrixGpu[4];
	centerMatrixBlock[13] = centerMatrixGpu[5];
	centerMatrixBlock[14] = centerMatrixGpu[6];
	centerMatrixBlock[15] = centerMatrixGpu[7];
	rightMatrixBlock[8] = rightMatrixGpu[0];
	rightMatrixBlock[9] = rightMatrixGpu[1];
	rightMatrixBlock[10] = rightMatrixGpu[2];
	rightMatrixBlock[11] = rightMatrixGpu[3];
	rightMatrixBlock[12] = rightMatrixGpu[4];
	rightMatrixBlock[13] = rightMatrixGpu[5];
	rightMatrixBlock[14] = rightMatrixGpu[6];
	rightMatrixBlock[15] = rightMatrixGpu[7];
	// rhsBlock[indexShift] = *rhsGpu;
        //leftMatrixBlock+indexShift = leftMatrixGpu;
        //centerMatrixBlock+indexShift = centerMatrixGpu;
	//rightMatrixBlock+indexShift = rightMatrixGpu;
    }
    extern __shared__ float sharedMemory[];
	
    // Need to reset shared Memory! Seems that what was left over from before in shared memory remains. Making all entries equal to one as per the initial condition
    // sharedMemory[threadIdx.x] = 1.0;
    // sharedMemory[threadIdx.x + blockDim.x] = 1.0;

    // printf("I am entering Lower Triangle From Shared in Shifted Diamond from thread %d and my value in Shared Memory is %f\n", threadIdx.x, sharedMemory[threadIdx.x]);
    
    // Perform down triangle similar to up triangle
    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                         leftMatrixBlock, centerMatrixBlock, rightMatrixBlock);

    printf("After finishing Lower Triangle From Shared, in thread %d my value in Shared Memory is %f\n", threadIdx.x, sharedMemory[threadIdx.x]);

    
    
    // Perform up triangle
    __jacobiBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                                         leftMatrixBlock, centerMatrixBlock,
                                         rightMatrixBlock, jacobiGrid);

    printf("After finishing Upper Triangle From Shared, in thread %d my xLeftBlock and xRightBlock values are %f and %f\n", threadIdx.x, xLeftBlock[threadIdx.x], xRightBlock[threadIdx.x]);

}

__global__
void _jacobiGpuDiamond(float * xLeftGpu, float * xRightGpu,
                       const float * rhsGpu,
		       const float * leftMatrixGpu, const float * centerMatrixGpu,
                       const float * rightMatrixGpu)
{
    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = (blockIdx.x == 0) ? 
                         xRightGpu + (blockDim.x * (gridDim.x-1)) : 
                         xRightGpu + blockShift - blockDim.x;
    float * xRightBlock = xLeftGpu + blockShift;

    const float * rhsBlock = rhsGpu + blockShift;
    const float * leftMatrixBlock = leftMatrixGpu + blockShift;
    const float * centerMatrixBlock = centerMatrixGpu + blockShift;
    const float * rightMatrixBlock = rightMatrixGpu + blockShift;
    
    extern __shared__ float sharedMemory[];
    
    //printf("Shared memory in GpuDiamond prints %f\n", sharedMemory[threadIdx.x]);

    // Perform down triangle similar to up triangle
    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                        leftMatrixBlock, centerMatrixBlock, rightMatrixBlock);

    printf("After finishing Lower Triangle From Shared in, in thread %d my value in Shared Memory is %f\n", threadIdx.x, sharedMemory[threadIdx.x]);
    
    // Perform up triangle
    __jacobiBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                                      leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, jacobiGrid);

    printf("After finishing Upper Triangle In GpuDiamond, in thread %d my xLeftBlock and xRightBlock values are %f and %f\n", threadIdx.x, xLeftBlock[threadIdx.x], xRightBlock[threadIdx.x]);
}

__global__
void _jacobiShiftedLowerTriangle(float * xLeftGpu, float * xRightGpu,
                       const float * rhsGpu,
		       const float * leftMatrixGpu, const float * centerMatrixGpu,
                       const float * rightMatrixGpu, int nGrids)
{
   // Same Implementation as ShiftedDiamond without call to UpperTriangle
   // Solution should exist in first half of shared Memory
}    

float * jacobiGpuSwept(const float * initX, const float * rhs, const float * leftMatrix, const float * centerMatrix, const float * rightMatrix, int nGrids, int nIters) { 
    // Determine number of threads and blocks 
    const int threadsPerBlock = 16;
    const int nBlocks = (int)ceil(nGrids / (float)threadsPerBlock);

    // Allocate memory for solution and inputs
    float *xLeftGpu, *xRightGpu;
    cudaMalloc(&xLeftGpu, sizeof(float) * threadsPerBlock * nBlocks);
    cudaMalloc(&xRightGpu, sizeof(float) * threadsPerBlock * nBlocks);
    float * x0Gpu, * rhsGpu, * leftMatrixGpu, * rightMatrixGpu, * centerMatrixGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nGrids);
    cudaMalloc(&rhsGpu, sizeof(float) * nGrids);
    cudaMalloc(&leftMatrixGpu, sizeof(float) * nGrids);
    cudaMalloc(&centerMatrixGpu, sizeof(float) * nGrids);
    cudaMalloc(&rightMatrixGpu, sizeof(float) * nGrids);

    // Allocate memory in the GPU
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu, leftMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu, centerMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu, rightMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);

    int sharedFloatsPerBlock = threadsPerBlock * 2;
    _jacobiGpuUpperTriangle <<<nBlocks, threadsPerBlock,
        sizeof(float) * sharedFloatsPerBlock>>>(
                xLeftGpu, xRightGpu,
                x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu,
                rightMatrixGpu, jacobiGrid);
    _jacobiGpuShiftedDiamond <<<nBlocks, threadsPerBlock,
            sizeof(float) * sharedFloatsPerBlock>>>(
                    xLeftGpu, xRightGpu,
                    rhsGpu, leftMatrixGpu, centerMatrixGpu,
                    rightMatrixGpu); 
    _jacobiGpuDiamond <<<nBlocks, threadsPerBlock,
                sizeof(float) * sharedFloatsPerBlock>>>(
                        xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu); 
    _jacobiGpuLowerTriangle <<<nBlocks, threadsPerBlock,
                sizeof(float) * sharedFloatsPerBlock>>>(
                        x0Gpu, xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu); 
    



    /*for (int iIter = 0; iIter < nIters; iIter += threadsPerBlock) {
        _jacobiGpuShiftedDiamond <<<nBlocks, threadsPerBlock,
            sizeof(float) * sharedFloatsPerBlock>>>(
                    xLeftGpu, xRightGpu,
                    rhsGpu, leftMatrixGpu, centerMatrixGpu,
                    rightMatrixGpu, nGrids, nIters - iIter);
        if (iIter + threadsPerBlock >= nIters) {
            //_jacobiGpuLowerTriangle <<<nBlocks, threadsPerBlock,
            //    sizeof(float) * sharedFloatsPerBlock>>>(
            //            x0Gpu, xLeftGpu, xRightGpu,
            //            rhsGpu, leftMatrixGpu, centerMatrixGpu,
            //            rightMatrixGpu, nGrids,
            //            nIters - iIter + threadsPerBlock / 2);
        }
        else {
            _jacobiGpuDiamond <<<nBlocks, threadsPerBlock,
                sizeof(float) * sharedFloatsPerBlock>>>(
                        xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu, nGrids,
                        nIters - iIter + threadsPerBlock / 2);
        }
    } */

    float * solution = new float[nGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids,
            cudaMemcpyDeviceToHost);

    cudaFree(xLeftGpu);
    cudaFree(xRightGpu);
    cudaFree(rhsGpu);
    cudaFree(leftMatrixGpu);
    cudaFree(centerMatrixGpu);
    cudaFree(rightMatrixGpu);

    return solution;
}

int main()
{
    int nGrids = 16;
    int nIters = 1;
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

    float * solutionCpu = jacobiCpu(initX, rhs,
            leftMatrix, centerMatrix, rightMatrix, nGrids, nIters);
    float * solutionGpuClassic = jacobiGpuClassic(initX, rhs,
            leftMatrix, centerMatrix, rightMatrix, nGrids, nIters);
    float * solutionGpuSwept = jacobiGpuSwept(initX, rhs,
            leftMatrix, centerMatrix, rightMatrix, nGrids, nIters); 

    for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
        printf("%f %f %f\n", solutionCpu[iGrid],
                             solutionGpuClassic[iGrid],
                             solutionGpuSwept[iGrid]);
    }

    delete[] initX;
    delete[] rhs;
    delete[] leftMatrix;
    delete[] centerMatrix;
    delete[] rightMatrix;
    delete[] solutionCpu;
    delete[] solutionGpuClassic;
     delete[] solutionGpuSwept;
}
