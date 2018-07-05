#include<utility>
#include<stdio.h>
#include<assert.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

__device__ __host__
float jacobiGrid(float leftMatrix, float centerMatrix, float rightMatrix,
                 float leftX, float centerX, float rightX, float centerRhs)
{
    return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX))
         / centerMatrix;
    // return centerX;
    // return (leftX + rightX) / 2.0;
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
    __syncthreads();
}

float * jacobiGpuClassic(const float * initX, const float * rhs,
                         const float * leftMatrix, const float * centerMatrix,
                         const float * rightMatrix, int nGrids, int nIters,
                         const int threadsPerBlock)
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
void __jacobiBlockUpperTriangleFromShared(
		float * xLeftBlock, float *xRightBlock, const float *rhsBlock,
		const float * leftMatrixBlock, const float * centerMatrixBlock,
                const float * rightMatrixBlock, int nGrids, int iGrid)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x; 

    for (int k = 0; k < blockDim.x/2; ++k) {
        if (k > 0) {
            if (threadIdx.x >= k && threadIdx.x <= blockDim.x-k-1) {
                float leftX = (iGrid > 0) ? x0[threadIdx.x - 1] : 0.0f;
                float centerX = x0[threadIdx.x];
                float rightX = (iGrid < nGrids - 1) ? x0[threadIdx.x + 1] : 0.0f;
                x1[threadIdx.x] = jacobiGrid(
				leftMatrixBlock[threadIdx.x],
				centerMatrixBlock[threadIdx.x],
                                rightMatrixBlock[threadIdx.x],
				leftX, centerX, rightX, rhsBlock[threadIdx.x]);
            }
	    float * tmp = x1; x1 = x0; x0 = tmp;
            __syncthreads();
        }
        if (threadIdx.x == k or threadIdx.x == k + 1) {
            xLeftBlock[k + threadIdx.x] = x0[threadIdx.x];
        }
        int reversedIdx = blockDim.x - threadIdx.x - 1;
        if (reversedIdx == k or reversedIdx == k + 1) {
            xRightBlock[k + reversedIdx] = x0[threadIdx.x];
        }
    }
}

__global__
void _jacobiGpuUpperTriangle(float * xLeftGpu, float *xRightGpu,
                             const float * x0Gpu, const float *rhsGpu, 
                             const float * leftMatrixGpu, const float *centerMatrixGpu,
                             const float * rightMatrixGpu, int nGrids)
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
    __syncthreads();

    __jacobiBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
    		                       leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid);
}

__device__ 
void __jacobiBlockLowerTriangleFromShared(
		const float * xLeftBlock, const float *xRightBlock, const float *rhsBlock,
		const float * leftMatrixBlock, const float * centerMatrixBlock,
                const float * rightMatrixBlock, int nGrids, int iGrid)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x;

    for (int k = blockDim.x/2; k > 0; --k) {
	if (k < blockDim.x/2) {
	    if (threadIdx.x >= k && threadIdx.x <= blockDim.x-k-1) {
                float leftX = (iGrid > 0) ? x0[threadIdx.x - 1] : 0.0f;
                float centerX = x0[threadIdx.x];
                float rightX = (iGrid < nGrids - 1) ? x0[threadIdx.x + 1] : 0.0f;
                x1[threadIdx.x] = jacobiGrid(leftMatrixBlock[threadIdx.x],
                                centerMatrixBlock[threadIdx.x],
                                rightMatrixBlock[threadIdx.x],
                                leftX, centerX, rightX, rhsBlock[threadIdx.x]);
	    }
	    __syncthreads();
 	    float * tmp = x1; x1 = x0; x0 = tmp;
        }
        if (threadIdx.x == k-1 or threadIdx.x == k-2) { 
	    x0[threadIdx.x] = xLeftBlock[blockDim.x-k-threadIdx.x-1];
        }
	int reversedIdx = blockDim.x -1 - threadIdx.x;
        if (reversedIdx == k-1 or reversedIdx == k-2) {
            x0[threadIdx.x] = xRightBlock[blockDim.x-k-reversedIdx-1];
        }
    }
}

__global__
void _jacobiGpuShiftedLowerTriangle(float * x0Gpu, float *xLeftGpu,
                             float * xRightGpu, float *rhsGpu, 
                             float * leftMatrixGpu, float *centerMatrixGpu,
                             float * rightMatrixGpu, int nGrids)
{
    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = xRightGpu + blockShift;
    float * xRightBlock = (blockIdx.x == (gridDim.x-1)) ?
                          xLeftGpu : 
                          xLeftGpu + blockShift + blockDim.x;

    int iGrid = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x/2;
    iGrid = (iGrid < nGrids) ? iGrid : threadIdx.x - blockDim.x/2;

    int indexShift = blockDim.x/2;
    float * x0Block = x0Gpu + blockShift + indexShift;
    float * rhsBlock = rhsGpu + blockShift + indexShift;
    float * leftMatrixBlock = leftMatrixGpu + blockShift + indexShift;
    float * centerMatrixBlock = centerMatrixGpu + blockShift + indexShift;
    float * rightMatrixBlock = rightMatrixGpu + blockShift + indexShift;
        
    if (blockDim.x == gridDim.x - 1) {
        memcpy(x0Block + blockDim.x/2, x0Gpu, sizeof(float)*blockDim.x/2);
        memcpy(rhsBlock + blockDim.x/2, rhsGpu, sizeof(float)*blockDim.x/2);
        memcpy(leftMatrixBlock + blockDim.x/2, leftMatrixGpu, sizeof(float)*blockDim.x/2);
        memcpy(centerMatrixBlock + blockDim.x/2, centerMatrixGpu, sizeof(float)*blockDim.x/2);
        memcpy(rightMatrixBlock + blockDim.x/2, rightMatrixGpu, sizeof(float)*blockDim.x/2);
    }
    
    extern __shared__ float sharedMemory[];
    
    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                         leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid);
    
    __syncthreads();

    x0Block[threadIdx.x] = sharedMemory[threadIdx.x];
 
   if (blockIdx.x == gridDim.x - 1) {
       memcpy(x0Gpu, sharedMemory + blockDim.x/2, sizeof(float)*blockDim.x/2);
   }
}

__global__       
void _jacobiGpuShiftedDiamond(float * xLeftGpu, float * xRightGpu,
                              float * rhsGpu, 
			      float * leftMatrixGpu, float * centerMatrixGpu,
                              float * rightMatrixGpu, int nGrids)
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
     
    if (blockIdx.x == gridDim.x-1) {
        memcpy(rhsBlock + blockDim.x/2, rhsGpu, sizeof(float)*blockDim.x/2);
        memcpy(leftMatrixBlock + blockDim.x/2, leftMatrixGpu, sizeof(float)*blockDim.x/2);
        memcpy(centerMatrixBlock + blockDim.x/2, centerMatrixGpu, sizeof(float)*blockDim.x/2);
        memcpy(rightMatrixBlock + blockDim.x/2, rightMatrixGpu, sizeof(float)*blockDim.x/2);
    }

    __syncthreads();
    
    extern __shared__ float sharedMemory[];
    
    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                         leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid);  

    __syncthreads(); 

    __jacobiBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                                       leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid);
}

__global__
void _jacobiGpuDiamond(float * xLeftGpu, float * xRightGpu,
                       const float * rhsGpu,
		       const float * leftMatrixGpu, const float * centerMatrixGpu,
                       const float * rightMatrixGpu, int nGrids)
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

    __syncthreads();

    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                        leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid);

    __syncthreads();
    
    __jacobiBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                                      leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid);
}

float * jacobiGpuSwept(const float * initX, const float * rhs, const float * leftMatrix, const float * centerMatrix, const float * rightMatrix, int nGrids, int nIters, const int threadsPerBlock, int nCycles) { 
    // Determine number of threads and blocks 
    // const int threadsPerBlock = 32;
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

    int sharedFloatsPerBlock = threadsPerBlock * 2;

    for (int i = 0; i < nCycles; i++) {
    _jacobiGpuUpperTriangle <<<nBlocks, threadsPerBlock,
        sizeof(float) * sharedFloatsPerBlock>>>(
                xLeftGpu, xRightGpu,
                x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu,
                rightMatrixGpu, nGrids);
    _jacobiGpuShiftedDiamond <<<nBlocks, threadsPerBlock,
            sizeof(float) * sharedFloatsPerBlock>>>(
                    xLeftGpu, xRightGpu,
                    rhsGpu, leftMatrixGpu, centerMatrixGpu,
                    rightMatrixGpu, nGrids); 
    _jacobiGpuDiamond <<<nBlocks, threadsPerBlock,
                sizeof(float) * sharedFloatsPerBlock>>>(
                        xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu, nGrids); 
    _jacobiGpuShiftedLowerTriangle <<<nBlocks, threadsPerBlock,
                sizeof(float) * sharedFloatsPerBlock>>>(
                        x0Gpu, xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu, nGrids);
    }

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

int main(int argc, char *argv[])
{
    // DEFINE the number of grid points, threads per Block (number of cycles of the swept rule set to 1)
    int nGrids = atoi(argv[1]); 
    const int threadsPerBlock = atoi(argv[2]); 
    int nCycles = 1;

    int nIters = 3*(threadsPerBlock/2-2) * nCycles; 

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

    cudaEvent_t startClassic, stopClassic;
    cudaEvent_t startSwept, stopSwept;
    float timeClassic;
    float timeSwept;
    cudaEventCreate(&startClassic);
    cudaEventCreate(&stopClassic);
    cudaEventCreate(&startSwept);
    cudaEventCreate(&stopSwept);

    float * solutionCpu = jacobiCpu(initX, rhs,
            leftMatrix, centerMatrix, rightMatrix, nGrids, nIters);

    cudaEventRecord(startClassic, 0);
    float * solutionGpuClassic = jacobiGpuClassic(initX, rhs,
            leftMatrix, centerMatrix, rightMatrix, nGrids, nIters, threadsPerBlock);
    cudaEventRecord(stopClassic, 0);
    cudaEventSynchronize(stopClassic);
    cudaEventElapsedTime(&timeClassic, startClassic, stopClassic);

    cudaEventRecord(startSwept, 0);
    float * solutionGpuSwept = jacobiGpuSwept(initX, rhs,
            leftMatrix, centerMatrix, rightMatrix, nGrids, nIters, threadsPerBlock, nCycles);
    cudaEventRecord(stopSwept, 0); 
    cudaEventSynchronize(stopSwept);
    cudaEventElapsedTime(&timeSwept, startSwept, stopSwept);

    for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
        printf("%f %f %f\n", solutionCpu[iGrid],
                             solutionGpuClassic[iGrid],
                             solutionGpuSwept[iGrid]); 
        //assert(solutionGpuClassic[iGrid] == solutionGpuSwept[iGrid]);
	/*if (abs(solutionGpuClassic[iGrid] - solutionGpuSwept[iGrid]) > 1e-5) {
	    printf("For grid point %d, Classic and Swept give %f and %f respectively\n", iGrid, solutionGpuClassic[iGrid], solutionGpuSwept[iGrid]);
	}*/
    }

    printf("The solution for %d grid points, with %d threads per block, and %d iterations is\n", 
           nGrids, threadsPerBlock, nIters);
 
    printf("The time (per time step) required for the Classic Iteration is: %f ms\n", timeClassic/nIters);
    printf("The time (per time step) required for the Swept Rule is: %f ms\n ", timeSwept/(3*(threadsPerBlock/2-2) * nCycles));
    printf("The swept rule takes %f times the time that the Classic Iteration takes\n", timeSwept/timeClassic);

    delete[] initX;
    delete[] rhs;
    delete[] leftMatrix;
    delete[] centerMatrix;
    delete[] rightMatrix;
    delete[] solutionCpu;
    delete[] solutionGpuClassic;
    delete[] solutionGpuSwept;
}
