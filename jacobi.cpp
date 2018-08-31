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
        }
        if (threadIdx.x == k or threadIdx.x == k + 1) {
            xLeftBlock[k + threadIdx.x] = x0[threadIdx.x];
        }
        int reversedIdx = blockDim.x - threadIdx.x - 1;
        if (reversedIdx == k or reversedIdx == k + 1) {
            xRightBlock[k + reversedIdx] = x0[threadIdx.x];
        }
	__syncthreads();
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
 	    float * tmp = x1; x1 = x0; x0 = tmp;
        }
        if (threadIdx.x == k-1 or threadIdx.x == k-2) { 
	    x0[threadIdx.x] = xLeftBlock[blockDim.x-k-threadIdx.x-1];
        }
	int reversedIdx = blockDim.x -1 - threadIdx.x;
        if (reversedIdx == k-1 or reversedIdx == k-2) {
            x0[threadIdx.x] = xRightBlock[blockDim.x-k-reversedIdx-1];
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
    x1[threadIdx.x] = jacobiGrid(leftMatrixBlock[threadIdx.x],
                                centerMatrixBlock[threadIdx.x],
                                rightMatrixBlock[threadIdx.x],
                                leftX, centerX, rightX, rhsBlock[threadIdx.x]);
    float * tmp = x1; x1 = x0; x0 = tmp;
}

__global__
void _jacobiGpuLowerTriangle(float * x0Gpu, float *xLeftGpu,
                             float * xRightGpu, float *rhsGpu, 
                             float * leftMatrixGpu, float *centerMatrixGpu,
                             float * rightMatrixGpu, int nGrids)
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
    
    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                         leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid);

    x0Block[threadIdx.x] = sharedMemory[threadIdx.x];

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
    
    extern __shared__ float sharedMemory[];
    
    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                         leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid);

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
    
    extern __shared__ float sharedMemory[];
    
    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                         leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid);  

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

    __jacobiBlockLowerTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                        leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid);
    
    __jacobiBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                                      leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, iGrid);
}

float * jacobiGpuSwept(const float * initX, const float * rhs, const float * leftMatrix, const float * centerMatrix, const float * rightMatrix, int nGrids, int nIters, const int threadsPerBlock) { 
    
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

    for (int i = 0; i < nIters/threadsPerBlock-1; i++) {
    _jacobiGpuDiamond <<<nBlocks, threadsPerBlock,
                sizeof(float) * sharedFloatsPerBlock>>>(
                        xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu, nGrids); 
    _jacobiGpuShiftedDiamond <<<nBlocks, threadsPerBlock,
            sizeof(float) * sharedFloatsPerBlock>>>(
                    xLeftGpu, xRightGpu,
                    rhsGpu, leftMatrixGpu, centerMatrixGpu,
                    rightMatrixGpu, nGrids); 
    }

    _jacobiGpuLowerTriangle <<<nBlocks, threadsPerBlock,
                sizeof(float) * sharedFloatsPerBlock>>>(
                        x0Gpu, xLeftGpu, xRightGpu,
                        rhsGpu, leftMatrixGpu, centerMatrixGpu,
                        rightMatrixGpu, nGrids);

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
    int nGrids = atoi(argv[1]); 
    const int threadsPerBlock = atoi(argv[2]); 
    int nIters = atoi(argv[3]);

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

    // CPU Implementation
    float * solutionCpu = jacobiCpu(initX, rhs,
            leftMatrix, centerMatrix, rightMatrix, nGrids, nIters);

    // Classic/Naive GPU Implementation
    float * solutionGpuClassic = jacobiGpuClassic(initX, rhs,
            leftMatrix, centerMatrix, rightMatrix, nGrids, nIters, threadsPerBlock);

    // Swept GPU Implementation
    float * solutionGpuSwept = jacobiGpuSwept(initX, rhs,
            leftMatrix, centerMatrix, rightMatrix, nGrids, nIters, threadsPerBlock);

    for (int iGrid = 0; iGrid < nGrids; ++iGrid) {
        printf("%d %f %f %f\n", iGrid, solutionCpu[iGrid],
                             solutionGpuClassic[iGrid],
                             solutionGpuSwept[iGrid]); 
    }

    printf("The solution for %d grid points, with %d threads per block, and %d iterations is\n", 
           nGrids, threadsPerBlock, nIters);

    delete[] initX;
    delete[] rhs;
    delete[] leftMatrix;
    delete[] centerMatrix;
    delete[] rightMatrix;
    delete[] solutionCpu;
    delete[] solutionGpuClassic;
    delete[] solutionGpuSwept;
}
