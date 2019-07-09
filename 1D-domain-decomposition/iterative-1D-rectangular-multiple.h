__device__
void __updateKernelM(const float * rhsBlock, const float * leftMatrixBlock, const float * centerMatrixBlock, const float * rightMatrixBlock, const int nGrids, const int nSub, const int numIterations, const int method)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory, * x1 = sharedMemory + nSub;
    
    int blockShift = nSub-2;
    int stride = blockDim.x;
    for (int k = 0; k < numIterations; k++) {
        for (int index = threadIdx.x + 1; index < nSub-1; index += stride) {
            int I = blockIdx.x * blockShift + index;
            if (I < nGrids-1) {
                x1[index] = iterativeOperation(leftMatrixBlock[index], centerMatrixBlock[index], rightMatrixBlock[index], x0[index-1], x0[index], x0[index+1], rhsBlock[index], 5, method);
            }
        }
        __syncthreads();
        float * tmp = x1; x1 = x0; x0 = tmp;
    }

/*    
    for (int k = 0; k < numIterations; k++) {
        int i = threadIdx.x + 1;
        int I = blockIdx.x * blockDim.x + i;
        if (I < nGrids-1) {
            x1[i] = iterativeOperation(leftMatrixBlock[i], centerMatrixBlock[i], rightMatrixBlock[i], x0[i-1], x0[i], x0[i+1], rhsBlock[i], 5, method);
        }
        __syncthreads();
        float * tmp = x1; x1 = x0; x0 = tmp;
    }
*/
}

__global__
void _updateM(float * x0Gpu, const float * rhsGpu, const float * leftMatrixGpu, const float * centerMatrixGpu, const float * rightMatrixGpu, const int nGrids, const int nSub, const int numIterations, const int method)
{
    // Move to shared memory
    extern __shared__ float sharedMemory[];

    int blockShift = nSub-2;
    int stride = blockDim.x;
    for (int index = threadIdx.x; index < nSub; index += stride) {
        int I = index + blockIdx.x * blockShift;
        if (I < nGrids) {
            sharedMemory[index] = x0Gpu[I];
            sharedMemory[index + nSub] = x0Gpu[I];
        }
    }

/*
    const int I = threadIdx.x + blockDim.x * blockIdx.x;
    const int i = threadIdx.x;
    if (I < nGrids) {
        sharedMemory[i] = x0Gpu[I];
        sharedMemory[i + nSub] = x0Gpu[I];
    }
    const int I2 = blockDim.x + (threadIdx.x + blockDim.x * blockIdx.x);
    const int i2 = blockDim.x + threadIdx.x;
    if (i2 < nSub && I2 < nGrids) { 
        sharedMemory[i2] = x0Gpu[I2];
        sharedMemory[i2 + nSub] = x0Gpu[I2];
    }
*/
    const float * rhsBlock = rhsGpu + blockShift * blockIdx.x;
    const float * leftMatrixBlock = leftMatrixGpu + blockShift * blockIdx.x;
    const float * centerMatrixBlock = centerMatrixGpu + blockShift * blockIdx.x;
    const float * rightMatrixBlock = rightMatrixGpu + blockShift * blockIdx.x;

    // Synchronize all threads before entering kernel
    __syncthreads();    

    // Update all inner points
    __updateKernelM(rhsBlock, leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, nSub, numIterations, method);

    // Move back to global memory
    for (int index = threadIdx.x + 1; index < nSub-1; index += stride) {
        int I = blockIdx.x * blockShift + index;
        if (I < nGrids-1) {
            x0Gpu[I] = sharedMemory[index];
        }
    }
    
/*
    // Move back to global memory
    if ((I+1) < nGrids) {
        x0Gpu[I+1] = sharedMemory[i+1];
    }
*/
}

float * iterativeGpuRectangularMultiple(const float * initX, const float * rhs, const float * leftMatrix, const float * centerMatrix, const float * rightMatrix, const int nGrids, const int threadsPerBlock, const int cycles, const int nInnerUpdates, int method, const int pointsPerThread)
{
    // Number of grid points handled by a subdomain
    const int nSub = pointsPerThread*(threadsPerBlock) + 2;

    // Number of blocks necessary
    const int numBlocks = ceil(((float)nGrids-2.0) / (float)(pointsPerThread*threadsPerBlock));

    // Allocate GPU memory via cudaMalloc
    float * x0Gpu, * rhsGpu, * leftMatrixGpu, * rightMatrixGpu, * centerMatrixGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nGrids);
    cudaMalloc(&rhsGpu, sizeof(float) * nGrids);
    cudaMalloc(&leftMatrixGpu, sizeof(float) * nGrids);
    cudaMalloc(&rightMatrixGpu, sizeof(float) * nGrids);
    cudaMalloc(&centerMatrixGpu, sizeof(float) * nGrids);
    
    // Copy contents to GPU
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu, leftMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu, centerMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu, rightMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);

    // Define amount of shared memory needed
    const int sharedBytes = 2 * nSub * sizeof(float);

    // Call kernel to allocate to sharedmemory and update points
    for (int step = 0; step < cycles; step++) {
        _updateM <<<numBlocks, threadsPerBlock, sharedBytes>>> (x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, nSub, nInnerUpdates, method);
    }

    float * solution = new float[nGrids];
    cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids,
            cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(x0Gpu);
    cudaFree(rhsGpu);
    cudaFree(leftMatrixGpu);
    cudaFree(centerMatrixGpu);
    cudaFree(rightMatrixGpu);

    return solution;
}

int iterativeGpuRectangularMultipleIterationCount(const float * initX, const float * rhs, const float * leftMatrix, const float * centerMatrix, const float * rightMatrix, const int nGrids, const int threadsPerBlock, const int TOL, const int nInnerUpdates, int method, const int pointsPerThread)
{
    // Number of grid points handled by a subdomain
    const int nSub = pointsPerThread*(threadsPerBlock) + 2;

    // Number of blocks necessary
    const int numBlocks = ceil(((float)nGrids-2.0) / (float)(pointsPerThread*threadsPerBlock));

    // Allocate GPU memory via cudaMalloc
    float * x0Gpu, * rhsGpu, * leftMatrixGpu, * rightMatrixGpu, * centerMatrixGpu;
    cudaMalloc(&x0Gpu, sizeof(float) * nGrids);
    cudaMalloc(&rhsGpu, sizeof(float) * nGrids);
    cudaMalloc(&leftMatrixGpu, sizeof(float) * nGrids);
    cudaMalloc(&rightMatrixGpu, sizeof(float) * nGrids);
    cudaMalloc(&centerMatrixGpu, sizeof(float) * nGrids);
    
    // Copy contents to GPU
    cudaMemcpy(x0Gpu, initX, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(rhsGpu, rhs, sizeof(float) * nGrids, cudaMemcpyHostToDevice);
    cudaMemcpy(leftMatrixGpu, leftMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(centerMatrixGpu, centerMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);
    cudaMemcpy(rightMatrixGpu, rightMatrix, sizeof(float) * nGrids,
            cudaMemcpyHostToDevice);

    // Define amount of shared memory needed
    const int sharedBytes = 2 * nSub * sizeof(float);

    // Call kernel to allocate to sharedmemory and update points
    float residual = 100.0;
    int nIters = 0;
    float * solution = new float[nGrids];
    while (residual > TOL) {
        _update <<<numBlocks, threadsPerBlock, sharedBytes>>> (x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, nSub, nInnerUpdates, method);
         nIters++;
         cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids, cudaMemcpyDeviceToHost);
         residual = Residual(solution, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
    }

    // Clean up
    delete[] solution;
    cudaFree(x0Gpu);
    cudaFree(rhsGpu);
    cudaFree(leftMatrixGpu);
    cudaFree(centerMatrixGpu);
    cudaFree(rightMatrixGpu);

    return nIters;
}
