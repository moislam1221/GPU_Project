__device__
void __gaussSeidelUpdateKernel(const float * rhsBlock, const float * leftMatrixBlock, const float * centerMatrixBlock, const float * rightMatrixBlock, const int nGrids, const int nSub, const int numIterations)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory; 
    
    int blockShift = nSub-2;
    const float omega = 2.0 / (1.0 + sin(PI / (float)(nGrids - 1)));
    int index;
    for (int k = 0; k < numIterations; k++) {
        if (k % 2 == 0) {
            index = 2 * (threadIdx.x + 1);
        }    
        else if (k % 2 == 1) {
            index = 2 * threadIdx.x + 1;
        }    
        int I = blockIdx.x * blockShift + index;
        if (I < nGrids-1) {
            x0[index] = relaxedJacobi(leftMatrixBlock[index], centerMatrixBlock[index], rightMatrixBlock[index], x0[index-1], x0[index], x0[index+1], rhsBlock[index], omega);
        }
        
        __syncthreads();
        if (I == 193 || I == 192) {
            printf("iGrid %d: (handled by blockIdx.x %d and threadIdx.x %d) Hello, my leftX centerX and rightX are %f %f %f\n", I, blockIdx.x, threadIdx.x,x0[index-1], x0[index], x0[index+1]);
        }
    }

/*    
    int blockShift = nSub-2;
    int stride = 2*blockDim.x;
    const float omega = 2.0 / (1.0 + sin(PI / (float)(nGrids - 1)));
    for (int k = 0; k < numIterations; k++) {
        for (int index = 2*(threadIdx.x) + 1 + (k % 2); index < nSub-1; index += stride) {
            int I = blockIdx.x * blockShift + index;
            if (I < nGrids-1) {
                x0[index] = relaxedJacobi(leftMatrixBlock[index], centerMatrixBlock[index], rightMatrixBlock[index], x0[index-1], x0[index], x0[index+1], rhsBlock[index], omega);
            }
        }
        __syncthreads();
    }
*/


}

__global__
void _gaussSeidelUpdate(float * x0Gpu, const float * rhsGpu, const float * leftMatrixGpu, const float * centerMatrixGpu, const float * rightMatrixGpu, const int nGrids, const int nSub, const int numIterations)
{
    // Move to shared memory
    extern __shared__ float sharedMemory[];

    int blockShift = nSub-2;
    int stride = blockDim.x;
    for (int index = threadIdx.x; index < nSub; index += stride) {
        int I = index + blockIdx.x * blockShift;
        if (I < nGrids) {
            sharedMemory[index] = x0Gpu[I];
        }
        __syncthreads();
    }
__syncthreads(); 
    const float * rhsBlock = rhsGpu + blockShift * blockIdx.x;
    const float * leftMatrixBlock = leftMatrixGpu + blockShift * blockIdx.x;
    const float * centerMatrixBlock = centerMatrixGpu + blockShift * blockIdx.x;
    const float * rightMatrixBlock = rightMatrixGpu + blockShift * blockIdx.x;

    __syncthreads();

    // Update all inner points
    __gaussSeidelUpdateKernel(rhsBlock, leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, nGrids, nSub, numIterations);

__syncthreads(); 
    // Move back to global memory
    for (int index = threadIdx.x + 1; index < nSub-1; index += stride) {
        int I = blockIdx.x * blockShift + index;
        if (I < nGrids-1) {
            x0Gpu[I] = sharedMemory[index];
            printf("In blockIdx.x %d, sharedMemory[index=%d] (global index %d) = %f\n", blockIdx.x, index, I, sharedMemory[index]);
        }
        __syncthreads();
    }
  __syncthreads(); 
    __syncthreads();
}

float * gaussSeidelGpuRectangular(const float * initX, const float * rhs, const float * leftMatrix, const float * centerMatrix, const float * rightMatrix, const int nGrids, const int threadsPerBlock, const int cycles, const int nInnerUpdates)
{
    // Number of grid points handled by a subdomain
    const int nSub = 2*(threadsPerBlock) + 2;

    // Number of blocks necessary
    const int numBlocks = ceil(((float)nGrids-2.0) / (float)(2*threadsPerBlock));

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
    const int sharedBytes = nSub * sizeof(float);

    float * solution = new float[nGrids];
    // Call kernel to allocate to sharedmemory and update points
    for (int step = 0; step < cycles; step++) {
        printf("Step %d now:\n", step);
        _gaussSeidelUpdate <<<numBlocks, threadsPerBlock, sharedBytes>>> (x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, nSub, nInnerUpdates);
        cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids, cudaMemcpyDeviceToHost);
        float residual = Residual(solution, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
        cudaDeviceSynchronize(); 
        //printf("residual = %f\n", residual);
    }

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

int gaussSeidelGpuRectangularIterationCount(const float * initX, const float * rhs, const float * leftMatrix, const float * centerMatrix, const float * rightMatrix, const int nGrids, const int threadsPerBlock, const int TOL, const int nInnerUpdates)
{
    // Number of grid points handled by a subdomain
    const int nSub = 2*(threadsPerBlock) + 2;
    
    // Number of blocks necessary
    const int numBlocks = ceil(((float)nGrids-2.0) / (float)(2*threadsPerBlock));

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
    const int sharedBytes = nSub * sizeof(float);

    // Call kernel to allocate to sharedmemory and update points
    float residual = 100.0;
    int nIters = 0;
    float * solution = new float[nGrids];
    while (residual > TOL) {
    //for (int i = 0; i < 10; i++) {
        _gaussSeidelUpdate <<<numBlocks, threadsPerBlock, sharedBytes>>> (x0Gpu, rhsGpu, leftMatrixGpu, centerMatrixGpu, rightMatrixGpu, nGrids, nSub, nInnerUpdates);
        nIters++;
        cudaMemcpy(solution, x0Gpu, sizeof(float) * nGrids, cudaMemcpyDeviceToHost);
        residual = Residual(solution, rhs, leftMatrix, centerMatrix, rightMatrix, nGrids);
        //printf("residual = %f\n", residual);
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
