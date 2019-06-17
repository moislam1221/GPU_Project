// Print values from the GPU

// Print solution from GPU
__device__
void printSolutionGPU(const float * solution, int nxGrids, int nyGrids) {

    for (int index = threadIdx.x + blockDim.x * threadIdx.y; index < nxGrids * nyGrids; index += blockDim.x * blockDim.y) {
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            printf("x0Gpu[%d] = %f\n", index, solution[index]);
        }
    }

}

// Print shared memory values
__device__
void printSharedMemoryContents(int blockIDX, int blockIDY, int xLength, int yLength) {

    extern __shared__ float sharedMemory[];

    for (int index = threadIdx.x + blockDim.x * threadIdx.y; index < xLength * yLength; index += blockDim.x * blockDim.y) {
        if (blockIdx.x == blockIDX && blockIdx.y == blockIDY) {
            printf("In shared[%d] = %f\n", index, sharedMemory[index]);
        }
    }
}

