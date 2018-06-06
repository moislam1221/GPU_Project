__device__ 
void __jacobiBlockUpperTriangleFromShared(
        float * xLeftBlock, float *xRightBlock, const float *rhsBlock,
        const float * leftMatrixBlock, const float * centerMatrixBlock,
        const float * rightMatrixBlock)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x;

    for (int k = 0; k < blockDim.x/2; ++k) {
        if (k > 0) {
            if (threadIdx.x >= k && threadIdx.x <= blockDim.x-k-1) {
                float leftX = x0[threadIdx.x - 1];
                float centerX = x0[threadIdx.x];
                float rightX = x0[threadIdx.x + 1];
                x1[threadIdx.x] = gridOp(leftMatrixBlock[threadIdx.x],
                                         centerMatrixBlock[threadIdx.x],
                                         rightMatrixBlock[threadIdx.x],
                                         leftX, centerX, rightX,
                                         rhsBlock[threadIdx.x]);
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
                             const float * rightMatrixGpu)
{
    int blockShift = blockDim.x * blockIdx.x;
    float * xLeftBlock = xLeftGpu + blockShift;
    float * xRightBlock = xRightGpu + blockShift;
    const float * x0Block = x0Gpu + blockShift;
    const float * rhsBlock = rhsGpu + blockShift;
    const float * leftMatrixBlock = leftMatrixGpu + blockShift;
    const float * centerMatrixBlock = centerMatrixGpu + blockShift;
    const float * rightMatrixBlock = rightMatrixGpu + blockShift;
  
    extern __shared__ float sharedMemory[];
    sharedMemory[threadIdx.x] = x0Block[threadIdx.x];
    __syncthreads();

    __jacobiBlockUpperTriangleFromShared(xLeftBlock, xRightBlock, rhsBlock,
                                         leftMatrixBlock, centerMatrixBlock,
                                         rightMatrixBlock);
}
