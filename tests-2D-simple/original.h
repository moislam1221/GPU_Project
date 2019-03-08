__device__
double increment(double centerX) {
     return centerX + 100.;
}

__device__
double jacobi(const double leftMatrix, const double centerMatrix, const double rightMatrix, const double topMatrix, const double bottomMatrix,
              const double leftX, const double centerX, const double rightX, const double topX, const double bottomX,
              const double centerRhs) {
    return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX));
}

__device__
void __iterativeBlockUpdateToLeftRight(double * xLeftBlock, double * xRightBlock, const double *rhsBlock, 
                             const double * leftMatrixBlock, const double *centerMatrixBlock, const double * rightMatrixBlock, 
			     const double * topMatrixBlock, const double * bottomMatrixBlock, int nxGrids, int nyGrids, int iGrid, int method)
{
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory; 
    int elemPerBlock = blockDim.x * blockDim.y;
    double * x1 = sharedMemory + elemPerBlock;
    int maxSteps = 1;
    int idx = threadIdx.x + threadIdx.y * blockDim.x;

    if ((threadIdx.x >= 1 && threadIdx.x <= blockDim.x-2) && (threadIdx.y >= 1 && threadIdx.y <= blockDim.y-2)) {
        for (int k = 0; k < maxSteps; k++) {
            // Define necessary constants
            double centerX = x0[idx];
	    // Perform update
	    x1[idx] = increment(centerX);
            // Synchronize
	    __syncthreads();
	    double * tmp; tmp = x0; x0 = x1;
	}
    }
    
    // Save xLeft, xRight, xTop, xBottom
    if (idx < (blockDim.x * blockDim.y)/2) {
        xLeftBlock[idx] = x0[threadIdx.x * blockDim.x + threadIdx.y];
	xRightBlock[idx] = x0[(blockDim.x-1-threadIdx.y) + threadIdx.x * blockDim.x];
    }
}

__device__
void __iterativeBlockUpdateToNorthSouth(double * xTopBlock, double * xBottomBlock, const double *rhsBlock, 
                             const double * leftMatrixBlock, const double *centerMatrixBlock, const double * rightMatrixBlock, 
			     const double * topMatrixBlock, const double * bottomMatrixBlock, int nxGrids, int nyGrids, int iGrid, int method)
{
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory; 
    int elemPerBlock = blockDim.x * blockDim.y;
    double * x1 = sharedMemory + elemPerBlock;
    int maxSteps = 1;
    int idx = threadIdx.x + threadIdx.y * blockDim.x;

    if ((threadIdx.x >= 1 && threadIdx.x <= blockDim.x-2) && (threadIdx.y >= 1 && threadIdx.y <= blockDim.y-2)) {
        for (int k = 0; k < maxSteps; k++) {
            // Define necessary constants
            double centerX = x0[idx];
	    // Perform update
	    x1[idx] = increment(centerX);
            // Synchronize
	    __syncthreads();
	    double * tmp; tmp = x0; x0 = x1;
	}
    }

    // Return values for xTop and xBottom here
    if (idx < (blockDim.x * blockDim.y)/2) {
        xBottomBlock[idx] = x0[idx];
	xTopBlock[idx] = x0[threadIdx.x + (blockDim.x)*(blockDim.x-1-threadIdx.y)];
    }
}

__global__
void _iterativeGpuOriginal(double * xLeftGpu, double *xRightGpu,
                             const double * x0Gpu, const double *rhsGpu, 
                             const double * leftMatrixGpu, const double *centerMatrixGpu, const double * rightMatrixGpu, 
			     const double * topMatrixGpu, const double * bottomMatrixGpu, int nxGrids, int nyGrids, int method)
{

    int xShift = blockDim.x * blockIdx.x;
    int yShift = blockDim.y * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;

    const double * x0Block = x0Gpu + blockShift;
    const double * rhsBlock = rhsGpu + blockShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift;

    int numElementsPerBlock = blockDim.x * blockDim.y;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = (numElementsPerBlock*blockID)/2;
    double * xLeftBlock = xLeftGpu + arrayShift;
    double * xRightBlock = xRightGpu + arrayShift;
    
    int idx = threadIdx.x + threadIdx.y * nxGrids;
    int iGrid = blockShift + idx;
    extern __shared__ double sharedMemory[];
    sharedMemory[threadIdx.x + threadIdx.y * blockDim.x] = x0Block[threadIdx.x + threadIdx.y * nxGrids];

    sharedMemory[threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y] = x0Block[threadIdx.x + threadIdx.y * nxGrids];
    
    __iterativeBlockUpdateToLeftRight(xLeftBlock, xRightBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, iGrid, method);
}

__global__
void _iterativeGpuHorizontalShift(double * xLeftGpu, double *xRightGpu, double * xTopGpu, double * xBottomGpu,
                                  const double * x0Gpu, const double *rhsGpu, 
                                  const double * leftMatrixGpu, const double *centerMatrixGpu, const double * rightMatrixGpu, 
			          const double * topMatrixGpu, const double * bottomMatrixGpu, int nxGrids, int nyGrids, int method)
{
    int xShift = blockDim.x * blockIdx.x;
    int yShift = blockDim.y * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    int horizontalShift = blockDim.x/2;

    const double * rhsBlock = rhsGpu + blockShift + horizontalShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift + horizontalShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift + horizontalShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift + horizontalShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift + horizontalShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift + horizontalShift;

    int numElementsPerBlock = (blockDim.x * blockDim.y)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
    
    double * xLeftBlock =  xRightGpu + arrayShift;
    double * xRightBlock = (blockIdx.x != gridDim.x-1) ?
                           xLeftGpu + arrayShift + numElementsPerBlock :
			   xLeftGpu + (numElementsPerBlock * blockIdx.y * gridDim.x);
    double * xBottomBlock = xBottomGpu + arrayShift;
    double * xTopBlock = xTopGpu + arrayShift;

    int idx = threadIdx.x + threadIdx.y * nxGrids;
    int iGrid = blockShift + idx;

    extern __shared__ double sharedMemory[];
    idx = threadIdx.x + threadIdx.y * blockDim.x;

    if (threadIdx.x < blockDim.x/2) {
        sharedMemory[idx] = xLeftBlock[threadIdx.y + (blockDim.x/2-1-threadIdx.x)*blockDim.y];
    }
    else {
        sharedMemory[idx] = xRightBlock[threadIdx.y + (threadIdx.x-(blockDim.x/2))*blockDim.y];
    }
    
    __iterativeBlockUpdateToNorthSouth(xTopBlock, xBottomBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, iGrid, method);
}

__global__
void _iterativeGpuVerticalandHorizontalShift(double * xLeftGpu, double *xRightGpu, double * xTopGpu, double * xBottomGpu,
                                const double * x0Gpu, const double *rhsGpu, 
                                const double * leftMatrixGpu, const double *centerMatrixGpu, const double * rightMatrixGpu, 
			        const double * topMatrixGpu, const double * bottomMatrixGpu, int nxGrids, int nyGrids, int method)
{
    int xShift = blockDim.x * blockIdx.x;
    int yShift = blockDim.y * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    int verticalShift = blockDim.y/2 * nxGrids;

    const double * rhsBlock = rhsGpu + blockShift + verticalShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift + verticalShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift + verticalShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift + verticalShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift + verticalShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift + verticalShift;

    int numElementsPerBlock = (blockDim.x * blockDim.y)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
    double * xBottomBlock = xTopGpu + arrayShift;
    double * xTopBlock = (blockIdx.y != gridDim.y-1) ?
                         xBottomGpu + arrayShift + numElementsPerBlock * gridDim.x :
			 xBottomGpu + (numElementsPerBlock * blockIdx.x);
    
    double * xLeftBlock = xLeftGpu + arrayShift;
    double * xRightBlock = xRightGpu + arrayShift;

    int idx = threadIdx.x + threadIdx.y * nxGrids;
    int iGrid = blockShift + idx;

    extern __shared__ double sharedMemory[];
    idx = threadIdx.x + threadIdx.y * blockDim.x;

    if (idx < numElementsPerBlock) {
        sharedMemory[idx] = xBottomBlock[threadIdx.x + (blockDim.y/2-1-threadIdx.y)*blockDim.x];
    }
    else {
        sharedMemory[idx] = xTopBlock[threadIdx.x + (threadIdx.y-(blockDim.y/2))*blockDim.x];
    }
    
    __iterativeBlockUpdateToLeftRight(xLeftBlock, xRightBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, iGrid, method);
}


__global__
void _iterativeGpuVerticalShift(double * xLeftGpu, double *xRightGpu, double * xTopGpu, double * xBottomGpu,
                                const double * x0Gpu, const double *rhsGpu, 
                                const double * leftMatrixGpu, const double *centerMatrixGpu, const double * rightMatrixGpu, 
			        const double * topMatrixGpu, const double * bottomMatrixGpu, int nxGrids, int nyGrids, int method)
{
    int xShift = blockDim.x * blockIdx.x;
    int yShift = blockDim.y * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    int verticalShift = blockDim.y/2 * nxGrids;

    const double * rhsBlock = rhsGpu + blockShift + verticalShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift + verticalShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift + verticalShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift + verticalShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift + verticalShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift + verticalShift;

    int numElementsPerBlock = (blockDim.x * blockDim.y)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
    
    double * xRightBlock =  xLeftGpu + arrayShift;
    double * xLeftBlock = (blockIdx.x != 0) ?
                           xRightGpu + arrayShift - numElementsPerBlock :
    			   xRightGpu + numElementsPerBlock * ((gridDim.x-1) + blockIdx.y * gridDim.x);
    
    double * xBottomBlock = xBottomGpu + arrayShift;
    double * xTopBlock = xTopGpu + arrayShift;

    int idx = threadIdx.x + threadIdx.y * nxGrids;
    int iGrid = blockShift + idx;

    extern __shared__ double sharedMemory[];
    idx = threadIdx.x + threadIdx.y * blockDim.x;

    if (threadIdx.x < blockDim.x/2) {
        sharedMemory[idx] = xLeftBlock[threadIdx.y + (blockDim.x/2-1-threadIdx.x)*blockDim.y];
    }
    else {
        sharedMemory[idx] = xRightBlock[threadIdx.y + (threadIdx.x-(blockDim.x/2))*blockDim.y];
    }

    __iterativeBlockUpdateToNorthSouth( xTopBlock, xBottomBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, iGrid, method);
}

__global__
void _finalSolution(double * xTopGpu, double * xBottomGpu, double * x0Gpu, int nxGrids)
{

    extern __shared__ double sharedMemory[];
    int numElementsPerBlock = (blockDim.x * blockDim.y)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;

    double * xTopBlock = xBottomGpu + arrayShift;
    double * xBottomBlock = (blockIdx.y != 0) ?
			    xTopGpu + (blockIdx.x + (blockIdx.y-1) * gridDim.x) * numElementsPerBlock :
			    xTopGpu + (gridDim.x * (gridDim.y-1) + blockIdx.x) * numElementsPerBlock;

    int xShift = blockDim.x * blockIdx.x;
    int yShift = blockDim.y * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    double * x0Block = x0Gpu + blockShift;

    int idx = threadIdx.x + threadIdx.y * blockDim.x;

    if (idx < (blockDim.x * blockDim.y)/2) {
        sharedMemory[idx + numElementsPerBlock] = xTopBlock[idx]; 
	sharedMemory[threadIdx.x + (blockDim.x)*(blockDim.x/2-1-threadIdx.y)] = xBottomBlock[idx];
    }

    double * x0 = x0Gpu + blockShift;

    idx = threadIdx.x + threadIdx.y * nxGrids;
    x0[threadIdx.x + threadIdx.y * nxGrids] = sharedMemory[threadIdx.x + threadIdx.y * blockDim.x];    
}

