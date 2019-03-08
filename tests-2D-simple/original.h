__device__
void __iterativeBlockUpdate(double * xLeftBlock, double * xRightBlock, const double *rhsBlock, 
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
	    // Perform update
            double centerX = x0[idx];
	    x1[idx] = centerX + 100;
	    __syncthreads();
	    double * tmp; tmp = x0; x0 = x1;
	}
    }
    
    // Save xLeft, xRight, xTop, xBottom
    //printf("In blockIdx.x %d and blockIdx.y %d, idx %d, x0[idx=%d] is %f \n", blockIdx.x, blockIdx.y, idx, idx, x0[idx]);
    if (idx < (blockDim.x * blockDim.y)/2) {
//        xBottomBlock[idx] = x0[idx];
//	xTopBlock[idx] = x0[threadIdx.x + (blockDim.x)*(blockDim.x-1-threadIdx.y)];
        xLeftBlock[idx] = x0[threadIdx.x * blockDim.x + threadIdx.y];
	xRightBlock[idx] = x0[(blockDim.x-1-threadIdx.y) + threadIdx.x * blockDim.x];
    }
}

__device__
void __iterativeBlockUpdateHorizontal(double * xLeftBlock, double * xRightBlock, double * xTopBlock, double * xBottomBlock, const double *rhsBlock, 
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
	    // Perform update
            double centerX = x0[idx];
	    x1[idx] = centerX + 100;
	    __syncthreads();
	    double * tmp; tmp = x0; x0 = x1;
	}
    }

    // Save xLeft, xRight, xTop, xBottom
    printf("HELLO: In blockIdx.x %d and blockIdx.y %d, idx %d, x0[idx=%d] is %f \n", blockIdx.x, blockIdx.y, idx, idx, x0[idx]);
/*    if (idx < (blockDim.x * blockDim.y)/2) {
        //xBottomBlock[idx] = x0[idx];
        //xTopBlock[idx] = x0[(elemPerBlock-1)-idx];
        //xLeftBlock[idx] = x0[threadIdx.x * blockDim.x + threadIdx.y];
        //xRightBlock[idx] = x0[(elemPerBlock-1)-(threadIdx.x * blockDim.x + threadIdx.y)];
        //xLeftBlock[idx] = x0[threadIdx.x * blockDim.x + threadIdx.y];
	xLeftBlock[idx] = x0[(blockDim.x-1-threadIdx.y) + threadIdx.x * blockDim.x];
        //xRightBlock[idx] = x0[(elemPerBlock-1)-(threadIdx.x * blockDim.x + threadIdx.y)];
	xRightBlock[idx] = x0[(blockDim.x-1-threadIdx.y) + threadIdx.x * blockDim.x];
    }
*/
/*    if (threadIdx.x < blockDim.x/2) {
        xLeftBlock[threadIdx.y + (blockDim.x/2-1-threadIdx.x)*blockDim.y] = x0[idx];
    }
    else {
        xRightBlock[threadIdx.y + (threadIdx.x-(blockDim.x/2))*blockDim.y] = x0[idx];
    }
*/
    // Return values for xTop and xBottom here
        if (idx < (blockDim.x * blockDim.y)/2) {
            xBottomBlock[idx] = x0[idx];
	    xTopBlock[idx] = x0[threadIdx.x + (blockDim.x)*(blockDim.x-1-threadIdx.y)];
        }
//    printf("HELLO: In blockIdx.x %d and blockIdx.y %d, idx %d, xBottom[idx=%d] is %f \n", blockIdx.x, blockIdx.y, idx, idx, xBottomBlock[idx]);
}

__device__
void __iterativeBlockUpdateVertical(double * xLeftBlock, double * xRightBlock, double *xTopBlock, double * xBottomBlock, const double *rhsBlock, 
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
	    // Perform update
            double centerX = x0[idx];
	    x1[idx] = centerX + 100;
	    __syncthreads();
	    double * tmp; tmp = x0; x0 = x1;
	}
    }

    // Save xLeft, xRight, xTop, xBottom
//    printf("HELLO: In blockIdx.x %d and blockIdx.y %d, idx %d, x0[idx=%d] is %f \n", blockIdx.x, blockIdx.y, idx, idx, x0[idx]);
/*    if (idx < (blockDim.x * blockDim.y)/2) {
        //xBottomBlock[idx] = x0[idx];
        //xTopBlock[idx] = x0[(elemPerBlock-1)-idx];
        //xLeftBlock[idx] = x0[threadIdx.x * blockDim.x + threadIdx.y];
        //xRightBlock[idx] = x0[(elemPerBlock-1)-(threadIdx.x * blockDim.x + threadIdx.y)];
        //xLeftBlock[idx] = x0[threadIdx.x * blockDim.x + threadIdx.y];
	xLeftBlock[idx] = x0[(blockDim.x-1-threadIdx.y) + threadIdx.x * blockDim.x];
        //xRightBlock[idx] = x0[(elemPerBlock-1)-(threadIdx.x * blockDim.x + threadIdx.y)];
	xRightBlock[idx] = x0[(blockDim.x-1-threadIdx.y) + threadIdx.x * blockDim.x];
    }

    int numElementsPerBlock = (blockDim.x * blockDim.y)/2;
    if (idx < numElementsPerBlock) {
        xBottomBlock[threadIdx.x + (blockDim.y/2-1-threadIdx.y)*blockDim.x] = x0[idx];
    }
    else {
        xTopBlock[threadIdx.x + (threadIdx.y-(blockDim.y/2))*blockDim.x] = x0[idx];
    }
*/
 /*   
    if (idx < (blockDim.x * blockDim.y)/2) {
        xBlock[idx] = x0[idx];
        xTopBlock[idx] = x0[threadIdx.x + (blockDim.x)*(blockDim.x-1-threadIdx.y)];
    }
*/

    if (idx < (blockDim.x * blockDim.y)/2) {
        xLeftBlock[idx] = x0[threadIdx.x * blockDim.x + threadIdx.y];
	xRightBlock[idx] = x0[(blockDim.x-1-threadIdx.y) + threadIdx.x * blockDim.x];
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
    //double * xTopBlock = xTopGpu + arrayShift;
    //double * xBottomBlock = xBottomGpu + arrayShift;
    
    int idx = threadIdx.x + threadIdx.y * nxGrids;
    int iGrid = blockShift + idx;
    extern __shared__ double sharedMemory[];
    sharedMemory[threadIdx.x + threadIdx.y * blockDim.x] = x0Block[threadIdx.x + threadIdx.y * nxGrids];

    //printf("The blockShift is %d and we put x0Block[idx=%d] into sharedMemory[idx=%d]\n", blockShift, threadIdx.x + threadIdx.y * nxGrids, threadIdx.x + threadIdx.y * blockDim.x);
    sharedMemory[threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y] = x0Block[threadIdx.x + threadIdx.y * nxGrids];
    //printf("In blockIdx.x %d and blockIdx.y %d, the %dth entry of sharedMemory is %f\n", blockIdx.x, blockIdx.y, threadIdx.x + threadIdx.y * blockDim.x, sharedMemory[threadIdx.x + threadIdx.y * blockDim.x]);
    __iterativeBlockUpdate(xLeftBlock, xRightBlock, rhsBlock,
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

    // const double * x0Block = x0Gpu + blockShift + horizontalShift;
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
//    printf("In blockIdx.x %d and blockIdx.y %d, the values in sharedMemory[idx=%d] is %f\n", blockIdx.x, blockIdx.y, idx, sharedMemory[idx]);
    __iterativeBlockUpdateHorizontal(xLeftBlock, xRightBlock, xTopBlock, xBottomBlock, rhsBlock,
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

    // const double * x0Block = x0Gpu + blockShift + verticalShift;
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
//    printf("In blockIdx.x %d and blockIdx.y %d, the values in sharedMemory[idx=%d] is %f\n", blockIdx.x, blockIdx.y, idx, sharedMemory[idx]);
    __iterativeBlockUpdateVertical(xLeftBlock, xRightBlock, xTopBlock, xBottomBlock, rhsBlock,
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

    // const double * x0Block = x0Gpu + blockShift + verticalShift;
    const double * rhsBlock = rhsGpu + blockShift + verticalShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift + verticalShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift + verticalShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift + verticalShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift + verticalShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift + verticalShift;

    int numElementsPerBlock = (blockDim.x * blockDim.y)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
//    double * xBottomBlock = xTopGpu + arrayShift;
//    double * xTopBlock = (blockIdx.y != gridDim.y-1) ?
//                         xBottomGpu + arrayShift + numElementsPerBlock * gridDim.x :
//			 xBottomGpu + (numElementsPerBlock * blockIdx.x);
    
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

    printf("VERTICAL SHIFT:In blockIdx.x %d and blockIdx.y %d, the values in sharedMemory[idx=%d] is %f\n", blockIdx.x, blockIdx.y, idx, sharedMemory[idx]);
    __iterativeBlockUpdateHorizontal(xLeftBlock, xRightBlock, xTopBlock, xBottomBlock, rhsBlock,
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

  //  printf("FINAL METHOD: In blockx %d and blocky %d, xTopBlock: %f, xBottomBlock: %f\n", blockIdx.x, blockIdx.y, xTopBlock[idx], xBottomBlock[idx]);

    // Fill in x0 based on xBottom and xTop
/*    if (idx < numElementsPerBlock) {
	x0Block[numElementsPerBlock + idx] = xTopBlock[idx];
        x0Block[threadIdx.x + (blockDim.y/2-1-threadIdx.y)] = xBottomBlock[idx];
    }
*/
    if (idx < (blockDim.x * blockDim.y)/2) {
        sharedMemory[idx + numElementsPerBlock] = xTopBlock[idx]; 
	sharedMemory[threadIdx.x + (blockDim.x)*(blockDim.x/2-1-threadIdx.y)] = xBottomBlock[idx];
    }
    printf("FINAL METHOD: In blockx %d and blocky %d, x0Block[idx=%d] is  %f\n", blockIdx.x, blockIdx.y, idx, sharedMemory[idx]);

    double * x0 = x0Gpu + blockShift;

    idx = threadIdx.x + threadIdx.y * nxGrids;
    x0[threadIdx.x + threadIdx.y * nxGrids] = sharedMemory[threadIdx.x + threadIdx.y * blockDim.x];
    
}

