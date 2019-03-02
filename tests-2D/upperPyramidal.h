__device__ 
void __iterativeBlockUpperPyramidalFromShared(
		double * xLeftBlock, double *xRightBlock, double *xTopBlock, double *xBottomBlock, const double *rhsBlock,
		const double * leftMatrixBlock, const double * centerMatrixBlock,
                const double * rightMatrixBlock, const double * topMatrixBlock, const double * bottomMatrixBlock,
	       	int nxGrids, int nyGrids, int iGrid, int method)
{
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory, * x1 = sharedMemory + blockDim.x * blockDim.y; 

    int idx = threadIdx.x + blockDim.x * threadIdx.y;
    
//    printf("Idx %d, initial solution %f\n", idx, x0[idx]);
    for (int k = 0; k <= blockDim.x/2-1; ++k) {
  //         printf("Time step %d\n", k); 
        if (threadIdx.x >= k && threadIdx.x <= blockDim.x-k-1 && threadIdx.y >= k && threadIdx.y <= blockDim.y-k-1) {
        
	// Bottom 
        if (threadIdx.y == k)
        {
	    xBottomBlock[threadIdx.x-k+(2*k)*(blockDim.x-(k-1))] = x0[idx];
     	}
	if (threadIdx.y == k + 1)
    	{
            xBottomBlock[threadIdx.x-k+(2*k)*(blockDim.x-k) + blockDim.x] = x0[idx];
    	}

	// Top
        if (threadIdx.y == blockDim.x - 1 - k)
    	{
	    xTopBlock[threadIdx.x-k+(2*k)*(blockDim.x-(k-1))] = x0[idx];
    	}
	if (threadIdx.y == blockDim.x - 2 - k)
    	{
            xTopBlock[threadIdx.x-k+(2*k)*(blockDim.x-k) + blockDim.x] = x0[idx];
      	}
	
        // Left
        if (threadIdx.x == k)
        {
            xLeftBlock[threadIdx.y-k + (2*k)*(blockDim.x-(k-1))] = x0[idx];
        }
        if (threadIdx.x == k + 1)
        {
            xLeftBlock[threadIdx.y-k + (2*k)*(blockDim.x-(k)) + blockDim.x] = x0[idx];
        }

        // Right
        if (threadIdx.x == blockDim.x - 1 - k)
        {
            xRightBlock[threadIdx.y-k + (2*k)*(blockDim.x-(k-1))] = x0[idx];
        }
        if (threadIdx.x == blockDim.x - 2 - k)
        {
            xRightBlock[threadIdx.y-k + (2*k)*(blockDim.x-(k)) + blockDim.x] = x0[idx];
        }    

	}

        if (threadIdx.x > k && threadIdx.x < blockDim.x-k-1 && threadIdx.y > k && threadIdx.y < blockDim.y-k-1) {
	    
	    double leftX = ((iGrid % nxGrids) == 0) ? 0.0f : x0[idx - 1];
            double centerX = x0[idx];
            double rightX = (((iGrid + 1) % nxGrids) == 0) ? 0.0f : x0[idx + 1];
	    double bottomX = (iGrid < nxGrids) ? 0.0f : x0[idx - blockDim.x];
            double topX = (iGrid < nxGrids*(nyGrids-1)) ? x0[idx + blockDim.x] : 0.0f;
            
	    double leftMat = leftMatrixBlock[idx];
            double centerMat = centerMatrixBlock[idx];
            double rightMat = rightMatrixBlock[idx];
      	    double topMat = topMatrixBlock[idx];
            double bottomMat = bottomMatrixBlock[idx];
            double rhs = rhsBlock[idx];
	    
            if (k % 2 == 0) {
                x1[idx] = centerX; /*iterativeOperation(leftMat, centerMat, rightMat, topMat, bottomMat, 
				                     leftX, centerX, rightX, topX, bottomX, rhs, iGrid, method); */
            }
	    else {
	        x1[idx] = centerX;/* iterativeOperation2(leftMat, centerMat, rightMat, topMat, bottomMat,
				                      leftX, centerX, rightX, topX, bottomX, rhs, iGrid, method); */
	    }
        }
        
	__syncthreads();	
    	double * tmp = x1; x1 = x0; x0 = tmp;
    
    } 

    //printf("Idx %d, Top %f, Bottom %f, Left %f, Right %f\n", idx, xTopBlock[idx], xBottomBlock[idx], xLeftBlock[idx], xRightBlock[idx]);    
    //printf("Idx %d, SharedMemoryValue %f\n", idx, x0[idx]);
    double * tmp = x1; x1 = x0; x0 = tmp;

}

__global__
void _iterativeGpuUpperPyramidal(double * xLeftGpu, double *xRightGpu, double * xTopGpu, double * xBottomGpu,
                             const double * x0Gpu, const double *rhsGpu, 
                             const double * leftMatrixGpu, const double *centerMatrixGpu, const double * rightMatrixGpu, 
			     const double * topMatrixGpu, const double * bottomMatrixGpu, int nxGrids, int nyGrids, int method)
{
    int xShift = blockDim.x * blockIdx.x;
    int yShift = blockDim.y * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;

    int numBridgeElemPerBlock = 2 * blockDim.x/2 * (blockDim.x/2 + 1);
    int blockID = blockIdx.x + gridDim.x * blockIdx.y;
    int bridgeShift = blockID * numBridgeElemPerBlock;

    double * xLeftBlock = xLeftGpu + bridgeShift;
    double * xRightBlock = xRightGpu + bridgeShift;
    double * xTopBlock = xTopGpu + bridgeShift;
    double * xBottomBlock = xBottomGpu + bridgeShift;

    const double * x0Block = x0Gpu + blockShift;
    const double * rhsBlock = rhsGpu + blockShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift;

    int idx = threadIdx.x + threadIdx.y * nxGrids;
    int iGrid = blockShift + idx;
    
    extern __shared__ double sharedMemory[];
    sharedMemory[threadIdx.x + threadIdx.y * blockDim.x] = x0Block[threadIdx.x + threadIdx.y * nxGrids];

    __iterativeBlockUpperPyramidalFromShared(xLeftBlock, xRightBlock, xTopBlock, xBottomBlock, rhsBlock,
    		                             leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
					     nxGrids, nyGrids, iGrid, method);
}
