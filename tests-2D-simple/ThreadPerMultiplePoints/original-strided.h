__device__
double increment(double centerX) {
     return centerX + 100.;
}

__device__
double jacobi(const double leftMatrix, const double centerMatrix, const double rightMatrix, const double topMatrix, const double bottomMatrix,
              const double leftX, const double centerX, const double rightX, const double topX, const double bottomX,
              const double centerRhs) {
    return (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
}

double normFromRow(double leftMatrix, double centerMatrix, double rightMatrix, double topMatrix, double bottomMatrix, double leftX, double centerX, double rightX,  double topX, double bottomX, double centerRhs) 
{
    return centerRhs - (leftMatrix*leftX + centerMatrix*centerX + rightMatrix*rightX + topMatrix*topX + bottomMatrix*bottomX);
}

double Residual(const double * solution, const double * rhs, const double * leftMatrix, const double * centerMatrix, const double * rightMatrix, const double * topMatrix, const double * bottomMatrix, int nxGrids, int nyGrids)
{
    int nDofs = nxGrids * nyGrids;
    double residual = 0.0;  

    for (int iGrid = 0; iGrid < nDofs; iGrid++) {
        double leftX = ((iGrid % nxGrids) == 0) ? 0.0 : solution[iGrid-1];
        double centerX = solution[iGrid];
        double rightX = ((iGrid + 1) % nxGrids == 0) ? 0.0 : solution[iGrid+1];
        double topX = (iGrid < nxGrids * (nyGrids - 1)) ? solution[iGrid + nxGrids] : 0.0;
        double bottomX = (iGrid >= nxGrids) ?  solution[iGrid-nxGrids] : 0.0;
        double residualContributionFromRow = normFromRow(leftMatrix[iGrid], centerMatrix[iGrid], rightMatrix[iGrid], topMatrix[iGrid], bottomMatrix[iGrid], leftX, centerX, rightX, topX, bottomX, rhs[iGrid]);
	residual = residual + residualContributionFromRow * residualContributionFromRow;
//	printf("For gridpoint %d, residual contribution is %f\n", iGrid, residualContributionFromRow);
    }
    residual = sqrt(residual);
    return residual;
}

__device__
void __iterativeBlockUpdateToLeftRight(double * xLeftBlock, double * xRightBlock, const double *rhsBlock, 
                             const double * leftMatrixBlock, const double *centerMatrixBlock, const double * rightMatrixBlock, 
			     const double * topMatrixBlock, const double * bottomMatrixBlock, int nxGrids, int nyGrids, int iGrid, int method, 
                             int subdomainLength)
{
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory; 
    int elemPerBlock = subdomainLength * subdomainLength;
    double * x1 = sharedMemory + elemPerBlock;
    int maxSteps = 1;
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    for (int k = 0; k < maxSteps; k++) {
        for (int idx = index; idx < elemPerBlock; idx += stride) {
           if ((idx % subdomainLength != 0) && ((idx+1) % subdomainLength != 0) && (idx > subdomainLength-1) && (idx < elemPerBlock-(subdomainLength-1))) {
                // Define necessary constants
                double centerRhs = rhsBlock[idx];
                double leftMatrix = leftMatrixBlock[idx];
                double centerMatrix = centerMatrixBlock[idx];
                double rightMatrix = rightMatrixBlock[idx];
                double topMatrix = topMatrixBlock[idx];
                double bottomMatrix = bottomMatrixBlock[idx];
                /* double leftX = ((iGrid % nxGrids) == 0) ? 0.0 : x0[idx-1];
                double centerX = x0[idx];
                double rightX = ((iGrid + 1) % nxGrids == 0) ? 0.0 : x0[idx+1];
                double topX = (iGrid < nxGrids * (nyGrids - 1)) ? x0[idx+blockDim.x] : 0.0;
                double bottomX = (iGrid >= nxGrids) ?  x0[idx-blockDim.x] : 0.0;
                */
                double leftX = x0[idx-1];
                double centerX = x0[idx];
                double rightX = x0[idx+1];
                double topX = x0[idx+blockDim.x];
                double bottomX = x0[idx-blockDim.x];
               
                //printf("In iGrid %d, idx = %d, left %f, right %f, center %f, top %f, bottom %f\n", iGrid, idx, leftX, rightX, centerX, topX, bottomX	);
	        // Perform updatE
   	        x1[idx] = increment(centerX);
                //x1[idx] = jacobi(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
                //                 leftX, centerX, rightX, topX, bottomX, centerRhs);
                // Synchronize
	        __syncthreads();
                printf("My idx is %d\n", idx);
                
//                double * tmp; tmp = x0; x0 = x1;
//              printf("Updated value in idx = %d is %f\n", idx, x1[idx]);
	    }
        }
                double * tmp; tmp = x0; x0 = x1;
    }

    for (int idx = index; idx < elemPerBlock/2; idx += stride) {
        xLeftBlock[idx] = x0[subdomainLength * (idx % subdomainLength) + (idx/subdomainLength)];
        xRightBlock[idx] = x0[subdomainLength * (idx % subdomainLength) - (idx/subdomainLength) + (subdomainLength-1)];
    }

    // Save xLeft, xRight, xTop, xBottom
    /* if (idx < (blockDim.x * blockDim.y)/2) {
        xLeftBlock[idx] = x0[threadIdx.x * blockDim.x + threadIdx.y];
	xRightBlock[idx] = x0[(blockDim.x-1-threadIdx.y) + threadIdx.x * blockDim.x];
    } */
}

__device__
void __iterativeBlockUpdateToNorthSouth(double * xTopBlock, double * xBottomBlock, const double *rhsBlock, 
                             const double * leftMatrixBlock, const double *centerMatrixBlock, const double * rightMatrixBlock, 
			     const double * topMatrixBlock, const double * bottomMatrixBlock, int nxGrids, int nyGrids, int iGrid, int method, int subdomainLength)
{
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory; 
    int elemPerBlock = subdomainLength * subdomainLength;
    double * x1 = sharedMemory + elemPerBlock;
    int maxSteps = 1;
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    for (int idx = index; idx < elemPerBlock; idx += stride) {
        printf("NorthToSouth: For blockIdx.x %d and blockIdx.y %d, the %dth entry of x0 is %f\n", blockIdx.x, blockIdx.y,  idx, x0[idx]);
        __syncthreads();
    }
    
    for (int k = 0; k < maxSteps; k++) {
        for (int idx = index; idx < elemPerBlock; idx += stride) {
            __syncthreads();
            printf("In the loop: For blockIdx.x %d and blockIdx.y %d, the %dth entry of x0 is %f\n", blockIdx.x, blockIdx.y,  idx, x0[idx]);
            if ((idx % subdomainLength != 0) && ((idx+1) % subdomainLength != 0) && (idx > subdomainLength-1) && (idx < elemPerBlock-subdomainLength-1)) {
                // Define necessary constants
                double centerRhs = rhsBlock[idx];
                double leftMatrix = leftMatrixBlock[idx];
                double centerMatrix = centerMatrixBlock[idx];
                double rightMatrix = rightMatrixBlock[idx];
                double topMatrix = topMatrixBlock[idx];
                double bottomMatrix = bottomMatrixBlock[idx];
                /* double leftX = ((iGrid % nxGrids) == 0) ? 0.0 : x0[idx-1];
                double centerX = x0[idx];
                double rightX = ((iGrid + 1) % nxGrids == 0) ? 0.0 : x0[idx+1];
                double topX = (iGrid < nxGrids * (nyGrids - 1)) ? x0[idx+blockDim.x] : 0.0;
                double bottomX = (iGrid >= nxGrids) ?  x0[idx-blockDim.x] : 0.0; */
                printf("idx is %d\n", idx);
                double leftX = x0[idx-1];
                double centerX = x0[idx];
                double rightX = x0[idx+1];
                double topX = x0[idx+blockDim.x];
                double bottomX = x0[idx-blockDim.x];
                // Perform update
                printf("x1[%d] before incrementing is %f and centerX is %f\n", idx, x1[idx], centerX);
     	        x1[idx] = increment(centerX);
                printf("x1[%d] is now %f\n", idx, x1[idx]);
                //x1[idx] = jacobi(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
                //                 leftX, centerX, rightX, topX, bottomX, centerRhs); 
                // Synchronize
	        __syncthreads();
//	        double * tmp; tmp = x0; x0 = x1;
                // Switch pointers only at the final step of the second foor loop
//                if (elemPerBlock - idx < 1*stride) {
//	            double * tmp; tmp = x0; x0 = x1;
//                printf("x0[%d] after switching is now %f\n", idx, x0[idx]);
               }
 //             printf("In blockIdx %d, blockIdy %d, iGrid %d, Updated value in idx = %d is %f\n", blockIdx.x, blockIdx.y, iGrid, idx, x1[idx]);
	    }
	            double * tmp; tmp = x0; x0 = x1;
        }   
    

/*    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    if (blockID == 0) {
        for (int i = 0; i < elemPerBlock; i++) {
            printf("Horizontal Shift - Block ID: %d, The %dth entry of x0 is %f\n", blockID, i, x0[i]);
        }
    }
*/

/*    
    for (int idx = index; idx < elemPerBlock; idx += stride) {
        printf("For blockIdx.x %d and blockIdx.y %d, threadIdx.x %d and threadIdx.y %d, the %dth entry of x0 is %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, idx, x0[idx]);
        __syncthreads();
    }
*/    
    // Return values for xTop and xBottom here
    for (int idx = index; idx < elemPerBlock/2; idx += stride) {
        xBottomBlock[idx] = x0[idx];
        //printf("For blockIdx.x %d and blockIdx.y %d, the %dth entry of xBottomBlock is %f\n", blockIdx.x, blockIdx.y, idx, xBottomBlock[idx]);
        //printf("When idx is %d, xTopBlock draws from x0[idx=%d]\n", idx, subdomainLength * (subdomainLength-1-idx/subdomainLength) + (idx % subdomainLength));
        xTopBlock[idx] = x0[subdomainLength * (subdomainLength-1-idx/subdomainLength) + (idx % subdomainLength)];
    }



/*
    if (idx < (blockDim.x * blockDim.y)/2) {
        xBottomBlock[idx] = x0[idx];
	xTopBlock[idx] = x0[threadIdx.x + (blockDim.x)*(blockDim.x-1-threadIdx.y)];
    }
*/
}

__global__
void _iterativeGpuOriginal(double * xLeftGpu, double *xRightGpu,
                             const double * x0Gpu, const double *rhsGpu, 
                             const double * leftMatrixGpu, const double *centerMatrixGpu, const double * rightMatrixGpu, 
			     const double * topMatrixGpu, const double * bottomMatrixGpu, int nxGrids, int nyGrids, int method, int subdomainLength)
{

    int xShift = subdomainLength * blockIdx.x;
    int yShift = subdomainLength * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;

    const double * x0Block = x0Gpu + blockShift;
    const double * rhsBlock = rhsGpu + blockShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift;

    int numElementsPerBlock = subdomainLength * subdomainLength;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = (numElementsPerBlock*blockID)/2;
    double * xLeftBlock = xLeftGpu + arrayShift;
    double * xRightBlock = xRightGpu + arrayShift;
    
    // int idx = threadIdx.x + threadIdx.y * nxGrids;
    // int iGrid = blockShift + idx;
    extern __shared__ double sharedMemory[];

    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    for (int idx = index; idx < numElementsPerBlock; idx += stride) {
        int Idx = (idx % subdomainLength) + (idx/subdomainLength) * nxGrids;
        sharedMemory[idx] = x0Block[Idx];
        sharedMemory[idx + numElementsPerBlock] = x0Block[Idx];
        // printf("In blockIdx %d, blockIdy %d, the value in sharedmemory[idx=%d] is %f\n", blockIdx.x, blockIdx.y, idx, sharedMemory[idx]); 
    }

    __iterativeBlockUpdateToLeftRight(xLeftBlock, xRightBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, nxGrids, method, subdomainLength);
}

__global__
void _iterativeGpuHorizontalShift(double * xLeftGpu, double *xRightGpu, double * xTopGpu, double * xBottomGpu,
                                  const double * x0Gpu, const double *rhsGpu, 
                                  const double * leftMatrixGpu, const double *centerMatrixGpu, const double * rightMatrixGpu, 
			          const double * topMatrixGpu, const double * bottomMatrixGpu, int nxGrids, int nyGrids, int method, int subdomainLength)
{
    int xShift = subdomainLength * blockIdx.x;
    int yShift = subdomainLength * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    int horizontalShift = subdomainLength/2;

    const double * rhsBlock = rhsGpu + blockShift; //+ horizontalShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift; //+ horizontalShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift; //+ horizontalShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift; //+ horizontalShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift; //+ horizontalShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift; //+ horizontalShift;

    int numElementsPerBlock = (subdomainLength * subdomainLength)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
    
    double * xLeftBlock =  xRightGpu + arrayShift;
    double * xRightBlock = (blockIdx.x != gridDim.x-1) ?
                           xLeftGpu + arrayShift + numElementsPerBlock :
			   xLeftGpu + (numElementsPerBlock * blockIdx.y * gridDim.x);
    double * xBottomBlock = xBottomGpu + arrayShift;
    double * xTopBlock = xTopGpu + arrayShift;

    // int idx = threadIdx.x + threadIdx.y * nxGrids;
    // int iGrid = blockShift + idx + horizontalShift;
   
    //if ((blockIdx.x == gridDim.x-1) && threadIdx.x >= (blockDim.x/2)) {
    //    iGrid = iGrid - nxGrids;
    //}
    
    // printf("In loop: I am idx %d and grid point %d\n", idx, iGrid);
    extern __shared__ double sharedMemory[];
    //idx = threadIdx.x + threadIdx.y * blockDim.x;
    
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    int idx = index;
    for (int idx = index; idx < subdomainLength * subdomainLength; idx += stride) {
        if (idx % subdomainLength < subdomainLength/2) {
            int Idx = ((subdomainLength-1)/2-(idx % subdomainLength)) * subdomainLength + idx/subdomainLength;
            sharedMemory[idx] = xLeftBlock[Idx];
            sharedMemory[idx + subdomainLength * subdomainLength] = xLeftBlock[Idx];
//            printf("In blockID %d, the %dth entry of shared memory is %f\n", blockID, idx, sharedMemory[idx]);
            printf("From left - Block ID is %d: sharedMemory[idx=%d] is equal to %f\n", blockID, idx, sharedMemory[idx]);
        }
        else {
            int Idx = ((idx % subdomainLength) - (subdomainLength-1)/2 - 1) * subdomainLength + idx/subdomainLength;
            sharedMemory[idx] = xRightBlock[Idx];
            sharedMemory[idx + subdomainLength * subdomainLength] = xRightBlock[Idx];
//            printf("In blockID %d, the %dth entry of shared memory is %f\n", blockID, idx, sharedMemory[idx]);
            printf("From right - Block ID is %d: sharedMemory[idx=%d] is equal to %f\n", blockID, idx, sharedMemory[idx]);
        }
    }

/*
    if (threadIdx.x < blockDim.x/2) {
        sharedMemory[idx] = xLeftBlock[threadIdx.y + (blockDim.x/2-1-threadIdx.x)*blockDim.y];
    }
    else {
        sharedMemory[idx] = xRightBlock[threadIdx.y + (threadIdx.x-(blockDim.x/2))*blockDim.y];
    }
*/    
    __iterativeBlockUpdateToNorthSouth(xTopBlock, xBottomBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, nxGrids, method, subdomainLength);
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
    int horizontalShift = blockDim.x/2;
    int verticalShift = blockDim.y/2 * nxGrids;

    const double * rhsBlock = rhsGpu + blockShift; //+ verticalShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift; //+ verticalShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift; //+ verticalShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift; //+ verticalShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift; //+ verticalShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift; //+ verticalShift;

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
    int iGrid = blockShift + verticalShift + horizontalShift + idx;

    if ((blockIdx.x == gridDim.x-1) && threadIdx.x >= (blockDim.x/2)) {
        iGrid = iGrid - nxGrids;
    }

    int nDofs = nxGrids * nyGrids;

    if ((blockIdx.y == gridDim.y-1) && threadIdx.y >= (blockDim.y/2)) {
        iGrid = iGrid - nDofs;
    } 

/*
    if ((blockIdx.x == gridDim.x-1) && (threadIdx.x >= (blockDim.x/2)) && (iGrid >= nDofs-1)) {
        iGrid = blockShift + verticalShift + horizontalShift + idx - nDofs - nxGrids; 
    }
*/   
    // printf("I am idx %d with tidx %d and tidy %d and grid point %d\n", idx, threadIdx.x, threadIdx.y, iGrid);

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
			   nxGrids, nyGrids, iGrid, method, method);
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

    const double * rhsBlock = rhsGpu + blockShift; //+ verticalShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift; //+ verticalShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift; //+ verticalShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift; //+ verticalShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift; //+ verticalShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift; //+ verticalShift;

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
    int nDofs = nxGrids * nyGrids;
    int iGrid = blockShift + verticalShift + threadIdx.y * nxGrids + threadIdx.x;
    iGrid = (iGrid >= nDofs) ? iGrid - nDofs : iGrid;

    // printf("In loop: I am idx %d and grid point %d\n", idx, iGrid);
    
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
			   nxGrids, nyGrids, iGrid, method, method);
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
//        printf("The %dth entry of xTopBlock is %f\n", idx, xTopBlock[idx]);
//        printf("xTopBlock[idx=%d] goes into sharedMemory[%d]\n", idx, idx+numElementsPerBlock);
        sharedMemory[idx + numElementsPerBlock] = xTopBlock[idx]; 
	sharedMemory[threadIdx.x + (blockDim.x)*(blockDim.x/2-1-threadIdx.y)] = xBottomBlock[idx];
    }

    __syncthreads();

//    printf("sharedMemory[idx=%d] is %f \n", idx, sharedMemory[idx]);
    
    double * x0 = x0Gpu + blockShift;

    idx = threadIdx.x + threadIdx.y * nxGrids;
    x0[threadIdx.x + threadIdx.y * nxGrids] = sharedMemory[threadIdx.x + threadIdx.y * blockDim.x];    
}

