__device__
float increment(float centerX) {
     return centerX + 100.;
}

__device__
float jacobi(const float leftMatrix, const float centerMatrix, const float rightMatrix, const float topMatrix, const float bottomMatrix,
              const float leftX, const float centerX, const float rightX, const float topX, const float bottomX,
              const float centerRhs) {
    float result = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
    return result;
}

float normFromRow(float leftMatrix, float centerMatrix, float rightMatrix, float topMatrix, float bottomMatrix, float leftX, float centerX, float rightX,  float topX, float bottomX, float centerRhs) 
{
    return centerRhs - (leftMatrix*leftX + centerMatrix*centerX + rightMatrix*rightX + topMatrix*topX + bottomMatrix*bottomX);
}

float Residual(const float * solution, const float * rhs, const float * leftMatrix, const float * centerMatrix, const float * rightMatrix, const float * topMatrix, const float * bottomMatrix, int nxGrids, int nyGrids)
{
    int nDofs = nxGrids * nyGrids;
    float residual = 0.0;  

    for (int iGrid = 0; iGrid < nDofs; iGrid++) {
        float leftX = ((iGrid % nxGrids) == 0) ? 0.0 : solution[iGrid-1];
        float centerX = solution[iGrid];
        float rightX = ((iGrid + 1) % nxGrids == 0) ? 0.0 : solution[iGrid+1];
        float topX = (iGrid < nxGrids * (nyGrids - 1)) ? solution[iGrid + nxGrids] : 0.0;
        float bottomX = (iGrid >= nxGrids) ?  solution[iGrid-nxGrids] : 0.0;
        float residualContributionFromRow = normFromRow(leftMatrix[iGrid], centerMatrix[iGrid], rightMatrix[iGrid], topMatrix[iGrid], bottomMatrix[iGrid], leftX, centerX, rightX, topX, bottomX, rhs[iGrid]);
	residual = residual + residualContributionFromRow * residualContributionFromRow;
//	printf("For gridpoint %d, residual contribution is %f\n", iGrid, residualContributionFromRow);
    }
    residual = sqrt(residual);
    return residual;
}

__device__
void __iterativeBlockUpdateToLeftRight(float * xLeftBlock, float * xRightBlock, const float *rhsBlock, 
                             const float * leftMatrixBlock, const float *centerMatrixBlock, const float * rightMatrixBlock, 
			     const float * topMatrixBlock, const float * bottomMatrixBlock, int nxGrids, int nyGrids, int iGrid, int method, 
                             int subdomainLength)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory; 
    int elemPerBlock = subdomainLength * subdomainLength;
    float * x1 = sharedMemory + elemPerBlock;
    int maxSteps = 1;
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    __syncthreads();
    for (int k = 0; k < maxSteps; k++) {
        for (int idx = index; idx < elemPerBlock; idx += stride) {
           if ((idx % subdomainLength != 0) && ((idx+1) % subdomainLength != 0) && (idx > subdomainLength-1) && (idx < elemPerBlock-(subdomainLength-1))) {
                // Define necessary constants
                float centerRhs = rhsBlock[idx];
                float leftMatrix = leftMatrixBlock[idx];
                float centerMatrix = centerMatrixBlock[idx];
                float rightMatrix = rightMatrixBlock[idx];
                float topMatrix = topMatrixBlock[idx];
                float bottomMatrix = bottomMatrixBlock[idx];
                /* float leftX = ((iGrid % nxGrids) == 0) ? 0.0 : x0[idx-1];
                float centerX = x0[idx];
                float rightX = ((iGrid + 1) % nxGrids == 0) ? 0.0 : x0[idx+1];
                float topX = (iGrid < nxGrids * (nyGrids - 1)) ? x0[idx+blockDim.x] : 0.0;
                float bottomX = (iGrid >= nxGrids) ?  x0[idx-blockDim.x] : 0.0;
                */
                float leftX = x0[idx-1];
                float centerX = x0[idx];
                float rightX = x0[idx+1];
                float topX = x0[idx+subdomainLength];
                float bottomX = x0[idx-subdomainLength];

//                printf("rhs is %f\n", centerRhs);
                
                // Apply boundary conditions
                // int Idx = (idx % subdomainLength) +(idx/subdomainLength) * nxGrids;

                // Bottom
                if (blockIdx.y == 0) {
                    if (idx < subdomainLength) {
                        leftX = 0.;
                    }
                }

                // Top 
                if (blockIdx.y == gridDim.y-1) {
                    if (idx >= subdomainLength * (subdomainLength - 1)) {
                        topX = 0.;
                    }
                }

                // Left
                if (blockIdx.x == 0) {
                    if (idx % subdomainLength == 0) {
                        leftX = 0.;
                    }
                }
 
                // Right
                if (blockIdx.x == gridDim.x-1) {
                    if ((idx+1) % subdomainLength == 0) {
                        rightX = 0.;
                    }
                }

               // printf("BlockIdx %d and blockIdy %d: In idx = %d, rhsBlock %f, center %f, left %f, right %f, top %f, bottom %f\n", blockIdx.x, blockIdx.y, idx, centerRhs, centerX,  leftX, rightX, topX, bottomX);
	        // Perform update
   	        // x1[idx] = increment(centerX);
                x1[idx] = jacobi(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
                                leftX, centerX, rightX, topX, bottomX, centerRhs); 
                // Synchronize
	        __syncthreads();
                
//                float * tmp; tmp = x0; x0 = x1;
//              printf("Updated value in idx = %d is %f\n", idx, x1[idx]);
	    }
        }
                float * tmp; tmp = x0; x0 = x1;
    }

    index = threadIdx.x + threadIdx.y * blockDim.x;
    stride = blockDim.x * blockDim.y;
    for (int idx = index; idx < elemPerBlock; idx += stride) {
        //printf("BlockIdx %d, BlockIdy %d: The value of x0[idx=%d] is %f\n", blockIdx.x, blockIdx.y, idx, x0[idx]);
    }

    __syncthreads();
    //printf("Hello\n"); 
    for (int idx = index; idx < elemPerBlock/2; idx += stride) {
        xLeftBlock[idx] = x0[subdomainLength * (idx % subdomainLength) + (idx/subdomainLength)];
        xRightBlock[idx] = x0[subdomainLength * (idx % subdomainLength) - (idx/subdomainLength) + (subdomainLength-1)];
        // printf("In blockIdx %d, blockIdy %d, xLeftBlock[idx=%d] = %f\n", blockIdx.x, blockIdx.y, idx, xLeftBlock[idx]);
    }
    
    __syncthreads();
    // Save xLeft, xRight, xTop, xBottom
    /* if (idx < (blockDim.x * blockDim.y)/2) {
        xLeftBlock[idx] = x0[threadIdx.x * blockDim.x + threadIdx.y];
	xRightBlock[idx] = x0[(blockDim.x-1-threadIdx.y) + threadIdx.x * blockDim.x];
    } */
}

__device__
void __iterativeBlockUpdateToNorthSouth(float * xTopBlock, float * xBottomBlock, const float *rhsBlock, 
                             const float * leftMatrixBlock, const float *centerMatrixBlock, const float * rightMatrixBlock, 
			     const float * topMatrixBlock, const float * bottomMatrixBlock, int nxGrids, int nyGrids, int iGrid, int method, int subdomainLength)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory; 
    int elemPerBlock = subdomainLength * subdomainLength;
    float * x1 = sharedMemory + elemPerBlock;
    int maxSteps = 1;
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    for (int idx = index; idx < elemPerBlock; idx += stride) {
        // printf("NorthToSouth: For blockIdx.x %d and blockIdx.y %d, the %dth entry of x0 is %f\n", blockIdx.x, blockIdx.y,  idx, x0[idx]);
        __syncthreads();
    }
    
    for (int k = 0; k < maxSteps; k++) {
        for (int idx = index; idx < elemPerBlock; idx += stride) {
            __syncthreads();
            //printf("In the loop: For blockIdx.x %d and blockIdx.y %d, the %dth entry of x0 is %f\n", blockIdx.x, blockIdx.y,  idx, x0[idx]);
            if ((idx % subdomainLength != 0) && ((idx+1) % subdomainLength != 0) && (idx > subdomainLength-1) && (idx < elemPerBlock-subdomainLength-1)) {
                // Define necessary constants
                float centerRhs = rhsBlock[idx];
                float leftMatrix = leftMatrixBlock[idx];
                float centerMatrix = centerMatrixBlock[idx];
                float rightMatrix = rightMatrixBlock[idx];
                float topMatrix = topMatrixBlock[idx];
                float bottomMatrix = bottomMatrixBlock[idx];
                /* float leftX = ((iGrid % nxGrids) == 0) ? 0.0 : x0[idx-1];
                float centerX = x0[idx];
                float rightX = ((iGrid + 1) % nxGrids == 0) ? 0.0 : x0[idx+1];
                float topX = (iGrid < nxGrids * (nyGrids - 1)) ? x0[idx+blockDim.x] : 0.0;
                float bottomX = (iGrid >= nxGrids) ?  x0[idx-blockDim.x] : 0.0; */
                //printf("idx is %d\n", idx);
                float leftX = x0[idx-1];
                float centerX = x0[idx];
                float rightX = x0[idx+1];
                float topX = x0[idx+blockDim.x];
                float bottomX = x0[idx-blockDim.x];

                // Apply boundary conditions
                // int Idx = (idx % subdomainLength) +(idx/subdomainLength) * nxGrids;

                // Bottom
                if (blockIdx.y == 0) {
                    if (idx < subdomainLength) {
                        leftX = 0.;
                    }
                }

                // Top 
                if (blockIdx.y == gridDim.y-1) {
                    if (idx >= subdomainLength * (subdomainLength - 1)) {
                        topX = 0.;
                    }
                }

                // Left
                if (blockIdx.x == 0) {
                    if (idx % subdomainLength == 0) {
                        leftX = 0.;
                    }
                }
 
                // Right
                if (blockIdx.x == gridDim.x-1) {
                    if ((idx+1) % subdomainLength == 0) {
                        rightX = 0.;
                    }
                }
                
                // Perform update
                //printf("x1[%d] before incrementing is %f and centerX is %f\n", idx, x1[idx], centerX);
     	        // x1[idx] = increment(centerX);
                //printf("x1[%d] is now %f\n", idx, x1[idx]);
                x1[idx] = jacobi(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
                                 leftX, centerX, rightX, topX, bottomX, centerRhs); 
                // Synchronize
	        __syncthreads();
//	        float * tmp; tmp = x0; x0 = x1;
                // Switch pointers only at the final step of the second foor loop
//                if (elemPerBlock - idx < 1*stride) {
//	            float * tmp; tmp = x0; x0 = x1;
//                printf("x0[%d] after switching is now %f\n", idx, x0[idx]);
               }
 //             printf("In blockIdx %d, blockIdy %d, iGrid %d, Updated value in idx = %d is %f\n", blockIdx.x, blockIdx.y, iGrid, idx, x1[idx]);
	    }
	            float * tmp; tmp = x0; x0 = x1;
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
void _iterativeGpuOriginal(float * xLeftGpu, float *xRightGpu,
                             const float * x0Gpu, const float *rhsGpu, 
                             const float * leftMatrixGpu, const float *centerMatrixGpu, const float * rightMatrixGpu, 
			     const float * topMatrixGpu, const float * bottomMatrixGpu, int nxGrids, int nyGrids, int method, int subdomainLength)
{

    __syncthreads();

    int xShift = subdomainLength * blockIdx.x;
    int yShift = subdomainLength * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;

    const float * x0Block = x0Gpu + blockShift;
    const float * rhsBlock = rhsGpu + blockShift;
    const float * leftMatrixBlock = leftMatrixGpu + blockShift;
    const float * centerMatrixBlock = centerMatrixGpu + blockShift;
    const float * rightMatrixBlock = rightMatrixGpu + blockShift;
    const float * topMatrixBlock = topMatrixGpu + blockShift;
    const float * bottomMatrixBlock = bottomMatrixGpu + blockShift;

    int numElementsPerBlock = subdomainLength * subdomainLength;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = (numElementsPerBlock*blockID)/2;
    float * xLeftBlock = xLeftGpu + arrayShift;
    float * xRightBlock = xRightGpu + arrayShift;
    
    // int idx = threadIdx.x + threadIdx.y * nxGrids;
    // int iGrid = blockShift + idx;
    extern __shared__ float sharedMemory[];

    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    // printf("My index is %d\n", index);
    for (int idx = index; idx < numElementsPerBlock; idx += stride) {
        int Idx = (idx % subdomainLength) + (idx/subdomainLength) * nxGrids;
        sharedMemory[idx] = x0Block[Idx];
        sharedMemory[idx + numElementsPerBlock] = x0Block[Idx];
        //printf("In blockIdx %d, blockIdy %d, the value in sharedmemory[idx=%d] is %f\n", blockIdx.x, blockIdx.y, idx, sharedMemory[idx]); 
    }
    __syncthreads();
    __iterativeBlockUpdateToLeftRight(xLeftBlock, xRightBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, nxGrids, method, subdomainLength);
    __syncthreads();


    //printf("Hello/n");
}

__global__
void _iterativeGpuHorizontalShift(float * xLeftGpu, float *xRightGpu, float * xTopGpu, float * xBottomGpu,
                                  const float * x0Gpu, const float *rhsGpu, 
                                  const float * leftMatrixGpu, const float *centerMatrixGpu, const float * rightMatrixGpu, 
			          const float * topMatrixGpu, const float * bottomMatrixGpu, int nxGrids, int nyGrids, int method, int subdomainLength)
{
    int xShift = subdomainLength * blockIdx.x;
    int yShift = subdomainLength * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    int horizontalShift = subdomainLength/2;

    const float * rhsBlock = rhsGpu + blockShift; //+ horizontalShift;
    const float * leftMatrixBlock = leftMatrixGpu + blockShift; //+ horizontalShift;
    const float * centerMatrixBlock = centerMatrixGpu + blockShift; //+ horizontalShift;
    const float * rightMatrixBlock = rightMatrixGpu + blockShift; //+ horizontalShift;
    const float * topMatrixBlock = topMatrixGpu + blockShift; //+ horizontalShift;
    const float * bottomMatrixBlock = bottomMatrixGpu + blockShift; //+ horizontalShift;

    int numElementsPerBlock = (subdomainLength * subdomainLength)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
    
    float * xLeftBlock =  xRightGpu + arrayShift;
    float * xRightBlock = (blockIdx.x != gridDim.x-1) ?
                           xLeftGpu + arrayShift + numElementsPerBlock :
			   xLeftGpu + (numElementsPerBlock * blockIdx.y * gridDim.x);
    float * xBottomBlock = xBottomGpu + arrayShift;
    float * xTopBlock = xTopGpu + arrayShift;

    // int idx = threadIdx.x + threadIdx.y * nxGrids;
    // int iGrid = blockShift + idx + horizontalShift;
   
    //if ((blockIdx.x == gridDim.x-1) && threadIdx.x >= (blockDim.x/2)) {
    //    iGrid = iGrid - nxGrids;
    //}
    
    // printf("In loop: I am idx %d and grid point %d\n", idx, iGrid);
    extern __shared__ float sharedMemory[];
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
            // printf("From left - Block ID is %d: sharedMemory[idx=%d] is equal to %f\n", blockID, idx, sharedMemory[idx]);
        }
        else {
            int Idx = ((idx % subdomainLength) - (subdomainLength-1)/2 - 1) * subdomainLength + idx/subdomainLength;
            sharedMemory[idx] = xRightBlock[Idx];
            sharedMemory[idx + subdomainLength * subdomainLength] = xRightBlock[Idx];
//            printf("In blockID %d, the %dth entry of shared memory is %f\n", blockID, idx, sharedMemory[idx]);
            // printf("From right - Block ID is %d: sharedMemory[idx=%d] is equal to %f\n", blockID, idx, sharedMemory[idx]);
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

    __syncthreads();
  
    __iterativeBlockUpdateToNorthSouth(xTopBlock, xBottomBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, nxGrids, method, subdomainLength);
}

__global__
void _iterativeGpuVerticalandHorizontalShift(float * xLeftGpu, float *xRightGpu, float * xTopGpu, float * xBottomGpu,
                                const float * x0Gpu, const float *rhsGpu, 
                                const float * leftMatrixGpu, const float *centerMatrixGpu, const float * rightMatrixGpu, 
			        const float * topMatrixGpu, const float * bottomMatrixGpu, int nxGrids, int nyGrids, int method, int subdomainLength)
{
    int xShift = subdomainLength * blockIdx.x;
    int yShift = subdomainLength * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    
    int horizontalShift = subdomainLength/2;
    int verticalShift = blockDim.y/2 * nxGrids;

    const float * rhsBlock = rhsGpu + blockShift; //+ verticalShift;
    const float * leftMatrixBlock = leftMatrixGpu + blockShift; //+ verticalShift;
    const float * centerMatrixBlock = centerMatrixGpu + blockShift; //+ verticalShift;
    const float * rightMatrixBlock = rightMatrixGpu + blockShift; //+ verticalShift;
    const float * topMatrixBlock = topMatrixGpu + blockShift; //+ verticalShift;
    const float * bottomMatrixBlock = bottomMatrixGpu + blockShift; //+ verticalShift;

    int numElementsPerBlock = (subdomainLength * subdomainLength)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
    float * xBottomBlock = xTopGpu + arrayShift;
    float * xTopBlock = (blockIdx.y != gridDim.y-1) ?
                         xBottomGpu + numElementsPerBlock * gridDim.x + arrayShift :
			 xBottomGpu + (numElementsPerBlock * blockIdx.x);
    int index3 = threadIdx.x + threadIdx.y * blockDim.x;
    int stride3 = blockDim.x * blockDim.y;
    for (int idx = index3; idx < subdomainLength * subdomainLength * 2; idx += stride3) {
        // printf("In blockIdx %d, blockIdy %d, xBottomGpu[idx=%d] = %f\n", blockIdx.x, blockIdx.y, idx, xBottomGpu[idx]);
    }

    int index2 = threadIdx.x + threadIdx.y * blockDim.x;
    int stride2 = blockDim.x * blockDim.y;
    for (int idx = index2; idx < subdomainLength * subdomainLength *2; idx += stride2) {
       // printf("In blockIdx %d, blockIdy %d, xTopBlock[idx=%d] = %f\n", blockIdx.x, blockIdx.y, idx, xTopBlock[idx]);
    }
    
    float * xLeftBlock = xLeftGpu + arrayShift;
    float * xRightBlock = xRightGpu + arrayShift;

/*    int idx = threadIdx.x + threadIdx.y * nxGrids;
    int iGrid = blockShift + verticalShift + horizontalShift + idx;

    if ((blockIdx.x == gridDim.x-1) && threadIdx.x >= (blockDim.x/2)) {
        iGrid = iGrid - nxGrids;
    }

    int nDofs = nxGrids * nyGrids;

    if ((blockIdx.y == gridDim.y-1) && threadIdx.y >= (blockDim.y/2)) {
        iGrid = iGrid - nDofs;
    } blockIdx.y
*/
/*
    if ((blockIdx.x == gridDim.x-1) && (threadIdx.x >= (blockDim.x/2)) && (iGrid >= nDofs-1)) {
        iGrid = blockShift + verticalShift + horizontalShift + idx - nDofs - nxGrids; 
    }
*/   
    // printf("I am idx %d with tidx %d and tidy %d and grid point %d\n", idx, threadIdx.x, threadIdx.y, iGrid);

    extern __shared__ float sharedMemory[];
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    for (int idx = index; idx < subdomainLength * subdomainLength; idx += stride) {
        if (idx < numElementsPerBlock) {
            sharedMemory[idx] = xBottomBlock[(subdomainLength/2-1-idx/subdomainLength) * subdomainLength + idx % subdomainLength];
            sharedMemory[idx + subdomainLength * subdomainLength] = xBottomBlock[(subdomainLength/2-1-idx/subdomainLength) * subdomainLength + idx % subdomainLength];
            //printf("In blockIdx.x %d and blockIdx.y %d: The %dth entry of sharedMemory is %f\n", blockIdx.x, blockIdx.y, idx, sharedMemory[idx]);
        }
        else {
            sharedMemory[idx] = xTopBlock[idx - numElementsPerBlock];
            sharedMemory[idx + subdomainLength * subdomainLength] = xTopBlock[idx - numElementsPerBlock];
            //printf("In blockIdx.x %d and blockIdx.y %d: The %dth entry of sharedMemory is %f\n", blockIdx.x, blockIdx.y, idx, sharedMemory[idx]);
           //printf("In blockIdx.x %d, blockidx.y %d, in idx %d we obtain value of %f\n", blockIdx.x, blockIdx.y, idx, sharedMemory[idx]); 
        }
    }
    __iterativeBlockUpdateToLeftRight(xLeftBlock, xRightBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, nyGrids, method, subdomainLength);
/*    __syncthreads(); 
    int index4 = threadIdx.x + threadIdx.y * blockDim.x;
    int stride4 = blockDim.x * blockDim.y;
    for (int idx = index4; idx < subdomainLength * subdomainLength/2; idx += stride4) {
        printf("In blockIdx %d, blockIdy %d, xLeftBlock[idx=%d] = %f\n", blockIdx.x, blockIdx.y, idx, xLeftBlock[idx]);
        printf("In blockIdx %d, blockIdy %d, xRightBlock[idx=%d] = %f\n", blockIdx.x, blockIdx.y, idx, xRightBlock[idx]);
    }*/
}


__global__
void _iterativeGpuVerticalShift(float * xLeftGpu, float *xRightGpu, float * xTopGpu, float * xBottomGpu,
                                const float * x0Gpu, const float *rhsGpu, 
                                const float * leftMatrixGpu, const float *centerMatrixGpu, const float * rightMatrixGpu, 
			        const float * topMatrixGpu, const float * bottomMatrixGpu, int nxGrids, int nyGrids, int method, int subdomainLength)
{
    int xShift = subdomainLength * blockIdx.x;
    int yShift = subdomainLength * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    int verticalShift = blockDim.y/2 * nxGrids;

    const float * rhsBlock = rhsGpu + blockShift; //+ verticalShift;
    const float * leftMatrixBlock = leftMatrixGpu + blockShift; //+ verticalShift;
    const float * centerMatrixBlock = centerMatrixGpu + blockShift; //+ verticalShift;
    const float * rightMatrixBlock = rightMatrixGpu + blockShift; //+ verticalShift;
    const float * topMatrixBlock = topMatrixGpu + blockShift; //+ verticalShift;
    const float * bottomMatrixBlock = bottomMatrixGpu + blockShift; //+ verticalShift;

    int numElementsPerBlock = (subdomainLength * subdomainLength)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
    
    float * xRightBlock =  xLeftGpu + arrayShift;
    float * xLeftBlock = (blockIdx.x != 0) ?
                           xRightGpu + arrayShift - numElementsPerBlock :
    			   xRightGpu + numElementsPerBlock * ((gridDim.x-1) + blockIdx.y * gridDim.x);
    
    float * xBottomBlock = xBottomGpu + arrayShift;
    float * xTopBlock = xTopGpu + arrayShift;

/*    int idx = threadIdx.x + threadIdx.y * nxGrids;
    int nDofs = nxGrids * nyGrids;
    int iGrid = blockShift + verticalShift + threadIdx.y * nxGrids + threadIdx.x;
    iGrid = (iGrid >= nDofs) ? iGrid - nDofs : iGrid;
*/
    // printf("In loop: I am idx %d and grid point %d\n", idx, iGrid);
/*    int index = threadIdx.x + threadIdx.y * blockDim.x;
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
*/    
    extern __shared__ float sharedMemory[];
//    idx = threadIdx.x + threadIdx.y * blockDim.x;

    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    int idx = index;
    for (int idx = index; idx < subdomainLength * subdomainLength; idx += stride) {
        if (idx % subdomainLength < subdomainLength/2) {
            int Idx = ((subdomainLength-1)/2-(idx % subdomainLength)) * subdomainLength + idx/subdomainLength;
            sharedMemory[idx] = xLeftBlock[Idx];
            sharedMemory[idx + subdomainLength * subdomainLength] = xLeftBlock[Idx];
//            printf("In blockID %d, the %dth entry of shared memory is %f\n", blockID, idx, sharedMemory[idx]);
            // printf("From left - Block ID is %d: sharedMemory[idx=%d] is equal to %f\n", blockID, idx, sharedMemory[idx]);
        }
        else {
            int Idx = ((idx % subdomainLength) - (subdomainLength-1)/2 - 1) * subdomainLength + idx/subdomainLength;
            sharedMemory[idx] = xRightBlock[Idx];
            sharedMemory[idx + subdomainLength * subdomainLength] = xRightBlock[Idx];
//            printf("In blockID %d, the %dth entry of shared memory is %f\n", blockID, idx, sharedMemory[idx]);
            // printf("From right - Block ID is %d: sharedMemory[idx=%d] is equal to %f\n", blockID, idx, sharedMemory[idx]);
        }
    }
    

/*    for (int idx = index; idx < subdomainLength * subdomainLength; idx += stride) {
        printf("Block Idx.x %d and blockIdx.y %d, Idx is %d\n", blockIdx.x, blockIdx.y, idx);
        if (idx % subdomainLength < subdomainLength/2) {
            int Idx = ((subdomainLength-1)/2-(idx % subdomainLength)) * subdomainLength + idx/subdomainLength;
            sharedMemory[idx] = xLeftBlock[Idx];
            sharedMemory[idx + subdomainLength * subdomainLength] = xLeftBlock[Idx];
            printf("In blockIdx %d, blockIdy %d, sharedMemory[idx=%d] is %f\n", blockIdx.x, blockIdx.y, idx, sharedMemory[idx]);
        }
        else {
            int Idx = ((idx % subdomainLength) - (subdomainLength-1)/2 - 1) * subdomainLength + idx/subdomainLength;
            sharedMemory[idx] = xRightBlock[threadIdx.y + (threadIdx.x-(blockDim.x/2))*blockDim.y];
            sharedMemory[idx + subdomainLength * subdomainLength] = xRightBlock[threadIdx.y + (threadIdx.x-(blockDim.x/2))*blockDim.y];
            printf("In blockIdx %d, blockIdy %d, sharedMemory[idx=%d] is %f\n", blockIdx.x, blockIdx.y, idx, sharedMemory[idx]);
        }
    }
*/
    __iterativeBlockUpdateToNorthSouth( xTopBlock, xBottomBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, nyGrids, method, subdomainLength);
}

__global__
void _finalSolution(float * xTopGpu, float * xBottomGpu, float * x0Gpu, int nxGrids, int subdomainLength)
{

    extern __shared__ float sharedMemory[];
    int numElementsPerBlock = (subdomainLength * subdomainLength)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;

    float * xTopBlock = xBottomGpu + arrayShift;
    float * xBottomBlock = (blockIdx.y != 0) ?
			    xTopGpu + (blockIdx.x + (blockIdx.y-1) * gridDim.x) * numElementsPerBlock :
			    xTopGpu + (gridDim.x * (gridDim.y-1) + blockIdx.x) * numElementsPerBlock;

    int xShift = subdomainLength * blockIdx.x;
    int yShift = subdomainLength * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    float * x0Block = x0Gpu + blockShift;

    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    for (int idx = index; idx < numElementsPerBlock; idx += stride) {
        sharedMemory[idx + numElementsPerBlock] = xTopBlock[idx]; 
	sharedMemory[(subdomainLength/2 - 1 - idx/subdomainLength) * subdomainLength + idx % subdomainLength] = xBottomBlock[idx];
    }

    for (int idx = index; idx < 2*2*numElementsPerBlock; idx += stride) {
       // printf("In blockIdx %d, blockIdy %d: Final method - The %dth entry of sharedMemory is %f\n", blockIdx.x, blockIdx.y, idx, sharedMemory[idx]);
    }
    __syncthreads();

//    printf("sharedMemory[idx=%d] is %f \n", idx, sharedMemory[idx]);
    
    float * x0 = x0Gpu + blockShift;
    
    for (int idx = index; idx < 2*numElementsPerBlock; idx += stride) {
        int Idx = (idx % subdomainLength) + (idx/subdomainLength) * nxGrids;
        x0Block[Idx] = sharedMemory[idx];
    }
}

