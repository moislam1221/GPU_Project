__device__
void boundaryConditions(int IGrid, int nxGrids, int nyGrids, float &leftX, float &rightX, float&bottomX, float &topX)
{
    // Left
    if (IGrid % nxGrids == 0) {
        leftX = 0.0;
    }               

    // Right
    if (((IGrid+1) % nxGrids) == 0) {
        rightX = 0.0;
    }               
    
    // Bottom
    if (IGrid < nxGrids) {
        bottomX = 0.0;
    }

    // Top
    if (IGrid >= (nxGrids * nyGrids - nxGrids)) {
        topX = 0.0;
    }

    return;
}


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
    }
    residual = sqrt(residual);
    return residual;
}

__device__
void __iterativeBlockUpdateToLeftRight(float * xLeftBlock, float * xRightBlock, const float *rhsBlock, 
                             const float * leftMatrixBlock, const float *centerMatrixBlock, const float * rightMatrixBlock, 
			     const float * topMatrixBlock, const float * bottomMatrixBlock, int nxGrids, int nyGrids, int iGrid, int method, 
                             int subdomainLength, bool diagonal)
{
    // Initialize shared memory and pointers to x0, x1 arrays containing Jacobi solutions
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory; 
    int elemPerBlock = subdomainLength * subdomainLength;
    float * x1 = sharedMemory + elemPerBlock;

    // Define number of Jacobi steps to take, and current index and stride value
    int maxSteps = 1;
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    // Perform Jacobi iterations
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
                float leftX = x0[idx-1];
                float centerX = x0[idx];
                float rightX = x0[idx+1];
                float topX = x0[idx+subdomainLength];
                float bottomX = x0[idx-subdomainLength];
                
                // Apply boundary conditions
                int step = idx / stride;
                int Idx = (stride % subdomainLength) + (stride/subdomainLength) * nxGrids;
                int IGrid  = iGrid + step * Idx;

                if (diagonal == true) {
                    int nDofs = nxGrids * nyGrids;
                    if ((blockIdx.y == gridDim.y-1) && idx/subdomainLength >= subdomainLength/2) {
                        IGrid = IGrid - nDofs;
                    }
                    if ((blockIdx.x == gridDim.x-1) && (idx % subdomainLength) >= (subdomainLength/2)) {
                        IGrid = IGrid - nxGrids;
                    }
                }

                boundaryConditions(iGrid, nxGrids, nyGrids, leftX, rightX, bottomX, topX);

/*

*/	        // Perform update
//   	        x1[idx] = increment(centerX);
                // x1[idx] = jacobi(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
                //                leftX, centerX, rightX, topX, bottomX, centerRhs); 

                // Synchronize
	        __syncthreads();
                
	    }
        }
                float * tmp; tmp = x0; x0 = x1;
    }

    index = threadIdx.x + threadIdx.y * blockDim.x;
    stride = blockDim.x * blockDim.y;
    for (int idx = index; idx < elemPerBlock; idx += stride) {
    }

    __syncthreads();
    for (int idx = index; idx < elemPerBlock/2; idx += stride) {
        xLeftBlock[idx] = x0[subdomainLength * (idx % subdomainLength) + (idx/subdomainLength)];
        xRightBlock[idx] = x0[subdomainLength * (idx % subdomainLength) - (idx/subdomainLength) + (subdomainLength-1)];
    }
    
}

__device__
void __iterativeBlockUpdateToNorthSouth(float * xTopBlock, float * xBottomBlock, const float *rhsBlock, 
                             const float * leftMatrixBlock, const float *centerMatrixBlock, const float * rightMatrixBlock, 
			     const float * topMatrixBlock, const float * bottomMatrixBlock, int nxGrids, int nyGrids, int iGrid, int method, int subdomainLength, bool vertical)
{
    extern __shared__ float sharedMemory[];
    float * x0 = sharedMemory; 
    int elemPerBlock = subdomainLength * subdomainLength;
    float * x1 = sharedMemory + elemPerBlock;
    int maxSteps = 1;
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    for (int idx = index; idx < elemPerBlock; idx += stride) {
        __syncthreads();
    }
    
    for (int k = 0; k < maxSteps; k++) {
        for (int idx = index; idx < elemPerBlock; idx += stride) {
            if ((idx % subdomainLength != 0) && ((idx+1) % subdomainLength != 0) && (idx > subdomainLength-1) && (idx < elemPerBlock-subdomainLength-1)) {
                // Define necessary constants
                float centerRhs = rhsBlock[idx];
                float leftMatrix = leftMatrixBlock[idx];
                float centerMatrix = centerMatrixBlock[idx];
                float rightMatrix = rightMatrixBlock[idx];
                float topMatrix = topMatrixBlock[idx];
                float bottomMatrix = bottomMatrixBlock[idx];
                float leftX = x0[idx-1];
                float centerX = x0[idx];
                float rightX = x0[idx+1];
                float topX = x0[idx+subdomainLength];
                float bottomX = x0[idx-subdomainLength];
                
                // Apply boundary conditions
                int step = idx / stride;
                int Idx = (stride % subdomainLength) + (stride/subdomainLength) * nxGrids;
                int IGrid  = iGrid + step * Idx;

                if (vertical == true) {
                    int nDofs = nxGrids * nyGrids;
                    if ((blockIdx.y == gridDim.y-1) && idx/subdomainLength >= subdomainLength/2) {
                        IGrid = IGrid - nDofs;
                    }
                }
                else {
                    if ((blockIdx.x == gridDim.x-1) && (idx % subdomainLength) >= (subdomainLength/2)) {
                        IGrid = IGrid - nxGrids;
                    }
                }
                
                boundaryConditions(iGrid, nxGrids, nyGrids, leftX, rightX, bottomX, topX);
 
                // Perform update
                //printf("x1[%d] before incrementing is %f and centerX is %f\n", idx, x1[idx], centerX);
     	        //x1[idx] = increment(centerX);
                //printf("x1[%d] is now %f\n", idx, x1[idx]);
                //x1[idx] = jacobi(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
                //                 leftX, centerX, rightX, topX, bottomX, centerRhs); 
                
                // Synchronize
	        __syncthreads();
               }
	    }
	            float * tmp; tmp = x0; x0 = x1;
        }   
    

    // Return values for xTop and xBottom here
    for (int idx = index; idx < elemPerBlock/2; idx += stride) {
        xBottomBlock[idx] = x0[idx];
        xTopBlock[idx] = x0[subdomainLength * (subdomainLength-1-idx/subdomainLength) + (idx % subdomainLength)];
    }

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
    
    extern __shared__ float sharedMemory[];

    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    for (int idx = index; idx < numElementsPerBlock; idx += stride) {
        int Idx = (idx % subdomainLength) + (idx/subdomainLength) * nxGrids;
        sharedMemory[idx] = x0Block[Idx];
        sharedMemory[idx + numElementsPerBlock] = x0Block[Idx];
    }
    int iGrid = blockShift + (index/subdomainLength) * nxGrids + index % subdomainLength;
    
    __iterativeBlockUpdateToLeftRight(xLeftBlock, xRightBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, iGrid, method, subdomainLength, false);
    __syncthreads();


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

    extern __shared__ float sharedMemory[];
    
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    int idx = index;
    for (int idx = index; idx < subdomainLength * subdomainLength; idx += stride) {
        if (idx % subdomainLength < subdomainLength/2) {
            int Idx = ((subdomainLength-1)/2-(idx % subdomainLength)) * subdomainLength + idx/subdomainLength;
            sharedMemory[idx] = xLeftBlock[Idx];
            sharedMemory[idx + subdomainLength * subdomainLength] = xLeftBlock[Idx];
        }
        else {
            int Idx = ((idx % subdomainLength) - (subdomainLength-1)/2 - 1) * subdomainLength + idx/subdomainLength;
            sharedMemory[idx] = xRightBlock[Idx];
            sharedMemory[idx + subdomainLength * subdomainLength] = xRightBlock[Idx];
        }
    }

    int iGrid = blockShift + (index/subdomainLength) * nxGrids + index % subdomainLength + horizontalShift;

    __syncthreads();
  
    __iterativeBlockUpdateToNorthSouth(xTopBlock, xBottomBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, iGrid, method, subdomainLength, false);
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
    int verticalShift = subdomainLength/2 * nxGrids;

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
    
    float * xLeftBlock = xLeftGpu + arrayShift;
    float * xRightBlock = xRightGpu + arrayShift;

    extern __shared__ float sharedMemory[];
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    for (int idx = index; idx < subdomainLength * subdomainLength; idx += stride) {
        if (idx < numElementsPerBlock) {
            sharedMemory[idx] = xBottomBlock[(subdomainLength/2-1-idx/subdomainLength) * subdomainLength + idx % subdomainLength];
            sharedMemory[idx + subdomainLength * subdomainLength] = xBottomBlock[(subdomainLength/2-1-idx/subdomainLength) * subdomainLength + idx % subdomainLength];
        }
        else {
            sharedMemory[idx] = xTopBlock[idx - numElementsPerBlock];
            sharedMemory[idx + subdomainLength * subdomainLength] = xTopBlock[idx - numElementsPerBlock];
        }
    }
    
    int iGrid = blockShift + (index/subdomainLength) * nxGrids + index % subdomainLength + horizontalShift + verticalShift;
    
    __iterativeBlockUpdateToLeftRight(xLeftBlock, xRightBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, iGrid, method, subdomainLength, true);
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
    int verticalShift = subdomainLength/2 * nxGrids;

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

    extern __shared__ float sharedMemory[];

    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;
    int idx = index;
    for (int idx = index; idx < subdomainLength * subdomainLength; idx += stride) {
        if (idx % subdomainLength < subdomainLength/2) {
            int Idx = ((subdomainLength-1)/2-(idx % subdomainLength)) * subdomainLength + idx/subdomainLength;
            sharedMemory[idx] = xLeftBlock[Idx];
            sharedMemory[idx + subdomainLength * subdomainLength] = xLeftBlock[Idx];
        }
        else {
            int Idx = ((idx % subdomainLength) - (subdomainLength-1)/2 - 1) * subdomainLength + idx/subdomainLength;
            sharedMemory[idx] = xRightBlock[Idx];
            sharedMemory[idx + subdomainLength * subdomainLength] = xRightBlock[Idx];
        }
    }

    int iGrid = blockShift + (index/subdomainLength) * nxGrids + index % subdomainLength + verticalShift;

    __iterativeBlockUpdateToNorthSouth( xTopBlock, xBottomBlock, rhsBlock,
    		           leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
			   nxGrids, nyGrids, iGrid, method, subdomainLength, true);
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

    __syncthreads();

    
    float * x0 = x0Gpu + blockShift;
    
    for (int idx = index; idx < 2*numElementsPerBlock; idx += stride) {
        int Idx = (idx % subdomainLength) + (idx/subdomainLength) * nxGrids;
        x0Block[Idx] = sharedMemory[idx];
    }
}

