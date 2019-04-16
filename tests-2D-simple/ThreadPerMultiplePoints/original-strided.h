__host__ __device__
void boundaryConditions(int IGrid, int nxGrids, int nyGrids, double &leftX, double &rightX, double&bottomX, double &topX)
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
double increment(double centerX) {
     return centerX + 100.;
}

__device__
double jacobi(const double leftMatrix, const double centerMatrix, const double rightMatrix, const double topMatrix, const double bottomMatrix,
              const double leftX, const double centerX, const double rightX, const double topX, const double bottomX,
              const double centerRhs) {
    double result = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
    return result;
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
/*        double leftX = ((iGrid % nxGrids) == 0) ? 0.0 : solution[iGrid-1];
        double centerX = solution[iGrid];
        double rightX = ((iGrid + 1) % nxGrids == 0) ? 0.0 : solution[iGrid+1];
        double topX = (iGrid < nxGrids * (nyGrids - 1)) ? solution[iGrid + nxGrids] : 0.0;
        double bottomX = (iGrid >= nxGrids) ?  solution[iGrid-nxGrids] : 0.0;
*/
	double leftX = solution[iGrid-1];
	double centerX = solution[iGrid];
	double rightX = solution[iGrid+1];
	double topX = solution[iGrid+nxGrids];
	double bottomX = solution[iGrid-nxGrids];
	
	boundaryConditions(iGrid, nxGrids, nyGrids, leftX, rightX, bottomX, topX);

        double residualContributionFromRow = normFromRow(leftMatrix[iGrid], centerMatrix[iGrid], rightMatrix[iGrid], topMatrix[iGrid], bottomMatrix[iGrid], leftX, centerX, rightX, topX, bottomX, rhs[iGrid]);
	residual = residual + residualContributionFromRow * residualContributionFromRow;
    }
    residual = sqrt(residual);
    return residual;
}

__device__
void __iterativeBlockUpdateToLeftRight(double * xLeftBlock, double * xRightBlock, const double *rhsBlock, 
                             const double * leftMatrixBlock, const double *centerMatrixBlock, const double * rightMatrixBlock, 
			     const double * topMatrixBlock, const double * bottomMatrixBlock, int nxGrids, int nyGrids, int iGrid, int method, 
                             int subdomainLength, bool diagonal)
{
    // Initialize shared memory and pointers to x0, x1 arrays containing Jacobi solutions
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory; 
    int elemPerBlock = subdomainLength * subdomainLength;
    double * x1 = sharedMemory + elemPerBlock;

    // Define number of Jacobi steps to take, and current index and stride value
    int maxSteps = 2;
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    // Perform Jacobi iterations
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
                double leftX = x0[idx-1];
                double centerX = x0[idx];
                double rightX = x0[idx+1];
                double topX = x0[idx+subdomainLength];
                double bottomX = x0[idx-subdomainLength];
                
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

                boundaryConditions(IGrid, nxGrids, nyGrids, leftX, rightX, bottomX, topX);

                __syncthreads();
	        // Perform update
   	        // x1[idx] = increment(centerX);
                x1[idx] = jacobi(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
                                 leftX, centerX, rightX, topX, bottomX, centerRhs); 

                // Synchronize
	        __syncthreads();
                
	    }
        }
        __syncthreads();
        double * tmp; tmp = x0; x0 = x1;
    }

    __syncthreads();
    index = threadIdx.x + threadIdx.y * blockDim.x;
    stride = blockDim.x * blockDim.y;
    for (int idx = index; idx < elemPerBlock/2; idx += stride) {
        xLeftBlock[idx] = x0[subdomainLength * (idx % subdomainLength) + (idx/subdomainLength)];
        xRightBlock[idx] = x0[subdomainLength * (idx % subdomainLength) - (idx/subdomainLength) + (subdomainLength-1)];
    }
    
}

__device__
void __iterativeBlockUpdateToNorthSouth(double * xTopBlock, double * xBottomBlock, const double *rhsBlock, 
                             const double * leftMatrixBlock, const double *centerMatrixBlock, const double * rightMatrixBlock, 
			     const double * topMatrixBlock, const double * bottomMatrixBlock, int nxGrids, int nyGrids, int iGrid, int method, int subdomainLength, bool vertical)
{
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory; 
    int elemPerBlock = subdomainLength * subdomainLength;
    double * x1 = sharedMemory + elemPerBlock;
    int maxSteps = 2;
    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    for (int idx = index; idx < elemPerBlock; idx += stride) {
        __syncthreads();
    }
    
    for (int k = 0; k < maxSteps; k++) {
        for (int idx = index; idx < elemPerBlock; idx += stride) {
            if ((idx % subdomainLength != 0) && ((idx+1) % subdomainLength != 0) && (idx > subdomainLength-1) && (idx < elemPerBlock-subdomainLength-1)) {
                // Define necessary constants
                double centerRhs = rhsBlock[idx];
                double leftMatrix = leftMatrixBlock[idx];
                double centerMatrix = centerMatrixBlock[idx];
                double rightMatrix = rightMatrixBlock[idx];
                double topMatrix = topMatrixBlock[idx];
                double bottomMatrix = bottomMatrixBlock[idx];
                double leftX = x0[idx-1];
                double centerX = x0[idx];
                double rightX = x0[idx+1];
                double topX = x0[idx+subdomainLength];
                double bottomX = x0[idx-subdomainLength];
                
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
                
                boundaryConditions(IGrid, nxGrids, nyGrids, leftX, rightX, bottomX, topX);
                
                __syncthreads();
                // Perform update
     	        //x1[idx] = increment(centerX);
                x1[idx] = jacobi(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix,
                                 leftX, centerX, rightX, topX, bottomX, centerRhs); 
                
                // Synchronize
	        __syncthreads();
               }
	    }
	    double * tmp; tmp = x0; x0 = x1;
        }   
    

    // Return values for xTop and xBottom here
    for (int idx = index; idx < elemPerBlock/2; idx += stride) {
        xBottomBlock[idx] = x0[idx];
        xTopBlock[idx] = x0[subdomainLength * (subdomainLength-1-idx/subdomainLength) + (idx % subdomainLength)];
    }

}

__global__
void _iterativeGpuOriginal(double * xLeftGpu, double *xRightGpu,
                             const double * x0Gpu, const double *rhsGpu, 
                             const double * leftMatrixGpu, const double *centerMatrixGpu, const double * rightMatrixGpu, 
			     const double * topMatrixGpu, const double * bottomMatrixGpu, int nxGrids, int nyGrids, int method, int subdomainLength)
{

    __syncthreads();

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
    
    extern __shared__ double sharedMemory[];

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

    extern __shared__ double sharedMemory[];
    
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
void _iterativeGpuVerticalandHorizontalShift(double * xLeftGpu, double *xRightGpu, double * xTopGpu, double * xBottomGpu,
                                const double * x0Gpu, const double *rhsGpu, 
                                const double * leftMatrixGpu, const double *centerMatrixGpu, const double * rightMatrixGpu, 
			        const double * topMatrixGpu, const double * bottomMatrixGpu, int nxGrids, int nyGrids, int method, int subdomainLength)
{
    int xShift = subdomainLength * blockIdx.x;
    int yShift = subdomainLength * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    
    int horizontalShift = subdomainLength/2;
    int verticalShift = subdomainLength/2 * nxGrids;

    const double * rhsBlock = rhsGpu + blockShift; //+ verticalShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift; //+ verticalShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift; //+ verticalShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift; //+ verticalShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift; //+ verticalShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift; //+ verticalShift;

    int numElementsPerBlock = (subdomainLength * subdomainLength)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
    double * xBottomBlock = xTopGpu + arrayShift;
    double * xTopBlock = (blockIdx.y != gridDim.y-1) ?
                         xBottomGpu + numElementsPerBlock * gridDim.x + arrayShift :
			 xBottomGpu + (numElementsPerBlock * blockIdx.x);
    
    double * xLeftBlock = xLeftGpu + arrayShift;
    double * xRightBlock = xRightGpu + arrayShift;

    extern __shared__ double sharedMemory[];
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
void _iterativeGpuVerticalShift(double * xLeftGpu, double *xRightGpu, double * xTopGpu, double * xBottomGpu,
                                const double * x0Gpu, const double *rhsGpu, 
                                const double * leftMatrixGpu, const double *centerMatrixGpu, const double * rightMatrixGpu, 
			        const double * topMatrixGpu, const double * bottomMatrixGpu, int nxGrids, int nyGrids, int method, int subdomainLength)
{
    int xShift = subdomainLength * blockIdx.x;
    int yShift = subdomainLength * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    int verticalShift = subdomainLength/2 * nxGrids;

    const double * rhsBlock = rhsGpu + blockShift; //+ verticalShift;
    const double * leftMatrixBlock = leftMatrixGpu + blockShift; //+ verticalShift;
    const double * centerMatrixBlock = centerMatrixGpu + blockShift; //+ verticalShift;
    const double * rightMatrixBlock = rightMatrixGpu + blockShift; //+ verticalShift;
    const double * topMatrixBlock = topMatrixGpu + blockShift; //+ verticalShift;
    const double * bottomMatrixBlock = bottomMatrixGpu + blockShift; //+ verticalShift;

    int numElementsPerBlock = (subdomainLength * subdomainLength)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;
    
    double * xRightBlock =  xLeftGpu + arrayShift;
    double * xLeftBlock = (blockIdx.x != 0) ?
                           xRightGpu + arrayShift - numElementsPerBlock :
    			   xRightGpu + numElementsPerBlock * ((gridDim.x-1) + blockIdx.y * gridDim.x);
    
    double * xBottomBlock = xBottomGpu + arrayShift;
    double * xTopBlock = xTopGpu + arrayShift;

    extern __shared__ double sharedMemory[];

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
void _finalSolution(double * xTopGpu, double * xBottomGpu, double * x0Gpu, int nxGrids, int subdomainLength)
{
    extern __shared__ double sharedMemory[];
    int numElementsPerBlock = (subdomainLength * subdomainLength)/2;
    int blockID = blockIdx.x + blockIdx.y * gridDim.x;
    int arrayShift = numElementsPerBlock*blockID;

    double * xTopBlock = xBottomGpu + arrayShift;
    double * xBottomBlock = (blockIdx.y != 0) ?
			    xTopGpu + (blockIdx.x + (blockIdx.y-1) * gridDim.x) * numElementsPerBlock :
			    xTopGpu + (gridDim.x * (gridDim.y-1) + blockIdx.x) * numElementsPerBlock;

    int xShift = subdomainLength * blockIdx.x;
    int yShift = subdomainLength * blockIdx.y;
    int blockShift = xShift + yShift * nxGrids;
    double * x0Block = x0Gpu + blockShift;

    int index = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    for (int idx = index; idx < numElementsPerBlock; idx += stride) {
        sharedMemory[idx + numElementsPerBlock] = xTopBlock[idx]; 
	sharedMemory[(subdomainLength/2 - 1 - idx/subdomainLength) * subdomainLength + idx % subdomainLength] = xBottomBlock[idx];
    }

    __syncthreads();

    
    double * x0 = x0Gpu + blockShift;
    
    for (int idx = index; idx < 2*numElementsPerBlock; idx += stride) {
        int Idx = (idx % subdomainLength) + (idx/subdomainLength) * nxGrids;
        x0Block[Idx] = sharedMemory[idx];
    }
}

