__device__
void __iterativeBlockLongitudinalBridgeFromShared(double * xLowerBlock, double * xUpperBlock, double * xEastBlock, double * xWestBlock, double * rhsBlock, double * leftMatrixBlock, double * centerMatrixBlock, double * rightMatrixBlock, double * topMatrixBlock, double * bottomMatrixBlock, int nxGrids, int nyGrids, int iGrid, int method)
{  
    extern __shared__ double sharedMemory[];
    double * x0 = sharedMemory;
    double * x1 = sharedMemory + blockDim.x * blockDim.x;
   
    int idx = threadIdx.x + threadIdx.y * blockDim.x;

    //printf("The %d entry of East Gpu is %f\n", idx, xEastBlock[idx]);
    //printf("The %d entry of West Gpu is %f\n", idx, xWestBlock[idx]);

    //printf("The upper block value in idx %d is %f\n", idx, xUpperBlock[idx]);
    //printf("The lower block value in idx %d is %f\n", idx, xLowerBlock[idx]);
    
    // At every step, load xLower and xUpper and fill in values
    for (int k = 0; k <= blockDim.x/2-1; k++) 
    {
//	printf("Entering k = %d now\n", k);

	// For all threads in x-range
	if (threadIdx.x >= k && threadIdx.x <= blockDim.x-1-k) {
		
	    if (threadIdx.y == blockDim.y/2-1-k)
	    {
                x0[idx] = xLowerBlock[threadIdx.x-k+(2*k)*(blockDim.x-(k-1))];
	        //printf("In blockIdx.x %d, blockIdx.y %d, threadIdx.x %d, threadIdx.y %d, idx %d and obtained from xLowerBlock %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, idx, threadIdx.x-k+(2*k)*(blockDim.x-(k-1)));
	    }
	    if (threadIdx.y == blockDim.y/2-2-k && k != blockDim.x/2-1)
	    {
	         x0[idx] = xLowerBlock[threadIdx.x-k+(2*k)*(blockDim.x-k)+blockDim.x];
	         //printf("In blockIdx.x %d, blockIdx.y %d, threadIdx.x %d, threadIdx.y %d, idx %d and obtained from xLowerBlock %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, idx, threadIdx.x-k+(2*k)*(blockDim.x-k)+blockDim.x);
	    }
	    if (threadIdx.y == blockDim.y/2+k)
	    {
	         x0[idx] = xUpperBlock[threadIdx.x-k+(2*k)*(blockDim.x-(k-1))];
	         //printf("In blockIdx.x %d, blockIdx.y %d, threadIdx.x %d, threadIdx.y %d, idx %d and obtained from xUpperBlock %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, idx, threadIdx.x-k+(2*k)*(blockDim.x-(k-1)));
	    }
	    if (threadIdx.y == blockDim.y/2+1+k && k != blockDim.x/2-1)
	    {
	         x0[idx] = xUpperBlock[threadIdx.x-k+(2*k)*(blockDim.x-k)+blockDim.x];
	         //printf("In blockIdx.x %d, blockIdx.y %d, threadIdx.x %d, threadIdx.y %d, idx %d and obtained from xUpperBlock %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, idx, threadIdx.x-k+(2*k)*(blockDim.x-k)+blockDim.x);
	    }
	}
	//    printf("Now x0[idx= %d] has value %f\n", idx, x0[idx]);

	if ((threadIdx.x == blockDim.x-2-k || threadIdx.x == blockDim.x-1-k) && (threadIdx.y >= blockDim.x/2-k && threadIdx.y <= blockDim.x/2+1+k))        {
	    int shift = 2*k*(k+1);
	    if (k != blockDim.x/2-1) {
	        if (threadIdx.x == blockDim.x-1-k) {
		    xEastBlock[blockDim.x/2+1+k-threadIdx.y+shift] = x0[idx]; // Right column
	        }
	        else {
		    xEastBlock[blockDim.x/2+1+k-threadIdx.y+shift+2*(k+1)] = x0[idx]; // Left column
	        }
	    }
	    else {
	        int numSharedElemPerBlock = blockDim.x * (blockDim.x / 2 + 1);
	        xEastBlock[shift] = xUpperBlock[numSharedElemPerBlock-1];
	        xEastBlock[shift+2*(k+1)] = xUpperBlock[numSharedElemPerBlock-2];
		    if (threadIdx.x == blockDim.x-1-k) {
		        xEastBlock[blockDim.x/2+1+k-threadIdx.y+shift] = x0[idx]; // Right column
		    }
		    else {
		        xEastBlock[blockDim.x/2+1+k-threadIdx.y+shift+2*(k+1)] = x0[idx]; // Left column
	         }   
   	    }	
        }
        
	if ((threadIdx.x == k || threadIdx.x == k+1) && (threadIdx.y + 1 >= blockDim.x/2-1-k && threadIdx.y <= blockDim.x/2-1+k))
	{
	    int shift = 2*k*(k+1);
	    if (k != blockDim.x/2-1) {
	        if (threadIdx.x == k) {
		    xWestBlock[(blockDim.x/2-1+k)-threadIdx.y+shift] = x0[idx]; // Left column
		}
		else {
		    xWestBlock[(blockDim.x/2-1+k)-threadIdx.y+shift+2*(k+1)] = x0[idx]; // Right column
		}
	    }
	    else {
	        int numSharedElemPerBlock = blockDim.x * (blockDim.x / 2 + 1);
	        xWestBlock[shift+2*(k+1)-1] = xLowerBlock[numSharedElemPerBlock-2];
	        xWestBlock[numSharedElemPerBlock-1] = xLowerBlock[numSharedElemPerBlock-1];
	        if (threadIdx.x == k) {
	            xWestBlock[(blockDim.x/2-1+k)-threadIdx.y+shift] = x0[idx]; // Left column
		}
		else {
		    xWestBlock[(blockDim.x/2-1+k)-threadIdx.y+shift+2*(k+1)] = x0[idx]; // Right column
		}
	    }
	}

	printf("Now the %d entry of WestBlock is %f\n", idx, xWestBlock[idx]);
	
	if (k != blockDim.x/2 && threadIdx.x > k && threadIdx.x < blockDim.x-1-k && threadIdx.y >= blockDim.y/2-1-k && threadIdx.y <= blockDim.y/2 + k)
	{
	    double leftX = x0[idx];
	    double centerX = x0[idx];
	    double rightX = x0[idx];
	    double topX = x0[idx];
            double bottomX = x0[idx];

            double leftMat = leftMatrixBlock[idx];
	    double centerMat = centerMatrixBlock[idx];
	    double rightMat = rightMatrixBlock[idx];
	    double topMat = topMatrixBlock[idx];
	    double bottomMat = bottomMatrixBlock[idx];
	    
	    if (k % 2 == 0) {
                x1[idx] = centerX; /* (leftMat, centerMat, rightMat, topMat, bottomMat, leftX, centerX, rightX, topX, bottomX,
					     rhs, iGrid, method); */
            }
	    else {
                x1[idx] = centerX; /*  iterativeOperation2(leftMat, centerMat, rightMat, topMat, bottomMat, leftX, centerX, rightX, topX, bottomX,
					     rhs, iGrid, method); */
            }
	}

	__syncthreads();
	double * tmp = x0; x1 = x0; x0 = tmp;
    }
}

__global__       
void _iterativeGpuLongitudinalBridge(double * xTopGpu, double * xBottomGpu, double * xEastGpu, double * xWestGpu, double * x0Gpu,
                                  double * rhsGpu, double * leftMatrixGpu, double * centerMatrixGpu, double * rightMatrixGpu, 
				  double * topMatrixGpu, double * bottomMatrixGpu, int nxGrids, int nyGrids, int method)
{
    int numSharedElemPerBlock = blockDim.x * (blockDim.x / 2 + 1);
    int blockID =  blockIdx.y * gridDim.x + blockIdx.x;
    int nDofs = nxGrids * nyGrids;

    int sharedShift = numSharedElemPerBlock * blockID;
    double * xLowerBlock = xTopGpu + sharedShift;
    double * xUpperBlock = (blockIdx.y == (gridDim.y-1)) ?
                           xBottomGpu + numSharedElemPerBlock * blockIdx.x : 
                           xBottomGpu + sharedShift + gridDim.x * numSharedElemPerBlock;

    int blockShift = (blockDim.x * blockDim.y) * blockID;
    int verticalShift = blockDim.y/2 * nxGrids;
    
    int idx = threadIdx.x + threadIdx.y * blockDim.x;
    int iGrid = blockShift + (threadIdx.x + threadIdx.y * nxGrids) + verticalShift;
    iGrid = (iGrid < nDofs) ? iGrid : iGrid - nDofs; 

    //printf("In entry %d, the value of xTopGpu is %f\n", idx, xTopGpu[idx]);

    double * rhsBlock = rhsGpu + blockShift + verticalShift;
    double * leftMatrixBlock = leftMatrixGpu + blockShift + verticalShift;
    double * centerMatrixBlock = centerMatrixGpu + blockShift + verticalShift;
    double * rightMatrixBlock = rightMatrixGpu + blockShift + verticalShift;
    double * topMatrixBlock = centerMatrixGpu + blockShift + verticalShift;
    double * bottomMatrixBlock = rightMatrixGpu + blockShift + verticalShift;
     
    double * xEastBlock = xEastGpu + blockShift;
    double * xWestBlock = xWestGpu + blockShift;

    extern __shared__ double sharedMemory[];
    
    __iterativeBlockLongitudinalBridgeFromShared(xLowerBlock, xUpperBlock, xEastBlock, xWestBlock, rhsBlock,
                                       leftMatrixBlock, centerMatrixBlock, rightMatrixBlock, topMatrixBlock, bottomMatrixBlock,
				       nxGrids, nyGrids, iGrid, method);  

}

