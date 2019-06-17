// Includes two helper functions
// 1 - Compute the Residual of the current solution (uses another helper function normFromRow)
// 2 - Print the solution in a 2D array 



// COMPUTE THE RESIDUAL //
///////////////////////////////////////////////////////////////////////////////////////////////////////////
float normFromRow(float leftMatrix, float centerMatrix, float rightMatrix, float topMatrix, float bottomMatrix, float leftX, float centerX, float rightX,  float topX, float bottomX, float centerRhs) 
{
    return centerRhs - (leftMatrix*leftX + centerMatrix*centerX + rightMatrix*rightX + topMatrix*topX + bottomMatrix*bottomX);
}


float Residual(const float * solution, const float * rhs, const float * matrixElements, const int nxGrids, const int nyGrids)
{
    // From matrixElements, extract the 5 entries of the matrix for 2D Poisson
    const float bottomMatrix = matrixElements[0];
    const float leftMatrix = matrixElements[1];
    const float centerMatrix = matrixElements[2];
    const float rightMatrix = matrixElements[3];
    const float topMatrix = matrixElements[4];

    // Declare variables used later
    float leftX, rightX, centerX, bottomX, topX;
    int dof;
    float residualContributionFromDOF;

    // Initialize the residual to be zero (the contribution of each DOF to the residual will be included later)
    float residual = 0.0;  

    // Loop over all grid points
    for (int iGrid = 0; iGrid < nyGrids; iGrid++) {
        for (int jGrid = 0; jGrid < nxGrids; jGrid++) {
            
            // Obtain index of current dof under consideration
            dof = jGrid + iGrid * nxGrids;
            
            // If this point lies on boundary, there is no contribution to residual (it satisfied BC so residual from here is zero)
            if (iGrid == 0 || iGrid == nxGrids - 1 || jGrid == 0 || jGrid == nyGrids - 1) {
                residualContributionFromDOF = 0.0f;
            }

            // Otherwise, compute residual contribution of the point! 
            else {
                leftX = solution[dof-1];
                centerX = solution[dof];
                rightX = solution[dof+1];
                bottomX = solution[dof-nxGrids];
                topX = solution[dof+nxGrids];
                residualContributionFromDOF = normFromRow(leftMatrix, centerMatrix, rightMatrix, topMatrix, bottomMatrix, leftX, centerX, rightX, topX, bottomX, rhs[dof]);
            }
         
            // Eventually obtain sum of squares of all residuals 
            residual = residual + residualContributionFromDOF * residualContributionFromDOF;

        }
    }

    // Take the square root to obtain L2 norm of residual 
    residual = sqrt(residual);
    return residual;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

// PRINT SOLUTION IN 2D ARRAY //
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void print2DSolution(const float * solution, const int nxGrids, const int nyGrids)
{
    // Declare degree of free variable
    int dof;

    // Loop over all grid points
    for (int iGrid = nyGrids-1; iGrid > -1; iGrid--) {
        for (int jGrid = 0; jGrid < nxGrids; jGrid++) {
            
            // Obtain index of current dof under consideration and print value
            dof = jGrid + iGrid * nxGrids;
            printf("%f ", solution[dof]); 
        }
        
        // Print newline to get to next row and continue printing values
        printf("\n"); 

    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
