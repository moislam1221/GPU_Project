float normFromRow(float leftMatrix, float centerMatrix, float rightMatrix, float leftX, float centerX, float rightX,  float centerRhs)
{
    return centerRhs - (leftMatrix*leftX + centerMatrix*centerX + rightMatrix*rightX);
}

float Residual(const float * solution, const float * rhs, const float * leftMatrix, const float * centerMatrix, const float * rightMatrix, int nGrids)
{
    int nDofs = nGrids;
    float residual = 0.0;

    float leftX, centerX, rightX, residualContributionFromRow;

    for (int iGrid = 0; iGrid < nDofs; iGrid++) {
        if (iGrid == 0 || iGrid == nGrids-1) {
            residualContributionFromRow = 0;
        }
        else {
            leftX = solution[iGrid - 1];
            centerX = solution[iGrid];
            rightX = solution[iGrid + 1];
            residualContributionFromRow = normFromRow(leftMatrix[iGrid], centerMatrix[iGrid], rightMatrix[iGrid], leftX, centerX, rightX, rhs[iGrid]);
        }

        residual = residual + residualContributionFromRow * residualContributionFromRow;
    }

    residual = sqrt(residual);
    return residual;
}

