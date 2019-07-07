float normFromRow(float leftMatrix, float centerMatrix, float rightMatrix, float leftX, float centerX, float rightX,  float centerRhs)
{
    return centerRhs - (leftMatrix*leftX + centerMatrix*centerX + rightMatrix*rightX);
}

float Residual(const float * solution, const float * rhs, const float * leftMatrix, const float * centerMatrix, const float * rightMatrix, int nGrids)
{
    int nDofs = nGrids;
    float residual = 0.0;
    for (int iGrid = 0; iGrid < nDofs; iGrid++) {
        float leftX = (iGrid > 0) ? solution[iGrid - 1] : 0.0f;
        float centerX = solution[iGrid];
        float rightX = (iGrid < nGrids - 1) ? solution[iGrid + 1] : 0.0f;
        float residualContributionFromRow = normFromRow(leftMatrix[iGrid], centerMatrix[iGrid], rightMatrix[iGrid], leftX, centerX, rightX, rhs[iGrid]);
        residual = residual + residualContributionFromRow * residualContributionFromRow;
        // printf("For gridpoint %d, residual contribution is %f\n", iGrid, residualContributionFromRow);
    }
    residual = sqrt(residual);
    return residual;
}

