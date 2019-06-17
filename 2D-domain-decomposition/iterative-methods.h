// To contain all iterative method functions (host and device)
// Right now we only have jacobi iteration

// Jacobi Iteration
__host__ __device__
float jacobi(const float leftMatrix, const float centerMatrix, const float rightMatrix, const float topMatrix, const float bottomMatrix,
              const float leftX, const float centerX, const float rightX, const float topX, const float bottomX,
              const float centerRhs) {
    float result = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX + topMatrix * topX + bottomMatrix * bottomX)) / centerMatrix;
    return result;
}

