enum method_type { JACOBI, GS, SOR };

__host__ __device__
float jacobi(const float leftMatrix, const float centerMatrix, const float rightMatrix, float leftX, float centerX, float rightX, const float centerRhs)
{
    return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix;
}

__host__ __device__
float relaxedJacobi(const float leftMatrix, const float centerMatrix, const float rightMatrix, float leftX, float centerX, float rightX, const float centerRhs, const float omega)
{
    return gridValue = relaxation*((centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix) + (1.0-relaxation)*centerX;
}


template <typename method_type>
__host__ __device__
float iterativeOperation(const float leftMatrix, const float centerMatrix, const float rightMatrix, float leftX, float centerX, float rightX, const float centerRhs, int gridPoint, method_type method)
{
    float gridValue = centerX;
    switch(method)
    {
        case JACOBI:
	    return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix;
	case GS:
	    if (gridPoint % 2 == 1) {
	        return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix;
	    }
	case SOR:
	    float relaxation = 1.9939;
	    if (gridPoint % 2 == 1) {
	        return gridValue = relaxation*((centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix) + (1.0-relaxation)*centerX;
	    }
    }
    return gridValue;
}


template <typename method_type>
__host__ __device__
float iterativeOperation2(const float leftMatrix, const float centerMatrix, const float rightMatrix, float leftX, float centerX, float rightX, const float centerRhs, int gridPoint, method_type method)
{
    float gridValue = centerX;
    switch(method)
    {
	case JACOBI:	
	    return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix;
	case GS:
	    if (gridPoint % 2 == 0) {
	        return gridValue = (centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix;
	    }
	case SOR:
	    float relaxation = 1.9939;
	    if (gridPoint % 2 == 0) {
	        return gridValue = relaxation*((centerRhs - (leftMatrix * leftX + rightMatrix * rightX)) / centerMatrix) + (1.0-relaxation)*centerX;
	    }
    }
    return gridValue;
}
