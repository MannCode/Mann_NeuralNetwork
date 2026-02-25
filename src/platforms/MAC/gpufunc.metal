
#include <metal_stdlib>
using namespace metal;

struct MatrixDims {
    uint M;
    uint N;
    uint K;
};

kernel void multiplyMatrix(
    device const float* matrixA [[ buffer(0) ]],
    device const float* matrixB [[ buffer(1) ]],
    device float* resultMatrix [[ buffer(2) ]],
    constant MatrixDims& dims [[ buffer(3) ]],
    uint2 gid [[ thread_position_in_grid ]] )
{
    float sum = 0.0f;

    for(uint _k=0; _k < dims.K; _k++) {
        float a = matrixA[gid.x * dims.K + _k];
        float b = matrixB[_k * dims.N + gid.y];
        sum += a * b;
    }
    
    resultMatrix[gid.x * dims.N + gid.y] = sum;
}
