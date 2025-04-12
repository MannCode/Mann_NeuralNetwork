#include <metal_stdlib>
using namespace metal;

kernel void matrix_multiply(const device float* a [[ buffer(0) ]],
                             const device float* b [[ buffer(1) ]],
                             device float* result [[ buffer(2) ]],
                             uint id [[ thread_position_in_grid ]],
                             uint widthA [[ buffer(3) ]],
                             uint widthB [[ buffer(4) ]]) {
    uint row = id / widthB;
    uint col = id % widthB;

    float sum = 0.0;
    for (uint k = 0; k < widthA; ++k) {
        sum += a[row * widthA + k] * b[k * widthB + col];
    }
    result[id] = sum;
}