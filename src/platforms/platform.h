#ifndef _PLATFORM
#define _PLATFORM

#include "utils.h"
#include "mann.h"

#include <assert.h>

namespace MTL
{
    class Device;
    class Library;
    class Function;
    class ComputePipelineState;
}

namespace NS
{
    class AutoreleasePool;
    class Error;
}


class MetalPlatform
{
public:
    NS::AutoreleasePool* pool;
    MTL::Device* device;
    NS::Error* error;
    MTL::Library* library;
    MTL::Function* multFunction;
    MTL::ComputePipelineState* multPipelineState;
    

    MetalPlatform();
    ~MetalPlatform();

    void matrixMultiply(const Mann::Matrix& A, const Mann::Matrix& B, Mann::Matrix& C);
};



#endif //_ PLATFORM