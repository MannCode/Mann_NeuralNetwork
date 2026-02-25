#include "../platform.h"



#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#define MP MetalPlatform

MP::MetalPlatform()
{
    pool = NS::AutoreleasePool::alloc()->init();

    device = MTL::CreateSystemDefaultDevice();
    assert(device);

    library = device->newLibrary(NS::String::string("build/default.metallib", NS::UTF8StringEncoding), &error);
    if (!library)
    {
        throw std::runtime_error("Failed to create Metal library.");
    }

    multFunction = library->newFunction(NS::String::string("multiplyMatrix", NS::UTF8StringEncoding));
    assert(multFunction);

    multPipelineState = device->newComputePipelineState(multFunction, &error);
    if (!multPipelineState)
    {
        throw std::runtime_error("Failed to create Metal compute pipeline state.");
    }
}

MP::~MetalPlatform()
{
    multPipelineState->release();
    multFunction->release();
    library->release();
    device->release();
    pool->release();
}

void MP::matrixMultiply(const Mann::Matrix& A, const Mann::Matrix& B, Mann::Matrix& C)
{
    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    assert(commandQueue);

    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    assert(commandBuffer);

    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder);

    unsigned int M = B.cols();
    unsigned int N = A.rows();
    unsigned int K = A.cols();

    MTL::Buffer* bufferA = device->newBuffer(A.data().data(), N * K * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferB = device->newBuffer(B.data().data(), K * M * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferC = device->newBuffer(N * M * sizeof(float), MTL::ResourceStorageModeShared);

    computeEncoder->setComputePipelineState(multPipelineState);
    computeEncoder->setBuffer(bufferA, 0, 0);
    computeEncoder->setBuffer(bufferB, 0, 1);
    computeEncoder->setBuffer(bufferC, 0, 2);

    unsigned int dims[3] = {M, N, K};
    computeEncoder->setBytes(dims, sizeof(dims), 3);

    MTL::Size threadPerGrid = MTL::Size::Make(M, N, 1);
    MTL::Size threadGroupSize = MTL::Size::Make(32, 32, 1);

    computeEncoder->dispatchThreads(threadPerGrid, threadGroupSize);
    computeEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    float* resultData = static_cast<float*>(bufferC->contents());
    for(int i = 0; i < N * M; i++)
    {
        C[i] = resultData[i];
    }

    bufferA->release();
    bufferB->release();
    bufferC->release();
    computeEncoder->release();
    commandBuffer->release();
    commandQueue->release();
}