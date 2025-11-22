#include "mannMetal.h"


MannMetal::MannMetal()
{
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cout << "Metal is not supported on this device" << std::endl;
        exit(1);
    }

    commandQueue = device->newCommandQueue();
}