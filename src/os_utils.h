#if defined(_WIN32) || defined(_WIN64) || defined(__linux__)
      
#elif defined(__APPLE__) && defined(__MACH__)
//        #define NS_PRIVATE_IMPLEMENTATION
        #define CA_PRIVATE_IMPLEMENTATION
//        #define MTL_PRIVATE_IMPLEMENTATION

        #include <Metal/Metal.hpp>
#endif
