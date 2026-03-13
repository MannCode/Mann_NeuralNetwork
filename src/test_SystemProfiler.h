#ifndef SYSTEM_PROFILER_H
#define SYSTEM_PROFILER_H
      
#include <vector>
#include <string>
#include <thread>
#include <mutex>

#include "utils.h"

class SystemProfiler
{
public:
    static const int BUFFER_SIZE = 120;

    struct Track {
        const char* name;
        float values[BUFFER_SIZE];
        float timestamps[BUFFER_SIZE];
        int head;
        float maxValue;
        float (*fn)(int index); // ptr to system query fn
        int gpuIndex = -1; // CPU&RAM = -1 and GPU >= 0
    };

    SystemProfiler();
    ~SystemProfiler();

    void renderGraphs();

private:
    std::vector<Track> tracks;
    bool running;
    std::thread worker;
    std::mutex swapMutex;

    void workerThread();
    void pushValue(Track& t, float v);

    static float getCPUUsage(int index);
    static float getRAMUsage(int index);
    static float getGPUUsage(int index);

    static int detectDXGIGpuCount();
    static int detectGpuCount();
};

#endif // SYSTEM_PROFILER_H
