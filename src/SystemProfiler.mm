#include "SystemProfiler.hpp"

#include <chrono>
#include <cstring>
#include <cmath>

#ifdef _WIN32
    #include <windows.h>
    #include <sysinfoapi.h>
    #include <pdh.h>
    #include <pdhmsg.h>
    #include <psapi.h>
    #include <dxgi1_6.h>
#elif __APPLE__
    #include <sys/sysctl.h>
    #include <mach/mach.h>
    #include <mach/mach_host.h>
    #import <Metal/Metal.h>
#else   // Linux
    #include <fstream>
    #include <sstream>
    #include <unistd.h>
    #include <nvml.h>
#endif

SystemProfiler::SystemProfiler() : running(true)
{
    Track cpu = {};
    cpu.name = "CPU Usage (%)";
    cpu.maxValue = 100.0f;
    cpu.fn = getCPUUsage;
    tracks.push_back(cpu);

    Track ram = {};
    ram.name = "Ram Usage (%)";
    ram.maxValue = 100.0f;
    ram.fn = getRAMUsage;
    tracks.push_back(ram);

    auto gpuCount = detectGpuCount();
    for (int i = 0; i < gpuCount; i++) {
        Track gpu = {};
        static std::string name; name = "GPU " + std::to_string(i);
        gpu.name = name.c_str();
        gpu.maxValue = 100.0f;
        gpu.fn = getGPUUsage;
        tracks.push_back(gpu);
    }

    worker = std::thread(&SystemProfiler::workerThread, this);
}

SystemProfiler::~SystemProfiler()
{
    running = false;
    if (worker.joinable()) worker.join();
}

void SystemProfiler::workerThread()
{
    while (running)
    {
        std::lock_guard<std::mutex> lock(swapMutex);

        float t = (float)std::chrono::duration_cast<
            std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        for (auto& tr : tracks)
            pushValue(tr, tr.fn(tr.gpuIndex));   // call CPU/GPU/RAM function

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void SystemProfiler::pushValue(Track& t, float v)
{
    int i = t.head;
    t.values[i] = v;
    t.timestamps[i] =  (float)i;
    t.head = (i + 1) % BUFFER_SIZE;
}

void SystemProfiler::renderGraphs()
{
    std::lock_guard<std::mutex> lock(swapMutex);

    float totalWidth = ImGui::GetContentRegionAvail().x;
    float spacing = ImGui::GetStyle().ItemSpacing.x;

    int columns = GetDynamicColumns(280.0f);

    float plotWidth = (totalWidth - (columns - 1) * spacing) / columns;
    ImVec2 plotSize(plotWidth, plotWidth * 0.75f);

    int index = 0;

    for (auto& tr : tracks)
    {
        if (ImPlot::BeginPlot(tr.name, plotSize))
        {
            ImPlot::SetupAxes("Time", "Usage %");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, BUFFER_SIZE, ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, tr.maxValue, ImGuiCond_Always);

            ImPlot::PlotLine(tr.name, tr.timestamps, tr.values, BUFFER_SIZE);

            ImPlot::EndPlot();
        }

        index++;

        if (index % columns != 0)
            ImGui::SameLine();
    }
}



float SystemProfiler::getCPUUsage(int)
{
    #ifdef _WIN32
        static PDH_HQUERY query;
        static PDH_HCOUNTER counter;
        static bool init = false;

        if (!init)
        {
            PdhOpenQuery(NULL, 0, &query);
            PdhAddCounter(query, TEXT("\\Processor(_Total)\\% Processor Time"), 0, &counter);
            PdhCollectQueryData(query);
            init = true;
            return 0;
        }

        PdhCollectQueryData(query);

        PDH_FMT_COUNTERVALUE val;
        PdhGetFormattedCounterValue(counter, PDH_FMT_DOUBLE, NULL, &val);

        return (float)val.doubleValue;
    #elif __APPLE__
        host_cpu_load_info_data_t cpuInfo;
        mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
        host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, (host_info_t)&cpuInfo, &count);

        static uint64_t lastIdle = 0;
        static uint64_t lastTotal = 0;

        uint64_t idle = cpuInfo.cpu_ticks[CPU_STATE_IDLE];
        uint64_t total = idle + cpuInfo.cpu_ticks[CPU_STATE_USER] +
                        cpuInfo.cpu_ticks[CPU_STATE_SYSTEM] +
                        cpuInfo.cpu_ticks[CPU_STATE_NICE];

        uint64_t dIdle = idle - lastIdle;
        uint64_t dTotal = total - lastTotal;

        lastIdle = idle;
        lastTotal = total;

        if (dTotal == 0) return 0.0f;
        return (1.0f - (float)dIdle / dTotal) * 100.0f;
    #else   // Linux
        static long long lastUser=0, lastNice=0, lastSys=0, lastIdle=0;

        std::ifstream f("/proc/stat");
        std::string cpu;
        long long user, nice, sys, idle;
        f >> cpu >> user >> nice >> sys >> idle;

        long long dUser = user - lastUser;
        long long dNice = nice - lastNice;
        long long dSys  = sys  - lastSys;
        long long dIdle = idle - lastIdle;

        lastUser = user; lastNice = nice; lastSys = sys; lastIdle = idle;

        long long total = dUser + dNice + dSys + dIdle;
        if (total == 0) return 0;

        return (float)(total - dIdle) / total * 100.0f;
    #endif
}

float SystemProfiler::getRAMUsage(int)
{
    #ifdef _WIN32
        MEMORYSTATUSEX  mem;
        mem.dwLength = sizeof(MEMORYSTATUSEX);
        GlobalMemoryStatusEx(&mem); 

        return (float)(mem.ullTotalPhys - mem.ullAvailPhys) / mem.ullTotalPhys * 100.0f;
    #elif __APPLE__
        vm_size_t page;
        mach_port_t host = mach_host_self();
        vm_statistics64_data_t stats;
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

        host_page_size(host, &page);
        host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&stats, &count);

        uint64_t used = (stats.active_count + stats.inactive_count + stats.wire_count) * page;

        uint64_t total;
        size_t len = sizeof(total);
        sysctlbyname("hw.memsize", &total, &len, NULL, 0);

        return (float)used / total * 100.0f;
    #else   // Linux
        std::ifstream file("/proc/meminfo");
        std::string line;
        long memTotal = 0, memAvailable = 0;

        while (std::getline(file, line)) {
            if (line.find("MemTotal:") == 0)
                sscanf(line, "MemTotal: %ld kB", &memTotal);
            else if (line.find("MemAvailable:") == 0)
                sscanf(line, "MemAvailable: %ld kB", &memAvailable);
        }

        return (float)(memTotal - memAvailable) / memTotal * 100.0f;
    #endif
}

int SystemProfiler::detectDXGIGpuCount()
{
#ifdef _WIN32
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory)))
        return 0;

    UINT index = 0;
    IDXGIAdapter1* adapter = nullptr;
    int count = 0;

    while (factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND)
    {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE))
            count++;

        adapter->Release();
        index++;
    }

    factory->Release();
    return count;
#else
    return 2;
#endif
}

int SystemProfiler::detectGpuCount()
{   
    #ifdef _WIN32
        return 0;
    #elif __APPLE__
        NSArray* devs = MTLCopyAllDevices();
        return (int)[devs count];
    #else   // Linux
        nvmlInit(); 
        unsigned int count = 0;
        nvmlDeviceGetCount(&count);
        return (int)count;
    #endif
}

float SystemProfiler::getGPUUsage(int gpuIndex)
{
    #ifdef _WIN32
        static PDH_HQUERY query = nullptr;
        static PDH_HCOUNTER counter;

        if (!query)
        {
            PdhOpenQuery(NULL, NULL, &query);
            PdhAddCounterW(query, L"\\GPU Engine(*)\\Utilization Percentage", 0, &counter);
            PdhCollectQueryData(query);
            Sleep(100);
        }

        PdhCollectQueryData(query);

        PDH_FMT_COUNTERVALUE value;
        PdhGetFormattedCounterValue(counter, PDH_FMT_DOUBLE, NULL, &value);

        return (float)value.doubleValue;
    #elif __APPLE__
        @autoreleasepool {
            NSArray* devs = MTLCopyAllDevices();
            if (gpuIndex >= [devs count]) return 0.0f;

            id<MTLDevice> dev = devs[gpuIndex];
            uint64_t used = dev.currentAllocatedSize;
            uint64_t total = [dev recommendedMaxWorkingSetSize];

            if (total == 0) return 0;

            return (float)used / total * 100.0f;
        }
    #else   // Linux
        nvmlDevice_t dev;

        if (nvmlDeviceGetHandleByIndex(gpuIndex, &dev) != NVML_SUCCESS)
            return 0.0f;

        nvmlUtilization_t util;
        if (nvmlDeviceGetUtilizationRates(dev, &util) != NVML_SUCCESS)
            return 0.0f;

        return (float)util.gpu;
    #endif
}
