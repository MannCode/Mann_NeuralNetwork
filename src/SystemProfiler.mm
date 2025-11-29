#include "SystemProfiler.hpp"

#include <chrono>
#include <cstring>
#include <cmath>

#include "mannlogger.hpp"

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

    std::stringstream outputText;

    int gpuCount = detectGpuCount();
    for (int i = 0; i < gpuCount; i++) {
        Track gpu = {};
        gpu.name = "GPU " + std::to_string(i);
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

        for (Track& tr : tracks)
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

    for (Track& tr : tracks)
    {
        if (ImPlot::BeginPlot(tr.name.c_str(), plotSize))
        {
            ImPlot::SetupAxes("Time", "Usage %");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, BUFFER_SIZE, ImGuiCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, tr.maxValue, ImGuiCond_Always);

            ImPlot::PlotLine(tr.name.c_str(), tr.timestamps, tr.values, BUFFER_SIZE);

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
        static FILETIME lastSysKernel, lastSysUser;
        static FILETIME lastProcKernel, lastProcUser;
        static bool initialized = false;

        FILETIME ftSysIdle, ftSysKernel, ftSysUser;
        FILETIME ftProcCreation, ftProcExit, ftProcKernel, ftProcUser;


        if (!initialized)
        {
            GetSystemTimes(&ftSysIdle, &ftSysKernel, &ftSysUser);
            GetProcessTimes(GetCurrentProcess(),
                            &ftProcCreation, &ftProcExit,
                            &ftProcKernel, &ftProcUser);
            lastSysKernel = ftSysKernel;
            lastSysUser   = ftSysUser;
            lastProcKernel = ftProcKernel;
            lastProcUser   = ftProcUser;
            initialized = true;
            return 0.0f;
        }

        GetSystemTimes(&ftSysIdle, &ftSysKernel, &ftSysUser);
        GetProcessTimes(GetCurrentProcess(),
                        &ftProcCreation, &ftProcExit,
                        &ftProcKernel, &ftProcUser);

        ULONGLONG sysKernelDiff = (*(ULONGLONG*)&ftSysKernel - *(ULONGLONG*)&lastSysKernel);
        ULONGLONG sysUserDiff   = (*(ULONGLONG*)&ftSysUser   - *(ULONGLONG*)&lastSysUser);

        ULONGLONG procKernelDiff = (*(ULONGLONG*)&ftProcKernel - *(ULONGLONG*)&lastProcKernel);
        ULONGLONG procUserDiff   = (*(ULONGLONG*)&ftProcUser   - *(ULONGLONG*)&lastProcUser);

        lastSysKernel = ftSysKernel;
        lastSysUser = ftSysUser;
        lastProcKernel = ftProcKernel;
        lastProcUser = ftProcUser;

        ULONGLONG sysTotal = sysKernelDiff + sysUserDiff;
        ULONGLONG procTotal = procKernelDiff + procUserDiff;

        if (sysTotal == 0) return 0.0f;

        float cpu = (float)procTotal / (float)sysTotal * 100.0f;
        return cpu;

    #else // __APPLE__
        return 0.0f;
    #endif
}

float SystemProfiler::getRAMUsage(int)
{
    #ifdef _WIN32
       PROCESS_MEMORY_COUNTERS_EX pmc = {};

        if (GetProcessMemoryInfo(GetCurrentProcess(),
            (PROCESS_MEMORY_COUNTERS*)&pmc,
            sizeof(pmc)))
        {
            SIZE_T used = pmc.PrivateUsage;

            MEMORYSTATUSEX mem = {};
            mem.dwLength = sizeof(mem);

            GlobalMemoryStatusEx(&mem);

            return (float)used / (float)mem.ullTotalPhys * 100.0f;
        }

        return 0.0f;
    #else __APPLE__
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
    #endif
}

int SystemProfiler::detectGpuCount()
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
    #elif __APPLE__
        NSArray* devs = MTLCopyAllDevices();
        return (int)[devs count];
    #endif
}

float SystemProfiler::getGPUUsage(int gpuIndex)
{
    #ifdef _WIN32
        static PDH_HQUERY query = nullptr;
        static PDH_HCOUNTER counter;
        static bool init = false;

        if (!init)
        {
            init = true;

            DWORD pid = GetCurrentProcessId();

            std::wstring counterPath =
                L"\\GPU Engine(pid_" +
                std::to_wstring(pid) + 
                L"_*)\\Utilization Percentage";

            PdhOpenQuery(NULL, 0, &query);
            PdhAddCounterW(query, counterPath.c_str(), 0, &counter);
            PdhCollectQueryData(query);
            return 0.0f;
        }

        PdhCollectQueryData(query);

        PDH_FMT_COUNTERVALUE value {};
        PdhGetFormattedCounterValue(counter, PDH_FMT_DOUBLE, NULL, &value);

        return (float)value.doubleValue;
    #else __APPLE__
       @autoreleasepool {
            NSArray* devs = MTLCopyAllDevices();
            if (gpuIndex >= [devs count]) return 0.0f;

            id<MTLDevice> dev = devs[gpuIndex];
            uint64_t used = dev.currentAllocatedSize;
            uint64_t total = [dev recommendedMaxWorkingSetSize];

            if (total == 0) return 0;

            return (float)used / total * 100.0 f;
        }
    #endif
}
