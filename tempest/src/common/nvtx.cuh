#ifndef NVTX_CUH
#define NVTX_CUH

// no-op when HAS_NVTX is undefined so direct uses keep compiling.

#include <cstdint>

#if defined(HAS_CUDA) && defined(HAS_NVTX)
#include <nvtx3/nvToolsExt.h>
#endif

namespace nvtx_colors {
    constexpr uint32_t index_blue    = 0xFF2E5BDA;
    constexpr uint32_t walk_green    = 0xFF2FB54E;
    constexpr uint32_t weight_orange = 0xFFD97F28;
    constexpr uint32_t edge_purple   = 0xFF8B3FCE;
    constexpr uint32_t io_grey       = 0xFF808080;
}

class NvtxRange {
public:
#if defined(HAS_CUDA) && defined(HAS_NVTX)
    explicit NvtxRange(const char* name, uint32_t color = 0xFF808080) {
        nvtxEventAttributes_t attr = {};
        attr.version = NVTX_VERSION;
        attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attr.colorType = NVTX_COLOR_ARGB;
        attr.color = color;
        attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attr.message.ascii = name;
        nvtxRangePushEx(&attr);
    }
    ~NvtxRange() { nvtxRangePop(); }
#else
    explicit NvtxRange(const char*) {}
    explicit NvtxRange(const char*, uint32_t) {}
    ~NvtxRange() = default;
#endif

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

#define NVTX_RANGE(name)                  NvtxRange _nvtx_range_(name)
#define NVTX_RANGE_COLORED(name, color)   NvtxRange _nvtx_range_(name, color)

#endif  // NVTX_CUH
