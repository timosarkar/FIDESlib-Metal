//
// FIDESlib Metal Backend - Kernel Launcher
// Provides C++ wrappers for launching Metal kernels from the CKKS Context.
//

#ifndef FIDESLIB_METAL_LAUNCHER_HPP
#define FIDESLIB_METAL_LAUNCHER_HPP

#import <Metal/Metal.h>
#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace FIDESlib {
namespace Metal {

// Forward declarations
class MetalStream;
class MetalBuffer;

// =============================================================================
// Kernel Pipeline Cache
// Metal compute pipelines are expensive to create, so we cache them
// =============================================================================

class MetalKernelLauncher {
public:
    MetalKernelLauncher(id<MTLDevice> device, id<MTLCommandQueue> queue);
    ~MetalKernelLauncher();

    // Device access
    id<MTLDevice> getDevice() const { return m_device; }
    id<MTLCommandQueue> getQueue() const { return m_queue; }

    // =============================================================================
    // Kernel Launch Helpers
    // =============================================================================

    // Simple 1D kernel launch
    void launchKernel(id<MTLComputePipelineState> pipeline,
                      MetalBuffer& arguments,
                      size_t gridSize,
                      size_t threadgroupSize);

    // 2D kernel launch
    void launchKernel2D(id<MTLComputePipelineState> pipeline,
                        MetalBuffer& arguments,
                        uint2 gridSize,
                        uint2 threadgroupSize);

    // =============================================================================
    // Add/Sub Kernels
    // =============================================================================

    void metalAdd(device void** a, device void** b,
                  const std::vector<int>& primeid_flattened,
                  const uint64_t* primes, uint64_t type_mask,
                  int primeid_init, size_t N, int numLimbs);

    void metalSub(device void** a, device void** b,
                  const std::vector<int>& primeid_flattened,
                  const uint64_t* primes, uint64_t type_mask,
                  int primeid_init, size_t N, int numLimbs);

    // =============================================================================
    // ModMult Kernels
    // =============================================================================

    void metalMult(device uint64_t* a, const uint64_t* b,
                   const uint64_t* primes, const uint64_t* barret_mu,
                   const uint32_t* prime_bits, int primeid, int algo,
                   size_t N);

    void metalMult3(device uint64_t* a, const uint64_t* b, const uint64_t* c,
                    const uint64_t* primes, const uint64_t* barret_mu,
                    const uint32_t* prime_bits, int primeid, int algo,
                    size_t N);

    void metalScalarMult(device uint64_t* a, uint64_t b,
                         const uint64_t* primes, const uint64_t* barret_mu,
                         const uint32_t* prime_bits, int primeid, int algo,
                         uint64_t shoup_b, size_t N);

    // =============================================================================
    // NTT Kernels
    // =============================================================================

    void metalBitReverse(device uint64_t* dat, uint32_t N, size_t N_elements);

    void metalNTT_1D(device uint64_t* dat,
                      const uint64_t* psi_table,
                      const uint64_t* primes,
                      uint32_t primeid,
                      uint32_t N,
                      uint32_t logN);

    void metalINTT_1D(device uint64_t* dat,
                        const uint64_t* psi_table,
                        const uint64_t* primes,
                        uint32_t primeid,
                        uint32_t N,
                        uint32_t logN,
                        uint64_t N_inv);

    // =============================================================================
    // Pipeline State Management
    // =============================================================================

    id<MTLComputePipelineState> getPipelineState(const std::string& kernelName);

    // Initialize pipelines from default library
    void initializePipelines();

private:
    id<MTLDevice> m_device;
    id<MTLCommandQueue> m_queue;
    id<MTLLibrary> m_defaultLibrary;

    std::map<std::string, id<MTLComputePipelineState>> m_pipelineCache;

    id<MTLComputePipelineState> createPipelineState(const std::string& functionName);
};

// =============================================================================
// Convenience RAII Scoped Timer
// =============================================================================

class MetalTimer {
public:
    MetalTimer(MetalStream& stream);
    ~MetalTimer();

    // Wait for completion and get elapsed time in milliseconds
    double elapsedMs();

private:
    MetalStream& m_stream;
    uint64_t m_startValue;
    uint64_t m_endValue;
};

// =============================================================================
// Global Launcher Instance
// =============================================================================

class MetalLauncherRegistry {
public:
    static MetalLauncherRegistry& instance();

    void initialize(int deviceIndex = 0);
    MetalKernelLauncher* getLauncher(int deviceIndex = -1);
    MetalKernelLauncher* getDefaultLauncher();

    int getDeviceCount() const;
    MetalDevice getDevice(int index = -1) const;

private:
    MetalLauncherRegistry() = default;

    std::map<int, std::unique_ptr<MetalKernelLauncher>> m_launchers;
    int m_defaultDevice = 0;
};

}}  // namespace FIDESlib::Metal

#endif  // FIDESLIB_METAL_LAUNCHER_HPP
