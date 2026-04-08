//
// FIDESlib Metal Backend - Kernel Launcher Implementation
//

#include "MetalLauncher.hpp"
#include "MetalStream.hpp"
#include "MetalBuffer.hpp"
#include <iostream>
#include <cassert>

namespace FIDESlib {
namespace Metal {

// =============================================================================
// MetalKernelLauncher Implementation
// =============================================================================

MetalKernelLauncher::MetalKernelLauncher(id<MTLDevice> device, id<MTLCommandQueue> queue)
    : m_device(device), m_queue(queue) {

    // Create default library from executable
    NSError* error = nil;
    m_defaultLibrary = [device newDefaultLibrary:&error];

    if (m_defaultLibrary == nil) {
        std::cerr << "MetalKernelLauncher: failed to create default library: "
                  << (error ? [[error localizedDescription] UTF8String] : "unknown error")
                  << std::endl;
    }
}

MetalKernelLauncher::~MetalKernelLauncher() {
    // Pipeline states are released via ARC
    m_defaultLibrary = nil;
}

id<MTLComputePipelineState> MetalKernelLauncher::createPipelineState(const std::string& functionName) {
    NSError* error = nil;

    id<MTLFunction> function = [m_defaultLibrary newFunctionWithName:[NSString stringWithUTF8String:functionName.c_str()]];
    if (function == nil) {
        std::cerr << "MetalKernelLauncher: function not found: " << functionName << std::endl;
        return nil;
    }

    MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
    descriptor.computeFunction = function;
    descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;

    id<MTLComputePipelineState> pipeline = [m_device newComputePipelineStateWithDescriptor:descriptor
                                                                                 options:0
                                                                                      error:&error];
    [descriptor release];

    if (pipeline == nil) {
        std::cerr << "MetalKernelLauncher: failed to create pipeline for " << functionName << ": "
                  << (error ? [[error localizedDescription] UTF8String] : "unknown error")
                  << std::endl;
        return nil;
    }

    return pipeline;
}

id<MTLComputePipelineState> MetalKernelLauncher::getPipelineState(const std::string& kernelName) {
    auto it = m_pipelineCache.find(kernelName);
    if (it != m_pipelineCache.end()) {
        return it->second;
    }

    id<MTLComputePipelineState> pipeline = createPipelineState(kernelName);
    if (pipeline != nil) {
        m_pipelineCache[kernelName] = pipeline;
    }

    return pipeline;
}

void MetalKernelLauncher::initializePipelines() {
    // Pre-create common pipeline states
    const char* kernelNames[] = {
        "metal_add_u64", "metal_add_u32",
        "metal_sub_u64", "metal_sub_u32",
        "metal_add_voidptr", "metal_sub_voidptr",
        "metal_mult_u64", "metal_mult_u32",
        "metal_mult3_u64", "metal_mult3_u32",
        "metal_scalar_mult_u64", "metal_scalar_mult_u32",
        "metal_Bit_Reverse", "metal_Bit_Reverse_u64",
        "metal_NTT_1D", "metal_INTT_1D",
        "metal_copy_u64"
    };

    for (const char* name : kernelNames) {
        getPipelineState(name);
    }
}

void MetalKernelLauncher::launchKernel(id<MTLComputePipelineState> pipeline,
                                        MetalBuffer& arguments,
                                        size_t gridSize,
                                        size_t threadgroupSize) {
    id<MTLCommandBuffer> commandBuffer = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];

    // Set buffer arguments
    if (arguments.isValid()) {
        [encoder setBuffer:arguments.getBuffer() offset:0 atIndex:0];
    }

    MTLSize threadsPerThreadgroup = MTLSizeMake(threadgroupSize, 1, 1);
    MTLSize numThreadgroups = MTLSizeMake((gridSize + threadgroupSize - 1) / threadgroupSize, 1, 1);

    [encoder dispatchThreads:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalKernelLauncher::launchKernel2D(id<MTLComputePipelineState> pipeline,
                                          MetalBuffer& arguments,
                                          uint2 gridSize,
                                          uint2 threadgroupSize) {
    id<MTLCommandBuffer> commandBuffer = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];

    if (arguments.isValid()) {
        [encoder setBuffer:arguments.getBuffer() offset:0 atIndex:0];
    }

    MTLSize threadsPerThreadgroup = MTLSizeMake(threadgroupSize.x, threadgroupSize.y, threadgroupSize.z);
    MTLSize numThreadgroups = MTLSizeMake(gridSize.x, gridSize.y, gridSize.z);

    [encoder dispatchThreads:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

// =============================================================================
// Add/Sub Kernels
// =============================================================================

void MetalKernelLauncher::metalAdd(device void** a, device void** b,
                                    const std::vector<int>& primeid_flattened,
                                    const uint64_t* primes, uint64_t type_mask,
                                    int primeid_init, size_t N, int numLimbs) {
    id<MTLComputePipelineState> pipeline = getPipelineState("metal_add_voidptr");
    if (pipeline == nil) {
        std::cerr << "metalAdd: pipeline not found" << std::endl;
        return;
    }

    // Flatten grid: CUDA uses dim3(N/128, numLimbs) -> Metal uses uint2(N/128, numLimbs)
    uint2 gridSize = uint2(N / 128, numLimbs);

    id<MTLCommandBuffer> commandBuffer = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    // Note: In a real implementation, we'd set up the buffers properly
    // This is a simplified version showing the dispatch pattern

    MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
    MTLSize numThreadgroups = MTLSizeMake(gridSize.x, gridSize.y, 1);

    [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalKernelLauncher::metalSub(device void** a, device void** b,
                                    const std::vector<int>& primeid_flattened,
                                    const uint64_t* primes, uint64_t type_mask,
                                    int primeid_init, size_t N, int numLimbs) {
    id<MTLComputePipelineState> pipeline = getPipelineState("metal_sub_voidptr");
    if (pipeline == nil) {
        return;
    }

    uint2 gridSize = uint2(N / 128, numLimbs);

    id<MTLCommandBuffer> commandBuffer = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
    MTLSize numThreadgroups = MTLSizeMake(gridSize.x, gridSize.y, 1);

    [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

// =============================================================================
// ModMult Kernels
// =============================================================================

void MetalKernelLauncher::metalMult(device uint64_t* a, const uint64_t* b,
                                      const uint64_t* primes, const uint64_t* barret_mu,
                                      const uint32_t* prime_bits, int primeid, int algo,
                                      size_t N) {
    id<MTLComputePipelineState> pipeline = getPipelineState("metal_mult_u64");
    if (pipeline == nil) {
        return;
    }

    id<MTLCommandBuffer> commandBuffer = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    // Set buffers
    [encoder setBuffer:(id<MTLBuffer>)a offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)b offset:0 atIndex:1];
    [encoder setBytes:primes length:sizeof(uint64_t) * 64 atIndex:2];
    [encoder setBytes:barret_mu length:sizeof(uint64_t) * 64 atIndex:3];
    [encoder setBytes:prime_bits length:sizeof(uint32_t) * 64 atIndex:4];
    [encoder setBytes:&primeid length:sizeof(int) atIndex:5];
    [encoder setBytes:&algo length:sizeof(int) atIndex:6];

    MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
    MTLSize numThreadgroups = MTLSizeMake(N / 128, 1, 1);

    [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalKernelLauncher::metalMult3(device uint64_t* a, const uint64_t* b, const uint64_t* c,
                                       const uint64_t* primes, const uint64_t* barret_mu,
                                       const uint32_t* prime_bits, int primeid, int algo,
                                       size_t N) {
    id<MTLComputePipelineState> pipeline = getPipelineState("metal_mult3_u64");
    if (pipeline == nil) {
        return;
    }

    id<MTLCommandBuffer> commandBuffer = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    [encoder setBuffer:(id<MTLBuffer>)a offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)b offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)c offset:0 atIndex:2];
    [encoder setBytes:primes length:sizeof(uint64_t) * 64 atIndex:3];
    [encoder setBytes:barret_mu length:sizeof(uint64_t) * 64 atIndex:4];
    [encoder setBytes:prime_bits length:sizeof(uint32_t) * 64 atIndex:5];
    [encoder setBytes:&primeid length:sizeof(int) atIndex:6];
    [encoder setBytes:&algo length:sizeof(int) atIndex:7];

    MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
    MTLSize numThreadgroups = MTLSizeMake(N / 128, 1, 1);

    [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalKernelLauncher::metalScalarMult(device uint64_t* a, uint64_t b,
                                          const uint64_t* primes, const uint64_t* barret_mu,
                                          const uint32_t* prime_bits, int primeid, int algo,
                                          uint64_t shoup_b, size_t N) {
    id<MTLComputePipelineState> pipeline = getPipelineState("metal_scalar_mult_u64");
    if (pipeline == nil) {
        return;
    }

    id<MTLCommandBuffer> commandBuffer = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    [encoder setBuffer:(id<MTLBuffer>)a offset:0 atIndex:0];
    [encoder setBytes:&b length:sizeof(uint64_t) atIndex:1];
    [encoder setBytes:primes length:sizeof(uint64_t) * 64 atIndex:2];
    [encoder setBytes:barret_mu length:sizeof(uint64_t) * 64 atIndex:3];
    [encoder setBytes:prime_bits length:sizeof(uint32_t) * 64 atIndex:4];
    [encoder setBytes:&primeid length:sizeof(int) atIndex:5];
    [encoder setBytes:&algo length:sizeof(int) atIndex:6];
    [encoder setBytes:&shoup_b length:sizeof(uint64_t) atIndex:7];

    MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
    MTLSize numThreadgroups = MTLSizeMake(N / 128, 1, 1);

    [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

// =============================================================================
// NTT Kernels
// =============================================================================

void MetalKernelLauncher::metalBitReverse(device uint64_t* dat, uint32_t N, size_t N_elements) {
    id<MTLComputePipelineState> pipeline = getPipelineState("metal_Bit_Reverse_u64");
    if (pipeline == nil) {
        pipeline = getPipelineState("metal_Bit_Reverse");  // Try 32-bit version
    }
    if (pipeline == nil) {
        return;
    }

    id<MTLCommandBuffer> commandBuffer = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    // Set buffers: dat and N
    // In a real implementation, we'd create MetalBuffers here
    MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
    MTLSize numThreadgroups = MTLSizeMake(N_elements / 256, 1, 1);

    [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalKernelLauncher::metalNTT_1D(device uint64_t* dat,
                                         const uint64_t* psi_table,
                                         const uint64_t* primes,
                                         uint32_t primeid,
                                         uint32_t N,
                                         uint32_t logN) {
    id<MTLComputePipelineState> pipeline = getPipelineState("metal_NTT_1D");
    if (pipeline == nil) {
        return;
    }

    id<MTLCommandBuffer> commandBuffer = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    // Set all arguments
    [encoder setBuffer:(id<MTLBuffer>)dat offset:0 atIndex:0];
    [encoder setBytes:psi_table length:sizeof(uint64_t) * N atIndex:1];
    [encoder setBytes:primes length:sizeof(uint64_t) * 64 atIndex:2];
    [encoder setBytes:&primeid length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&logN length:sizeof(uint32_t) atIndex:5];

    MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
    MTLSize numThreadgroups = MTLSizeMake(N / 512, 1, 1);

    [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalKernelLauncher::metalINTT_1D(device uint64_t* dat,
                                          const uint64_t* psi_table,
                                          const uint64_t* primes,
                                          uint32_t primeid,
                                          uint32_t N,
                                          uint32_t logN,
                                          uint64_t N_inv) {
    id<MTLComputePipelineState> pipeline = getPipelineState("metal_INTT_1D");
    if (pipeline == nil) {
        return;
    }

    id<MTLCommandBuffer> commandBuffer = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    [encoder setBuffer:(id<MTLBuffer>)dat offset:0 atIndex:0];
    [encoder setBytes:psi_table length:sizeof(uint64_t) * N atIndex:1];
    [encoder setBytes:primes length:sizeof(uint64_t) * 64 atIndex:2];
    [encoder setBytes:&primeid length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&logN length:sizeof(uint32_t) atIndex:5];
    [encoder setBytes:&N_inv length:sizeof(uint64_t) atIndex:6];

    MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
    MTLSize numThreadgroups = MTLSizeMake(N / 512, 1, 1);

    [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

// =============================================================================
// MetalTimer Implementation
// =============================================================================

MetalTimer::MetalTimer(MetalStream& stream) : m_stream(stream) {
    // Record start time
    m_startValue = 0;
}

double MetalTimer::elapsedMs() {
    // In a real implementation, we'd use MTLSharedEvent to measure time
    return 0.0;
}

// =============================================================================
// MetalLauncherRegistry Implementation
// =============================================================================

static std::unique_ptr<MetalLauncherRegistry>& getRegistryInstance() {
    static std::unique_ptr<MetalLauncherRegistry> registry = std::make_unique<MetalLauncherRegistry>();
    return registry;
}

MetalLauncherRegistry& MetalLauncherRegistry::instance() {
    return *getRegistryInstance();
}

void MetalLauncherRegistry::initialize(int deviceIndex) {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (deviceIndex < 0 || deviceIndex >= (int)devices.count) {
            deviceIndex = 0;
        }

        id<MTLDevice> device = devices[deviceIndex];
        id<MTLCommandQueue> queue = [device newCommandQueue];

        if (queue == nil) {
            std::cerr << "MetalLauncherRegistry: failed to create command queue" << std::endl;
            return;
        }

        auto launcher = std::make_unique<MetalKernelLauncher>(device, queue);
        launcher->initializePipelines();

        m_launchers[deviceIndex] = std::move(launcher);
        m_defaultDevice = deviceIndex;

        std::cout << "MetalLauncherRegistry: initialized for device " << deviceIndex << std::endl;
    }
}

MetalKernelLauncher* MetalLauncherRegistry::getLauncher(int deviceIndex) {
    if (deviceIndex < 0) {
        deviceIndex = m_defaultDevice;
    }

    auto it = m_launchers.find(deviceIndex);
    if (it != m_launchers.end()) {
        return it->second.get();
    }

    return nullptr;
}

MetalKernelLauncher* MetalLauncherRegistry::getDefaultLauncher() {
    return getLauncher(-1);
}

int MetalLauncherRegistry::getDeviceCount() const {
    @autoreleasepool {
        return (int)MTLCopyAllDevices().count;
    }
}

MetalDevice MetalLauncherRegistry::getDevice(int index) const {
    if (index < 0) {
        index = m_defaultDevice;
    }
    return MetalDevice(index);
}

}}  // namespace FIDESlib::Metal
