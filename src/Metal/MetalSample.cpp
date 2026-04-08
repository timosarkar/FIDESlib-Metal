//
// FIDESlib Metal Backend - Sample Program
// Demonstrates basic Metal device enumeration and buffer operations
//

#include "MetalDevice.hpp"
#include "MetalBuffer.hpp"
#include "MetalStream.hpp"
#include <iostream>
#include <vector>
#include <cstdint>

using namespace fides::metal;

void print_device_info() {
    std::cout << "=== Metal Device Information ===" << std::endl;

    int numDevices = MetalDevice::getNumDevices();
    std::cout << "Number of Metal devices: " << numDevices << std::endl;

    for (int i = 0; i < numDevices; i++) {
        MetalDevice dev = MetalDevice::getDevice(i);
        auto props = dev.getProperties();

        std::cout << "\nDevice " << i << ":" << std::endl;
        std::cout << "  Name: " << props.name << std::endl;
        std::cout << "  Compute Units: " << props.computeUnits << std::endl;
        std::cout << "  Max Threads Per Threadgroup: " << props.maxThreadsPerThreadgroup << std::endl;
        std::cout << "  Max Threadgroup Memory: " << (props.maxThreadgroupMemory / 1024) << " KB" << std::endl;
        std::cout << "  Threadgroup Size: " << props.threadgroupSize << std::endl;
    }
}

void test_buffer_operations() {
    std::cout << "\n=== Buffer Operations Test ===" << std::endl;

    MetalDevice device = MetalDevice::getDevice(0);
    id<MTLDevice> mtlDevice = device.get();

    // Allocate a buffer
    const size_t N = 1024;
    size_t bufferSize = N * sizeof(uint64_t);

    id<MTLBuffer> buffer = [mtlDevice newBufferWithLength:bufferSize
                                                options:MTLResourceStorageModeShared];

    if (!buffer) {
        std::cout << "Failed to allocate buffer" << std::endl;
        return;
    }

    std::cout << "Allocated buffer of size " << bufferSize << " bytes" << std::endl;

    // Write data to buffer
    uint64_t* data = (uint64_t*)buffer.contents;
    for (size_t i = 0; i < N; i++) {
        data[i] = i * 2;
    }

    // Read back and verify
    std::cout << "Writing and reading back " << N << " uint64_t values..." << std::endl;

    bool allCorrect = true;
    for (size_t i = 0; i < N; i++) {
        if (data[i] != i * 2) {
            std::cout << "  Mismatch at index " << i << ": " << data[i] << " vs " << (i * 2) << std::endl;
            allCorrect = false;
        }
    }

    if (allCorrect) {
        std::cout << "  All values verified correctly!" << std::endl;
    }

    // Test with different sizes
    std::vector<size_t> sizes = {64, 256, 1024, 4096};
    for (auto size : sizes) {
        id<MTLBuffer> testBuffer = [mtlDevice newBufferWithLength:size
                                                      options:MTLResourceStorageModeShared];
        if (testBuffer) {
            std::cout << "  Allocated " << size << " bytes: OK" << std::endl;
        } else {
            std::cout << "  Allocated " << size << " bytes: FAIL" << std::endl;
        }
    }
}

void test_stream_operations() {
    std::cout << "\n=== Stream Operations Test ===" << std::endl;

    MetalDevice device = MetalDevice::getDevice(0);
    id<MTLDevice> mtlDevice = device.get();

    // Create command queue
    id<MTLCommandQueue> queue = [mtlDevice newCommandQueue];
    if (!queue) {
        std::cout << "Failed to create command queue" << std::endl;
        return;
    }
    std::cout << "Created command queue: OK" << std::endl;

    // Create command buffer
    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    if (!cmdBuffer) {
        std::cout << "Failed to create command buffer" << std::endl;
        return;
    }
    std::cout << "Created command buffer: OK" << std::endl;
}

void test_modular_arithmetic_cpu() {
    std::cout << "\n=== Modular Arithmetic (CPU) ===" << std::endl;

    // These are the Metal math functions run on CPU for verification
    auto metal_modadd = [](uint64_t a, uint64_t b, uint64_t mod) -> uint64_t {
        uint64_t result = a + b;
        if (result >= mod) result -= mod;
        return result;
    };

    auto metal_modsub = [](uint64_t a, uint64_t b, uint64_t mod) -> uint64_t {
        uint64_t result = a - b;
        if (result > a) result += mod;
        return result;
    };

    uint64_t mod = 13;

    // Test addition
    std::cout << "Modular addition (mod=" << mod << "):" << std::endl;
    std::cout << "  " << 3 << " + " << 7 << " = " << metal_modadd(3, 7, mod) << " (expected 10)" << std::endl;
    std::cout << "  " << 10 << " + " << 5 << " = " << metal_modadd(10, 5, mod) << " (expected 2)" << std::endl;
    std::cout << "  " << 12 << " + " << 1 << " = " << metal_modadd(12, 1, mod) << " (expected 0)" << std::endl;

    // Test subtraction
    std::cout << "Modular subtraction (mod=" << mod << "):" << std::endl;
    std::cout << "  " << 3 << " - " << 7 << " = " << metal_modsub(3, 7, mod) << " (expected 9)" << std::endl;
    std::cout << "  " << 7 << " - " << 3 << " = " << metal_modsub(7, 3, mod) << " (expected 4)" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "FIDESlib Metal Backend - Sample Program" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    print_device_info();
    test_buffer_operations();
    test_stream_operations();
    test_modular_arithmetic_cpu();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Sample program completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
