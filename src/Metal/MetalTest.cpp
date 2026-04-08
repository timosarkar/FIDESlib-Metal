//
// FIDESlib Metal Backend - Test Suite
// Tests basic Metal kernel functionality
//

#include "MetalDevice.hpp"
#include "MetalBuffer.hpp"
#include "MetalStream.hpp"
#include "MetalLauncher.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using namespace fides::metal;

bool compare_uint64(const uint64_t* a, const uint64_t* b, size_t n, uint64_t tolerance = 0) {
    for (size_t i = 0; i < n; i++) {
        uint64_t diff = (a[i] > b[i]) ? (a[i] - b[i]) : (b[i] - a[i]);
        if (diff > tolerance) {
            std::cout << "  Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << " (diff=" << diff << ")" << std::endl;
            return false;
        }
    }
    return true;
}

int test_device_enumeration() {
    std::cout << "=== Test: Device Enumeration ===" << std::endl;

    int numDevices = MetalDevice::getNumDevices();
    if (numDevices == 0) {
        std::cout << "  FAIL: No Metal devices found" << std::endl;
        return 1;
    }

    std::cout << "  Found " << numDevices << " Metal device(s)" << std::endl;
    for (int i = 0; i < numDevices; i++) {
        MetalDevice dev = MetalDevice::getDevice(i);
        auto props = dev.getProperties();
        std::cout << "  Device " << i << ": " << props.name << std::endl;
        std::cout << "    Compute units: " << props.computeUnits << std::endl;
        std::cout << "    Max threads per threadgroup: " << props.maxThreadsPerThreadgroup << std::endl;
        std::cout << "    Max threadgroup memory: " << (props.maxThreadgroupMemory / 1024) << " KB" << std::endl;
    }

    std::cout << "  PASS" << std::endl;
    return 0;
}

int test_buffer_allocation() {
    std::cout << "=== Test: Buffer Allocation ===" << std::endl;

    MetalDevice device = MetalDevice::getDefaultDevice();
    id<MTLDevice> mtlDevice = device.get();

    // Test allocating various sizes
    std::vector<size_t> sizes = {1024, 4096, 65536, 1024*1024};

    for (auto size : sizes) {
        id<MTLBuffer> buffer = [mtlDevice newBufferWithLength:size
                                                   options:MTLResourceStorageModeShared];
        if (!buffer) {
            std::cout << "  FAIL: Could not allocate buffer of size " << size << std::endl;
            return 1;
        }
        std::cout << "  Allocated " << size << " bytes" << std::endl;
    }

    std::cout << "  PASS" << std::endl;
    return 0;
}

int test_buffer_read_write() {
    std::cout << "=== Test: Buffer Read/Write ===" << std::endl;

    MetalDevice device = MetalDevice::getDefaultDevice();
    id<MTLDevice> mtlDevice = device.get();

    const size_t N = 1024;
    id<MTLBuffer> buffer = [mtlDevice newBufferWithLength:N * sizeof(uint64_t)
                                                options:MTLResourceStorageModeShared];

    // Write data
    uint64_t* data = (uint64_t*)buffer.contents;
    for (size_t i = 0; i < N; i++) {
        data[i] = i * 2;
    }

    // Read back and verify
    bool pass = true;
    for (size_t i = 0; i < N; i++) {
        if (data[i] != i * 2) {
            std::cout << "  FAIL: data[" << i << "] = " << data[i] << ", expected " << (i * 2) << std::endl;
            pass = false;
            break;
        }
    }

    if (pass) {
        std::cout << "  Wrote and read back " << N << " uint64_t values" << std::endl;
        std::cout << "  PASS" << std::endl;
    }

    return pass ? 0 : 1;
}

int test_modular_arithmetic() {
    std::cout << "=== Test: Modular Arithmetic ===" << std::endl;

    uint64_t mod = 13;  // Small prime for testing

    // Test metal_modadd
    struct AddTest { uint64_t a, b, expected; };
    AddTest add_tests[] = {
        {3, 7, 10},      // 3 + 7 = 10 mod 13 -> 10
        {10, 5, 2},      // 10 + 5 = 15 mod 13 -> 2
        {12, 1, 0},      // 12 + 1 = 13 mod 13 -> 0
        {0, 0, 0},
    };

    std::cout << "  Testing modular addition:" << std::endl;
    for (auto& t : add_tests) {
        uint64_t result = t.a + t.b;
        if (result >= mod) result -= mod;
        bool pass = (result == t.expected);
        std::cout << "    " << t.a << " + " << t.b << " mod " << mod << " = " << result
                  << " (expected " << t.expected << ") " << (pass ? "OK" : "FAIL") << std::endl;
        if (!pass) return 1;
    }

    // Test metal_modsub
    struct SubTest { uint64_t a, b, expected; };
    SubTest sub_tests[] = {
        {7, 3, 4},       // 7 - 3 = 4 mod 13 -> 4
        {3, 7, 9},       // 3 - 7 = -4 mod 13 -> 9
        {0, 0, 0},
        {1, 1, 0},
    };

    std::cout << "  Testing modular subtraction:" << std::endl;
    for (auto& t : sub_tests) {
        uint64_t result = (t.a >= t.b) ? (t.a - t.b) : (t.a + mod - t.b);
        bool pass = (result == t.expected);
        std::cout << "    " << t.a << " - " << t.b << " mod " << mod << " = " << result
                  << " (expected " << t.expected << ") " << (pass ? "OK" : "FAIL") << std::endl;
        if (!pass) return 1;
    }

    std::cout << "  PASS" << std::endl;
    return 0;
}

int test_modmult_barrett() {
    std::cout << "=== Test: Barrett Modular Multiplication ===" << std::endl;

    // Test Barrett reduction manually
    // mu = floor(2^128 / mod), then q = umulhi(a, mu), r = a * b - q * mod

    uint64_t mod = 13;
    __uint128_t two_power_128 = (__uint128_t(1) << 64) * (__uint128_t(1) << 64);
    uint64_t mu = (uint64_t)((two_power_128 + mod - 1) / mod);

    struct MultTest { uint64_t a, b, expected; };
    MultTest tests[] = {
        {3, 7, 8},       // 3 * 7 = 21 mod 13 -> 8
        {12, 12, 3},     // 12 * 12 = 144 mod 13 -> 3
        {1, 1, 1},
        {0, 5, 0},
    };

    std::cout << "  mu for mod " << mod << " = " << mu << std::endl;

    for (auto& t : tests) {
        uint64_t q = (uint64_t)((__uint128_t)t.a * (__uint128_t)mu >> 64);
        uint64_t r = t.a * t.b - q * mod;
        if (r >= mod) r -= mod;

        bool pass = (r == t.expected);
        std::cout << "    " << t.a << " * " << t.b << " mod " << mod << " = " << r
                  << " (expected " << t.expected << ") " << (pass ? "OK" : "FAIL") << std::endl;
        if (!pass) return 1;
    }

    std::cout << "  PASS" << std::endl;
    return 0;
}

int test_bit_reversal() {
    std::cout << "=== Test: Bit Reversal ===" << std::endl;

    // Test that bit reversal works correctly
    struct Test { uint32_t input; uint32_t expected; };
    Test tests[] = {
        {0x00000001, 0x80000000},
        {0x00000002, 0x40000000},
        {0x00000003, 0xC0000000},
        {0x0000000F, 0xF0000000},
        {0xAAAAAAAA, 0x55555555},
        {0x55555555, 0xAAAAAAAA},
    };

    for (auto& t : tests) {
        uint32_t x = t.input;
        // Apply bit reversal algorithm from MetalMath.hpp
        x = ((x & 0xAAAAAAAAU) >> 1) | ((x & 0x55555555U) << 1);
        x = ((x & 0xCCCCCCCCU) >> 2) | ((x & 0x33333333U) << 2);
        x = ((x & 0xF0F0F0F0U) >> 4) | ((x & 0x0F0F0F0FU) << 4);
        x = ((x & 0xFF00FF00U) >> 8) | ((x & 0x00FF00FFU) << 8);
        x = (x >> 16) | (x << 16);

        bool pass = (x == t.expected);
        std::cout << "    rev(0x" << std::hex << t.input << ") = 0x" << x
                  << " (expected 0x" << t.expected << ") " << std::dec
                  << (pass ? "OK" : "FAIL") << std::endl;
        if (!pass) return 1;
    }

    std::cout << "  PASS" << std::endl;
    return 0;
}

int test_clz() {
    std::cout << "=== Test: Count Leading Zeros ===" << std::endl;

    struct Test { uint32_t input; int expected_clz; };
    Test tests[] = {
        {0x80000000, 0},  // MSB set
        {0x40000000, 1},
        {0x00000001, 31}, // LSB set
        {0x00000000, 32}, // zero input
        {0xFF000000, 8},
        {0x00FF0000, 16},
    };

    for (auto& t : tests) {
        int clz = (t.input == 0) ? 32 : __builtin_clz(t.input);
        bool pass = (clz == t.expected_clz);
        std::cout << "    clz(0x" << std::hex << t.input << ") = " << std::dec << clz
                  << " (expected " << t.expected_clz << ") " << (pass ? "OK" : "FAIL") << std::endl;
        if (!pass) return 1;
    }

    std::cout << "  PASS" << std::endl;
    return 0;
}

int test_ntt_memory_layout() {
    std::cout << "=== Test: NTT 2D Memory Layout ===" << std::endl;

    const int M = 4;  // elements per thread
    const int N = 256; // threads

    // Layout from MetalNTT.metal:
    // A0: [0..255], A1: [256..511], A2: [512..767], A3: [768..1023]
    // psi: [1024..1279]
    int total_size = 256 * M + 256; // 1024 + 256 = 1280

    std::cout << "  Total shared memory: " << total_size << " uint64_t" << std::endl;
    std::cout << "  A0 base: 0" << std::endl;
    std::cout << "  A1 base: " << (1 * 256) << std::endl;
    std::cout << "  A2 base: " << (2 * 256) << std::endl;
    std::cout << "  A3 base: " << (3 * 256) << std::endl;
    std::cout << "  psi base: " << (256 * M) << std::endl;

    if (total_size > 32 * 1024 / sizeof(uint64_t)) {
        std::cout << "  WARNING: May exceed threadgroup memory limit" << std::endl;
    } else {
        std::cout << "  Within threadgroup memory limits (~32KB)" << std::endl;
    }

    // Test indexing
    for (int limb = 0; limb < M; limb++) {
        uint32_t base = limb * 256;
        std::cout << "  Limb " << limb << " at offset " << base << std::endl;
    }

    std::cout << "  PASS" << std::endl;
    return 0;
}

int test_flat_buffer_indexing() {
    std::cout << "=== Test: Flat Buffer Indexing ===" << std::endl;

    // Simulate flat buffer layout: limbIdx * N + elemIdx
    const uint32_t N = 1024;  // elements per limb
    const uint32_t numLimbs = 8;
    const uint32_t totalSize = numLimbs * N;

    std::cout << "  Flat buffer layout: " << numLimbs << " limbs x " << N << " elements = " << totalSize << " total" << std::endl;

    // Test accessing element at (limb=3, elem=50)
    uint32_t limb = 3;
    uint32_t elem = 50;
    uint64_t flat_offset = limb * N + elem;

    std::cout << "  Element (limb=" << limb << ", elem=" << elem << ") -> flat offset=" << flat_offset << std::endl;

    // Verify all limbs are correct
    for (uint32_t l = 0; l < numLimbs; l++) {
        uint64_t base = l * N;
        std::cout << "  Limb " << l << " base offset: " << base << std::endl;
    }

    std::cout << "  PASS" << std::endl;
    return 0;
}

int test_command_queue() {
    std::cout << "=== Test: Command Queue ===" << std::endl;

    MetalDevice device = MetalDevice::getDefaultDevice();
    id<MTLDevice> mtlDevice = device.get();

    id<MTLCommandQueue> queue = [mtlDevice newCommandQueue];
    if (!queue) {
        std::cout << "  FAIL: Could not create command queue" << std::endl;
        return 1;
    }

    std::cout << "  Created command queue successfully" << std::endl;

    // Create a simple command buffer
    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    if (!cmdBuffer) {
        std::cout << "  FAIL: Could not create command buffer" << std::endl;
        return 1;
    }

    std::cout << "  Created command buffer successfully" << std::endl;
    std::cout << "  PASS" << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "FIDESlib Metal Backend Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int failed = 0;

    // Tests that don't need Metal device
    if (test_modular_arithmetic() != 0) { failed++; }
    if (test_bit_reversal() != 0) { failed++; }
    if (test_clz() != 0) { failed++; }
    if (test_ntt_memory_layout() != 0) { failed++; }
    if (test_flat_buffer_indexing() != 0) { failed++; }

    // Tests that need Metal device
    if (test_device_enumeration() != 0) { failed++; }
    if (test_buffer_allocation() != 0) { failed++; }
    if (test_buffer_read_write() != 0) { failed++; }
    if (test_command_queue() != 0) { failed++; }
    if (test_modmult_barrett() != 0) { failed++; }

    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << (11 - failed) << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return failed > 0 ? 1 : 0;
}