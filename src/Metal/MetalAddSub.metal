//
// FIDESlib Metal Backend - Add/Subtract Kernels (Metal Shading Language)
// Ported from AddSub.cu
//

#include <metal_stdlib>
#include "MetalMath.hpp"

using namespace metal;

// =============================================================================
// Helper Functions (inlineable in Metal)
// =============================================================================

template <typename T>
inline T metal_modadd(const T a, const T b, constant uint64_t* primes, int primeId) {
    T prime_p = primes[primeId];
    T tmp0 = a + b;
    return tmp0 - prime_p * (tmp0 >= prime_p);
}

template <typename T>
inline T metal_modsub(const T a, const T b, constant uint64_t* primes, int primeId) {
    T prime_p = primes[primeId];
    T tmp0 = a - b;
    return tmp0 + prime_p * (tmp0 >= prime_p);
}

// =============================================================================
// Basic Add/Subtract Kernels (single prime per kernel launch)
// grid.x = N / blockSize.x, block.x = 128 (typical)
// =============================================================================

kernel void metal_add_u64(device uint64_t* a [[buffer(0)]],
                          constant uint64_t* b [[buffer(1)]],
                          constant uint64_t* primes [[buffer(2)]],
                          constant int& primeId [[buffer(3)]],
                          uint idx [[thread_position_in_grid]]) {
    a[idx] = metal_modadd(a[idx], b[idx], primes, primeId);
}

kernel void metal_add_u32(device uint32_t* a [[buffer(0)]],
                          constant uint32_t* b [[buffer(1)]],
                          constant uint64_t* primes [[buffer(2)]],
                          constant int& primeId [[buffer(3)]],
                          uint idx [[thread_position_in_grid]]) {
    a[idx] = metal_modadd(a[idx], b[idx], primes, primeId);
}

kernel void metal_sub_u64(device uint64_t* a [[buffer(0)]],
                          constant uint64_t* b [[buffer(1)]],
                          constant uint64_t* primes [[buffer(2)]],
                          constant int& primeId [[buffer(3)]],
                          uint idx [[thread_position_in_grid]]) {
    a[idx] = metal_modsub(a[idx], b[idx], primes, primeId);
}

kernel void metal_sub_u32(device uint32_t* a [[buffer(0)]],
                          constant uint32_t* b [[buffer(1)]],
                          constant uint64_t* primes [[buffer(2)]],
                          constant int& primeId [[buffer(3)]],
                          uint idx [[thread_position_in_grid]]) {
    a[idx] = metal_modsub(a[idx], b[idx], primes, primeId);
}

// =============================================================================
// Copy Kernel
// =============================================================================

kernel void metal_copy_u64(device uint64_t* dst [[buffer(0)]],
                            constant uint64_t* src [[buffer(1)]],
                            constant uint32_t& size [[buffer(2)]],
                            uint idx [[thread_position_in_grid]]) {
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

kernel void metal_copy_u32(device uint32_t* dst [[buffer(0)]],
                            constant uint32_t* src [[buffer(1)]],
                            constant uint32_t& size [[buffer(2)]],
                            uint idx [[thread_position_in_grid]]) {
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

// =============================================================================
// Zero Kernel
// =============================================================================

kernel void metal_zero_u64(device uint64_t* data [[buffer(0)]],
                            constant uint32_t& size [[buffer(1)]],
                            uint idx [[thread_position_in_grid]]) {
    if (idx < size) {
        data[idx] = 0;
    }
}

kernel void metal_zero_u32(device uint32_t* data [[buffer(0)]],
                            constant uint32_t& size [[buffer(1)]],
                            uint idx [[thread_position_in_grid]]) {
    if (idx < size) {
        data[idx] = 0;
    }
}

// =============================================================================
// Scalar Add/Subtract Kernels
// Adds/subtracts a scalar value (same for all elements)
// =============================================================================

kernel void metal_scalar_add_u64(device uint64_t* a [[buffer(0)]],
                                 constant uint64_t& scalar [[buffer(1)]],
                                 constant uint64_t* primes [[buffer(2)]],
                                 constant int& primeId [[buffer(3)]],
                                 uint idx [[thread_position_in_grid]]) {
    a[idx] = metal_modadd(a[idx], scalar, primes, primeId);
}

kernel void metal_scalar_sub_u64(device uint64_t* a [[buffer(0)]],
                                 constant uint64_t& scalar [[buffer(1)]],
                                 constant uint64_t* primes [[buffer(2)]],
                                 constant int& primeId [[buffer(3)]],
                                 uint idx [[thread_position_in_grid]]) {
    a[idx] = metal_modsub(a[idx], scalar, primes, primeId);
}

// =============================================================================
// Multi-Limb Add/Subtract Kernels
// Uses flat buffer layout with stride calculations
// Grid: (N/128, numLimbs) as uint2
// =============================================================================

kernel void metal_add_flat(device uint64_t* a [[buffer(0)]],
                           device uint64_t* b [[buffer(1)]],
                           constant uint32_t& N [[buffer(2)]],
                           constant uint32_t& numLimbs [[buffer(3)]],
                           constant uint64_t* primes [[buffer(4)]],
                           constant int* primeid_flattened [[buffer(5)]],
                           uint2 idx [[thread_position_in_grid]]) {

    uint limbIdx = idx.y;
    if (limbIdx >= numLimbs) return;

    uint elemIdx = idx.x;
    if (elemIdx >= N/128) return;

    uint offset = limbIdx * N;
    uint primeid = primeid_flattened[limbIdx];

    uint64_t val_a = a[offset + elemIdx];
    uint64_t val_b = b[offset + elemIdx];

    a[offset + elemIdx] = metal_modadd(val_a, val_b, primes, primeid);
}

kernel void metal_sub_flat(device uint64_t* a [[buffer(0)]],
                           device uint64_t* b [[buffer(1)]],
                           constant uint32_t& N [[buffer(2)]],
                           constant uint32_t& numLimbs [[buffer(3)]],
                           constant uint64_t* primes [[buffer(4)]],
                           constant int* primeid_flattened [[buffer(5)]],
                           uint2 idx [[thread_position_in_grid]]) {

    uint limbIdx = idx.y;
    if (limbIdx >= numLimbs) return;

    uint elemIdx = idx.x;
    if (elemIdx >= N/128) return;

    uint offset = limbIdx * N;
    uint primeid = primeid_flattened[limbIdx];

    uint64_t val_a = a[offset + elemIdx];
    uint64_t val_b = b[offset + elemIdx];

    a[offset + elemIdx] = metal_modsub(val_a, val_b, primes, primeid);
}

// =============================================================================
// 32-bit Multi-Limb Kernels
// =============================================================================

kernel void metal_add_flat_u32(device uint32_t* a [[buffer(0)]],
                                device uint32_t* b [[buffer(1)]],
                                constant uint32_t& N [[buffer(2)]],
                                constant uint32_t& numLimbs [[buffer(3)]],
                                constant uint64_t* primes [[buffer(4)]],
                                constant int* primeid_flattened [[buffer(5)]],
                                uint2 idx [[thread_position_in_grid]]) {

    uint limbIdx = idx.y;
    if (limbIdx >= numLimbs) return;

    uint elemIdx = idx.x;
    if (elemIdx >= N/128) return;

    uint offset = limbIdx * N;
    uint primeid = primeid_flattened[limbIdx];

    uint32_t val_a = a[offset + elemIdx];
    uint32_t val_b = b[offset + elemIdx];

    a[offset + elemIdx] = metal_modadd(val_a, val_b, primes, primeid);
}

kernel void metal_sub_flat_u32(device uint32_t* a [[buffer(0)]],
                                device uint32_t* b [[buffer(1)]],
                                constant uint32_t& N [[buffer(2)]],
                                constant uint32_t& numLimbs [[buffer(3)]],
                                constant uint64_t* primes [[buffer(4)]],
                                constant int* primeid_flattened [[buffer(5)]],
                                uint2 idx [[thread_position_in_grid]]) {

    uint limbIdx = idx.y;
    if (limbIdx >= numLimbs) return;

    uint elemIdx = idx.x;
    if (elemIdx >= N/128) return;

    uint offset = limbIdx * N;
    uint primeid = primeid_flattened[limbIdx];

    uint32_t val_a = a[offset + elemIdx];
    uint32_t val_b = b[offset + elemIdx];

    a[offset + elemIdx] = metal_modsub(val_a, val_b, primes, primeid);
}
