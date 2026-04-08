//
// FIDESlib Metal Backend - Mathematical Intrinsics
// Apple Metal does not have hardware equivalents for some CUDA intrinsics.
// This header provides software implementations for use in .metal files.
//

#ifndef FIDESLIB_METAL_MATH_HPP
#define FIDESLIB_METAL_MATH_HPP

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Bit Reversal
// =============================================================================

// Software bit-reversal for 32-bit values (no __brev in Metal)
inline uint32_t metal_brev(uint32_t x) {
    x = ((x & 0xAAAAAAAAU) >> 1) | ((x & 0x55555555U) << 1);
    x = ((x & 0xCCCCCCCCU) >> 2) | ((x & 0x33333333U) << 2);
    x = ((x & 0xF0F0F0F0U) >> 4) | ((x & 0x0F0F0F0FU) << 4);
    x = ((x & 0xFF00FF00U) >> 8) | ((x & 0x00FF00FFU) << 8);
    x = (x >> 16) | (x << 16);
    return x;
}

// Software bit-reversal for 64-bit values
inline uint64_t metal_brev64(uint64_t x) {
    x = ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1) | ((x & 0x5555555555555555ULL) << 1);
    x = ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2) | ((x & 0x3333333333333333ULL) << 2);
    x = ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4);
    x = ((x & 0xFF00FF00FF00FF00ULL) >> 8) | ((x & 0x00FF00FF00FF00FFULL) << 8);
    x = ((x & 0xFFFF0000FFFF0000ULL) >> 16) | ((x & 0x0000FFFF0000FFFFULL) << 16);
    return (x >> 32) | (x << 32);
}

// =============================================================================
// Count Leading Zeros
// =============================================================================

// Metal has ctz but not clz - implement clz using ctz
inline uint32_t metal_clz(uint32_t x) {
    return x == 0 ? 32 : __builtin_clz(x);
}

inline uint32_t metal_clzll(uint64_t x) {
    return x == 0 ? 64 : __builtin_clzll(x);
}

// =============================================================================
// High Multiplication (upper bits of product)
// =============================================================================

// Metal equivalent for __umulhi(a, b) - upper 32 bits of 32x32 unsigned multiply
inline uint32_t metal_umulhi(uint32_t a, uint32_t b) {
    return uint32_t((uint64_t(a) * uint64_t(b)) >> 32);
}

// Metal equivalent for __umul64hi(a, b) - upper 64 bits of 64x64 unsigned multiply
inline uint64_t metal_umul64hi(uint64_t a, uint64_t b) {
    return uint64_t((__uint128_t(a) * __uint128_t(b)) >> 64);
}

// =============================================================================
// SIMD Shuffle Wrappers (Metal uses different naming than CUDA)
// =============================================================================

// simd_shuffle(val, srcLane) - shuffle across SIMD group
template <typename T>
T metal_shuffle(T val, uint simd_url) {
    return simd_shuffle(val, simd_url);
}

// metal_shuffle_up - shuffle with offset (upward)
template <typename T>
T metal_shuffle_up(T val, uint delta) {
    return simd_shuffle_up(val, delta);
}

// metal_shuffle_down - shuffle with offset (downward)
template <typename T>
T metal_shuffle_down(T val, uint delta) {
    return simd_shuffle_down(val, delta);
}

// metal_shuffle_xor - shuffle with XOR pattern
template <typename T>
T metal_shuffle_xor(T val, uint mask) {
    return simd_shuffle_xor(val, mask);
}

// =============================================================================
// Atomic Operations - Metal equivalents
// Note: Metal atomic_fetch_add_explicit takes 3 args in some versions
// =============================================================================

// atomic_fetch_add - simplified version
inline int metal_atomic_add(device atomic_int* addr, int val) {
    return atomic_fetch_add_explicit(addr, val, memory_order_relaxed);
}

inline uint32_t metal_atomic_add_u32(device atomic_uint* addr, uint32_t val) {
    return atomic_fetch_add_explicit(addr, val, memory_order_relaxed);
}

// =============================================================================
// Popcount
// =============================================================================

inline uint32_t metal_popc(uint32_t x) {
    return popcount(x);
}

inline uint32_t metal_popcll(uint64_t x) {
    return popcount(x);
}

// =============================================================================
// Modular Arithmetic (for CKKS)
// =============================================================================

// Barrett reduction precomputation: mu = floor(2^128 / mod)
inline uint64_t metal_barrett_mu(uint64_t mod) {
    __uint128_t two_power_128 = (__uint128_t(1) << 64) * (__uint128_t(1) << 64);
    return (uint64_t)((two_power_128 + mod - 1) / mod);
}

// Modular addition
inline uint64_t metal_modadd(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t result = a + b;
    if (result >= mod) result -= mod;
    return result;
}

// Modular subtraction
inline uint64_t metal_modsub(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t result = a - b;
    if (result > a) result += mod;  // Underflow occurred
    return result;
}

// Modular multiplication - Barrett reduction
inline uint64_t metal_modmult_barrett(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t mu = metal_barrett_mu(mod);
    uint64_t q = metal_umul64hi(a, mu);
    uint64_t r = a * b - q * mod;
    if (r >= mod) r -= mod;
    return r;
}

// Shoup's algorithm precomputation: b * 2^64 / mod
inline uint64_t metal_shoup_precompute(uint64_t b, uint64_t mod) {
    __uint128_t t = (__uint128_t(1) << 64);
    t = (t + mod - 1) / mod;
    t = t * b;
    return uint64_t(t >> 64);
}

// Modular multiplication - Shoup's algorithm
inline uint64_t metal_modmult_shoup(uint64_t a, uint64_t b, uint64_t mod, uint64_t shoup_b) {
    uint64_t q = metal_umul64hi(a, shoup_b);
    uint64_t r = a * b - q * mod;
    if (r >= mod) r -= mod;
    return r;
}

// =============================================================================
// Algorithm enum for modular multiplication
// =============================================================================

enum MetalALGO {
    METAL_ALGO_NATIVE = 0,
    METAL_ALGO_NONE = 1,
    METAL_ALGO_SHOUP = 2,
    METAL_ALGO_BARRETT = 3,
    METAL_ALGO_BARRETT_FP64 = 4
};

// Unified modular multiplication dispatch
template <MetalALGO algo>
inline uint64_t metal_modmult(uint64_t a, uint64_t b, uint64_t mod, uint64_t mu, uint64_t shoup_b) {
    if (algo == METAL_ALGO_SHOUP) {
        return metal_modmult_shoup(a, b, mod, shoup_b);
    } else {
        return metal_modmult_barrett(a, b, mod);
    }
}

#endif  // FIDESLIB_METAL_MATH_HPP
