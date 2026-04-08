//
// FIDESlib Metal Backend - Modular Multiplication Kernels (Metal Shading Language)
// Ported from ModMult.cu and ModMult.cuh
//

#include <metal_stdlib>
#include "MetalMath.hpp"

using namespace metal;

// =============================================================================
// Modular Multiplication Helper Functions (inlineable in Metal)
// These mirror the CUDA versions in ModMult.cuh
// =============================================================================

// 64-bit integer improved Barrett modular multiplication implementation (p < 2^62)
inline uint64_t metal_neal_mult_64(uint64_t op1, uint64_t op2, uint64_t mu,
                                    uint64_t prime, uint32_t qbit) {
    __uint128_t c = (__uint128_t)op1 * op2;
    uint64_t rx = c >> (qbit - 2);
    uint64_t rb = metal_umul64hi(rx << (62 - qbit), mu) >> 1;
    rb *= prime;
    uint64_t c_lo = c;
    c_lo -= rb;
    c_lo -= prime * (c_lo >= prime);
    return c_lo;
}

// 32-bit integer improved Barrett modular multiplication implementation (p < 2^30)
inline uint32_t metal_neal_mult_32(uint32_t op1, uint32_t op2, uint32_t mu,
                                    uint32_t prime, uint32_t qbit) {
    uint64_t c = (uint64_t)op1 * op2;
    uint32_t rx = c >> (qbit - 2);
    uint32_t rb = metal_umulhi(rx << (30 - qbit), mu) >> 1;
    rb *= prime;
    uint32_t c_lo = c;
    c_lo -= rb;
    c_lo -= prime * (c_lo >= prime);
    return c_lo;
}

// 64-bit integer Shoup modular multiplication
// Requires: psi = op2 * 2^64 / prime, prime < 2^63
inline uint64_t metal_shoup_mult_64(uint64_t op1, uint64_t op2, uint64_t psi,
                                    uint64_t prime) {
    uint64_t c = metal_umul64hi(op1, psi);
    uint64_t c_lo = op1 * op2 - c * prime;
    c_lo -= prime * (c_lo >= prime);
    return c_lo;
}

// 32-bit integer Shoup modular multiplication
// Requires: psi = op2 * 2^32 / prime, prime < 2^31
inline uint32_t metal_shoup_mult_32(uint32_t op1, uint32_t op2, uint32_t psi,
                                    uint32_t prime) {
    uint32_t c = metal_umulhi(op1, psi);
    uint32_t c_lo = op1 * op2 - c * prime;
    c_lo -= prime * (c_lo >= prime);
    return c_lo;
}

// 53-bit integer improved Barrett modular multiplication (pure integer version)
// Uses scaled multiplication approach to avoid floating point
inline uint64_t metal_neal_mult_53(uint64_t op1, uint64_t op2, uint64_t mu,
                                    uint64_t prime, uint32_t qbit) {
    // Scale operands to use more bits
    uint64_t op1_scaled = op1 << (53 - qbit);
    uint64_t op2_scaled = op2 << (53 - qbit);

    // Use 128-bit multiply and Barrett reduction
    __uint128_t c = (__uint128_t)op1_scaled * op2_scaled;
    uint64_t rx = c >> 64;  // Upper 64 bits
    uint64_t rb = metal_umul64hi(rx, mu) >> 1;
    rb *= prime;
    uint64_t c_lo = (uint64_t)c;
    c_lo -= rb;
    c_lo -= prime * (c_lo >= prime);
    return c_lo;
}

// Unified modular multiply dispatch
inline uint64_t metal_modmult(uint64_t op1, uint64_t op2, uint64_t prime,
                              uint64_t mu, uint64_t shoup, int algo) {
    if (algo == METAL_ALGO_SHOUP) {
        return metal_shoup_mult_64(op1, op2, shoup, prime);
    } else if (algo == METAL_ALGO_BARRETT) {
        return metal_neal_mult_64(op1, op2, mu, prime, 62);
    } else {
        return metal_neal_mult_64(op1, op2, mu, prime, 62);
    }
}

// =============================================================================
// Modular Multiplication Kernels
// Grid: N/128 threads
// =============================================================================

kernel void metal_mult_u64(device uint64_t* a [[buffer(0)]],
                          device uint64_t* b [[buffer(1)]],
                          constant uint64_t* primes [[buffer(2)]],
                          constant uint64_t* barret_mu [[buffer(3)]],
                          constant uint32_t* prime_bits [[buffer(4)]],
                          constant int& primeid [[buffer(5)]],
                          constant int& algo [[buffer(6)]],
                          uint idx [[thread_position_in_grid]]) {

    uint64_t op1 = a[idx];
    uint64_t op2 = b[idx];
    uint64_t prime = primes[primeid];
    uint64_t mu = barret_mu[primeid];

    a[idx] = metal_modmult(op1, op2, prime, mu, 0, algo);
}

kernel void metal_mult_u32(device uint32_t* a [[buffer(0)]],
                          device uint32_t* b [[buffer(1)]],
                          constant uint64_t* primes [[buffer(2)]],
                          constant uint64_t* barret_mu [[buffer(3)]],
                          constant uint32_t* prime_bits [[buffer(4)]],
                          constant int& primeid [[buffer(5)]],
                          constant int& algo [[buffer(6)]],
                          uint idx [[thread_position_in_grid]]) {

    uint32_t op1 = a[idx];
    uint32_t op2 = b[idx];
    uint32_t prime = (uint32_t)primes[primeid];
    uint32_t mu = (uint32_t)barret_mu[primeid];

    a[idx] = metal_neal_mult_32(op1, op2, mu, prime, 30);
}

// Three-operand multiply: a = b * c
kernel void metal_mult3_u64(device uint64_t* a [[buffer(0)]],
                            device uint64_t* b [[buffer(1)]],
                            device uint64_t* c [[buffer(2)]],
                            constant uint64_t* primes [[buffer(3)]],
                            constant uint64_t* barret_mu [[buffer(4)]],
                            constant uint32_t* prime_bits [[buffer(5)]],
                            constant int& primeid [[buffer(6)]],
                            constant int& algo [[buffer(7)]],
                            uint idx [[thread_position_in_grid]]) {

    uint64_t op1 = b[idx];
    uint64_t op2 = c[idx];
    uint64_t prime = primes[primeid];
    uint64_t mu = barret_mu[primeid];

    a[idx] = metal_modmult(op1, op2, prime, mu, 0, algo);
}

// Scalar multiply: a = a * scalar
kernel void metal_scalar_mult_u64(device uint64_t* a [[buffer(0)]],
                                  constant uint64_t& scalar [[buffer(1)]],
                                  constant uint64_t* primes [[buffer(2)]],
                                  constant uint64_t* barret_mu [[buffer(3)]],
                                  constant uint32_t* prime_bits [[buffer(4)]],
                                  constant int& primeid [[buffer(5)]],
                                  constant int& algo [[buffer(6)]],
                                  constant uint64_t& shoup_scalar [[buffer(7)]],
                                  uint idx [[thread_position_in_grid]]) {

    uint64_t op1 = a[idx];
    uint64_t prime = primes[primeid];
    uint64_t mu = barret_mu[primeid];

    if (algo == METAL_ALGO_SHOUP) {
        a[idx] = metal_shoup_mult_64(op1, scalar, shoup_scalar, prime);
    } else {
        a[idx] = metal_neal_mult_64(op1, scalar, mu, prime, 62);
    }
}

// =============================================================================
// Flat Multi-Limb Multiplication Kernels
// Grid: (N/128, numLimbs) as uint2
// =============================================================================

kernel void metal_mult_flat(device uint64_t* a [[buffer(0)]],
                            device uint64_t* b [[buffer(1)]],
                            constant uint32_t& N [[buffer(2)]],
                            constant uint32_t& numLimbs [[buffer(3)]],
                            constant uint64_t* primes [[buffer(4)]],
                            constant uint64_t* barret_mu [[buffer(5)]],
                            constant int* primeid_flattened [[buffer(6)]],
                            constant int& algo [[buffer(7)]],
                            uint2 idx [[thread_position_in_grid]]) {

    uint limbIdx = idx.y;
    if (limbIdx >= numLimbs) return;

    uint elemIdx = idx.x;
    uint N_per_limb = N / 128;
    if (elemIdx >= N_per_limb) return;

    uint offset = limbIdx * N;
    uint primeid = primeid_flattened[limbIdx];
    uint64_t prime = primes[primeid];
    uint64_t mu = barret_mu[primeid];

    uint64_t op1 = a[offset + elemIdx];
    uint64_t op2 = b[offset + elemIdx];

    a[offset + elemIdx] = metal_modmult(op1, op2, prime, mu, 0, algo);
}
