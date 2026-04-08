//
// FIDESlib Metal Backend - Convolution and Modulus Switching Kernels
// Ported from Conv.cu
//
// These kernels handle:
// - ModDown2: Modulus switching down (RNS CRT decomposition)
// - DecompAndModUpConv: Decomposition and modulus up with convolution
//

#include <metal_stdlib>
#include "MetalMath.hpp"

using namespace metal;

// =============================================================================
// Constants (must match CUDA ConstantsGPU.cuh)
// =============================================================================

#ifndef MAXP
#define MAXP 64
#endif

// =============================================================================
// Modular Reduction
// =============================================================================

// Modular reduce using 128-bit intermediate
inline uint64_t metal_modreduce(__uint128_t a, uint64_t mod) {
    uint64_t result = (uint64_t)(a % mod);
    return result;
}

// =============================================================================
// ModDown2 Kernel
// Performs modulus switching down using precomputed scales and matrix
// Used in key switching to reduce from Q*P to Q
//
// Grid: (N/128, K) where K is the digit size
// Threadgroup: 128 x K threads
// Shared memory: sizeof(uint64_t) * (K + blockDim.y) * blockDim.y
// Uses flat buffer layout: b[K*N], a[n*N]
// =============================================================================

kernel void metal_ModDown2(
    device uint64_t* a [[buffer(0)]],         // Output flat buffer: n*N
    device uint64_t* b [[buffer(1)]],         // Input flat buffer: K*N
    constant uint64_t* primes [[buffer(2)]],
    constant uint64_t* ModDown_pre_scale [[buffer(3)]],     // Pre-scale factors
    constant uint64_t* ModDown_pre_scale_shoup [[buffer(4)]], // Shoup pre-scale
    constant uint64_t* ModDown_matrix [[buffer(5)]],        // Decomposition matrix
    constant uint64_t* ModDown_matrix_shoup [[buffer(6)]],  // Shoup matrix
    constant uint32_t& K [[buffer(7)]],           // Digit size
    constant uint32_t& L [[buffer(8)]],           // Base modulus size
    constant uint32_t& n [[buffer(9)]],           // Number of moduli
    constant uint32_t& N [[buffer(10)]],          // Elements per limb
    constant uint32_t& primeid_init [[buffer(11)]], // Starting primeid
    constant uint32_t* primeid_flattened [[buffer(12)]], // Flattened primeid lookup
    constant uint32_t& algo [[buffer(13)]],       // Algorithm: BARRETT, SHOUP, etc.
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    threadgroup uint64_t* shared_mem [[threadgroup(0)]]) {

    const uint elemIdx = gid.x;
    const uint K_dim = K;
    const uint L_base = L;
    const uint N_per_limb = N;
    const uint tid = lid.x;

    // Shared memory layout: buff[tid + 128 * i] for i in [0, K)
    threadgroup uint64_t* buff = shared_mem;

    // Step 1: Load and pre-scale all K limbs
    for (uint i = tid; i < K_dim; i += 128) {
        uint primeid = L_base + i;
        uint buff_idx = tid + 128 * i;

        if (primeid < MAXP && elemIdx < N_per_limb) {
            uint64_t val = b[i * N_per_limb + elemIdx];
            uint64_t scale = ModDown_pre_scale[primeid];

            if (algo == 3) {  // ALGO_SHOUP
                uint64_t scale_shoup = ModDown_pre_scale_shoup[primeid];
                buff[buff_idx] = metal_modmult_shoup(val, scale, primes[primeid], scale_shoup);
            } else {
                buff[buff_idx] = metal_modmult_barrett(val, scale, primes[primeid]);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Step 2: Apply decomposition matrix
    // For each output modulus j, compute sum over i: buff[i] * matrix[i][j]
    for (uint j = tid; j < n; j += 128) {
        uint primeid_j = primeid_flattened[primeid_init + j];

        __uint128_t res = 0;
        for (uint i = 0; i < K_dim; i++) {
            uint buff_idx = i * 128 + tid;
            uint matrix_idx = K_dim * i + j;  // MODDOWN_MATRIX(i, j) equivalent

            if (buff_idx < K_dim * 128 && matrix_idx < K_dim * n && elemIdx < N_per_limb) {
                res += (__uint128_t)buff[buff_idx] * ModDown_matrix[matrix_idx];
            }
        }

        // Final reduction
        if (elemIdx < N_per_limb) {
            a[j * N_per_limb + elemIdx] = metal_modreduce(res, primes[primeid_j]);
        }
    }
}

// =============================================================================
// DecompAndModUpConv Kernel
// Decomposes CRT representation and performs modulus up with convolution
// Used to go from Q to Q*P (extended modulus)
//
// Grid: (N/128, num_limbs)
// Threadgroup: 128 x threadgroup_size
// Shared memory: sizeof(uint64_t) * threadgroup_size * K
// Uses flat buffer layout
// =============================================================================

kernel void metal_DecompAndModUpConv(
    device uint64_t* a [[buffer(0)]],         // Output flat buffer
    device uint64_t* b [[buffer(1)]],         // Input flat buffer
    constant uint64_t* primes [[buffer(2)]],
    constant uint64_t* DecompAndModUp_pre_scale [[buffer(3)]],    // Pre-scale
    constant uint64_t* DecompAndModUp_pre_scale_shoup [[buffer(4)]], // Shoup pre-scale
    constant uint64_t* DecompAndModUp_matrix [[buffer(5)]],       // Up-matrix
    constant uint64_t* DecompAndModUp_matrix_shoup [[buffer(6)]], // Shoup matrix
    constant uint32_t& d [[buffer(7)]],            // Current digit
    constant uint32_t& n [[buffer(8)]],            // Number of output moduli
    constant uint32_t& n_d_n [[buffer(9)]],        // Number of source moduli for digit d
    constant uint32_t* primeid_digit_from [[buffer(10)]], // Source primeids for digit d
    constant uint32_t* primeid_digit_to [[buffer(11)]],   // Target primeids for digit d
    constant uint32_t* num_primeid_digit_to [[buffer(12)]], // Count of target moduli per digit
    constant uint32_t& N [[buffer(13)]],           // Elements per limb
    constant uint32_t& algo [[buffer(14)]],         // Algorithm
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    threadgroup uint64_t* shared_mem [[threadgroup(0)]]) {

    const uint elemIdx = gid.x;
    const uint n_sources = n_d_n;  // Number of source moduli for this digit
    const uint N_per_limb = N;
    const uint tid = lid.x;

    threadgroup uint64_t* buff = shared_mem;

    // Step 1: Load and pre-scale source moduli
    for (uint i_ = tid; i_ < n_sources; i_ += 128) {
        uint primeid = primeid_digit_from[d * MAXP + i_];

        if (primeid < MAXP && elemIdx < N_per_limb) {
            uint64_t val = b[i_ * N_per_limb + elemIdx];
            uint scale_idx = d * MAXP * MAXP + (n_sources - 1) * MAXP + primeid;  // MODUPIDX_SCALE

            uint64_t scale = DecompAndModUp_pre_scale[scale_idx];
            uint64_t buff_idx = tid + 128 * i_;

            if (algo == 3) {  // ALGO_SHOUP
                uint64_t scale_shoup = DecompAndModUp_pre_scale_shoup[scale_idx];
                buff[buff_idx] = metal_modmult_shoup(val, scale, primes[primeid], scale_shoup);
            } else {
                buff[buff_idx] = metal_modmult_barrett(val, scale, primes[primeid]);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Step 2: Apply up-matrix and accumulate
    uint num_targets = num_primeid_digit_to[d * MAXP + n - 1];

    for (uint j_ = tid; j_ < num_targets; j_ += 128) {
        uint primeid_j = primeid_digit_to[d * MAXP + j_];

        // Skip if primeid_j is not in valid range
        if (primeid_j >= MAXP || elemIdx >= N_per_limb) {
            continue;
        }

        __uint128_t res = 0;

        for (uint i_ = 0; i_ < n_sources; i_++) {
            uint buff_idx = i_ * 128 + tid;
            uint matrix_idx = (n - 1) * MAXP * MAXP * MAXP + d * MAXP * MAXP + i_ * MAXP + primeid_j;

            if (buff_idx < n_sources * 128 && matrix_idx < MAXP * MAXP * MAXP * MAXP) {
                res += (__uint128_t)buff[buff_idx] * DecompAndModUp_matrix[matrix_idx];
            }
        }

        a[primeid_j * N_per_limb + elemIdx] = metal_modreduce(res, primes[primeid_j]);
    }
}

// =============================================================================
// conv1 Kernel (Simple)
// Single element modular multiplication by q_hat_inv
// Used in CRT reconstruction
// =============================================================================

kernel void metal_conv1(
    device uint64_t* a [[buffer(0)]],
    constant uint64_t& q_hat_inv [[buffer(1)]],
    constant uint64_t* primes [[buffer(2)]],
    constant uint32_t& primeid [[buffer(3)]],
    uint idx [[thread_position_in_grid]]) {

    a[idx] = metal_modmult_barrett(a[idx], q_hat_inv, primes[primeid]);
}

// =============================================================================
// SwitchModulus Kernel
// Switches a value from old modulus om to new modulus nm
// Used in modulus switching operations
// =============================================================================

kernel void metal_SwitchModulus(
    device uint64_t* src [[buffer(0)]],
    device uint64_t* res [[buffer(1)]],
    constant uint64_t* primes [[buffer(2)]],
    constant uint32_t& old_primeid [[buffer(3)]],
    constant uint32_t& new_primeid [[buffer(4)]],
    uint idx [[thread_position_in_grid]]) {

    uint64_t old_mod = primes[old_primeid];
    uint64_t new_mod = primes[new_primeid];
    uint64_t val = src[idx];
    uint64_t halfQ = old_mod >> 1;

    uint64_t result;

    if (new_mod > old_mod) {
        // New modulus larger: just add difference if needed
        uint64_t diff = new_mod - old_mod;
        if (val > halfQ) {
            result = val + diff;
        } else {
            result = val;
        }
    } else {
        // New modulus smaller: need reduction
        uint64_t diff = new_mod - (old_mod % new_mod);
        if (val > halfQ) {
            result = val + diff;
        } else {
            result = val;
        }
        if (result >= new_mod) {
            result = result % new_mod;
        }
    }

    res[idx] = result;
}

// =============================================================================
// ModUp Diagonal Kernel
// Performs modulus up along a specific diagonal of the decomposition matrix
// Uses flat buffer layout: a[num_targets*N], b[num_sources*N]
// =============================================================================

kernel void metal_ModUpDiag(
    device uint64_t* a [[buffer(0)]],         // Output flat buffer
    device uint64_t* b [[buffer(1)]],         // Input flat buffer
    constant uint64_t* primes [[buffer(2)]],
    constant uint64_t* pre_scale [[buffer(3)]],
    constant uint64_t* pre_scale_shoup [[buffer(4)]],
    constant uint32_t& digit [[buffer(5)]],
    constant uint32_t& num_sources [[buffer(6)]],
    constant uint32_t& num_targets [[buffer(7)]],
    constant uint32_t* source_primeids [[buffer(8)]],
    constant uint32_t* target_primeids [[buffer(9)]],
    constant uint32_t& N [[buffer(10)]],
    constant uint32_t& algo [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]]) {

    const uint elemIdx = gid.x;
    const uint target_idx = gid.y;
    const uint N_per_limb = N;

    if (target_idx >= num_targets || elemIdx >= N_per_limb) return;

    uint target_primeid = target_primeids[digit * MAXP + target_idx];
    if (target_primeid >= MAXP) return;

    __uint128_t res = 0;

    for (uint src_idx = 0; src_idx < num_sources; src_idx++) {
        uint source_primeid = source_primeids[digit * MAXP + src_idx];
        if (source_primeid >= MAXP) continue;

        uint scale_idx = digit * MAXP * MAXP + src_idx * MAXP + target_primeid;

        uint64_t val = b[src_idx * N_per_limb + elemIdx];
        uint64_t scale = pre_scale[scale_idx];

        uint64_t scaled_val;
        if (algo == 3) {  // ALGO_SHOUP
            scaled_val = metal_modmult_shoup(val, scale, primes[source_primeid], pre_scale_shoup[scale_idx]);
        } else {
            scaled_val = metal_modmult_barrett(val, scale, primes[source_primeid]);
        }

        res += scaled_val;
    }

    a[target_primeid * N_per_limb + elemIdx] = metal_modreduce(res, primes[target_primeid]);
}