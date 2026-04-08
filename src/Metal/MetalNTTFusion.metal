//
// FIDESlib Metal Backend - NTT Fusion Kernels
// Ported from NTTfusions.cuh
// These are fused NTT operations for CKKS multiplication and key switching
//

#include <metal_stdlib>
#include "MetalMath.hpp"

using namespace metal;

#ifndef MAXP
#define MAXP 64
#endif

#define NEGACYCLIC 1

// =============================================================================
// Simple NTT Fusion Kernels
// These are essential fused operations that combine multiple NTT steps
// =============================================================================

// Forward NTT with pre-scaling (combines scale and first butterfly stage)
// Used in key switching and multiplication
kernel void metal_NTT_fuse_forward(
    device uint64_t* data [[buffer(0)]],
    device uint64_t* res [[buffer(1)]],
    threadgroup uint64_t* scratch [[threadgroup(0)]],
    constant uint64_t* globals_psi [[buffer(2)]],
    constant uint64_t* globals_psi_shoup [[buffer(3)]],
    constant uint64_t* globals_psi_no [[buffer(4)]],
    constant uint64_t* globals_root [[buffer(5)]],
    constant uint64_t* globals_root_shoup [[buffer(6)]],
    constant uint64_t* primes [[buffer(7)]],
    constant uint32_t& primeid [[buffer(8)]],
    constant uint32_t& N [[buffer(9)]],
    constant uint32_t& logN [[buffer(10)]],
    constant uint32_t& gridDim_x [[buffer(11)]],
    constant uint32_t& algo [[buffer(12)]],       // METAL_ALGO_* enum
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]) {

    const int M = 4;
    const uint tid = lid.x;
    const uint j = tid << 1;
    const uint blockIdx_x = gid.x;
    const uint numThreads = lid.x;

    threadgroup uint64_t shared[256 * M + 256];

    bool use_shoup = (algo == 3);
    uint64_t mod = primes[primeid];

    // Load psi twiddle factor
    uint64_t psi_shoup_tid = 0;
    if (use_shoup && globals_psi_shoup != nullptr) {
        psi_shoup_tid = globals_psi_shoup[primeid * 256 + tid];
    }
    uint64_t psi_twiddle = globals_psi[primeid * 256 + tid];

    // Load data (simplified - no transposition for fused kernel)
    {
        const uint offset_2t = blockIdx_x * numThreads * M + tid;
        for (int i = 0; i < M; i++) {
            shared[i * 256 + tid] = data[offset_2t * 2 + i * 2];
            shared[i * 256 + tid + numThreads] = data[offset_2t * 2 + i * 2 + 1];
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Negacyclic scale
    if (NEGACYCLIC) {
        uint64_t root_val = globals_root[primeid];
        uint64_t root_shoup = (globals_root_shoup != nullptr) ? globals_root_shoup[primeid] : 0;
        uint64_t fourth_root = psi_twiddle;

        for (int i = 0; i < M; i++) {
            uint64_t val0 = shared[i * 256 + tid];
            uint64_t val1 = shared[i * 256 + tid + numThreads];

            if (use_shoup) {
                val0 = metal_modmult_shoup(val0, root_val, mod, root_shoup);
                val1 = metal_modmult_shoup(val1, fourth_root, mod, root_shoup);
            } else {
                val0 = metal_modmult_barrett(val0, root_val, mod);
                val1 = metal_modmult_barrett(val1, fourth_root, mod);
            }

            shared[i * 256 + tid] = val0;
            shared[i * 256 + tid + numThreads] = val1;
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Butterfly iterations
    int m = numThreads;
    int maskPsi = m;

    // First iteration - no twiddle
    for (int i = 0; i < M; i++) {
        uint64_t val0 = shared[i * 256 + tid];
        uint64_t val1 = shared[i * 256 + tid + m];
        shared[i * 256 + tid] = metal_modadd(val0, val1, mod);
        shared[i * 256 + tid + m] = metal_modsub(val0, val1, mod);
    }

    m >>= 1;
    maskPsi |= (maskPsi >> 1);
    int log_psi = metal_clz(numThreads) - 2;

    for (; m >= 1; m >>= 1, log_psi--, maskPsi |= (maskPsi >> 1)) {
        threadgroup_barrier(mem_flags::mem_device);

        const uint mask = m - 1;
        int j1 = (mask & tid) | ((~mask & tid) << 1);
        int j2 = j1 + m;
        const uint psiid = (tid & maskPsi) >> log_psi;
        const uint64_t psiaux = globals_psi[primeid * 256 + psiid];
        uint64_t psiaux_shoup = use_shoup ? globals_psi_shoup[primeid * 256 + psiid] : 0;

        for (int i = 0; i < M; i++) {
            uint64_t c = shared[i * 256 + j1];
            uint64_t d = shared[i * 256 + j2];

            // CT butterfly
            if (use_shoup) {
                d = metal_modmult_shoup(d, psiaux, mod, psiaux_shoup);
            } else {
                d = metal_modmult_barrett(d, psiaux, mod);
            }
            uint64_t sum = metal_modadd(c, d, mod);
            uint64_t diff = metal_modsub(c, d, mod);

            shared[i * 256 + j1] = sum;
            shared[i * 256 + j2] = diff;
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Store result
    const uint offset_2t = blockIdx_x * numThreads * M + tid;
    for (int i = 0; i < M; i++) {
        res[offset_2t * 2 + i * 2] = shared[i * 256 + tid];
        res[offset_2t * 2 + i * 2 + 1] = shared[i * 256 + tid + 1];
    }
}

// =============================================================================
// Inverse NTT with post-scaling (combines last butterfly and scale)
// =============================================================================

kernel void metal_INTT_fuse_backward(
    device uint64_t* data [[buffer(0)]],
    device uint64_t* res [[buffer(1)]],
    threadgroup uint64_t* scratch [[threadgroup(0)]],
    constant uint64_t* globals_inv_psi [[buffer(2)]],
    constant uint64_t* globals_inv_psi_shoup [[buffer(3)]],
    constant uint64_t* globals_inv_psi_no [[buffer(4)]],
    constant uint64_t* globals_root [[buffer(5)]],
    constant uint64_t* globals_root_shoup [[buffer(6)]],
    constant uint64_t* globals_N_inv [[buffer(7)]],
    constant uint64_t* primes [[buffer(8)]],
    constant uint32_t& primeid [[buffer(9)]],
    constant uint32_t& N [[buffer(10)]],
    constant uint32_t& logN [[buffer(11)]],
    constant uint32_t& gridDim_x [[buffer(12)]],
    constant uint32_t& algo [[buffer(13)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]) {

    const int M = 4;
    const uint tid = lid.x;
    const uint j = tid << 1;
    const uint blockIdx_x = gid.x;
    const uint numThreads = lid.x;

    threadgroup uint64_t shared[256 * M + 256];

    bool use_shoup = (algo == 3);
    uint64_t mod = primes[primeid];

    uint64_t psi_shoup_tid = 0;
    if (use_shoup && globals_inv_psi_shoup != nullptr) {
        psi_shoup_tid = globals_inv_psi_shoup[primeid * 256 + tid];
    }
    uint64_t inv_psi_twiddle = globals_inv_psi[primeid * 256 + tid];

    // Load data
    {
        const uint offset_2t = blockIdx_x * numThreads * M + tid;
        for (int i = 0; i < M; i++) {
            shared[i * 256 + tid] = data[offset_2t * 2 + i * 2];
            shared[i * 256 + tid + numThreads] = data[offset_2t * 2 + i * 2 + 1];
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // GS butterfly loop
    int m = 1;
    int maskPsi = 1;
    uint log_psi = 0;

    for (; m < (int)numThreads; m <<= 1, maskPsi = (maskPsi << 1) | maskPsi, ++log_psi) {
        threadgroup_barrier(mem_flags::mem_device);

        const uint mask = m - 1;
        const uint j1 = (mask & tid) | (((~mask) << 1) & (tid << 1));
        const uint j2 = j1 | m;

        const uint psiid = (tid & maskPsi) >> log_psi;
        const uint64_t psiaux = globals_inv_psi[primeid * 256 + psiid];
        uint64_t psiaux_shoup = use_shoup ? globals_inv_psi_shoup[primeid * 256 + psiid] : 0;

        for (int i = 0; i < M; i++) {
            uint64_t a = shared[i * 256 + j1];
            uint64_t b = shared[i * 256 + j2];

            uint64_t c = metal_modadd(a, b, mod);
            uint64_t d = metal_modsub(a, b, mod);

            if (use_shoup) {
                d = metal_modmult_shoup(d, psiaux, mod, psiaux_shoup);
            } else {
                d = metal_modmult_barrett(d, psiaux, mod);
            }

            shared[i * 256 + j1] = c;
            shared[i * 256 + j2] = d;
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Final stage
    for (int i = 0; i < M; i++) {
        uint64_t val0 = shared[i * 256 + tid];
        uint64_t val1 = shared[i * 256 + tid + m];
        shared[i * 256 + tid] = metal_modadd(val0, val1, mod);
        shared[i * 256 + tid + m] = metal_modsub(val0, val1, mod);
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Scale by N_inv and output
    uint64_t N_inv = globals_N_inv[primeid];
    const uint offset_2t = blockIdx_x * numThreads * M + tid;
    for (int i = 0; i < M; i++) {
        uint64_t val0 = metal_modmult_barrett(shared[i * 256 + tid], N_inv, mod);
        uint64_t val1 = metal_modmult_barrett(shared[i * 256 + tid + 1], N_inv, mod);
        res[offset_2t * 2 + i * 2] = val0;
        res[offset_2t * 2 + i * 2 + 1] = val1;
    }
}

// =============================================================================
// Multiply and Save Fusion (INTT * pointwise multiply and store)
// =============================================================================

kernel void metal_mult_and_save(
    device uint64_t* dat [[buffer(0)]],
    device uint64_t* dat2 [[buffer(1)]],
    device uint64_t* res [[buffer(2)]],
    device uint64_t* res0 [[buffer(3)]],        // For fusion output
    device uint64_t* res1 [[buffer(4)]],        // For fusion output
    threadgroup uint64_t* scratch [[threadgroup(0)]],
    constant uint64_t* globals_inv_psi [[buffer(5)]],
    constant uint64_t* globals_inv_psi_shoup [[buffer(6)]],
    constant uint64_t* globals_N_inv [[buffer(7)]],
    constant uint64_t* primes [[buffer(8)]],
    constant uint32_t& primeid [[buffer(9)]],
    constant uint32_t& N [[buffer(10)]],
    constant uint32_t& logN [[buffer(11)]],
    constant uint32_t& gridDim_x [[buffer(12)]],
    constant uint32_t& algo [[buffer(13)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]) {

    const int M = 4;
    const uint tid = lid.x;
    const uint j = tid << 1;
    const uint blockIdx_x = gid.x;
    const uint numThreads = lid.x;

    threadgroup uint64_t shared[256 * M + 256];

    bool use_shoup = (algo == 3);
    uint64_t mod = primes[primeid];

    const uint offset_2t = blockIdx_x * numThreads * M + tid;

    // Load and multiply
    uint64_t c1_[2], c1tilde_[2];
    c1_[0] = dat[offset_2t * 2];
    c1_[1] = dat[offset_2t * 2 + 1];
    c1tilde_[0] = dat2[offset_2t * 2];
    c1tilde_[1] = dat2[offset_2t * 2 + 1];

    uint64_t in1_0 = metal_modmult_barrett(c1_[0], c1tilde_[0], mod);
    uint64_t in1_1 = metal_modmult_barrett(c1_[1], c1tilde_[1], mod);

    for (int i = 0; i < M; i++) {
        shared[i * 256 + j] = in1_0;
        shared[i * 256 + j + 1] = in1_1;
    }

    threadgroup_barrier(mem_flags::mem_device);

    // GS butterfly loop
    int m = 1;
    int maskPsi = 1;
    uint log_psi = 0;

    for (; m < (int)numThreads; m <<= 1, maskPsi = (maskPsi << 1) | maskPsi, ++log_psi) {
        threadgroup_barrier(mem_flags::mem_device);

        const uint mask = m - 1;
        const uint j1 = (mask & tid) | (((~mask) << 1) & (tid << 1));
        const uint j2 = j1 | m;

        const uint psiid = (tid & maskPsi) >> log_psi;
        const uint64_t psiaux = globals_inv_psi[primeid * 256 + psiid];
        uint64_t psiaux_shoup = use_shoup ? globals_inv_psi_shoup[primeid * 256 + psiid] : 0;

        for (int i = 0; i < M; i++) {
            uint64_t a = shared[i * 256 + j1];
            uint64_t b = shared[i * 256 + j2];

            uint64_t c = metal_modadd(a, b, mod);
            uint64_t d = metal_modsub(a, b, mod);

            if (use_shoup) {
                d = metal_modmult_shoup(d, psiaux, mod, psiaux_shoup);
            } else {
                d = metal_modmult_barrett(d, psiaux, mod);
            }

            shared[i * 256 + j1] = c;
            shared[i * 256 + j2] = d;
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Final add/sub
    for (int i = 0; i < M; i++) {
        uint64_t val0 = shared[i * 256 + tid];
        uint64_t val1 = shared[i * 256 + tid + m];
        shared[i * 256 + tid] = metal_modadd(val0, val1, mod);
        shared[i * 256 + tid + m] = metal_modsub(val0, val1, mod);
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Scale and output
    uint64_t N_inv = globals_N_inv[primeid];
    for (int i = 0; i < M; i++) {
        uint64_t val0 = metal_modmult_barrett(shared[i * 256 + tid], N_inv, mod);
        uint64_t val1 = metal_modmult_barrett(shared[i * 256 + tid + 1], N_inv, mod);
        res[offset_2t * 2 + i * 2] = val0;
        res[offset_2t * 2 + i * 2 + 1] = val1;
    }
}

// =============================================================================
// Square and Save Fusion (INTT * square and store)
// =============================================================================

kernel void metal_square_and_save(
    device uint64_t* dat [[buffer(0)]],
    device uint64_t* res [[buffer(1)]],
    threadgroup uint64_t* scratch [[threadgroup(0)]],
    constant uint64_t* globals_inv_psi [[buffer(2)]],
    constant uint64_t* globals_inv_psi_shoup [[buffer(3)]],
    constant uint64_t* globals_N_inv [[buffer(4)]],
    constant uint64_t* primes [[buffer(5)]],
    constant uint32_t& primeid [[buffer(6)]],
    constant uint32_t& N [[buffer(7)]],
    constant uint32_t& logN [[buffer(8)]],
    constant uint32_t& gridDim_x [[buffer(9)]],
    constant uint32_t& algo [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]) {

    const int M = 4;
    const uint tid = lid.x;
    const uint j = tid << 1;
    const uint blockIdx_x = gid.x;
    const uint numThreads = lid.x;

    threadgroup uint64_t shared[256 * M + 256];

    bool use_shoup = (algo == 3);
    uint64_t mod = primes[primeid];

    const uint offset_2t = blockIdx_x * numThreads * M + tid;

    // Load and square
    uint64_t val0 = dat[offset_2t * 2];
    uint64_t val1 = dat[offset_2t * 2 + 1];
    uint64_t sq0 = metal_modmult_barrett(val0, val0, mod);
    uint64_t sq1 = metal_modmult_barrett(val1, val1, mod);

    for (int i = 0; i < M; i++) {
        shared[i * 256 + j] = sq0;
        shared[i * 256 + j + 1] = sq1;
    }

    threadgroup_barrier(mem_flags::mem_device);

    // GS butterfly
    int m = 1;
    int maskPsi = 1;
    uint log_psi = 0;

    for (; m < (int)numThreads; m <<= 1, maskPsi = (maskPsi << 1) | maskPsi, ++log_psi) {
        threadgroup_barrier(mem_flags::mem_device);

        const uint mask = m - 1;
        const uint j1 = (mask & tid) | (((~mask) << 1) & (tid << 1));
        const uint j2 = j1 | m;

        const uint psiid = (tid & maskPsi) >> log_psi;
        const uint64_t psiaux = globals_inv_psi[primeid * 256 + psiid];
        uint64_t psiaux_shoup = use_shoup ? globals_inv_psi_shoup[primeid * 256 + psiid] : 0;

        for (int i = 0; i < M; i++) {
            uint64_t a = shared[i * 256 + j1];
            uint64_t b = shared[i * 256 + j2];

            uint64_t c = metal_modadd(a, b, mod);
            uint64_t d = metal_modsub(a, b, mod);

            if (use_shoup) {
                d = metal_modmult_shoup(d, psiaux, mod, psiaux_shoup);
            } else {
                d = metal_modmult_barrett(d, psiaux, mod);
            }

            shared[i * 256 + j1] = c;
            shared[i * 256 + j2] = d;
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    for (int i = 0; i < M; i++) {
        uint64_t val0 = shared[i * 256 + tid];
        uint64_t val1 = shared[i * 256 + tid + m];
        shared[i * 256 + tid] = metal_modadd(val0, val1, mod);
        shared[i * 256 + tid + m] = metal_modsub(val0, val1, mod);
    }

    threadgroup_barrier(mem_flags::mem_device);

    uint64_t N_inv = globals_N_inv[primeid];
    for (int i = 0; i < M; i++) {
        uint64_t v0 = metal_modmult_barrett(shared[i * 256 + tid], N_inv, mod);
        uint64_t v1 = metal_modmult_barrett(shared[i * 256 + tid + 1], N_inv, mod);
        res[offset_2t * 2 + i * 2] = v0;
        res[offset_2t * 2 + i * 2 + 1] = v1;
    }
}

// =============================================================================
// Rescale Fusion (multiply by delta and reduce)
// =============================================================================

kernel void metal_rescale_fuse(
    device uint64_t* data [[buffer(0)]],
    device uint64_t* res [[buffer(1)]],
    constant uint64_t& delta [[buffer(2)]],
    constant uint64_t* primes [[buffer(3)]],
    constant uint32_t& N [[buffer(4)]],
    constant uint32_t& numLimbs [[buffer(5)]],
    constant int* primeid_flattened [[buffer(6)]],
    uint2 idx [[thread_position_in_grid]]) {

    uint limbIdx = idx.y;
    uint elemIdx = idx.x;
    if (limbIdx >= numLimbs || elemIdx >= N/128) return;

    uint offset = limbIdx * N;
    uint primeid = primeid_flattened[limbIdx];
    uint64_t mod = primes[primeid];

    uint64_t val = data[offset + elemIdx];
    val = metal_modmult_barrett(val, delta, mod);
    res[offset + elemIdx] = val;
}

// =============================================================================
// ModDown Fusion (modulus switching down)
// =============================================================================

kernel void metal_moddown_fuse(
    device uint64_t* data [[buffer(0)]],
    device uint64_t* res [[buffer(1)]],
    constant uint64_t* primes [[buffer(2)]],
    constant uint32_t& N [[buffer(3)]],
    constant uint32_t& numLimbs [[buffer(4)]],
    constant int* primeid_flattened [[buffer(5)]],
    constant uint32_t& primeid_init [[buffer(6)]],
    uint2 idx [[thread_position_in_grid]]) {

    uint limbIdx = idx.y;
    uint elemIdx = idx.x;
    if (limbIdx >= numLimbs || elemIdx >= N/128) return;

    uint offset = limbIdx * N;
    uint primeid = primeid_flattened[primeid_init + limbIdx];
    uint64_t mod = primes[primeid];

    uint64_t val = data[offset + elemIdx];
    if (val >= mod) val = val % mod;
    res[offset + elemIdx] = val;
}