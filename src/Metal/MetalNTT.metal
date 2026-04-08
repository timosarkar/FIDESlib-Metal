//
// FIDESlib Metal Backend - Complete NTT/INTT Kernels
// Ported from NTT.cu, NTThelper.cuh, and NTTfusions.cuh
//

#include <metal_stdlib>
#include "MetalMath.hpp"

using namespace metal;

// =============================================================================
// Metal Constants
// =============================================================================

#ifndef MAXP
#define MAXP 64
#endif

#define NEGACYCLIC 1

// =============================================================================
// 2D NTT Kernel (main forward transform)
// =============================================================================

kernel void metal_NTT_2D(
    device uint64_t* dat [[buffer(0)]],
    device uint64_t* res [[buffer(1)]],
    constant uint64_t* globals_psi [[buffer(2)]],
    constant uint64_t* globals_psi_shoup [[buffer(3)]],
    constant uint64_t* globals_psi_no [[buffer(4)]],
    constant uint64_t* globals_inv_psi [[buffer(5)]],
    constant uint64_t* globals_inv_psi_shoup [[buffer(6)]],
    constant uint64_t* globals_inv_psi_no [[buffer(7)]],
    constant uint64_t* globals_inv_psi_middle_scale [[buffer(8)]],
    constant uint64_t* globals_root [[buffer(9)]],
    constant uint64_t* globals_root_shoup [[buffer(10)]],
    constant uint64_t* primes [[buffer(11)]],
    constant uint64_t* P [[buffer(12)]],
    constant uint32_t& primeid [[buffer(13)]],
    constant uint32_t& N [[buffer(14)]],
    constant uint32_t& logN [[buffer(15)]],
    constant uint32_t& gridDim_x [[buffer(16)]],
    constant uint32_t& algo [[buffer(17)]],
    constant uint32_t& mode [[buffer(18)]],
    constant uint32_t& primeid_rescale [[buffer(19)]],
    constant uint32_t& is_second [[buffer(20)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]) {

    const int M = 4;
    const uint tid = lid.x;
    const uint j = tid << 1;
    const uint blockIdx_x = gid.x;
    const uint numThreads = lid.x;

    // Shared memory: 256*M + 256 for psi twiddle factors
    threadgroup uint64_t shared[256 * M + 256];

    bool use_shoup = (algo == 3);
    uint64_t mod = primes[primeid];

    // Load psi twiddle factors
    uint64_t psi_shoup_tid = 0;
    if (use_shoup && globals_psi_shoup != nullptr) {
        psi_shoup_tid = globals_psi_shoup[primeid * 256 + tid];
    }
    uint64_t psi_twiddle = globals_psi[primeid * 256 + tid];

    // =====================================================================
    // Stage 1: Load transposed data
    // =====================================================================
    {
        const uint col_init = j & ~2;

        for (int i = 0; i < M; i++) {
            const uint pos_transp = M * gridDim_x * (col_init + i) + M * blockIdx_x + (j & 2);
            const uint pos_res = col_init + i;

            shared[i * 256 + pos_res * 2] = dat[pos_transp];
            shared[i * 256 + pos_res * 2 + 1] = dat[pos_transp + 1];
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // =====================================================================
    // Forward negacyclic scale (if !second && NEGACYCLIC)
    // =====================================================================
    if (!is_second && NEGACYCLIC) {
        const uint32_t block_pos = (blockIdx_x * M);

        uint64_t aux_3 = globals_psi_no[(tid & 1) * (gridDim_x * M) + block_pos];

        uint64_t aux;
        if (use_shoup) {
            aux = metal_modmult_shoup(aux_3, psi_twiddle, mod, psi_shoup_tid);
        } else {
            aux = metal_modmult_barrett(aux_3, psi_twiddle, mod);
        }

        uint64_t root_val = globals_root[primeid];
        uint64_t root_shoup = (globals_root_shoup != nullptr) ? globals_root_shoup[primeid] : 0;
        uint64_t fourth_root = globals_psi[primeid * 256 + 1];
        uint64_t fourth_root_shoup = (globals_psi_shoup != nullptr) ? globals_psi_shoup[primeid * 256 + 1] : 0;

        for (int i = 0; i < M; i++) {
            if (i > 0) {
                if (use_shoup) {
                    aux = metal_modmult_shoup(aux, root_val, mod, root_shoup);
                } else {
                    aux = metal_modmult_barrett(aux, root_val, mod);
                }
            }

            uint64_t aux2;
            if (use_shoup) {
                aux2 = metal_modmult_shoup(aux, fourth_root, mod, fourth_root_shoup);
            } else {
                aux2 = metal_modmult_barrett(aux, fourth_root, mod);
            }

            uint64_t val0 = shared[i * 256 + tid];
            uint64_t val1 = shared[i * 256 + tid + numThreads];

            if (use_shoup) {
                val0 = metal_modmult_shoup(val0, aux, mod, root_shoup);
                val1 = metal_modmult_shoup(val1, aux2, mod, root_shoup);
            } else {
                val0 = metal_modmult_barrett(val0, aux, mod);
                val1 = metal_modmult_barrett(val1, aux2, mod);
            }

            shared[i * 256 + tid] = val0;
            shared[i * 256 + tid + numThreads] = val1;
        }
    }

    // =====================================================================
    // Main butterfly loop (Cooley-Tukey)
    // =====================================================================

    int m = numThreads;
    int maskPsi = m;

    // Iteration 0 - no twiddle factor
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

        if (m >= 32) {
            threadgroup_barrier(mem_flags::mem_device);
        }

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

    // =====================================================================
    // Stage 3: Output with optional fusion
    // =====================================================================

    if (!is_second) {
        const uint32_t logBD_out = 32 - metal_clz(numThreads) + 3;
        const uint32_t mask_lo_exp = ((N >> 1) | ((N >> metal_clz(numThreads)) - 1));
        const uint32_t clzN = metal_clz(N) + 2;
        const uint32_t block_pos = (blockIdx_x * M);
        const uint col_init = j & ~2;

        for (int i = 0; i < M; i++) {
            uint64_t aux[2];

            for (int k = 0; k < 2; k++) {
                uint32_t br_j = metal_brev(j + k) >> (32 - logBD_out);
                uint32_t exp = block_pos * br_j;
                uint32_t hi_exp_br = metal_brev(exp << clzN) & (numThreads - 1);
                uint32_t lo_exp = exp & mask_lo_exp;

                uint64_t psi_no_val = globals_psi_no[lo_exp * 2];
                uint64_t psi_hi = globals_psi[primeid * 256 + hi_exp_br];
                uint64_t psi_hi_shoup = use_shoup ? globals_psi_shoup[primeid * 256 + hi_exp_br] : 0;

                if (use_shoup) {
                    aux[k] = metal_modmult_shoup(psi_hi, psi_no_val, mod, psi_hi_shoup);
                } else {
                    aux[k] = metal_modmult_barrett(psi_hi, psi_no_val, mod);
                }
            }

            uint64_t val0 = shared[i * 256 + j];
            uint64_t val1 = shared[i * 256 + j + 1];

            val0 = metal_modmult_barrett(val0, aux[0], mod);
            val1 = metal_modmult_barrett(val1, aux[1], mod);

            const uint pos_transp = M * gridDim_x * (col_init + i) + M * blockIdx_x + (j & 2);
            res[pos_transp] = val0;
            res[pos_transp + 1] = val1;
        }
    } else {
        for (int i = 0; i < M; i++) {
            const uint offset_2t = numThreads * M + tid;
            res[offset_2t * 2] = shared[i * 256 + tid * 2];
            res[offset_2t * 2 + 1] = shared[i * 256 + tid * 2 + 1];
        }
    }
}

// =============================================================================
// 2D INTT Kernel (main inverse transform)
// =============================================================================

kernel void metal_INTT_2D(
    device uint64_t* dat [[buffer(0)]],
    device uint64_t* dat2 [[buffer(1)]],
    device uint64_t* res [[buffer(2)]],
    device uint64_t* res0 [[buffer(3)]],
    device uint64_t* res1 [[buffer(4)]],
    device uint64_t* kska [[buffer(5)]],
    device uint64_t* kskb [[buffer(6)]],
    device uint64_t* c0 [[buffer(7)]],
    device uint64_t* c0tilde [[buffer(8)]],
    constant uint64_t* globals_inv_psi [[buffer(9)]],
    constant uint64_t* globals_inv_psi_shoup [[buffer(10)]],
    constant uint64_t* globals_inv_psi_no [[buffer(11)]],
    constant uint64_t* globals_inv_psi_middle_scale [[buffer(12)]],
    constant uint64_t* globals_root [[buffer(13)]],
    constant uint64_t* globals_root_shoup [[buffer(14)]],
    constant uint64_t* globals_N_inv [[buffer(15)]],
    constant uint64_t* primes [[buffer(16)]],
    constant uint64_t* P [[buffer(17)]],
    constant uint32_t& primeid [[buffer(18)]],
    constant uint32_t& N [[buffer(19)]],
    constant uint32_t& logN [[buffer(20)]],
    constant uint32_t& gridDim_x [[buffer(21)]],
    constant uint32_t& algo [[buffer(22)]],
    constant uint32_t& mode [[buffer(23)]],
    constant uint32_t& primeid_rescale [[buffer(24)]],
    constant uint32_t& is_second [[buffer(25)]],
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

    // Load inv_psi twiddle factors
    uint64_t psi_shoup_tid = 0;
    if (use_shoup && globals_inv_psi_shoup != nullptr) {
        psi_shoup_tid = globals_inv_psi_shoup[primeid * 256 + tid];
    }
    uint64_t inv_psi_twiddle = globals_inv_psi[primeid * 256 + tid];

    // =====================================================================
    // Stage 1: Load and optional fusion
    // =====================================================================

    if (mode == 2) {  // INTT_MULT_AND_SAVE
        if (dat2 != nullptr) {
            const uint offset_2t = blockIdx_x * numThreads * M + tid;

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
        }
    } else {
        const uint col_init = j & ~2;
        for (int i = 0; i < M; i++) {
            const uint pos_transp = M * gridDim_x * (col_init + i) + M * blockIdx_x + (j & 2);
            const uint pos_res = col_init + i;

            shared[i * 256 + pos_res * 2] = dat[pos_transp];
            shared[i * 256 + pos_res * 2 + 1] = dat[pos_transp + 1];
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // =====================================================================
    // Backward negacyclic scale (if second && NEGACYCLIC)
    // =====================================================================

    if (is_second && NEGACYCLIC) {
        const uint32_t block_pos = (blockIdx_x * M);

        uint64_t aux_3 = globals_inv_psi_no[(tid & 1) * (gridDim_x * M) + block_pos];
        aux_3 = metal_modmult_shoup(aux_3, globals_N_inv[primeid], mod, psi_shoup_tid);

        uint64_t aux;
        if (use_shoup) {
            aux = metal_modmult_shoup(aux_3, inv_psi_twiddle, mod, psi_shoup_tid);
        } else {
            aux = metal_modmult_barrett(aux_3, inv_psi_twiddle, mod);
        }

        uint64_t inv_root_val = globals_root[primeid];
        uint64_t inv_root_shoup = (globals_root_shoup != nullptr) ? globals_root_shoup[primeid] : 0;
        uint64_t fourth_root = globals_inv_psi[primeid * 256 + 1];
        uint64_t fourth_root_shoup = (globals_inv_psi_shoup != nullptr) ?
            globals_inv_psi_shoup[primeid * 256 + 1] : 0;

        for (int i = 0; i < M; i++) {
            if (i > 0) {
                if (use_shoup) {
                    aux = metal_modmult_shoup(aux, inv_root_val, mod, inv_root_shoup);
                } else {
                    aux = metal_modmult_barrett(aux, inv_root_val, mod);
                }
            }

            uint64_t aux2;
            if (use_shoup) {
                aux2 = metal_modmult_shoup(aux, fourth_root, mod, fourth_root_shoup);
            } else {
                aux2 = metal_modmult_barrett(aux, fourth_root, mod);
            }

            uint64_t val0 = shared[i * 256 + tid];
            uint64_t val1 = shared[i * 256 + tid + numThreads];

            if (use_shoup) {
                val0 = metal_modmult_shoup(val0, aux, mod, inv_root_shoup);
                val1 = metal_modmult_shoup(val1, aux2, mod, inv_root_shoup);
            } else {
                val0 = metal_modmult_barrett(val0, aux, mod);
                val1 = metal_modmult_barrett(val1, aux2, mod);
            }

            shared[i * 256 + tid] = val0;
            shared[i * 256 + tid + numThreads] = val1;
        }
    }

    // =====================================================================
    // Main GS butterfly loop (Gentleman-Sande inverse)
    // =====================================================================

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

            // GS butterfly
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

    // Final add/subtract stage
    for (int i = 0; i < M; i++) {
        uint64_t val0 = shared[i * 256 + tid];
        uint64_t val1 = shared[i * 256 + tid + m];
        shared[i * 256 + tid] = metal_modadd(val0, val1, mod);
        shared[i * 256 + tid + m] = metal_modsub(val0, val1, mod);
    }

    threadgroup_barrier(mem_flags::mem_device);

    // =====================================================================
    // Stage 3: Output with transposition
    // =====================================================================

    {
        const uint col_init = j & ~2;

        for (int i = 0; i < M; i++) {
            uint64_t val0 = shared[i * 256 + col_init * 2];
            uint64_t val1 = shared[i * 256 + col_init * 2 + 1];
            uint64_t val2 = shared[i * 256 + col_init * 2 + numThreads];
            uint64_t val3 = shared[i * 256 + col_init * 2 + numThreads + 1];

            const uint pos_trasp = (M * gridDim_x) * (col_init + i) + M * blockIdx_x + (j & 2);
            res[pos_trasp * 2] = val0;
            res[pos_trasp * 2 + 1] = val1;
            res[(pos_trasp + 1) * 2] = val2;
            res[(pos_trasp + 1) * 2 + 1] = val3;
        }
    }
}

// =============================================================================
// Bit-Reversal Kernel
// =============================================================================

kernel void metal_Bit_Reverse(device uint32_t* dat [[buffer(0)]],
                               constant uint32_t& N [[buffer(1)]],
                               uint idx [[thread_position_in_grid]]) {
    uint32_t br_idx = metal_brev(idx) >> (32 - metal_clz(N) - 1);
    if (br_idx > idx) {
        uint32_t a = dat[idx];
        uint32_t b = dat[br_idx];
        dat[idx] = b;
        dat[br_idx] = a;
    }
}

kernel void metal_Bit_Reverse_u64(device uint64_t* dat [[buffer(0)]],
                                   constant uint32_t& N [[buffer(1)]],
                                   uint idx [[thread_position_in_grid]]) {
    uint32_t br_idx = metal_brev(idx) >> (32 - metal_clz(N) - 1);
    if (br_idx > idx) {
        uint64_t a = dat[idx];
        uint64_t b = dat[br_idx];
        dat[idx] = b;
        dat[br_idx] = a;
    }
}

// =============================================================================
// Key Switching Kernels
// =============================================================================

kernel void metal_ksk_dot(
    device uint64_t* kska [[buffer(0)]],
    device uint64_t* kskb [[buffer(1)]],
    device uint64_t* res0 [[buffer(2)]],
    device uint64_t* res1 [[buffer(3)]],
    device uint64_t* A [[buffer(4)]],
    constant uint64_t* primes [[buffer(5)]],
    constant uint32_t& primeid [[buffer(6)]],
    constant uint32_t& N [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]) {

    const int M = 4;
    const uint tid = lid.x;
    const uint blockIdx_x = gid.x;
    const uint j = tid << 1;

    uint64_t mod = primes[primeid];

    uint offset_2t = blockIdx_x * lid.x * M + tid;

    uint64_t ksk2_0 = kska[offset_2t * 2];
    uint64_t ksk2_1 = kska[offset_2t * 2 + 1];
    uint64_t in2_0 = metal_modmult_barrett(A[j], ksk2_0, mod);
    uint64_t in2_1 = metal_modmult_barrett(A[j + 1], ksk2_1, mod);
    res1[offset_2t * 2] = in2_0;
    res1[offset_2t * 2 + 1] = in2_1;

    uint64_t ksk1_0 = kskb[offset_2t * 2];
    uint64_t ksk1_1 = kskb[offset_2t * 2 + 1];
    uint64_t in1_0 = metal_modmult_barrett(A[j], ksk1_0, mod);
    uint64_t in1_1 = metal_modmult_barrett(A[j + 1], ksk1_1, mod);
    res0[offset_2t * 2] = in1_0;
    res0[offset_2t * 2 + 1] = in1_1;
}

// =============================================================================
// Utility Kernels
// =============================================================================

kernel void metal_copy_u64_ntt(device uint64_t* dst [[buffer(0)]],
                            constant uint64_t* src [[buffer(1)]],
                            constant uint32_t& size [[buffer(2)]],
                            uint idx [[thread_position_in_grid]]) {
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

kernel void metal_zero_u64_ntt(device uint64_t* data [[buffer(0)]],
                            constant uint32_t& size [[buffer(1)]],
                            uint idx [[thread_position_in_grid]]) {
    if (idx < size) {
        data[idx] = 0;
    }
}