//
// FIDESlib Metal Backend - Element-wise Batch Kernels (Metal Shading Language)
// Ported from CKKS/ElemenwiseBatchKernels.cu
//
// Uses flat buffer layout: all limbs in one buffer, indexed as limb*N + elem
//

#include <metal_stdlib>
#include "MetalMath.hpp"

using namespace metal;

#ifndef MAXP
#define MAXP 64
#endif

// =============================================================================
// Flat Buffer Element-wise Kernels
// Grid: (N/128, numLimbs) as uint2
// Buffer layout: flat[N * numLimbs] where elem at limb*N + elemIdx
// =============================================================================

// l = l * l1 + l2
kernel void metal_mult1Add2(device uint64_t* l [[buffer(0)]],
                             device uint64_t* l1 [[buffer(1)]],
                             device uint64_t* l2 [[buffer(2)]],
                             constant uint64_t* primes [[buffer(3)]],
                             constant uint32_t& N [[buffer(4)]],
                             constant uint32_t& numLimbs [[buffer(5)]],
                             constant int* primeid_flattened [[buffer(6)]],
                             constant uint32_t& primeid_init [[buffer(7)]],
                             uint2 idx [[thread_position_in_grid]]) {

    uint limbIdx = idx.y;
    uint elemIdx = idx.x;
    if (limbIdx >= numLimbs || elemIdx >= N/128) return;

    uint offset = limbIdx * N;
    uint primeid = primeid_flattened[primeid_init + limbIdx];
    uint64_t mod = primes[primeid];

    uint64_t val_l = l[offset + elemIdx];
    uint64_t val_l1 = l1[offset + elemIdx];
    uint64_t val_l2 = l2[offset + elemIdx];

    uint64_t res = metal_modmult_barrett(val_l, val_l1, mod);
    res = metal_modadd(res, val_l2, mod);
    l[offset + elemIdx] = res;
}

// l = l * l1 (simple multiply)
kernel void metal_Mult(device uint64_t* l [[buffer(0)]],
                         device uint64_t* l1 [[buffer(1)]],
                         device uint64_t* l2 [[buffer(2)]],
                         constant uint64_t* primes [[buffer(3)]],
                         constant uint32_t& N [[buffer(4)]],
                         constant uint32_t& numLimbs [[buffer(5)]],
                         constant int* primeid_flattened [[buffer(6)]],
                         constant uint32_t& primeid_init [[buffer(7)]],
                         uint2 idx [[thread_position_in_grid]]) {

    uint limbIdx = idx.y;
    uint elemIdx = idx.x;
    if (limbIdx >= numLimbs || elemIdx >= N/128) return;

    uint offset = limbIdx * N;
    uint primeid = primeid_flattened[primeid_init + limbIdx];
    uint64_t mod = primes[primeid];

    l[offset + elemIdx] = metal_modmult_barrett(l1[offset + elemIdx], l2[offset + elemIdx], mod);
}

// l = l1 * l1 (square)
kernel void metal_square(device uint64_t* l [[buffer(0)]],
                           device uint64_t* l1 [[buffer(1)]],
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

    uint64_t val = l1[offset + elemIdx];
    l[offset + elemIdx] = metal_modmult_barrett(val, val, mod);
}

// l = l + l1 (add)
kernel void metal_Add(device uint64_t* l [[buffer(0)]],
                        device uint64_t* l1 [[buffer(1)]],
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

    l[offset + elemIdx] = metal_modadd(l[offset + elemIdx], l1[offset + elemIdx], mod);
}

// l = l - l1 (subtract)
kernel void metal_Sub(device uint64_t* l [[buffer(0)]],
                        device uint64_t* l1 [[buffer(1)]],
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

    l[offset + elemIdx] = metal_modsub(l[offset + elemIdx], l1[offset + elemIdx], mod);
}

// l = binomial sampling: l = l * scalar + base
kernel void metal_binomialMult(device uint64_t* l [[buffer(0)]],
                                constant uint64_t& scalar [[buffer(1)]],
                                constant uint64_t& base [[buffer(2)]],
                                constant uint64_t* primes [[buffer(3)]],
                                constant uint32_t& N [[buffer(4)]],
                                constant uint32_t& numLimbs [[buffer(5)]],
                                constant int* primeid_flattened [[buffer(6)]],
                                constant uint32_t& primeid_init [[buffer(7)]],
                                uint2 idx [[thread_position_in_grid]]) {

    uint limbIdx = idx.y;
    uint elemIdx = idx.x;
    if (limbIdx >= numLimbs || elemIdx >= N/128) return;

    uint offset = limbIdx * N;
    uint primeid = primeid_flattened[primeid_init + limbIdx];
    uint64_t mod = primes[primeid];

    uint64_t val = l[offset + elemIdx];
    val = metal_modmult_barrett(val, scalar, mod);
    val = metal_modadd(val, base, mod);
    l[offset + elemIdx] = val;
}

// Scalar multiply: l = l * scalar
kernel void metal_scalarMult(device uint64_t* l [[buffer(0)]],
                               constant uint64_t& scalar [[buffer(1)]],
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

    l[offset + elemIdx] = metal_modmult_barrett(l[offset + elemIdx], scalar, mod);
}

// Scalar add: l = l + scalar
kernel void metal_scalarAdd(device uint64_t* l [[buffer(0)]],
                              constant uint64_t& scalar [[buffer(1)]],
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

    l[offset + elemIdx] = metal_modadd(l[offset + elemIdx], scalar, mod);
}

// element-wise negate
kernel void metal_negate(device uint64_t* l [[buffer(0)]],
                          constant uint64_t* primes [[buffer(1)]],
                          constant uint32_t& N [[buffer(2)]],
                          constant uint32_t& numLimbs [[buffer(3)]],
                          constant int* primeid_flattened [[buffer(4)]],
                          constant uint32_t& primeid_init [[buffer(5)]],
                          uint2 idx [[thread_position_in_grid]]) {

    uint limbIdx = idx.y;
    uint elemIdx = idx.x;
    if (limbIdx >= numLimbs || elemIdx >= N/128) return;

    uint offset = limbIdx * N;
    uint primeid = primeid_flattened[primeid_init + limbIdx];
    uint64_t mod = primes[primeid];

    l[offset + elemIdx] = metal_modsub(0, l[offset + elemIdx], mod);
}

// Copy: l = l1
kernel void metal_copy(device uint64_t* l [[buffer(0)]],
                         device uint64_t* l1 [[buffer(1)]],
                         constant uint32_t& N [[buffer(2)]],
                         constant uint32_t& numLimbs [[buffer(3)]],
                         uint2 idx [[thread_position_in_grid]]) {

    uint limbIdx = idx.y;
    uint elemIdx = idx.x;
    if (limbIdx >= numLimbs || elemIdx >= N/128) return;

    uint offset = limbIdx * N;
    l[offset + elemIdx] = l1[offset + elemIdx];
}

// Zero: l = 0
kernel void metal_zero(device uint64_t* l [[buffer(0)]],
                         constant uint32_t& N [[buffer(1)]],
                         constant uint32_t& numLimbs [[buffer(2)]],
                         uint2 idx [[thread_position_in_grid]]) {

    uint limbIdx = idx.y;
    uint elemIdx = idx.x;
    if (limbIdx >= numLimbs || elemIdx >= N/128) return;

    uint offset = limbIdx * N;
    l[offset + elemIdx] = 0;
}

// =============================================================================
// Dot Product (point-wise multiply and sum across limbs)
// res = sum over limbs: l0[elem] * l1[elem] mod primes[limb]
// =============================================================================

kernel void metal_dotProductPt(device uint64_t* res [[buffer(0)]],
                                 device uint64_t* l0 [[buffer(1)]],
                                 device uint64_t* l1 [[buffer(2)]],
                                 constant uint64_t* primes [[buffer(3)]],
                                 constant uint32_t& N [[buffer(4)]],
                                 constant uint32_t& numLimbs [[buffer(5)]],
                                 constant int* primeid_flattened [[buffer(6)]],
                                 uint elemIdx [[thread_position_in_grid]]) {

    if (elemIdx >= N/128) return;

    __uint128_t sum = 0;

    for (uint limbIdx = 0; limbIdx < numLimbs; limbIdx++) {
        uint offset = limbIdx * N;
        uint primeid = primeid_flattened[limbIdx];
        uint64_t mod = primes[primeid];

        uint64_t prod = metal_modmult_barrett(l0[offset + elemIdx], l1[offset + elemIdx], mod);
        sum += prod;
    }

    // Reduce final sum (approximate - use first prime)
    uint64_t mod = primes[primeid_flattened[0]];
    res[elemIdx] = (uint64_t)(sum % mod);
}

// =============================================================================
// Rescaling: multiply by delta/mod and reduce
// =============================================================================

kernel void metal_rescale(device uint64_t* l [[buffer(0)]],
                            constant uint64_t& delta [[buffer(1)]],
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

    l[offset + elemIdx] = metal_modmult_barrett(l[offset + elemIdx], delta, mod);
}

// =============================================================================
// Precomputation helper kernels (for Shoup's algorithm)
// =============================================================================

kernel void metal_precompute_shoup(device uint64_t* output [[buffer(0)]],
                                    constant uint64_t* input [[buffer(1)]],
                                    constant uint64_t* primes [[buffer(2)]],
                                    constant uint32_t& size [[buffer(3)]],
                                    uint idx [[thread_position_in_grid]]) {

    if (idx >= size) return;

    uint64_t mod = primes[idx];
    output[idx] = metal_shoup_precompute(input[idx], mod);
}