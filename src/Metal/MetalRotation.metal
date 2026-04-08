//
// FIDESlib Metal Backend - Rotation/Automorphism Kernels
// Ported from Rotation.cuh
// Uses flat buffer layout: elem at limbIdx * N + slotIdx
//

#include <metal_stdlib>
#include "MetalMath.hpp"

using namespace metal;

#ifndef MAXP
#define MAXP 64
#endif

// =============================================================================
// Autormophism Slot Calculation
// Computes the rotation index for a given slot
// =============================================================================

inline uint32_t automorph_slot_core(const int n_bits, const int index, const uint32_t slot) {
    uint32_t j = slot;

    // Bit reversal using metal_brev from MetalMath.hpp
    j = metal_brev(j) >> (32 - n_bits);

    uint32_t jTmp = (j << 1) + 1;
    uint32_t rotIndex = ((jTmp * index) & ((1 << (n_bits + 1)) - 1)) >> 1;

    // Bit reversal again
    rotIndex = metal_brev(rotIndex) >> (32 - n_bits);

    return rotIndex;
}

// =============================================================================
// Single Limb Automorphism Kernel
// Grid: N/256 threads
// Threadgroup: 256 threads
// =============================================================================

kernel void metal_automorph_u64(device uint64_t* a [[buffer(0)]],
                                 device uint64_t* a_rot [[buffer(1)]],
                                 constant int& N [[buffer(2)]],
                                 constant int& logN [[buffer(3)]],
                                 constant int& index [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]) {

    uint32_t j = gid;

    if (j >= (uint32_t)N) return;

    uint32_t rotIndex = automorph_slot_core(logN, index, j);

    a_rot[rotIndex] = a[j];
}

kernel void metal_automorph_u32(device uint32_t* a [[buffer(0)]],
                                 device uint32_t* a_rot [[buffer(1)]],
                                 constant int& N [[buffer(2)]],
                                 constant int& logN [[buffer(3)]],
                                 constant int& index [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]) {

    uint32_t j = gid;

    if (j >= (uint32_t)N) return;

    uint32_t rotIndex = automorph_slot_core(logN, index, j);

    a_rot[rotIndex] = a[j];
}

// =============================================================================
// Multi-Limb Automorphism Kernel
// Grid: (N/256, numLimbs) as uint2
// Buffer layout: flat[N * numLimbs] where elem at limbIdx * N + slotIdx
// =============================================================================

kernel void metal_automorph_flat(device uint64_t* a [[buffer(0)]],
                                  device uint64_t* a_rot [[buffer(1)]],
                                  constant int& k [[buffer(2)]],        // rotation index
                                  constant int& N [[buffer(3)]],
                                  constant int& logN [[buffer(4)]],
                                  constant int& numLimbs [[buffer(5)]],
                                  uint2 gid [[thread_position_in_grid]]) {

    int limbIdx = gid.y;
    if (limbIdx >= numLimbs) return;

    uint32_t slotIdx = gid.x;
    if (slotIdx >= (uint32_t)(N / 256)) return;

    uint32_t j = slotIdx * 256;
    if (j >= (uint32_t)N) return;

    uint32_t rotIndex = automorph_slot_core(logN, k, j);

    uint offset = limbIdx * N;
    a_rot[offset + rotIndex] = a[offset + j];
}

// =============================================================================
// In-place Automorphism
// =============================================================================

kernel void metal_automorph_inplace_u64(device uint64_t* data [[buffer(0)]],
                                          constant int& N [[buffer(1)]],
                                          constant int& logN [[buffer(2)]],
                                          constant int& index [[buffer(3)]],
                                          uint gid [[thread_position_in_grid]]) {

    uint32_t j = gid;

    if (j >= (uint32_t)N) return;

    uint32_t rotIndex = automorph_slot_core(logN, index, j);

    uint64_t val = data[j];
    data[rotIndex] = val;
}

// =============================================================================
// Automorphism with Teleswap (optimized for CKKS)
// =============================================================================

kernel void metal_automorph_teleswap(device uint64_t* a [[buffer(0)]],
                                      device uint64_t* a_rot [[buffer(1)]],
                                      constant int& N [[buffer(2)]],
                                      constant int& logN [[buffer(3)]],
                                      constant int& index [[buffer(4)]],
                                      uint gid [[thread_position_in_grid]]) {

    uint32_t j = gid;

    if (j >= (uint32_t)N) return;

    uint32_t rotIndex = automorph_slot_core(logN, index, j);

    a_rot[rotIndex] = a[j];
}

// =============================================================================
// Multi-Limb Teleswap
// =============================================================================

kernel void metal_automorph_teleswap_flat(device uint64_t* a [[buffer(0)]],
                                           device uint64_t* a_rot [[buffer(1)]],
                                           constant int& N [[buffer(2)]],
                                           constant int& logN [[buffer(3)]],
                                           constant int& index [[buffer(4)]],
                                           constant int& numLimbs [[buffer(5)]],
                                           uint2 gid [[thread_position_in_grid]]) {

    int limbIdx = gid.y;
    if (limbIdx >= numLimbs) return;

    uint32_t slotIdx = gid.x;
    if (slotIdx >= (uint32_t)(N / 256)) return;

    uint32_t j = slotIdx * 256;
    if (j >= (uint32_t)N) return;

    uint32_t rotIndex = automorph_slot_core(logN, index, j);

    uint offset = limbIdx * N;
    a_rot[offset + rotIndex] = a[offset + j];
}

// =============================================================================
// Batch Automorphism Kernels
// Grid: (N/256, numLimbs) as uint2, processes multiple ciphertexts
// =============================================================================

kernel void metal_automorph_batch_flat(device uint64_t* a [[buffer(0)]],
                                        device uint64_t* a_rot [[buffer(1)]],
                                        constant int& N [[buffer(2)]],
                                        constant int& logN [[buffer(3)]],
                                        constant int& index [[buffer(4)]],
                                        constant int& numLimbs [[buffer(5)]],
                                        constant int& batchSize [[buffer(6)]],
                                        uint2 gid [[thread_position_in_grid]]) {

    int limbIdx = gid.y;
    int slotBatchIdx = gid.x;

    int slotsPerLimb = N / 256;
    if (slotBatchIdx >= slotsPerLimb * batchSize) return;

    int slotIdx = slotBatchIdx % slotsPerLimb;
    int batchIdx = slotBatchIdx / slotsPerLimb;

    uint32_t j = slotIdx * 256 + (gid.x % slotsPerLimb);

    if (j >= (uint32_t)N) return;

    uint32_t rotIndex = automorph_slot_core(logN, index, j);

    uint offset = (batchIdx * numLimbs + limbIdx) * N;
    a_rot[offset + rotIndex] = a[offset + j];
}

// =============================================================================
// Automorphism with Mask
// =============================================================================

kernel void metal_automorph_mask(device uint64_t* a [[buffer(0)]],
                                  device uint64_t* a_rot [[buffer(1)]],
                                  device uint64_t* mask [[buffer(2)]],
                                  constant int& N [[buffer(3)]],
                                  constant int& logN [[buffer(4)]],
                                  constant int& index [[buffer(5)]],
                                  uint gid [[thread_position_in_grid]]) {

    uint32_t j = gid;

    if (j >= (uint32_t)N) return;

    uint32_t rotIndex = automorph_slot_core(logN, index, j);

    uint64_t val = a[j];
    uint64_t m = mask[j];

    a_rot[rotIndex] = val & m;
}

// =============================================================================
// Multi-Limb Mask Automorphism
// =============================================================================

kernel void metal_automorph_mask_flat(device uint64_t* a [[buffer(0)]],
                                       device uint64_t* a_rot [[buffer(1)]],
                                       device uint64_t* mask [[buffer(2)]],
                                       constant int& N [[buffer(3)]],
                                       constant int& logN [[buffer(4)]],
                                       constant int& index [[buffer(5)]],
                                       constant int& numLimbs [[buffer(6)]],
                                       uint2 gid [[thread_position_in_grid]]) {

    int limbIdx = gid.y;
    if (limbIdx >= numLimbs) return;

    uint32_t slotIdx = gid.x;
    if (slotIdx >= (uint32_t)(N / 256)) return;

    uint32_t j = slotIdx * 256;
    if (j >= (uint32_t)N) return;

    uint32_t rotIndex = automorph_slot_core(logN, index, j);

    uint offset = limbIdx * N;
    uint64_t val = a[offset + j];
    uint64_t m = mask[limbIdx * N + j];  // Mask also flat

    a_rot[offset + rotIndex] = val & m;
}

// =============================================================================
// Conjugate Kernel (special case of automorphism with index = N/2 + 1)
// =============================================================================

kernel void metal_conjugate_u64(device uint64_t* a [[buffer(0)]],
                                device uint64_t* a_conj [[buffer(1)]],
                                constant int& N [[buffer(2)]],
                                constant int& logN [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {

    uint32_t j = gid;

    if (j >= (uint32_t)N) return;

    int conjugateIndex = (N / 2) + 1;
    uint32_t rotIndex = automorph_slot_core(logN, conjugateIndex, j);

    a_conj[rotIndex] = a[j];
}

// =============================================================================
// Multi-Limb Conjugate
// =============================================================================

kernel void metal_conjugate_flat(device uint64_t* a [[buffer(0)]],
                                  device uint64_t* a_conj [[buffer(1)]],
                                  constant int& N [[buffer(2)]],
                                  constant int& logN [[buffer(3)]],
                                  constant int& numLimbs [[buffer(4)]],
                                  uint2 gid [[thread_position_in_grid]]) {

    int limbIdx = gid.y;
    if (limbIdx >= numLimbs) return;

    uint32_t slotIdx = gid.x;
    if (slotIdx >= (uint32_t)(N / 256)) return;

    uint32_t j = slotIdx * 256;
    if (j >= (uint32_t)N) return;

    int conjugateIndex = (N / 2) + 1;
    uint32_t rotIndex = automorph_slot_core(logN, conjugateIndex, j);

    uint offset = limbIdx * N;
    a_conj[offset + rotIndex] = a[offset + j];
}

// =============================================================================
// Rotate and Add (in-place rotation and accumulation)
// =============================================================================

kernel void metal_rotadd_u64(device uint64_t* data [[buffer(0)]],
                              device uint64_t* to_add [[buffer(1)]],
                              constant int& N [[buffer(2)]],
                              constant int& logN [[buffer(3)]],
                              constant int& index [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {

    uint32_t j = gid;

    if (j >= (uint32_t)N) return;

    uint32_t rotIndex = automorph_slot_core(logN, index, j);

    uint64_t orig = data[j];
    uint64_t add = to_add[rotIndex];

    data[j] = metal_modadd(orig, add, 0);  // mod will be passed separately
}

// Multi-limb rotate and add
kernel void metal_rotadd_flat(device uint64_t* data [[buffer(0)]],
                                device uint64_t* to_add [[buffer(1)]],
                                constant uint64_t* primes [[buffer(2)]],
                                constant int& N [[buffer(3)]],
                                constant int& logN [[buffer(4)]],
                                constant int& index [[buffer(5)]],
                                constant int& numLimbs [[buffer(6)]],
                                constant int* primeid_flattened [[buffer(7)]],
                                uint2 gid [[thread_position_in_grid]]) {

    int limbIdx = gid.y;
    if (limbIdx >= numLimbs) return;

    uint32_t slotIdx = gid.x;
    if (slotIdx >= (uint32_t)(N / 256)) return;

    uint32_t j = slotIdx * 256;
    if (j >= (uint32_t)N) return;

    uint32_t rotIndex = automorph_slot_core(logN, index, j);

    uint offset = limbIdx * N;
    uint primeid = primeid_flattened[limbIdx];
    uint64_t mod = primes[primeid];

    uint64_t orig = data[offset + j];
    uint64_t add = to_add[offset + rotIndex];

    data[offset + j] = metal_modadd(orig, add, mod);
}