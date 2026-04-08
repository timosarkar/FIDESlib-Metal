//
// FIDESlib Metal Backend - Modular Arithmetic Example
// Demonstrates use of Metal math helpers for CKKS operations
//

#include <iostream>
#include <vector>
#include <cstdint>
#include <iomanip>

// =============================================================================
// Metal Math Helper Implementations (CPU reference)
// These match the implementations in MetalMath.hpp
// =============================================================================

// Unsigned multiply high - upper 64 bits of 64x64 product
uint64_t umul64hi(uint64_t a, uint64_t b) {
    return (uint64_t)((__uint128_t)a * (__uint128_t)b >> 64);
}

// Count leading zeros (32-bit)
uint32_t clz(uint32_t x) {
    return x == 0 ? 32 : __builtin_clz(x);
}

// Count leading zeros (64-bit)
uint32_t clzll(uint64_t x) {
    return x == 0 ? 64 : __builtin_clzll(x);
}

// Population count
uint32_t popcount(uint32_t x) {
    return __builtin_popcount(x);
}

// Bit reversal (32-bit)
uint32_t brev32(uint32_t x) {
    x = ((x & 0xAAAAAAAAU) >> 1) | ((x & 0x55555555U) << 1);
    x = ((x & 0xCCCCCCCCU) >> 2) | ((x & 0x33333333U) << 2);
    x = ((x & 0xF0F0F0F0U) >> 4) | ((x & 0x0F0F0F0FU) << 4);
    x = ((x & 0xFF00FF00U) >> 8) | ((x & 0x00FF00FFU) << 8);
    x = (x >> 16) | (x << 16);
    return x;
}

// Bit reversal (64-bit)
uint64_t brev64(uint64_t x) {
    x = ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1) | ((x & 0x5555555555555555ULL) << 1);
    x = ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2) | ((x & 0x3333333333333333ULL) << 2);
    x = ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4);
    x = ((x & 0xFF00FF00FF00FF00ULL) >> 8) | ((x & 0x00FF00FF00FF00FFULL) << 8);
    x = (x >> 16) | (x << 16);
    return (x >> 32) | (x << 32);
}

// Modular addition
uint64_t mod_add(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t result = a + b;
    if (result >= mod) result -= mod;
    return result;
}

// Modular subtraction
uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t result = a - b;
    if (result > a) result += mod;
    return result;
}

// Barrett reduction precomputation: mu = floor(2^64 / mod)
uint64_t barrett_mu(uint64_t mod) {
    return UINT64_MAX / mod;  // floor(2^64 / mod)
}

// Barrett modular multiplication
uint64_t barrett_mul(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t mu = barrett_mu(mod);
    uint64_t q = umul64hi(a, mu);
    __uint128_t t = (__uint128_t)a * (__uint128_t)b;
    uint64_t r = (uint64_t)t - q * mod;
    if (r >= mod) r -= mod;
    return r;
}

// Shoup precomputation: b * 2^64 / mod
uint64_t shoup_precompute(uint64_t b, uint64_t mod) {
    __uint128_t t = (__uint128_t)1 << 64;
    t = (t + mod - 1) / mod;
    t = t * b;
    return (uint64_t)(t >> 64);
}

// Shoup modular multiplication
uint64_t shoup_mul(uint64_t a, uint64_t b, uint64_t mod, uint64_t shoup_b) {
    uint64_t q = umul64hi(a, shoup_b);
    uint64_t r = a * b - q * mod;
    if (r >= mod) r -= mod;
    return r;
}

// =============================================================================
// Sample 1: Basic Modular Arithmetic
// =============================================================================

void sample_basic_modular() {
    std::cout << "\n=== Sample 1: Basic Modular Arithmetic ===" << std::endl;

    uint64_t mod = 13;

    std::cout << "Modulus: " << mod << std::endl;

    // Modular addition
    std::cout << "\nModular Addition:" << std::endl;
    std::cout << "  (" << 3 << " + " << 7 << ") mod " << mod << " = "
              << mod_add(3, 7, mod) << " (expected 10 mod 13 = 10)" << std::endl;
    std::cout << "  (" << 10 << " + " << 5 << ") mod " << mod << " = "
              << mod_add(10, 5, mod) << " (expected 15 mod 13 = 2)" << std::endl;
    std::cout << "  (" << 12 << " + " << 1 << ") mod " << mod << " = "
              << mod_add(12, 1, mod) << " (expected 13 mod 13 = 0)" << std::endl;

    // Modular subtraction
    std::cout << "\nModular Subtraction:" << std::endl;
    std::cout << "  (" << 3 << " - " << 7 << ") mod " << mod << " = "
              << mod_sub(3, 7, mod) << " (expected -4 mod 13 = 9)" << std::endl;
    std::cout << "  (" << 7 << " - " << 3 << ") mod " << mod << " = "
              << mod_sub(7, 3, mod) << " (expected 4)" << std::endl;
    std::cout << "  (" << 0 << " - " << 1 << ") mod " << mod << " = "
              << mod_sub(0, 1, mod) << " (expected -1 mod 13 = 12)" << std::endl;
}

// =============================================================================
// Sample 2: Barrett vs Naive Multiplication
// =============================================================================

void sample_barrett() {
    std::cout << "\n=== Sample 2: Barrett Modular Multiplication ===" << std::endl;

    // Large 64-bit modulus (CKKS-sized)
    uint64_t mod = 1073807353;  // ~30-bit prime

    std::cout << "Modulus: " << mod << " (30-bit prime)" << std::endl;

    uint64_t a = 123456789;
    uint64_t b = 987654321;
    uint64_t expected = (uint64_t)((__uint128_t)a * b % mod);

    uint64_t barrett_result = barrett_mul(a, b, mod);

    std::cout << "\nComputing " << a << " * " << b << " mod " << mod << std::endl;
    std::cout << "  Expected: " << expected << std::endl;
    std::cout << "  Barrett:  " << barrett_result << std::endl;
    std::cout << "  Match: " << (barrett_result == expected ? "YES" : "NO") << std::endl;

    // Demonstrate Barrett precomputation
    uint64_t mu = barrett_mu(mod);
    std::cout << "\nBarrett precomputation:" << std::endl;
    std::cout << "  mu = floor(2^64 / " << mod << ") = " << mu << std::endl;
    std::cout << "  Verification: 2^64 / " << mod << " = " << (double)(UINT64_MAX + 1) / mod << std::endl;

    // Test multiple values
    std::cout << "\nMultiple Barrett multiplications:" << std::endl;
    std::vector<std::pair<uint64_t, uint64_t>> tests = {
        {1, 1},
        {123, 456},
        {mod - 1, mod - 1},
        {a, b},
        {mod / 2, 2}
    };

    for (auto [x, y] : tests) {
        uint64_t result = barrett_mul(x, y, mod);
        uint64_t expected = (uint64_t)((__uint128_t)x * y % mod);
        std::cout << "  " << x << " * " << y << " mod " << mod << " = " << result
                  << " (expected " << expected << ") "
                  << (result == expected ? "[OK]" : "[FAIL]") << std::endl;
    }
}

// =============================================================================
// Sample 3: Shoup Multiplication (Fast with Precomputation)
// =============================================================================

void sample_shoup() {
    std::cout << "\n=== Sample 3: Shoup Modular Multiplication ===" << std::endl;

    uint64_t mod = 1073807353;

    std::cout << "Modulus: " << mod << std::endl;

    // Precompute for b = 7
    uint64_t b = 7;
    uint64_t shoup_b = shoup_precompute(b, mod);

    std::cout << "\nShoup precomputation for b = " << b << ":" << std::endl;
    std::cout << "  shoup_b = floor(b * 2^64 / " << mod << ") = " << shoup_b << std::endl;

    // Now multiply by various a values using Shoup
    std::cout << "\nShoup multiplication with precomputed b=" << b << ":" << std::endl;
    std::cout << "  a = 3:  " << shoup_mul(3, b, mod, shoup_b)
              << " (expected " << (3ULL * b % mod) << ")" << std::endl;
    std::cout << "  a = 13: " << shoup_mul(13, b, mod, shoup_b)
              << " (expected " << (13ULL * b % mod) << ")" << std::endl;
    std::cout << "  a = 99: " << shoup_mul(99, b, mod, shoup_b)
              << " (expected " << (99ULL * b % mod) << ")" << std::endl;

    // Compare speed: Shoup vs Barrett
    std::cout << "\nPerformance comparison (precomputation amortized):" << std::endl;
    std::cout << "  Barrett: ~2 multiplications per op (a*mu >> 64, a*b, q*mod)" << std::endl;
    std::cout << "  Shoup:   ~1 multiplication per op (a*shoup_b >> 64, a*b, q*mod)" << std::endl;
    std::cout << "  Shoup is faster when b is reused across multiple multiplications" << std::endl;

    // Verify Shoup matches Barrett
    std::cout << "\nVerification: Shoup == Barrett:" << std::endl;
    int match_count = 0;
    for (uint64_t a = 0; a < 100; a++) {
        uint64_t barrett = barrett_mul(a, b, mod);
        uint64_t shoup = shoup_mul(a, b, mod, shoup_b);
        if (barrett == shoup) match_count++;
    }
    std::cout << "  Matched " << match_count << "/100 cases" << std::endl;
}

// =============================================================================
// Sample 4: Bit Operations (for NTT/Memory Access)
// =============================================================================

void sample_bit_operations() {
    std::cout << "\n=== Sample 4: Bit Operations (for NTT) ===" << std::endl;

    // Bit reversal for NTT
    std::cout << "Bit Reversal (for FFT/NTT):" << std::endl;
    uint32_t test32 = 0x12345678;
    uint32_t rev32 = brev32(test32);
    std::cout << "  0x" << std::hex << test32 << " -> 0x" << rev32 << std::dec << std::endl;

    uint64_t test64 = 0x123456789ABCDEF0ULL;
    uint64_t rev64 = brev64(test64);
    std::cout << "  0x" << std::hex << test64 << " -> 0x" << rev64 << std::dec << std::endl;

    // Count leading zeros (for normalization)
    std::cout << "\nCount Leading Zeros:" << std::endl;
    std::cout << "  clz(0x80000000) = " << clz(0x80000000) << " (MSB set)" << std::endl;
    std::cout << "  clz(0x00000001) = " << clz(0x00000001) << " (LSB set)" << std::endl;
    std::cout << "  clz(0xFFFFFFFF) = " << clz(0xFFFFFFFF) << " (all ones)" << std::endl;
    std::cout << "  clz(0) = " << clz(0) << " (zero)" << std::endl;

    std::cout << "  clzll(0x8000000000000000) = " << clzll(0x8000000000000000ULL) << " (64-bit MSB)" << std::endl;
    std::cout << "  clzll(0x0000000000000001) = " << clzll(0x0000000000000001ULL) << " (64-bit LSB)" << std::endl;

    // Population count (for various algorithms)
    std::cout << "\nPopulation Count:" << std::endl;
    std::cout << "  popcount(0x00000000) = " << popcount(0x00000000) << std::endl;
    std::cout << "  popcount(0xFFFFFFFF) = " << popcount(0xFFFFFFFF) << std::endl;
    std::cout << "  popcount(0xAAAAAAAA) = " << popcount(0xAAAAAAAA) << " (alternating bits)" << std::endl;
    std::cout << "  popcount(0x55555555) = " << popcount(0x55555555) << " (alternating bits)" << std::endl;
    std::cout << "  popcount(0x12345678) = " << popcount(0x12345678) << std::endl;
}

// =============================================================================
// Sample 5: High-Precision Multiplication
// =============================================================================

void sample_high_precision() {
    std::cout << "\n=== Sample 5: High-Precision Multiplication ===" << std::endl;

    // umul64hi returns upper 64 bits of 128-bit product
    std::cout << "Upper 64 bits of 64x64 multiplication:" << std::endl;

    uint64_t a = 0x123456789ABCDEF0ULL;
    uint64_t b = 0xFEDCBA9876543210ULL;

    uint64_t hi = umul64hi(a, b);
    __uint128_t full = (__uint128_t)a * (__uint128_t)b;
    uint64_t lo = (uint64_t)full;

    std::cout << "  a = 0x" << std::hex << a << std::dec << std::endl;
    std::cout << "  b = 0x" << std::hex << b << std::dec << std::endl;
    std::cout << "  a * b (128-bit) = 0x" << std::hex << (uint64_t)(full >> 64)
              << "_" << std::setw(16) << std::setfill('0') << (uint64_t)full << std::dec << std::endl;
    std::cout << "  umul64hi(a, b) = 0x" << std::hex << hi << std::dec << std::endl;
    std::cout << "  low 64 bits    = 0x" << std::hex << lo << std::dec << std::endl;

    // Use case: computing floor(a * b / 2^64) in Barrett reduction
    std::cout << "\nUse in Barrett reduction:" << std::endl;
    uint64_t mod = 1073807353;
    uint64_t mu = barrett_mu(mod);

    // q = floor(a * mu / 2^64)
    uint64_t q = umul64hi(a, mu);
    std::cout << "  a = " << a << std::endl;
    std::cout << "  mu = " << mu << std::endl;
    std::cout << "  q = floor(a * mu / 2^64) = " << q << std::endl;
}

// =============================================================================
// Sample 6: CKKS RNS Arithmetic
// =============================================================================

void sample_ckks_rns() {
    std::cout << "\n=== Sample 6: CKKS RNS Arithmetic ===" << std::endl;

    // CKKS uses RNS (Residue Number System) with multiple moduli
    // Each modulus operates independently

    std::vector<uint64_t> moduli = {
        1073807353,   // ~30 bits
        1073799977,   // ~30 bits
        1073741827,   // ~30 bits
        1073738933    // ~30 bits
    };

    std::cout << "CKKS RNS moduli:" << std::endl;
    for (size_t i = 0; i < moduli.size(); i++) {
        std::cout << "  Limb " << i << ": " << moduli[i] << std::endl;
    }

    // Input values (same logical value, different RNS representations)
    std::vector<uint64_t> a = {123456789, 123456789, 123456789, 123456789};
    std::vector<uint64_t> b = {987654321, 987654321, 987654321, 987654321};

    std::cout << "\nPerforming RNS multiplication (a * b mod each limb):" << std::endl;
    std::vector<uint64_t> c(moduli.size());

    for (size_t i = 0; i < moduli.size(); i++) {
        c[i] = barrett_mul(a[i], b[i], moduli[i]);
        uint64_t expected = (uint64_t)((__uint128_t)a[i] * b[i] % moduli[i]);
        std::cout << "  Limb " << i << ": " << a[i] << " * " << b[i] << " mod " << moduli[i]
                  << " = " << c[i] << " (expected " << expected << ") "
                  << (c[i] == expected ? "[OK]" : "[FAIL]") << std::endl;
    }

    // Demonstrate that operations are independent per limb
    std::cout << "\nIndependence demonstration:" << std::endl;
    std::cout << "  Each limb can be multiplied independently" << std::endl;
    std::cout << "  No cross-limb communication needed for multiplication" << std::endl;
    std::cout << "  This is the advantage of RNS representation" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "FIDESlib Metal - Modular Arithmetic Examples" << std::endl;
    std::cout << "========================================" << std::endl;

    sample_basic_modular();
    sample_barrett();
    sample_shoup();
    sample_bit_operations();
    sample_high_precision();
    sample_ckks_rns();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Modular arithmetic examples completed!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
