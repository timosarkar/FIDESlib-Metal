//
// FIDESlib Metal Backend - NTT Example
// Demonstrates Number Theoretic Transform concepts on Metal GPU
//

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <complex>
#include <iomanip>

using Complex = std::complex<double>;

// =============================================================================
// CKKS/NTT Parameters
// =============================================================================

struct NTTParams {
    uint32_t N;          // Polynomial degree (1024, 2048, 4096, 8192)
    uint32_t prime;      // Modulus (must be prime, prime = 1 mod 2N)
    uint32_t root;       // Primitive root of order 2N
    uint32_t invN;       // Modular inverse of N
    uint32_t invRoot;   // Modular inverse of root
};

const NTTParams PARAMS_1024 = {
    .N = 1024,
    .prime = 1073807353,   // 2^20 * 1023 + 1
    .root = 31,            // Primitive root of order 2048
    .invN = 0,
    .invRoot = 0
};

// =============================================================================
// Modular Arithmetic
// =============================================================================

uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) result = (uint64_t)((__uint128_t)result * base % mod);
        base = (uint64_t)((__uint128_t)base * base % mod);
        exp >>= 1;
    }
    return result;
}

uint64_t mod_inv(uint64_t a, uint64_t mod) {
    int64_t t = 0, new_t = 1;
    uint64_t r = mod, new_r = a;
    while (new_r != 0) {
        uint64_t q = r / new_r;
        uint64_t tmp = t - (int64_t)q * new_t;
        t = new_t;
        new_t = tmp;
        tmp = r - q * new_r;
        r = new_r;
        new_r = tmp;
    }
    if (r > 1) return 0;
    if (t < 0) t += mod;
    return t;
}

uint64_t mod_add(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t result = a + b;
    if (result >= mod) result -= mod;
    return result;
}

uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t result = a - b;
    if (result > a) result += mod;
    return result;
}

uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t mod) {
    return (uint64_t)((__uint128_t)a * b % mod);
}

// =============================================================================
// Flat Buffer Layout for Multi-Limb Representation
// =============================================================================

class FlatBuffer {
public:
    FlatBuffer(size_t numLimbs, size_t N) : numLimbs_(numLimbs), N_(N) {
        data_.resize(numLimbs * N);
    }

    uint32_t& at(size_t limb, size_t idx) {
        return data_[limb * N_ + idx];
    }

    const uint32_t& at(size_t limb, size_t idx) const {
        return data_[limb * N_ + idx];
    }

    uint32_t* limb_ptr(size_t limb) {
        return &data_[limb * N_];
    }

    size_t size() const { return data_.size(); }
    size_t numLimbs() const { return numLimbs_; }
    size_t N() const { return N_; }

private:
    size_t numLimbs_;
    size_t N_;
    std::vector<uint32_t> data_;
};

// =============================================================================
// Sample 1: Flat Buffer Layout
// =============================================================================

void sample_flat_buffer_layout() {
    std::cout << "\n=== Sample 1: Flat Buffer Layout ===" << std::endl;

    const size_t NUM_LIMBS = 4;
    const size_t N = 1024;

    std::cout << "CKKS uses RNS (Residue Number System) with multiple moduli" << std::endl;
    std::cout << "Each limb operates independently with a different prime modulus" << std::endl;

    std::cout << "\nParameters:" << std::endl;
    std::cout << "  Polynomial degree N = " << N << std::endl;
    std::cout << "  Number of limbs = " << NUM_LIMBS << std::endl;
    std::cout << "  Total coefficients = " << (NUM_LIMBS * N) << std::endl;

    std::vector<uint32_t> primes = {
        1073807353,   // Limb 0: ~30 bits
        1073799977,   // Limb 1: ~30 bits
        1073741827,   // Limb 2: ~30 bits
        1073738933    // Limb 3: ~30 bits
    };

    FlatBuffer poly(NUM_LIMBS, N);

    // Initialize with sample values
    for (size_t limb = 0; limb < NUM_LIMBS; limb++) {
        for (size_t i = 0; i < N; i++) {
            poly.at(limb, i) = (uint32_t)((i + limb * 100) % primes[limb]);
        }
    }

    std::cout << "\nFlat buffer memory layout:" << std::endl;
    std::cout << "  Indexing: element at (limb, slot) -> flat[limb * N + slot]" << std::endl;

    for (size_t limb = 0; limb < NUM_LIMBS; limb++) {
        size_t base = limb * N;
        std::cout << "  Limb " << limb << " (prime=" << primes[limb] << "):" << std::endl;
        std::cout << "    Memory range: [" << base << ", " << (base + N - 1) << "]" << std::endl;
        std::cout << "    Sample values:" << std::endl;
        for (size_t i = 0; i < 4; i++) {
            std::cout << "      poly[" << limb << "][" << i << "] = " << poly.at(limb, i)
                      << " (flat[" << (base + i) << "])" << std::endl;
        }
    }

    std::cout << "\nGPU Memory Access Pattern:" << std::endl;
    std::cout << "  Thread " << 0 << " accesses flat[" << 0 << "], flat[" << N << "], flat[" << (2*N) << "], flat[" << (3*N) << "]" << std::endl;
    std::cout << "  Thread " << 1 << " accesses flat[" << 1 << "], flat[" << (N+1) << "], flat[" << (2*N+1) << "], flat[" << (3*N+1) << "]" << std::endl;
    std::cout << "  Thread " << 5 << " accesses flat[" << 5 << "], flat[" << (N+5) << "], flat[" << (2*N+5) << "], flat[" << (3*N+5) << "]" << std::endl;
}

// =============================================================================
// Sample 2: NTT Butterfly Operation
// =============================================================================

void sample_ntt_butterfly() {
    std::cout << "\n=== Sample 2: NTT Butterfly Operation ===" << std::endl;

    uint32_t prime = 1073807353;

    std::cout << "The butterfly is the fundamental operation in NTT:" << std::endl;
    std::cout << "  Cooley-Tukey (decimation-in-time):" << std::endl;
    std::cout << "    u = a + b" << std::endl;
    std::cout << "    v = (a - b) * twiddle" << std::endl;
    std::cout << "  Gentleman-Sande (decimation-in-frequency):" << std::endl;
    std::cout << "    u = a + b * twiddle" << std::endl;
    std::cout << "    v = (a - b) * twiddle" << std::endl;

    std::cout << "\nExample butterfly (twiddle = 1):" << std::endl;
    uint32_t a = 123, b = 456;
    uint32_t u = mod_add(a, b, prime);
    uint32_t v = mod_mul(mod_sub(a, b, prime), 1, prime);
    std::cout << "  Input: a=" << a << ", b=" << b << std::endl;
    std::cout << "  Output: u=" << u << " (a+b), v=" << v << " (a-b)" << std::endl;

    std::cout << "\nButterfly stages for N=8:" << std::endl;
    std::cout << "  Stage 1: butterflies of size 2 (stride 1)" << std::endl;
    std::cout << "  Stage 2: butterflies of size 4 (stride 2)" << std::endl;
    std::cout << "  Stage 3: butterflies of size 8 (stride 4)" << std::endl;

    // Show butterfly pairs
    std::vector<uint32_t> data = {0, 1, 2, 3, 4, 5, 6, 7};
    std::cout << "\n  Initial: [0, 1, 2, 3, 4, 5, 6, 7]" << std::endl;

    // Stage 1
    std::cout << "  After stage 1 (twiddles=1):" << std::endl;
    for (size_t i = 0; i < 4; i++) {
        uint32_t u = mod_add(data[2*i], data[2*i+1], prime);
        uint32_t v = mod_sub(data[2*i], data[2*i+1], prime);
        data[2*i] = u;
        data[2*i+1] = v;
    }
    std::cout << "    [" << data[0] << ", " << data[1] << ", "
              << data[2] << ", " << data[3] << ", "
              << data[4] << ", " << data[5] << ", "
              << data[6] << ", " << data[7] << "]" << std::endl;
}

// =============================================================================
// Sample 3: Bit Reversal Permutation
// =============================================================================

void sample_bit_reversal() {
    std::cout << "\n=== Sample 3: Bit Reversal Permutation ===" << std::endl;

    const uint32_t N = 16;  // 4-bit indices

    std::cout << "NTT requires bit-reversal permutation before butterflies:" << std::endl;
    std::cout << "  Index bits are reversed" << std::endl;
    std::cout << "  For N=" << N << " (4 bits), indices 0-15 are rearranged" << std::endl;

    uint32_t n_bits = 0;
    uint32_t temp = N;
    while (temp > 1) { temp >>= 1; n_bits++; }

    std::cout << "\nBit reversal for N=" << N << " (" << n_bits << "-bit indices):" << std::endl;
    std::cout << "  Index -> Bit-reversed" << std::endl;

    for (uint32_t i = 0; i < 16; i++) {
        // Reverse n_bits bits
        uint32_t rev = 0;
        for (uint32_t b = 0; b < n_bits; b++) {
            rev = (rev << 1) | ((i >> b) & 1);
        }
        if (i <= 15) {
            std::cout << "  " << std::setw(2) << i << " -> " << std::setw(2) << rev;
            if (i < 8) std::cout << "   ";
            else std::cout << "  ";
            // Show binary
            for (uint32_t b = n_bits; b > 0; b--) {
                std::cout << ((i >> (b-1)) & 1);
            }
            std::cout << " -> ";
            for (uint32_t b = n_bits; b > 0; b--) {
                std::cout << ((rev >> (b-1)) & 1);
            }
            std::cout << std::endl;
        }
    }

    std::cout << "\nIn-place bit reversal algorithm:" << std::endl;
    std::cout << "  for i = 1 to N-1:" << std::endl;
    std::cout << "    j = reverse_bits(i, n_bits)" << std::endl;
    std::cout << "    if i < j: swap(a[i], a[j])" << std::endl;
}

// =============================================================================
// Sample 4: Twiddle Factor Powers
// =============================================================================

void sample_twiddle_factors() {
    std::cout << "\n=== Sample 4: Twiddle Factor Powers ===" << std::endl;

    uint32_t N = 8;
    uint32_t prime = 1073807353;
    uint32_t root = 31;  // Primitive root for order 2N

    std::cout << "Twiddle factors are powers of primitive root:" << std::endl;
    std::cout << "  root = " << root << std::endl;
    std::cout << "  order = 2N = " << (2*N) << std::endl;

    // Compute twiddle = root^((order)/N) = root^(2N/N) = root^2
    uint32_t twiddle_base = mod_pow(root, (prime - 1) / (2 * N), prime);
    std::cout << "  twiddle_base = root^((p-1)/2N) = " << twiddle_base << std::endl;

    std::cout << "\nTwiddle factors for N=" << N << ":" << std::endl;
    std::cout << "  W^k = twiddle_base^k for k=0.." << (N-1) << std::endl;
    for (uint32_t k = 0; k < N; k++) {
        uint32_t w = mod_pow(twiddle_base, k, prime);
        std::cout << "  W^" << k << " = " << w << std::endl;
    }

    std::cout << "\nVerification: W^N = " << mod_pow(twiddle_base, N, prime)
              << " (should be 1 for order N)" << std::endl;
}

// =============================================================================
// Sample 5: NTT vs Convolution
// =============================================================================

void sample_ntt_vs_convolution() {
    std::cout << "\n=== Sample 5: NTT vs Convolution ===" << std::endl;

    std::cout << "Naive polynomial multiplication: O(N^2)" << std::endl;
    std::cout << "  c[k] = sum of a[i] * b[k-i] for all valid i" << std::endl;

    std::cout << "\nNTT-based multiplication: O(N log N)" << std::endl;
    std::cout << "  1. NTT(a) -> A" << std::endl;
    std::cout << "  2. NTT(b) -> B" << std::endl;
    std::cout << "  3. C = pointwise_mul(A, B)" << std::endl;
    std::cout << "  4. INTT(C) -> c" << std::endl;

    std::cout << "\nExample for N=4 (degree 3 polynomials):" << std::endl;
    std::cout << "  a = [a0, a1, a2, a3]" << std::endl;
    std::cout << "  b = [b0, b1, b2, b3]" << std::endl;
    std::cout << "  Convolution c = a * b (degree 6):" << std::endl;
    std::cout << "    c[0] = a0*b0" << std::endl;
    std::cout << "    c[1] = a0*b1 + a1*b0" << std::endl;
    std::cout << "    c[2] = a0*b2 + a1*b1 + a2*b0" << std::endl;
    std::cout << "    c[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0" << std::endl;
    std::cout << "    c[4] = a1*b3 + a2*b2 + a3*b1" << std::endl;
    std::cout << "    c[5] = a2*b3 + a3*b2" << std::endl;
    std::cout << "    c[6] = a3*b3" << std::endl;

    std::cout << "\nWith NTT, each pointwise multiply replaces O(N) operations" << std::endl;
    std::cout << "Total complexity: 3*O(N log N) + O(N) ≈ O(N log N)" << std::endl;
}

// =============================================================================
// Sample 6: Slot Structure
// =============================================================================

void sample_slot_structure() {
    std::cout << "\n=== Sample 6: Slot Structure ===" << std::endl;

    uint32_t N = 8;  // Polynomial degree

    std::cout << "CKKS encodes complex numbers in polynomial slots:" << std::endl;
    std::cout << "  N coefficients -> N/2 complex slots" << std::endl;
    std::cout << "  For N=" << N << ", we have " << (N/2) << " complex slots" << std::endl;

    std::cout << "\nSlot encoding:" << std::endl;
    std::cout << "  Polynomial: p(x) = a0 + a1*x + a2*x^2 + ..." << std::endl;
    std::cout << "  Evaluated at special roots of unity" << std::endl;
    std::cout << "  Roots: zeta^k for k=0..N-1" << std::endl;
    std::cout << "  Slot k = p(zeta^k)" << std::endl;

    std::cout << "\nMemory layout for slots:" << std::endl;
    std::cout << "  In flat buffer, even indices = real parts" << std::endl;
    std::cout << "  In flat buffer, odd indices = imaginary parts" << std::endl;
    std::cout << "  slot[k].real = buffer[2*k]" << std::endl;
    std::cout << "  slot[k].imag = buffer[2*k+1]" << std::endl;

    std::cout << "\nFor N=1024, numSlots=512:" << std::endl;
    std::cout << "  Can encrypt and operate on 512 complex numbers simultaneously" << std::endl;
    std::cout << "  This is the SIMD property of CKKS" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "FIDESlib Metal - NTT Examples" << std::endl;
    std::cout << "========================================" << std::endl;

    sample_flat_buffer_layout();
    sample_ntt_butterfly();
    sample_bit_reversal();
    sample_twiddle_factors();
    sample_ntt_vs_convolution();
    sample_slot_structure();

    std::cout << "\n========================================" << std::endl;
    std::cout << "NTT examples completed!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
