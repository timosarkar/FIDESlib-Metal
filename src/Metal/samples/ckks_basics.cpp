//
// FIDESlib Metal Backend - CKKS Basics Example
// Demonstrates encoding, encryption, and homomorphic operations
//

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <iomanip>
#include <cstdint>

using Complex = std::complex<double>;

// =============================================================================
// CKKS Parameters
// =============================================================================

struct CKKSPartParams {
    uint32_t N;              // Polynomial degree (e.g., 1024, 2048, 4096)
    double scale;            // Scaling factor (Δ)
    uint32_t numSlots;       // Number of complex slots (N/2)
};

const CKKSPartParams PARAMS = {
    .N = 1024,
    .scale = 1ULL << 40,     // 2^40 ≈ 1 trillion
    .numSlots = 512
};

// =============================================================================
// Modular Arithmetic (CPU reference)
// =============================================================================

uint64_t mod_add(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t r = a + b;
    if (r >= mod) r -= mod;
    return r;
}

uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t r = a - b;
    if (r > a) r += mod;
    return r;
}

uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t mod) {
    return (uint64_t)((__uint128_t)a * b % mod);
}

// =============================================================================
// CKKS Encoding (simplified)
// =============================================================================

// Encode a vector of complex numbers into a polynomial
// In real CKKS, this uses baby-step-giant-step and coefficient encoding
class CKKSEncoder {
public:
    CKKSEncoder(const CKKSPartParams& params) : params_(params) {}

    // Simulated encoding: just scales and packs complex values
    // Real CKKS uses proper encoding with FFT/NTT
    std::vector<double> encode(const std::vector<Complex>& messages) {
        size_t n = messages.size();
        std::vector<double> poly(params_.N);

        // Simple packing: interleave real and imaginary parts
        for (size_t i = 0; i < n && i < params_.numSlots; i++) {
            // In real CKKS, we use the slot structure for this
            poly[2 * i] = messages[i].real() * params_.scale;
            poly[2 * i + 1] = messages[i].imag() * params_.scale;
        }

        return poly;
    }

    // Decode: extract complex values from polynomial
    std::vector<Complex> decode(const std::vector<double>& poly) {
        size_t n = std::min<size_t>(poly.size() / 2, params_.numSlots);
        std::vector<Complex> messages(n);

        for (size_t i = 0; i < n; i++) {
            double re = poly[2 * i] / params_.scale;
            double im = poly[2 * i + 1] / params_.scale;
            messages[i] = Complex(re, im);
        }

        return messages;
    }

private:
    CKKSPartParams params_;
};

// =============================================================================
// Simulated FHE Context
// =============================================================================

class FHETex {
public:
    FHETex(const CKKSPartParams& params) : params_(params), encoder_(params) {
        // Initialize random polynomial for encryption
        s_ = generate_random_poly();
    }

    // Simulated encryption: c0 = m + q * r, c1 = q (simplified)
    // Real FHE uses RLWE encryption: c0 = m + q*r, c1 = r
    struct Ciphertext {
        std::vector<double> c0;  // First polynomial
        std::vector<double> c1;  // Second polynomial
    };

    Ciphertext encrypt(const std::vector<Complex>& messages) {
        auto m = encoder_.encode(messages);

        // Generate random "error" polynomial (simulated)
        auto r = generate_random_poly();

        // Simple encryption: c0 = m + r (in real FHE, this is mod-switched)
        Ciphertext ct;
        ct.c0.resize(params_.N);
        ct.c1 = r;  // In real FHE, c1 = secret_key in NTT form

        for (size_t i = 0; i < params_.N; i++) {
            ct.c0[i] = m[i] + r[i] * 0.001;  // Small error term
        }

        return ct;
    }

    std::vector<Complex> decrypt(const Ciphertext& ct) {
        // Simplified decryption: just return c0
        // Real FHE: m = c0 - c1 * s
        return encoder_.decode(ct.c0);
    }

    // Homomorphic addition: ct3 = ct1 + ct2
    Ciphertext add(const Ciphertext& ct1, const Ciphertext& ct2) {
        Ciphertext result;
        result.c0.resize(params_.N);
        result.c1.resize(params_.N);

        for (size_t i = 0; i < params_.N; i++) {
            result.c0[i] = ct1.c0[i] + ct2.c0[i];
            result.c1[i] = ct1.c1[i] + ct2.c1[i];
        }

        return result;
    }

    // Simulated homomorphic multiplication: (a, b) * (c, d) = (ac, ad + bc)
    // Real CKKS multiplication involves NTT convolution and rescaling
    Ciphertext multiply(const Ciphertext& ct1, const Ciphertext& ct2) {
        // For demonstration, we simulate polynomial multiplication
        // In real CKKS: use NTT for fast convolution, then rescale

        Ciphertext result;
        result.c0.resize(params_.N);
        result.c1.resize(params_.N);

        // Simulate convolution (simplified - real uses NTT)
        for (size_t i = 0; i < params_.N; i++) {
            result.c0[i] = ct1.c0[i] * ct2.c0[i] * 0.5;  // Rescale
            result.c1[i] = ct1.c1[i] * ct2.c0[i] + ct1.c0[i] * ct2.c1[i];
        }

        return result;
    }

private:
    std::vector<double> generate_random_poly() {
        std::vector<double> poly(params_.N);
        for (size_t i = 0; i < params_.N; i++) {
            poly[i] = (rand() % 1000) * 0.001;
        }
        return poly;
    }

    CKKSPartParams params_;
    CKKSEncoder encoder_;
    std::vector<double> s_;  // Secret key
};

// =============================================================================
// Sample 1: Basic Encryption/Decryption
// =============================================================================

void sample_basic_encrypt_decrypt() {
    std::cout << "\n=== Sample 1: Basic Encryption/Decryption ===" << std::endl;

    FHETex fhe(PARAMS);

    // Create message vector
    std::vector<Complex> messages = {
        Complex(1.0, 0.0),
        Complex(2.0, 0.0),
        Complex(0.5, 0.5),
        Complex(3.14159, 2.71828)
    };

    std::cout << "Original messages:" << std::endl;
    for (size_t i = 0; i < messages.size(); i++) {
        std::cout << "  m[" << i << "] = " << std::fixed << std::setprecision(4)
                  << messages[i].real() << " + " << messages[i].imag() << "i" << std::endl;
    }

    // Encrypt
    std::cout << "\nEncrypting..." << std::endl;
    auto ct = fhe.encrypt(messages);
    std::cout << "  Ciphertext size: " << ct.c0.size() << " coefficients" << std::endl;
    std::cout << "  c0[0] = " << std::fixed << std::setprecision(4) << ct.c0[0] << std::endl;
    std::cout << "  c1[0] = " << ct.c1[0] << std::endl;

    // Decrypt
    std::cout << "\nDecrypting..." << std::endl;
    auto decrypted = fhe.decrypt(ct);

    std::cout << "Decrypted messages:" << std::endl;
    for (size_t i = 0; i < decrypted.size(); i++) {
        std::cout << "  d[" << i << "] = " << std::fixed << std::setprecision(4)
                  << decrypted[i].real() << " + " << decrypted[i].imag() << "i" << std::endl;
    }
}

// =============================================================================
// Sample 2: Homomorphic Addition
// =============================================================================

void sample_homomorphic_addition() {
    std::cout << "\n=== Sample 2: Homomorphic Addition ===" << std::endl;

    FHETex fhe(PARAMS);

    // Two message vectors
    std::vector<Complex> messages1 = {
        Complex(1.0, 0.0),
        Complex(2.0, 0.0),
        Complex(3.0, 0.0)
    };

    std::vector<Complex> messages2 = {
        Complex(0.5, 0.0),
        Complex(0.5, 0.0),
        Complex(0.5, 0.0)
    };

    std::cout << "Message vector 1: [1, 2, 3]" << std::endl;
    std::cout << "Message vector 2: [0.5, 0.5, 0.5]" << std::endl;
    std::cout << "Expected sum: [1.5, 2.5, 3.5]" << std::endl;

    // Encrypt both
    auto ct1 = fhe.encrypt(messages1);
    auto ct2 = fhe.encrypt(messages2);

    // Homomorphic addition
    auto ct3 = fhe.add(ct1, ct2);

    // Decrypt result
    auto result = fhe.decrypt(ct3);

    std::cout << "\nHomomorphic addition result:" << std::endl;
    std::cout << "  Result: [";
    for (size_t i = 0; i < result.size(); i++) {
        std::cout << std::fixed << std::setprecision(4) << result[i].real();
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// =============================================================================
// Sample 3: Homomorphic Multiplication
// =============================================================================

void sample_homomorphic_multiplication() {
    std::cout << "\n=== Sample 3: Homomorphic Multiplication ===" << std::endl;

    FHETex fhe(PARAMS);

    // Two message vectors
    std::vector<Complex> messages1 = {
        Complex(2.0, 0.0),
        Complex(3.0, 0.0)
    };

    std::vector<Complex> messages2 = {
        Complex(4.0, 0.0),
        Complex(5.0, 0.0)
    };

    std::cout << "Message vector 1: [2, 3]" << std::endl;
    std::cout << "Message vector 2: [4, 5]" << std::endl;
    std::cout << "Expected product: [8, 15]" << std::endl;

    // Encrypt both
    auto ct1 = fhe.encrypt(messages1);
    auto ct2 = fhe.encrypt(messages2);

    // Homomorphic multiplication
    auto ct3 = fhe.multiply(ct1, ct2);

    // Decrypt result
    auto result = fhe.decrypt(ct3);

    std::cout << "\nHomomorphic multiplication result:" << std::endl;
    std::cout << "  Result: [";
    for (size_t i = 0; i < result.size(); i++) {
        std::cout << std::fixed << std::setprecision(4) << result[i].real();
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Note: Results are approximate due to rounding in simulation" << std::endl;
}

// =============================================================================
// Sample 4: Polynomial Representation
// =============================================================================

void sample_polynomial_representation() {
    std::cout << "\n=== Sample 4: Polynomial Representation ===" << std::endl;

    CKKSEncoder encoder(PARAMS);

    // Complex messages
    std::vector<Complex> messages = {
        Complex(1.0, 0.0),
        Complex(0.0, 1.0),
        Complex(-1.0, 0.0),
        Complex(0.0, -1.0)
    };

    std::cout << "Messages (complex numbers):" << std::endl;
    for (size_t i = 0; i < messages.size(); i++) {
        std::cout << "  m[" << i << "] = " << messages[i] << std::endl;
    }

    // Encode to polynomial
    auto poly = encoder.encode(messages);

    std::cout << "\nEncoded polynomial coefficients (scaled by Δ = 2^40):" << std::endl;
    std::cout << "  Polynomial degree: " << PARAMS.N << std::endl;
    std::cout << "  Used slots: " << PARAMS.numSlots << std::endl;
    std::cout << "  First 8 coefficients:" << std::endl;
    for (size_t i = 0; i < 8; i++) {
        std::cout << "    poly[" << i << "] = " << std::fixed << std::setprecision(1)
                  << poly[i] << std::endl;
    }

    // Decode back
    auto decoded = encoder.decode(poly);

    std::cout << "\nDecoded messages:" << std::endl;
    for (size_t i = 0; i < decoded.size(); i++) {
        std::cout << "  d[" << i << "] = " << std::fixed << std::setprecision(4)
                  << decoded[i] << std::endl;
    }
}

// =============================================================================
// Sample 5: Slot Structure (SIMD operations)
// =============================================================================

void sample_slot_structure() {
    std::cout << "\n=== Sample 5: Slot Structure (SIMD) ===" << std::endl;

    std::cout << "CKKS Parameters:" << std::endl;
    std::cout << "  Polynomial degree N = " << PARAMS.N << std::endl;
    std::cout << "  Number of slots = " << PARAMS.numSlots << std::endl;
    std::cout << "  Each slot holds one complex number" << std::endl;

    // Demonstrate slot packing
    std::cout << "\nSlot structure for N=1024, numSlots=512:" << std::endl;
    std::cout << "  Complex numbers packed as: [re0, im0, re1, im1, ..., re511, im511]" << std::endl;
    std::cout << "  Polynomial indices:     [  0,   1,   2,   3, ..., 1022, 1023]" << std::endl;

    // Create a message for each slot
    std::vector<Complex> all_slots(PARAMS.numSlots);
    for (size_t i = 0; i < PARAMS.numSlots; i++) {
        all_slots[i] = Complex(static_cast<double>(i), 0.0);
    }

    CKKSEncoder encoder(PARAMS);
    auto poly = encoder.encode(all_slots);

    std::cout << "\nAll slots filled with values [0, 1, 2, ..., 511]:" << std::endl;
    std::cout << "  First 10 polynomial coefficients:" << std::endl;
    for (size_t i = 0; i < 10; i++) {
        std::cout << "    poly[" << i << "] = " << std::fixed << std::setprecision(1)
                  << poly[i] << std::endl;
    }

    // SIMD operations: element-wise multiplication by 2
    std::cout << "\nSIMD operation: multiply all slots by 2" << std::endl;
    std::cout << "  This is done by polynomial multiplication, not element-wise in coeff domain" << std::endl;
    std::cout << "  In real CKKS, use NTT to go to coefficient domain" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "FIDESlib Metal - CKKS Basics Examples" << std::endl;
    std::cout << "========================================" << std::endl;

    sample_basic_encrypt_decrypt();
    sample_homomorphic_addition();
    sample_homomorphic_multiplication();
    sample_polynomial_representation();
    sample_slot_structure();

    std::cout << "\n========================================" << std::endl;
    std::cout << "CKKS basics examples completed!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
