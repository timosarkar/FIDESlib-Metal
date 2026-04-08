//
// FIDESlib Metal Backend - Rotation Example
// Demonstrates automorphism/rotation operations in CKKS
//

#include <iostream>
#include <vector>
#include <cstdint>
#include <complex>

using Complex = std::complex<double>;

// =============================================================================
// CKKS Rotation Parameters
// =============================================================================

struct RotationParams {
    uint32_t N;          // Polynomial degree (must be power of 2)
    uint32_t m;          // Cyclotomic index (2 * N for power-of-2)
    uint32_t phi;        // Euler's totient = N for power-of-2
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

uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t mod) {
    return (uint64_t)((__uint128_t)a * b % mod);
}

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

// =============================================================================
// Automorphism Group
// =============================================================================

// In CKKS with cyclotomic polynomial x^N + 1, the automorphism group is:
// Gal(K/Q) = {sigma_k : x -> x^k mod (x^N + 1) | gcd(k, 2N) = 1}
//
// For power-of-2 N, this means k must be odd.
// The rotation by k positions corresponds to automorphism sigma_k.

class AutomorphismGroup {
public:
    AutomorphismGroup(uint32_t N) : N_(N), m_(2 * N), phi_(N) {
        // Find primitive (2N)-th root of unity
        // For N = power of 2, primitive root is 5 or similar
        find_primitive_root();
    }

    // Get all valid automorphism indices (odd numbers < 2N)
    std::vector<uint32_t> get_valid_indices() const {
        std::vector<uint32_t> indices;
        for (uint32_t k = 1; k < m_; k += 2) {
            if (gcd(k, m_) == 1) {  // k coprime to 2N
                indices.push_back(k);
            }
        }
        return indices;
    }

    // Apply automorphism sigma_k to polynomial coefficients
    // In CKKS, sigma_k acts on the slots by multiplication
    // For power-of-2, slots correspond to powers of primitive root
    std::vector<uint64_t> apply_automorphism(
        const std::vector<uint64_t>& poly,
        uint32_t k,
        uint64_t prime,
        uint64_t root) {

        uint32_t N = (uint32_t)poly.size();
        std::vector<uint64_t> result(N);

        // For automorphism x -> x^k mod (x^N + 1)
        // If poly(x) = sum a_i * x^i, then sigma_k(poly) = sum a_i * x^(k*i mod 2N)
        // But since x^N = -1, we have x^(2N) = 1
        // So x^(k*i) = x^((k*i) mod 2N)

        for (uint32_t i = 0; i < N; i++) {
            uint32_t new_idx = (k * i) % (2 * N_);
            if (new_idx >= N) {
                new_idx = 2 * N_ - new_idx;  // x^N = -1, so x^(N+t) = -x^t
                result[new_idx] = mod_sub(0, poly[i], prime);
            } else {
                result[new_idx] = poly[i];
            }
        }

        return result;
    }

    // Compute rotation on the slots
    // For CKKS, rotation by r slots corresponds to automorphism with k = root^r
    uint32_t compute_rotation_index(uint32_t rotation_amount, uint64_t root, uint64_t mod) {
        // Find k such that sigma_k corresponds to rotation by r slots
        // This depends on the specific CKKS parameterization
        // For standard CKKS with power-of-2, k = 5^rotation (or similar)

        // Simplified: just return the rotation amount if valid
        return rotation_amount % (2 * N_);
    }

    // Verify automorphism property: sigma_a o sigma_b = sigma_(a*b mod 2N)
    bool verify_group_property() const {
        auto indices = get_valid_indices();
        uint32_t k1 = indices[0];
        uint32_t k2 = indices[1];
        uint32_t k_combined = (k1 * k2) % m_;

        // Check if k_combined is also a valid automorphism index
        return gcd(k_combined, m_) == 1;
    }

    uint32_t N() const { return N_; }
    uint32_t m() const { return m_; }

private:
    void find_primitive_root() {
        // Find primitive root of unity of order 2N
        // For 2N = power of 2, we need a primitive (2N)-th root
        // This is complex - for now use a simple heuristic
        for (uint32_t g = 2; g < prime_; g++) {
            if (mod_pow(g, m_, prime_) == 1 &&
                mod_pow(g, m_ / 2, prime_) != 1) {
                primitive_root_ = g;
                break;
            }
        }
    }

    uint32_t gcd(uint32_t a, uint32_t b) const {
        while (b != 0) {
            uint32_t t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    uint32_t N_;
    uint32_t m_;
    uint32_t phi_;
    uint32_t primitive_root_ = 5;  // Common choice
    const uint32_t prime_ = 1073807353;  // Example prime
};

// =============================================================================
// Rotation Simulation for CKKS Slots
// =============================================================================

class CKKSRotations {
public:
    CKKSRotations(uint32_t N, uint32_t num_slots) : N_(N), num_slots_(num_slots) {}

    // Rotate slots by r positions to the left
    // In CKKS, this corresponds to automorphism sigma_k where k = g^r
    std::vector<Complex> rotate_left(const std::vector<Complex>& slots, uint32_t r) {
        uint32_t n = (uint32_t)slots.size();
        std::vector<Complex> result(n);

        r = r % n;  // Wrap rotation
        for (uint32_t i = 0; i < n; i++) {
            result[i] = slots[(i + r) % n];
        }

        return result;
    }

    // Rotate slots by r positions to the right
    std::vector<Complex> rotate_right(const std::vector<Complex>& slots, uint32_t r) {
        uint32_t n = (uint32_t)slots.size();
        std::vector<Complex> result(n);

        r = r % n;
        for (uint32_t i = 0; i < n; i++) {
            result[(i + r) % n] = slots[i];
        }

        return result;
    }

    // Conjugate: negate imaginary part
    // In CKKS, conjugation is automorphism k = -1 mod 2N
    std::vector<Complex> conjugate(const std::vector<Complex>& slots) {
        std::vector<Complex> result(slots.size());
        for (size_t i = 0; i < slots.size(); i++) {
            result[i] = std::conj(slots[i]);
        }
        return result;
    }

    // Teleswap rotation pattern (used in bootstrapping)
    // Rotates first half forward, second half backward
    std::vector<Complex> teleswap(const std::vector<Complex>& slots) {
        uint32_t n = (uint32_t)slots.size();
        uint32_t half = n / 2;
        std::vector<Complex> result(n);

        for (uint32_t i = 0; i < half; i++) {
            result[i] = slots[i + half];           // First half comes from second half
            result[i + half] = slots[i];            // Second half comes from first half
        }

        return result;
    }

private:
    uint32_t N_;
    uint32_t num_slots_;
};

// =============================================================================
// Sample 1: Basic Slot Rotation
// =============================================================================

void sample_basic_rotation() {
    std::cout << "\n=== Sample 1: Basic Slot Rotation ===" << std::endl;

    const uint32_t N = 16;  // 16 slots (simplified)
    CKKSRotations rot(N, N);

    // Create a vector of complex numbers
    std::vector<Complex> slots(N);
    for (uint32_t i = 0; i < N; i++) {
        slots[i] = Complex(static_cast<double>(i), 0.0);
    }

    std::cout << "Original slots:" << std::endl;
    for (uint32_t i = 0; i < N; i++) {
        std::cout << "  slot[" << i << "] = " << slots[i].real() << std::endl;
    }

    // Rotate left by 3
    auto rotated = rot.rotate_left(slots, 3);

    std::cout << "\nAfter rotating left by 3:" << std::endl;
    for (uint32_t i = 0; i < N; i++) {
        std::cout << "  slot[" << i << "] = " << rotated[i].real() << std::endl;
    }
    std::cout << "  Note: slot[0] now has original slot[13]" << std::endl;

    // Rotate right by 2
    auto rotated_right = rot.rotate_right(slots, 2);

    std::cout << "\nAfter rotating right by 2:" << std::endl;
    for (uint32_t i = 0; i < N; i++) {
        std::cout << "  slot[" << i << "] = " << rotated_right[i].real() << std::endl;
    }
    std::cout << "  Note: slot[2] now has original slot[0]" << std::endl;
}

// =============================================================================
// Sample 2: Complex Conjugation
// =============================================================================

void sample_conjugation() {
    std::cout << "\n=== Sample 2: Complex Conjugation ===" << std::endl;

    const uint32_t N = 8;
    CKKSRotations rot(N, N);

    // Create complex slots
    std::vector<Complex> slots = {
        Complex(1.0, 2.0),
        Complex(3.0, 4.0),
        Complex(5.0, 6.0),
        Complex(7.0, 8.0),
        Complex(-1.0, -2.0),
        Complex(-3.0, -4.0),
        Complex(-5.0, -6.0),
        Complex(-7.0, -8.0)
    };

    std::cout << "Original complex slots:" << std::endl;
    for (uint32_t i = 0; i < N; i++) {
        std::cout << "  slot[" << i << "] = " << slots[i] << std::endl;
    }

    // Apply conjugation
    auto conjugated = rot.conjugate(slots);

    std::cout << "\nAfter conjugation (imaginary parts negated):" << std::endl;
    for (uint32_t i = 0; i < N; i++) {
        std::cout << "  slot[" << i << "] = " << conjugated[i] << std::endl;
    }

    std::cout << "\nUse case: Getting real part of encrypted complex number" << std::endl;
    std::cout << "  In FHE, encrypt a + bi and its conjugate a - bi" << std::endl;
    std::cout << "  Adding them gives 2a (twice the real part)" << std::endl;
    std::cout << "  Subtracting them gives 2bi (twice the imaginary part)" << std::endl;
}

// =============================================================================
// Sample 3: Teleswap Pattern
// =============================================================================

void sample_teleswap() {
    std::cout << "\n=== Sample 3: Teleswap Rotation ===" << std::endl;

    const uint32_t N = 8;
    CKKSRotations rot(N, N);

    std::vector<Complex> slots(N);
    for (uint32_t i = 0; i < N; i++) {
        slots[i] = Complex(static_cast<double>(i + 1), 0.0);  // [1, 2, 3, 4, 5, 6, 7, 8]
    }

    std::cout << "Original slots: [1, 2, 3, 4, 5, 6, 7, 8]" << std::endl;
    std::cout << "  First half: [1, 2, 3, 4]" << std::endl;
    std::cout << "  Second half: [5, 6, 7, 8]" << std::endl;

    // Apply teleswap
    auto swapped = rot.teleswap(slots);

    std::cout << "\nAfter teleswap: [5, 6, 7, 8, 1, 2, 3, 4]" << std::endl;
    std::cout << "  Result first half: [5, 6, 7, 8] (from original second half)" << std::endl;
    std::cout << "  Result second half: [1, 2, 3, 4] (from original first half)" << std::endl;

    for (uint32_t i = 0; i < N; i++) {
        std::cout << "  slot[" << i << "] = " << swapped[i].real() << std::endl;
    }

    std::cout << "\nUse case: Bootstrapping operations that need to swap halves" << std::endl;
}

// =============================================================================
// Sample 4: Automorphism Group Structure
// =============================================================================

void sample_automorphism_group() {
    std::cout << "\n=== Sample 4: Automorphism Group Structure ===" << std::endl;

    const uint32_t N = 8;
    AutomorphismGroup aut(N);

    std::cout << "Automorphism group for N=" << N << ", 2N=" << (2*N) << std::endl;
    std::cout << "  Cyclotomic polynomial: x^" << N << " + 1" << std::endl;
    std::cout << "  Automorphisms: sigma_k where gcd(k, 2N) = 1" << std::endl;

    auto indices = aut.get_valid_indices();

    std::cout << "\nValid automorphism indices (odd numbers coprime to 2N):" << std::endl;
    for (auto k : indices) {
        std::cout << "  k = " << k << std::endl;
    }

    std::cout << "\nNumber of automorphisms: " << indices.size() << std::endl;
    std::cout << "  (Should equal phi(2N)/2 = N/2 for power-of-2 case)" << std::endl;

    std::cout << "\nComposition property:" << std::endl;
    std::cout << "  sigma_a o sigma_b = sigma_(a*b mod 2N)" << std::endl;
    std::cout << "  Example: sigma_1 o sigma_3 = sigma_3" << std::endl;
}

// =============================================================================
// Sample 5: Rotation Chain
// =============================================================================

void sample_rotation_chain() {
    std::cout << "\n=== Sample 5: Rotation Chain (Mobile Operation) ===" << std::endl;

    const uint32_t N = 8;
    CKKSRotations rot(N, N);

    // Initial vector
    std::vector<Complex> slots(N);
    for (uint32_t i = 0; i < N; i++) {
        slots[i] = Complex(static_cast<double>(i), 0.0);
    }

    std::cout << "Initial slots: [0, 1, 2, 3, 4, 5, 6, 7]" << std::endl;

    // Apply consecutive left rotations
    auto r1 = rot.rotate_left(slots, 1);
    auto r2 = rot.rotate_left(r1, 1);
    auto r3 = rot.rotate_left(r2, 1);

    std::cout << "\nAfter consecutive left rotations:" << std::endl;
    std::cout << "  After 1: [" << r1[0].real() << ", " << r1[1].real() << ", " << r1[2].real() << ", "
              << r1[3].real() << ", " << r1[4].real() << ", " << r1[5].real() << ", "
              << r1[6].real() << ", " << r1[7].real() << "]" << std::endl;
    std::cout << "  After 2: [" << r2[0].real() << ", " << r2[1].real() << ", " << r2[2].real() << ", "
              << r2[3].real() << ", " << r2[4].real() << ", " << r2[5].real() << ", "
              << r2[6].real() << ", " << r2[7].real() << "]" << std::endl;
    std::cout << "  After 3: [" << r3[0].real() << ", " << r3[1].real() << ", " << r3[2].real() << ", "
              << r3[3].real() << ", " << r3[4].real() << ", " << r3[5].real() << ", "
              << r3[6].real() << ", " << r3[7].real() << "]" << std::endl;

    // Verify: rotating by 3 is same as rotating by 1 three times
    auto direct_r3 = rot.rotate_left(slots, 3);
    bool match = true;
    for (uint32_t i = 0; i < N; i++) {
        if (r3[i].real() != direct_r3[i].real()) {
            match = false;
            break;
        }
    }
    std::cout << "\nVerification: rotate(slots, 3) == rotate(rotate(rotate(slots, 1), 1), 1): "
              << (match ? "PASS" : "FAIL") << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "FIDESlib Metal - Rotation Examples" << std::endl;
    std::cout << "========================================" << std::endl;

    sample_basic_rotation();
    sample_conjugation();
    sample_teleswap();
    sample_automorphism_group();
    sample_rotation_chain();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Rotation examples completed!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
