# FIDESlib Metal Backend - Samples

This directory contains real-world sample programs demonstrating FIDESlib Metal backend usage for CKKS homomorphic encryption operations.

## Samples

### 1. ntt_example.cpp
Demonstrates Number Theoretic Transform (NTT) concepts:
- Flat buffer layout for multi-limb representation
- Butterfly operations (Cooley-Tukey, Gentleman-Sande)
- Bit reversal permutation
- Slot structure for SIMD operations

### 2. ckks_basics.cpp
Basic CKKS operations on encrypted data:
- Encoding plaintext to polynomial
- Encryption (simulated)
- Homomorphic addition
- Homomorphic multiplication
- Decryption and decoding

### 3. rotation_example.cpp
Rotation/automorphism operations:
- Automorphism transformation
- Rotation patterns for different slot counts
- Conjugation for complex numbers
- Teleswap patterns

### 4. modular_arithmetic.cpp
Modular arithmetic operations using Metal math helpers:
- Modular addition/subtraction
- Barrett reduction
- Shoup multiplication
- Bit operations for NTT

### 5. bmi_calculator.cpp (NEW)
Privacy-preserving BMI calculator demonstrating FHE:
- Basic BMI calculation on encrypted data
- Batch BMI for multiple users (SIMD)
- Height anonymization for privacy
- Group average BMI without individual disclosure
- BMI trend analysis over time

**Run BMI calculator:**
```bash
./build/bmi_calculator
```

This shows how FHE enables:
- Health data stays encrypted on user's device
- Server computes BMI without seeing raw data
- Only authorized party can decrypt results

## Building Samples

```bash
# Compile math tests (CPU-only, no Metal device required)
clang++ -std=c++20 -o build/metal_math_test src/Metal/MetalMathTest.cpp
./build/metal_math_test

# Full Metal samples (require Objective-C++ and Metal framework)
clang++ -std=c++20 -framework Metal -fobjc-arc \
    src/Metal/samples/*.cpp \
    src/Metal/MetalDevice.mm \
    src/Metal/MetalBuffer.mm \
    src/Metal/MetalStream.mm \
    -o build/samples
```

## Understanding the Samples

### Flat Buffer Layout

CKKS polynomials are represented as multi-limb buffers in GPU memory:
```
Element at (limb, slot) -> flat[index]
where: index = limb * N + slot
```

For N=1024 slots and 8 limbs:
- Limb 0: indices 0-1023
- Limb 1: indices 1024-2047
- etc.

### NTT Parameters

Standard NTT parameters for CKKS:
- N = 1024, 2048, 4096, 8192
- Primes: 1073807353 (30-bit), 1073799977 (30-bit), etc.
- Primitive roots: 31, 30, etc.

### CKKS Encoding

CKKS encodes complex vectors into polynomials:
1. Scale the complex numbers by a large factor Δ
2. Embed into a polynomial of degree N/2
3. Use the slot structure for SIMD operations

### Homomorphic Operations

After encryption, you can perform:
- **Addition**: Element-wise polynomial addition
- **Multiplication**: Polynomial multiplication (NTT domain)
- **Rotation**: Automorphism x → x^k mod Φ(M)
- **Conjugation**: x → x^{-1} mod Φ(M)

## Prerequisites

- macOS with Xcode command line tools
- Metal-compatible GPU
- Understanding of CKKS homomorphic encryption
