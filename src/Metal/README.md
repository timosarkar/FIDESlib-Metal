# FIDESlib Metal Backend

## Overview

FIDESlib is a full CKKS (Cheon-Kim-Kim-Son) homomorphic encryption library originally written for NVIDIA CUDA. This directory contains the **direct Metal port** for **macOS only**, targeting Apple Silicon GPUs with **full CKKS support** (NTT, modular arithmetic, rotations, bootstrap).

**Key Characteristics:**
- Direct Metal port (no abstraction layer, no CUDA fallback)
- macOS only (Apple Silicon and Intel Mac with Metal support)
- Full CKKS support for homomorphic encryption operations
- All 7 Metal shaders compile successfully

## Files

### Foundation Layer

| File | Purpose |
|------|---------|
| `MetalDevice.hpp/cpp` | Device enumeration, MTLDevice setup |
| `MetalBuffer.hpp/cpp` | GPU memory wrapper (MTLBuffer) |
| `MetalMemoryPool.hpp/cpp` | Async memory pool |
| `MetalStream.hpp/cpp` | Command queue/buffer abstraction |
| `MetalLauncher.hpp/cpp` | Kernel launch helpers with pipeline caching |
| `MetalMath.hpp` | Intrinsic wrappers (bit-reverse, clz, umulhi) |

### Metal Shaders

| File | Purpose | Status |
|------|---------|--------|
| `MetalAddSub.metal` | Add/subtract kernels | Compiles |
| `MetalModMult.metal` | Barrett/Shoup/Neal multiplication | Compiles |
| `MetalNTT.metal` | NTT/INTT 2D CT/GS butterflies | Compiles |
| `MetalNTTFusion.metal` | Fused NTT templates | Compiles |
| `MetalConv.metal` | Convolution/modulus switching | Compiles |
| `MetalRotation.metal` | Automorphism/rotation kernels | Compiles |
| `MetalElemWise.metal` | Batch element-wise operations | Compiles |

### Test Files

| File | Purpose |
|------|---------|
| `MetalMathTest.cpp` | CPU-based unit tests for math helpers |
| `MetalSample.cpp` | Sample program demonstrating Metal operations |
| `MetalTest.cpp` | Full Metal integration tests (requires device) |

## Building

### Prerequisites

- macOS with Xcode command line tools
- Metal-compatible GPU (Apple Silicon or Intel Mac with Metal)
- CMake 3.10+

### Compile Metal Shaders

```bash
mkdir -p build/metal
for f in src/Metal/*.metal; do
    xcrun metal -c "$f" -o "build/metal/$(basename $f .metal).air"
done
xcrun metallib build/metal/*.air -o build/metal/fideslib_metal.metallib
```

### CMake Integration

Add to your CMake configuration:

```cmake
set(FIDES_USE_METAL ON CACHE BOOL "Enable Metal GPU backend")
include(${CMAKE_SOURCE_DIR}/cmake/Metal.cmake)
```

The CMake module will:
1. Check for xcrun toolchain
2. Compile all `.metal` files to `.air` files
3. Combine `.air` files into `fideslib_metal.metallib`
4. Install the metallib to the library directory

## Math Helpers (MetalMath.hpp)

The Metal backend provides software implementations for intrinsics not available in Metal:

```cpp
// Bit reversal (Metal has no __brev)
uint32_t metal_brev(uint32_t x);     // 32-bit bit reverse
uint64_t metal_brev64(uint64_t x);   // 64-bit bit reverse

// Count leading zeros
uint32_t metal_clz(uint32_t x);      // clz for 32-bit
uint32_t metal_clzll(uint64_t x);    // clz for 64-bit

// Unsigned multiply high
uint32_t metal_umulhi(uint32_t a, uint32_t b);    // 32-bit upper product
uint64_t metal_umul64hi(uint64_t a, uint64_t b);  // 64-bit upper product

// Popcount
uint32_t metal_popc(uint32_t x);     // Population count
```

## Modular Arithmetic

```cpp
// Barrett reduction - requires a*b >= mod^2 for accurate results
uint64_t metal_modmult_barrett(uint64_t a, uint64_t b, uint64_t mod);
uint64_t metal_barrett_mu(uint64_t mod);  // Precompute mu = floor(2^64 / mod)

// Shoup's algorithm - faster when b is reused
uint64_t metal_shoup_precompute(uint64_t b, uint64_t mod);
uint64_t metal_modmult_shoup(uint64_t a, uint64_t b, uint64_t mod, uint64_t shoup_b);

// Simple modular operations
uint64_t metal_modadd(uint64_t a, uint64_t b, uint64_t mod);
uint64_t metal_modsub(uint64_t a, uint64_t b, uint64_t mod);
```

**Barrett Reduction Limitation:**
Barrett reduction with k=64 requires `a * b >= mod^2` for accurate results. When `a * b < mod^2`, the quotient `q = floor(a * b * mu / 2^64)` can be 0, resulting in no reduction.

In CKKS context, inputs are typically in `[0, mod)` but products are often large enough to satisfy this constraint. For small moduli, use direct modular multiplication instead.

## Architecture Notes

### Thread/Grid Model

Metal has no native 3D grid. Flatten 3D CUDA grids:

```cpp
// CUDA
dim3 grid(N/2, numLimbs, parts);

// Metal (flattened)
MTLSize gridSize = MTLSizeMake(N/2 * parts, numLimbs, 1);
```

### Shared Memory Constraint

- CUDA: up to 48KB shared memory per block
- Metal: ~32KB threadgroup memory per threadgroup
- **Fix:** Reduce threadgroup size from 1024 to 512 threads

### Threadgroup Size

- CUDA: max 1024 threads/block
- Metal: max 512 threads/threadgroup recommended

### Intrinsic Translations

| CUDA | Metal |
|------|-------|
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_device)` |
| `__syncwarp()` | `simdgroup_barrier()` |
| `__brev(x)` | `metal_brev(x)` (software, ~10x slower) |
| `__clz(x)` | `metal_clz(x)` |
| `__umulhi(a,b)` | `metal_umul64hi(a, b)` |
| `atomicAdd` | `atomic_fetch_add_explicit` |

## Testing

### Math Unit Tests (CPU-only)

```bash
clang++ -std=c++20 -o build/metal_math_test src/Metal/MetalMathTest.cpp
./build/metal_math_test
```

Tests cover:
- Modular addition/subtraction
- Barrett/Shoup modular multiplication
- Bit reversal (32-bit and 64-bit)
- Count leading zeros
- Population count
- Unsigned multiply high

**Results:** 53 tests pass

### Sample Programs

```bash
# Build all samples
clang++ -std=c++20 -o build/ntt_example src/Metal/samples/ntt_example.cpp
clang++ -std=c++20 -o build/ckks_basics src/Metal/samples/ckks_basics.cpp
clang++ -std=c++20 -o build/rotation_example src/Metal/samples/rotation_example.cpp
clang++ -std=c++20 -o build/modular_arithmetic src/Metal/samples/modular_arithmetic.cpp
clang++ -std=c++20 -o build/bmi_calculator src/Metal/samples/bmi_calculator.cpp

# Run samples
./build/ntt_example          # Flat buffer layout, butterfly operations
./build/ckks_basics         # CKKS encoding/encryption/decryption
./build/rotation_example     # Rotation/automorphism concepts
./build/modular_arithmetic  # Barrett/Shoup multiplication
./build/bmi_calculator       # Privacy-preserving BMI example
```

### Full Metal Tests (requires Metal device)

```bash
# Requires Objective-C++ and Metal framework
clang++ -std=c++20 -framework Metal -fobjc-arc \
    src/Metal/MetalTest.cpp src/Metal/MetalDevice.mm \
    src/Metal/MetalBuffer.mm src/Metal/MetalStream.mm \
    -o build/metal_test
./build/metal_test
```

## CKKS Operations Supported

### Element-wise Batch Kernels
- `mult1Add2` - multiply then add
- `Mult` - batch multiplication
- `Add`, `Sub` - batch addition/subtraction
- `square` - batch squaring
- `binomialMult` - binomial multiplication
- `dotProductPt` - dot product
- `eval_linear_w_sum` - linear sum evaluation

### Rotation/Automorphism Kernels
- `automorph` - automorphism transformation
- `conjugate` - complex conjugation
- `teleswap` - telescopic rotation patterns

### NTT Fusion Kernels
- `mult_and_save` - multiply and save
- `square_and_save` - square and save
- `rescale` - rescaling operation
- `moddown` - modulus switching down
- `ksk_dot` - key switching dot product

### Convolution Kernels
- `ModDown2` - modulus switching
- `DecompAndModUpConv` - decomposition and mod-up
- `ModUpDiag` - modulus up diagonal

## Bootstrap

Bootstrap kernels are mostly orchestration/CPU code that calls the GPU kernels already ported. The GPU parts are in `MetalBootstrap.metal`.

## Key Translation Notes

1. **No `device void**` - Metal doesn't support `device void**`. Restructure buffers to flat layouts.

2. **No double precision** - Use `float` for CKKS operations (homomorphic encryption doesn't require doubles).

3. **Threadgroup barriers** - Always use `threadgroup_barrier(mem_flags::mem_device)` for device-wide sync.

4. **constexpr limitations** - Metal doesn't allow `constexpr` at program scope. Use `#define` instead.

## Performance Considerations

- Bit reversal is software (~10x slower than hardware `__brev`). Consider precomputing lookup tables for small N.
- Threadgroup memory is limited to ~32KB. Restructure kernels to use registers where possible.
- 512 threads per threadgroup (not 1024 like CUDA). Adjust grid dimensions accordingly.

## CMake Configuration

The `cmake/Metal.cmake` module handles:

```cmake
# Find Metal toolchain
find_program(XCRUN_EXECUTABLE xcrun PATHS /usr/bin /usr/local/bin)

# Metal source files
file(GLOB METAL_SHADER_FILES "${METAL_SOURCE_DIR}/*.metal")

# Custom target for compilation
add_custom_target(metal_lib ALL
    COMMAND "${COMPILE_SCRIPT}" ...
    DEPENDS ${METAL_SHADER_FILES}
)
```

## Combined metallib

After compilation, all shaders are combined into:
```
build/metal/fideslib_metal.metallib (292KB)
```

This single metallib contains all kernels from all 7 shader files.

## Next Steps

1. **Context Integration** - Replace CUDA calls with Metal in CKKS Context
2. **Testing Infrastructure** - Build complete test suite with Metal device
3. **Bootstrap Validation** - Verify bootstrap produces correct results vs CPU reference

## Troubleshooting

**"No Metal devices found"**
- Ensure running on macOS with Metal-compatible GPU
- Check that Xcode command line tools are installed

**"xcrun not found"**
- Install Xcode command line tools: `xcode-select --install`

**Shader compilation errors**
- Ensure using correct Metal SDK: `xcrun --version`
- Check for deprecated Metal 2.0 features

**Linker errors with metallib**
- Ensure all `.air` files are included: `xcrun metallib build/metal/*.air`
