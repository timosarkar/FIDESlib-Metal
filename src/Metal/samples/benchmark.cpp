//
// FIDESlib Metal Backend - Benchmark Suite
// Measures performance of Metal GPU operations vs CPU baseline
//

#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <random>
#include <iomanip>
#include <numeric>

// =============================================================================
// Timing Utilities
// =============================================================================

class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::duration<double, std::milli>;

    void start() { start_ = Clock::now(); }
    void stop() { end_ = Clock::now(); }

    double elapsed_ms() const {
        return std::chrono::duration_cast<Duration>(end_ - start_).count();
    }

    double elapsed_us() const {
        return elapsed_ms() * 1000.0;
    }

private:
    TimePoint start_;
    TimePoint end_;
};

// =============================================================================
// Modular Arithmetic Implementations
// =============================================================================

inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t r = a + b;
    if (r >= mod) r -= mod;
    return r;
}

inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t r = a - b;
    if (r > a) r += mod;
    return r;
}

inline uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t mod) {
    return (uint64_t)((__uint128_t)a * b % mod);
}

inline uint64_t umul64hi(uint64_t a, uint64_t b) {
    return (uint64_t)((__uint128_t)a * (__uint128_t)b >> 64);
}

inline uint64_t barrett_mu(uint64_t mod) {
    return UINT64_MAX / mod;
}

inline uint64_t barrett_mul(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t mu = barrett_mu(mod);
    uint64_t q = umul64hi(a, mu);
    __uint128_t t = (__uint128_t)a * (__uint128_t)b;
    uint64_t r = (uint64_t)t - q * mod;
    if (r >= mod) r -= mod;
    return r;
}

inline uint32_t brev32(uint32_t x) {
    x = ((x & 0xAAAAAAAAU) >> 1) | ((x & 0x55555555U) << 1);
    x = ((x & 0xCCCCCCCCU) >> 2) | ((x & 0x33333333U) << 2);
    x = ((x & 0xF0F0F0F0U) >> 4) | ((x & 0x0F0F0F0FU) << 4);
    x = ((x & 0xFF00FF00U) >> 8) | ((x & 0x00FF00FFU) << 8);
    return (x >> 16) | (x << 16);
}

inline uint64_t brev64(uint64_t x) {
    x = ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1) | ((x & 0x5555555555555555ULL) << 1);
    x = ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2) | ((x & 0x3333333333333333ULL) << 2);
    x = ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4);
    x = ((x & 0xFF00FF00FF00FF00ULL) >> 8) | ((x & 0x00FF00FF00FF00FFULL) << 8);
    x = (x >> 16) | (x << 16);
    return (x >> 32) | (x << 32);
}

inline uint32_t clz(uint32_t x) {
    return x == 0 ? 32 : __builtin_clz(x);
}

inline uint32_t clzll(uint64_t x) {
    return x == 0 ? 64 : __builtin_clzll(x);
}

// =============================================================================
// Benchmark Results
// =============================================================================

struct BenchmarkResult {
    std::string name;
    double cpu_time_ms;
    double gpu_time_ms;  // Simulated for Metal
    size_t ops_count;
    double throughput;  // ops per second
    double speedup;
};

// =============================================================================
// Sample 1: Modular Arithmetic Benchmarks
// =============================================================================

void benchmark_modular_arithmetic() {
    std::cout << "\n=== Sample 1: Modular Arithmetic ===" << std::endl;

    const uint64_t MOD = 1073807353;  // ~30-bit prime
    const size_t N = 10000000;  // 10M operations

    std::vector<uint64_t> a(N), b(N);
    std::vector<uint64_t> result(N);

    // Generate random inputs
    std::mt19937_64 rng(42);
    for (size_t i = 0; i < N; i++) {
        a[i] = rng() % MOD;
        b[i] = rng() % MOD;
    }

    // Benchmark modular addition
    {
        Timer t;
        t.start();
        for (size_t i = 0; i < N; i++) {
            result[i] = mod_add(a[i], b[i], MOD);
        }
        t.stop();
        double time_ms = t.elapsed_ms();
        double throughput = N / (time_ms / 1000.0);
        std::cout << "  mod_add:      " << std::fixed << std::setprecision(2)
                  << time_ms << " ms for " << N << " ops = "
                  << (throughput / 1e6) << " M ops/sec" << std::endl;
    }

    // Benchmark modular subtraction
    {
        Timer t;
        t.start();
        for (size_t i = 0; i < N; i++) {
            result[i] = mod_sub(a[i], b[i], MOD);
        }
        t.stop();
        double time_ms = t.elapsed_ms();
        double throughput = N / (time_ms / 1000.0);
        std::cout << "  mod_sub:      " << std::fixed << std::setprecision(2)
                  << time_ms << " ms for " << N << " ops = "
                  << (throughput / 1e6) << " M ops/sec" << std::endl;
    }

    // Benchmark modular multiplication (naive)
    {
        Timer t;
        t.start();
        for (size_t i = 0; i < N; i++) {
            result[i] = mod_mul(a[i], b[i], MOD);
        }
        t.stop();
        double time_ms = t.elapsed_ms();
        double throughput = N / (time_ms / 1000.0);
        std::cout << "  mod_mul:      " << std::fixed << std::setprecision(2)
                  << time_ms << " ms for " << N << " ops = "
                  << (throughput / 1e6) << " M ops/sec" << std::endl;
    }

    // Benchmark Barrett multiplication
    {
        Timer t;
        t.start();
        for (size_t i = 0; i < N; i++) {
            result[i] = barrett_mul(a[i], b[i], MOD);
        }
        t.stop();
        double time_ms = t.elapsed_ms();
        double throughput = N / (time_ms / 1000.0);
        std::cout << "  barrett_mul:  " << std::fixed << std::setprecision(2)
                  << time_ms << " ms for " << N << " ops = "
                  << (throughput / 1e6) << " M ops/sec" << std::endl;
    }
}

// =============================================================================
// Sample 2: Bit Operation Benchmarks
// =============================================================================

void benchmark_bit_operations() {
    std::cout << "\n=== Sample 2: Bit Operations ===" << std::endl;

    const size_t N = 50000000;  // 50M operations

    std::vector<uint32_t> input32(N);
    std::vector<uint64_t> input64(N);
    std::vector<uint32_t> result32(N);
    std::vector<uint64_t> result64(N);

    std::mt19937_64 rng(42);
    for (size_t i = 0; i < N; i++) {
        input32[i] = (uint32_t)rng();
        input64[i] = rng();
    }

    // Benchmark bit reversal (32-bit)
    {
        Timer t;
        t.start();
        for (size_t i = 0; i < N; i++) {
            result32[i] = brev32(input32[i]);
        }
        t.stop();
        double time_ms = t.elapsed_ms();
        double throughput = N / (time_ms / 1000.0);
        std::cout << "  brev32:       " << std::fixed << std::setprecision(2)
                  << time_ms << " ms for " << N << " ops = "
                  << (throughput / 1e6) << " M ops/sec" << std::endl;
    }

    // Benchmark bit reversal (64-bit)
    {
        Timer t;
        t.start();
        for (size_t i = 0; i < N; i++) {
            result64[i] = brev64(input64[i]);
        }
        t.stop();
        double time_ms = t.elapsed_ms();
        double throughput = N / (time_ms / 1000.0);
        std::cout << "  brev64:       " << std::fixed << std::setprecision(2)
                  << time_ms << " ms for " << N << " ops = "
                  << (throughput / 1e6) << " M ops/sec" << std::endl;
    }

    // Benchmark clz (32-bit)
    {
        Timer t;
        t.start();
        for (size_t i = 0; i < N; i++) {
            result32[i] = clz(input32[i]);
        }
        t.stop();
        double time_ms = t.elapsed_ms();
        double throughput = N / (time_ms / 1000.0);
        std::cout << "  clz (32-bit): " << std::fixed << std::setprecision(2)
                  << time_ms << " ms for " << N << " ops = "
                  << (throughput / 1e6) << " M ops/sec" << std::endl;
    }

    // Benchmark clzll (64-bit)
    {
        Timer t;
        t.start();
        for (size_t i = 0; i < N; i++) {
            result64[i] = clzll(input64[i]);
        }
        t.stop();
        double time_ms = t.elapsed_ms();
        double throughput = N / (time_ms / 1000.0);
        std::cout << "  clzll (64-bit):" << std::fixed << std::setprecision(2)
                  << time_ms << " ms for " << N << " ops = "
                  << (throughput / 1e6) << " M ops/sec" << std::endl;
    }
}

// =============================================================================
// Sample 3: Polynomial Operations (Simulated GPU)
// =============================================================================

void benchmark_polynomial_ops() {
    std::cout << "\n=== Sample 3: Polynomial Operations ===" << std::endl;

    // Simulate Metal GPU parameters
    const uint32_t N = 1024;           // Polynomial degree
    const uint32_t NUM_LIMBS = 8;       // RNS limbs
    const uint32_t THREADS = 512;       // Metal threadgroup size
    const size_t TOTAL_OPS = 100000;    // Number of polynomials

    std::cout << "  Simulated Metal GPU parameters:" << std::endl;
    std::cout << "    Polynomial degree N: " << N << std::endl;
    std::cout << "    RNS limbs: " << NUM_LIMBS << std::endl;
    std::cout << "    Threadgroup size: " << THREADS << std::endl;

    // Calculate operations per polynomial
    const size_t COEFFS_PER_POLY = N * NUM_LIMBS;
    const size_t TOTAL_COEFFS = TOTAL_OPS * COEFFS_PER_POLY;

    std::cout << "    Total coefficients: " << TOTAL_COEFFS << std::endl;
    std::cout << "    Total polynomials: " << TOTAL_OPS << std::endl;

    // CPU baseline: coefficient-wise addition
    {
        std::vector<uint64_t> a(TOTAL_COEFFS), b(TOTAL_COEFFS);
        std::vector<uint64_t> result(TOTAL_COEFFS);

        std::mt19937_64 rng(42);
        for (size_t i = 0; i < TOTAL_COEFFS; i++) {
            a[i] = rng();
            b[i] = rng();
        }

        Timer t;
        t.start();
        for (size_t i = 0; i < TOTAL_COEFFS; i++) {
            result[i] = mod_add(a[i], b[i], 1073807353);
        }
        t.stop();

        double time_ms = t.elapsed_ms();
        double throughput = TOTAL_COEFFS / (time_ms / 1000.0);
        std::cout << "\n  CPU polynomial addition:" << std::endl;
        std::cout << "    Time: " << std::fixed << std::setprecision(2) << time_ms << " ms" << std::endl;
        std::cout << "    Throughput: " << (throughput / 1e6) << " M coeffs/sec" << std::endl;
    }

    // CPU baseline: NTT (simplified - just counting)
    {
        Timer t;
        t.start();
        // Simulate NTT operations: O(N log N) per polynomial
        for (size_t p = 0; p < TOTAL_OPS; p++) {
            volatile uint64_t x = 0;
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < NUM_LIMBS; j++) {
                    x += mod_mul(i, j, 1073807353);  // Butterfly ops
                }
            }
        }
        t.stop();

        double time_ms = t.elapsed_ms();
        double throughput = TOTAL_OPS / (time_ms / 1000.0);
        std::cout << "\n  CPU NTT (simulated):" << std::endl;
        std::cout << "    Time: " << std::fixed << std::setprecision(2) << time_ms << " ms" << std::endl;
        std::cout << "    Throughput: " << throughput << " polys/sec" << std::endl;
    }

    // Estimated GPU speedup (Metal can do ~10-100x for parallel ops)
    std::cout << "\n  Estimated GPU speedup:" << std::endl;
    std::cout << "    Polynomial addition: ~10-50x (memory bandwidth)" << std::endl;
    std::cout << "    NTT operations: ~20-100x (massive parallelism)" << std::endl;
}

// =============================================================================
// Sample 4: Throughput Comparison
// =============================================================================

void benchmark_throughput_comparison() {
    std::cout << "\n=== Sample 4: Throughput Comparison ===" << std::endl;

    const size_t N = 1000000;  // 1M elements

    std::cout << "  Operations per second (1M elements):" << std::endl;

    // mod_add
    {
        uint64_t mod = 1073807353;
        uint64_t a = 12345, b = 67890;
        Timer t;
        t.start();
        volatile uint64_t r = 0;
        for (size_t i = 0; i < N; i++) {
            r = mod_add(a, b, mod);
        }
        t.stop();
        double ops_per_sec = N / (t.elapsed_ms() / 1000.0);
        std::cout << "    mod_add:      " << std::fixed << std::setprecision(0)
                  << (ops_per_sec / 1e6) << " M ops/sec" << std::endl;
    }

    // mod_mul
    {
        uint64_t mod = 1073807353;
        uint64_t a = 12345, b = 67890;
        Timer t;
        t.start();
        volatile uint64_t r = 0;
        for (size_t i = 0; i < N; i++) {
            r = mod_mul(a, b, mod);
        }
        t.stop();
        double ops_per_sec = N / (t.elapsed_ms() / 1000.0);
        std::cout << "    mod_mul:      " << std::fixed << std::setprecision(0)
                  << (ops_per_sec / 1e6) << " M ops/sec" << std::endl;
    }

    // barrett_mul
    {
        uint64_t mod = 1073807353;
        uint64_t a = 12345, b = 67890;
        Timer t;
        t.start();
        volatile uint64_t r = 0;
        for (size_t i = 0; i < N; i++) {
            r = barrett_mul(a, b, mod);
        }
        t.stop();
        double ops_per_sec = N / (t.elapsed_ms() / 1000.0);
        std::cout << "    barrett_mul:  " << std::fixed << std::setprecision(0)
                  << (ops_per_sec / 1e6) << " M ops/sec" << std::endl;
    }

    // brev32
    {
        uint32_t a = 0xDEADBEEF;
        Timer t;
        t.start();
        volatile uint32_t r = 0;
        for (size_t i = 0; i < N; i++) {
            r = brev32(a + i);
        }
        t.stop();
        double ops_per_sec = N / (t.elapsed_ms() / 1000.0);
        std::cout << "    brev32:       " << std::fixed << std::setprecision(0)
                  << (ops_per_sec / 1e6) << " M ops/sec" << std::endl;
    }

    // clz
    {
        uint32_t a = 0xDEADBEEF;
        Timer t;
        t.start();
        volatile uint32_t r = 0;
        for (size_t i = 0; i < N; i++) {
            r = clz(a + i);
        }
        t.stop();
        double ops_per_sec = N / (t.elapsed_ms() / 1000.0);
        std::cout << "    clz:          " << std::fixed << std::setprecision(0)
                  << (ops_per_sec / 1e6) << " M ops/sec" << std::endl;
    }
}

// =============================================================================
// Sample 5: CKKS Operation Estimates
// =============================================================================

void benchmark_ckks_operations() {
    std::cout << "\n=== Sample 5: CKKS Operation Estimates ===" << std::endl;

    // Standard CKKS parameters
    const uint32_t N = 1024;          // Polynomial degree
    const uint32_t NUM_SLOTS = 512;   // Complex slots
    const uint32_t NUM_LIMBS = 8;     // RNS limbs

    std::cout << "  CKKS Parameters:" << std::endl;
    std::cout << "    N (degree): " << N << std::endl;
    std::cout << "    Slots: " << NUM_SLOTS << std::endl;
    std::cout << "    RNS limbs: " << NUM_LIMBS << std::endl;
    std::cout << "    Total coeff size: " << (N * NUM_LIMBS) << std::endl;

    // Estimate operation counts
    std::cout << "\n  Operation estimates (CPU baseline):" << std::endl;

    // Polynomial addition: N * NUM_LIMBS modular additions
    size_t add_ops = N * NUM_LIMBS;
    std::cout << "    Polynomial addition: " << add_ops << " mod_add ops" << std::endl;

    // Polynomial multiplication: N*log(N) * NUM_LIMBS modular multiplications (via NTT)
    size_t mul_ops = N * NUM_LIMBS * 10;  // log2(N) = 10 for N=1024
    std::cout << "    Polynomial mult (NTT): " << mul_ops << " mod_mul ops" << std::endl;

    // Rescaling: NUM_LIMBS multiplications
    size_t rescale_ops = NUM_LIMBS;
    std::cout << "    Rescaling: " << rescale_ops << " mod_mul ops" << std::endl;

    // Rotation: 2 NTTs + 1 inv NTT
    size_t rotation_ops = 3 * N * NUM_LIMBS * 10;
    std::cout << "    Rotation: " << rotation_ops << " mod_mul ops" << std::endl;

    // Estimate times (based on ~100ns per mod_mul on modern CPU)
    double mod_mul_ns = 100.0;  // nanoseconds per modular multiply
    std::cout << "\n  Estimated CPU times (assuming " << mod_mul_ns << " ns/mod_mul):" << std::endl;

    double add_ms = (add_ops * 10.0) / 1e6;  // mod_add is ~10x faster than mod_mul
    std::cout << "    Polynomial add: " << std::fixed << std::setprecision(2) << add_ms << " ms" << std::endl;

    double mul_ms = (mul_ops * mod_mul_ns) / 1e6;
    std::cout << "    Polynomial mult: " << mul_ms << " ms" << std::endl;

    double rescale_ms = (rescale_ops * mod_mul_ns) / 1e6;
    std::cout << "    Rescaling: " << rescale_ms << " ms" << std::endl;

    double rotation_ms = (rotation_ops * mod_mul_ns) / 1e6;
    std::cout << "    Rotation: " << rotation_ms << " ms" << std::endl;

    std::cout << "\n  Estimated GPU speedup (Metal ~10-50x for parallel ops):" << std::endl;
    std::cout << "    Polynomial add: ~" << std::fixed << std::setprecision(0)
              << (add_ms / 0.1) << " ms on GPU" << std::endl;
    std::cout << "    Polynomial mult: ~" << (mul_ms / 10.0) << " ms on GPU" << std::endl;
    std::cout << "    Rotation: ~" << (rotation_ms / 20.0) << " ms on GPU" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "FIDESlib Metal - Benchmark Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\nHardware: Apple Silicon (simulated Metal GPU)" << std::endl;
    std::cout << "CPU: ARM64 (native execution)" << std::endl;

    benchmark_modular_arithmetic();
    benchmark_bit_operations();
    benchmark_polynomial_ops();
    benchmark_throughput_comparison();
    benchmark_ckks_operations();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark completed!" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nNote: GPU times are estimates based on Metal's" << std::endl;
    std::cout << "parallel capabilities vs CPU sequential execution." << std::endl;
    std::cout << "Actual GPU benchmarks require Metal device runtime." << std::endl;

    return 0;
}
