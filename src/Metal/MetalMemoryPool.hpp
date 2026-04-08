//
// FIDESlib Metal Backend - Memory Pool
// Async memory pool for Metal GPU allocations.
//

#ifndef FIDESLIB_METAL_MEMORY_POOL_HPP
#define FIDESLIB_METAL_MEMORY_POOL_HPP

#import <Metal/Metal.h>
#include <map>
#include <mutex>
#include <vector>
#include <cstddef>

namespace FIDESlib {
namespace Metal {

class MetalMemoryPool {
public:
    static MetalMemoryPool& getInstance(id<MTLDevice> device);

    // Disable copy/move
    MetalMemoryPool(const MetalMemoryPool&) = delete;
    MetalMemoryPool& operator=(const MetalMemoryPool&) = delete;
    MetalMemoryPool(MetalMemoryPool&&) = delete;
    MetalMemoryPool& operator=(MetalMemoryPool&&) = delete;

    // Allocation
    void* alloc(size_t bytes);
    void free(void* ptr, size_t bytes);

    // Aligned allocation for FHE (requires 16-byte alignment)
    void* allocAligned(size_t bytes, size_t alignment = 16);

    // Synchronization
    void synchronize();

    id<MTLDevice> getDevice() const { return m_device; }

private:
    MetalMemoryPool(id<MTLDevice> device);

    id<MTLDevice> m_device;

    // Memory pool: size -> list of free blocks
    std::map<size_t, std::vector<void*>> m_freeList;
    std::map<void*, size_t> m_allocations;  // ptr -> size for freeing

    std::mutex m_lock;

    // Default pool size for small allocations
    static constexpr size_t POOL_BLOCK_SIZE = 64 * 1024;  // 64KB
    static constexpr size_t MAX_POOL_BLOCKS = 1024;
};

// RAII wrapper for pooled memory
class MetalMemoryBlock {
public:
    MetalMemoryBlock() : m_ptr(nullptr), m_size(0), m_pool(nullptr) {}
    MetalMemoryBlock(void* ptr, size_t size, MetalMemoryPool* pool)
        : m_ptr(ptr), m_size(size), m_pool(pool) {}

    ~MetalMemoryBlock() {
        if (m_ptr && m_pool) {
            m_pool->free(m_ptr, m_size);
        }
    }

    void* get() const { return m_ptr; }
    size_t size() const { return m_size; }
    explicit operator bool() const { return m_ptr != nullptr; }

private:
    void* m_ptr;
    size_t m_size;
    MetalMemoryPool* m_pool;
};

}}  // namespace FIDESlib::Metal

#endif  // FIDESLIB_METAL_MEMORY_POOL_HPP
