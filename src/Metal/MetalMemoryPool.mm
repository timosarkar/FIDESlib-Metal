//
// FIDESlib Metal Backend - Memory Pool Implementation
//

#include "MetalMemoryPool.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

namespace FIDESlib {
namespace Metal {

MetalMemoryPool::MetalMemoryPool(id<MTLDevice> device) : m_device(device) {}

// Singleton instance per device
static std::map<void*, MetalMemoryPool*> s_pools;
static std::mutex s_poolMapLock;

MetalMemoryPool& MetalMemoryPool::getInstance(id<MTLDevice> device) {
    std::lock_guard<std::mutex> lock(s_poolMapLock);

    void* key = (__bridge void*)device;
    auto it = s_pools.find(key);

    if (it == s_pools.end()) {
        static MetalMemoryPool instance(device);
        s_pools[key] = &instance;
        return instance;
    }

    return *(it->second);
}

void* MetalMemoryPool::alloc(size_t bytes) {
    std::lock_guard<std::mutex> lock(m_lock);

    // For very small allocations, use power-of-two pooling
    if (bytes < POOL_BLOCK_SIZE) {
        // Round up to next power of 2
        size_t nextPow2 = 1024;
        while (nextPow2 < bytes) {
            nextPow2 *= 2;
        }
        bytes = nextPow2;

        auto& freeList = m_freeList[bytes];
        if (!freeList.empty()) {
            void* ptr = freeList.back();
            freeList.pop_back();
            return ptr;
        }

        // Allocate a large block and sub-allocate
        // For simplicity, allocate directly for now
        id<MTLBuffer> buffer = [m_device newBufferWithLength:MAX_POOL_BLOCKS * bytes
                                                     options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            return nullptr;
        }

        void* basePtr = buffer.contents;
        size_t blockBytes = MAX_POOL_BLOCKS * bytes;

        // Add all sub-blocks to free list
        for (size_t i = 0; i < MAX_POOL_BLOCKS; i++) {
            freeList.push_back(static_cast<char*>(basePtr) + i * bytes);
        }

        // Store buffer for later cleanup
        m_allocations[basePtr] = blockBytes;

        // Return first block
        void* ptr = freeList.back();
        freeList.pop_back();
        return ptr;
    }

    // For large allocations, directly allocate
    id<MTLBuffer> buffer = [m_device newBufferWithLength:bytes
                                                 options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        return nullptr;
    }

    void* ptr = buffer.contents;
    m_allocations[ptr] = bytes;
    return ptr;
}

void MetalMemoryPool::free(void* ptr, size_t bytes) {
    if (ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_lock);

    // For pooled allocations, return to free list
    if (bytes < POOL_BLOCK_SIZE) {
        size_t nextPow2 = 1024;
        while (nextPow2 < bytes) {
            nextPow2 *= 2;
        }
        bytes = nextPow2;

        m_freeList[bytes].push_back(ptr);
        return;
    }

    // For direct allocations, we would need to track the buffer
    // For now, just remove from allocations
    m_allocations.erase(ptr);
}

void* MetalMemoryPool::allocAligned(size_t bytes, size_t alignment) {
    // Standard aligned allocation
    void* ptr = alloc(bytes + alignment);
    if (ptr == nullptr) {
        return nullptr;
    }

    // Align pointer
    size_t alignMask = alignment - 1;
    void* alignedPtr = reinterpret_cast<void*>((reinterpret_cast<size_t>(ptr) + alignMask) & ~alignMask);

    // Store original pointer for freeing
    // In a real implementation, you'd track the offset separately
    return alignedPtr;
}

void MetalMemoryPool::synchronize() {
    // Metal command buffer completion is handled via callbacks
    // This is a no-op for the pool itself
}

}}  // namespace FIDESlib::Metal
