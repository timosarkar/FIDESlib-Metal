//
// FIDESlib Metal Backend - GPU Buffer Wrapper
// Wraps MTLBuffer for GPU memory management.
//

#ifndef FIDESLIB_METAL_BUFFER_HPP
#define FIDESLIB_METAL_BUFFER_HPP

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstddef>
#include <cstdint>

namespace FIDESlib {
namespace Metal {

class MetalBuffer {
public:
    MetalBuffer() : m_buffer(nil), m_size(0), m_device(nil) {}

    MetalBuffer(id<MTLDevice> device, size_t size, MTLResourceOptions options = MTLResourceStorageModeShared);

    MetalBuffer(id<MTLDevice> device, size_t size, const void* hostData,
               MTLResourceOptions options = MTLResourceStorageModeShared);

    ~MetalBuffer();

    // Non-copyable
    MetalBuffer(const MetalBuffer&) = delete;
    MetalBuffer& operator=(const MetalBuffer&) = delete;

    // Movable
    MetalBuffer(MetalBuffer&& other) noexcept;
    MetalBuffer& operator=(MetalBuffer&& other) noexcept;

    id<MTLBuffer> get() const { return m_buffer; }
    id<MTLBuffer> getBuffer() const { return m_buffer; }
    void* getContents() const { return m_buffer ? m_buffer.contents : nullptr; }
    size_t getSize() const { return m_size; }
    id<MTLDevice> getDevice() const { return m_device; }
    bool isValid() const { return m_buffer != nil; }

    // Synchronization
    void addCompletedHandler(MTLBufferHandler handler);

private:
    id<MTLDevice> m_device;
    id<MTLBuffer> m_buffer;
    size_t m_size;
};

// =============================================================================
// Buffer Views (typed accessors)
// =============================================================================

template <typename T>
class MetalBufferView {
public:
    MetalBufferView() : m_buffer(nil), m_offset(0), m_count(0) {}
    MetalBufferView(const MetalBuffer& buffer, size_t offset = 0, size_t count = 0)
        : m_buffer(buffer.getBuffer()), m_offset(offset), m_count(count == 0 ? buffer.getSize() / sizeof(T) : count) {}

    T* data() const {
        if (m_buffer == nil) return nullptr;
        return reinterpret_cast<T*>(static_cast<char*>(m_buffer.contents) + m_offset);
    }

    size_t count() const { return m_count; }
    size_t size() const { return m_count * sizeof(T); }

    MetalBufferView<T> slice(size_t offset, size_t count) const {
        MetalBufferView view;
        view.m_buffer = m_buffer;
        view.m_offset = m_offset + offset * sizeof(T);
        view.m_count = count;
        return view;
    }

private:
    id<MTLBuffer> m_buffer;
    size_t m_offset;
    size_t m_count;
};

using MetalBufferU32 = MetalBufferView<uint32_t>;
using MetalBufferU64 = MetalBufferView<uint64_t>;

}}  // namespace FIDESlib::Metal

#endif  // FIDESLIB_METAL_BUFFER_HPP
