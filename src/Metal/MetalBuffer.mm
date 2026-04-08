//
// FIDESlib Metal Backend - GPU Buffer Implementation
//

#include "MetalBuffer.hpp"
#include <cstring>
#include <stdexcept>

namespace FIDESlib {
namespace Metal {

MetalBuffer::MetalBuffer(id<MTLDevice> device, size_t size, MTLResourceOptions options)
    : m_device(device), m_size(size) {
    if (device == nil) {
        return;
    }
    m_buffer = [device newBufferWithLength:size options:options];
    if (m_buffer == nil) {
        throw std::runtime_error("MetalBuffer: failed to allocate " + std::to_string(size) + " bytes");
    }
}

MetalBuffer::MetalBuffer(id<MTLDevice> device, size_t size, const void* hostData, MTLResourceOptions options)
    : m_device(device), m_size(size) {
    if (device == nil) {
        return;
    }

    // For shared storage, we can directly copy
    if (options & MTLResourceStorageModeShared) {
        m_buffer = [device newBufferWithBytes:hostData length:size options:options];
    } else {
        // For private storage, allocate and fill viablit encoder
        m_buffer = [device newBufferWithLength:size options:options];
        if (m_buffer && hostData) {
            // Copy to shared shadow buffer, then blit
            // For simplicity, use shared for now
            memcpy(m_buffer.contents, hostData, size);
        }
    }
}

MetalBuffer::~MetalBuffer() {
    if (m_buffer) {
        // MTLBuffer is managed by ARC
        m_buffer = nil;
    }
}

MetalBuffer::MetalBuffer(MetalBuffer&& other) noexcept
    : m_device(other.m_device), m_buffer(other.m_buffer), m_size(other.m_size) {
    other.m_buffer = nil;
    other.m_size = 0;
}

MetalBuffer& MetalBuffer::operator=(MetalBuffer&& other) noexcept {
    if (this != &other) {
        m_device = other.m_device;
        m_buffer = other.m_buffer;
        m_size = other.m_size;
        other.m_buffer = nil;
        other.m_size = 0;
    }
    return *this;
}

void MetalBuffer::addCompletedHandler(MTLBufferHandler handler) {
    if (m_buffer) {
        [m_buffer addCompletedHandler:handler];
    }
}

}}  // namespace FIDESlib::Metal
