//
// FIDESlib Metal Backend - Stream / Command Queue Wrapper
// Wraps MTLCommandQueue and MTLCommandBuffer for GPU operations.
//

#ifndef FIDESLIB_METAL_STREAM_HPP
#define FIDESLIB_METAL_STREAM_HPP

#import <Metal/Metal.h>
#include <functional>
#include <atomic>

namespace FIDESlib {
namespace Metal {

class MetalStream {
public:
    MetalStream(id<MTLDevice> device, int queuePriority = 0);
    ~MetalStream();

    // Non-copyable
    MetalStream(const MetalStream&) = delete;
    MetalStream& operator=(const MetalStream&) = delete;

    // Movable
    MetalStream(MetalStream&& other) noexcept;
    MetalStream& operator=(MetalStream&& other) noexcept;

    // Accessors
    id<MTLDevice> getDevice() const { return m_device; }
    id<MTLCommandQueue> getQueue() const { return m_queue; }
    id<MTLCommandBuffer> getCurrentBuffer() const { return m_currentBuffer; }

    // Command encoding
    id<MTLComputeCommandEncoder> computeEncoder();
    void endCurrentEncoder();

    // Synchronization
    void commit();
    void waitUntilCompleted();
    void synchronize();

    // Event-based synchronization
    void wait(MetalStream& other);
    void signal(int value);

    // Graph capture support
    void capture_begin();
    void capture_end();
    bool isCapturing() const { return m_capturing; }

    // Stream wait on raw command queue
    void wait(id<MTLCommandQueue> otherQueue, uint64_t value);

private:
    void ensureBuffer();

    id<MTLDevice> m_device;
    id<MTLCommandQueue> m_queue;
    id<MTLCommandBuffer> m_currentBuffer;
    id<MTLComputeCommandEncoder> m_currentEncoder;
    id<MTLSharedEvent> m_event;
    uint64_t m_eventValue;
    bool m_capturing = false;
};

// =============================================================================
// Stream Synchronization RAII Wrapper
// =============================================================================

class MetalStreamSync {
public:
    explicit MetalStreamSync(MetalStream& stream) : m_stream(stream) {
        m_stream.commit();
    }
    ~MetalStreamSync() {
        m_stream.waitUntilCompleted();
    }

private:
    MetalStream& m_stream;
};

}}  // namespace FIDESlib::Metal

#endif  // FIDESLIB_METAL_STREAM_HPP
