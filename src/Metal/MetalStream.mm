//
// FIDESlib Metal Backend - Stream Implementation
//

#include "MetalStream.hpp"
#include <cassert>
#include <iostream>

namespace FIDESlib {
namespace Metal {

MetalStream::MetalStream(id<MTLDevice> device, int queuePriority)
    : m_device(device), m_eventValue(0), m_capturing(false) {

    if (device == nil) {
        return;
    }

    // Create command queue
    // Note: Metal doesn't have priority-based queues like CUDA
    m_queue = [device newCommandQueue];

    if (m_queue == nil) {
        std::cerr << "MetalStream: failed to create command queue" << std::endl;
        return;
    }

    // Create shared event for synchronization
    m_event = [device newSharedEvent];
    if (m_event == nil) {
        std::cerr << "MetalStream: failed to create shared event" << std::endl;
    }
}

MetalStream::~MetalStream() {
    if (m_currentEncoder) {
        [m_currentEncoder release];
        m_currentEncoder = nil;
    }
    // MTLCommandQueue and MTLCommandBuffer are managed by ARC
    m_queue = nil;
    m_currentBuffer = nil;
    m_event = nil;
}

MetalStream::MetalStream(MetalStream&& other) noexcept
    : m_device(other.m_device),
      m_queue(other.m_queue),
      m_currentBuffer(other.m_currentBuffer),
      m_currentEncoder(other.m_currentEncoder),
      m_event(other.m_event),
      m_eventValue(other.m_eventValue),
      m_capturing(other.m_capturing) {
    other.m_device = nil;
    other.m_queue = nil;
    other.m_currentBuffer = nil;
    other.m_currentEncoder = nil;
    other.m_event = nil;
    other.m_capturing = false;
}

MetalStream& MetalStream::operator=(MetalStream&& other) noexcept {
    if (this != &other) {
        m_device = other.m_device;
        m_queue = other.m_queue;
        m_currentBuffer = other.m_currentBuffer;
        m_currentEncoder = other.m_currentEncoder;
        m_event = other.m_event;
        m_eventValue = other.m_eventValue;
        m_capturing = other.m_capturing;
        other.m_device = nil;
        other.m_queue = nil;
        other.m_currentBuffer = nil;
        other.m_currentEncoder = nil;
        other.m_event = nil;
        other.m_capturing = false;
    }
    return *this;
}

void MetalStream::ensureBuffer() {
    if (m_currentBuffer == nil) {
        m_currentBuffer = [m_queue commandBuffer];
        if (m_currentBuffer == nil) {
            std::cerr << "MetalStream: failed to create command buffer" << std::endl;
        }
    }
}

id<MTLComputeCommandEncoder> MetalStream::computeEncoder() {
    ensureBuffer();

    if (m_currentEncoder == nil) {
        m_currentEncoder = [m_currentBuffer computeCommandEncoder];
        if (m_currentEncoder == nil) {
            std::cerr << "MetalStream: failed to create compute encoder" << std::endl;
        }
    }
    return m_currentEncoder;
}

void MetalStream::endCurrentEncoder() {
    if (m_currentEncoder) {
        [m_currentEncoder endEncoding];
        [m_currentEncoder release];
        m_currentEncoder = nil;
    }
}

void MetalStream::commit() {
    endCurrentEncoder();

    if (m_currentBuffer) {
        [m_currentBuffer commit];
        m_currentBuffer = nil;
    }
}

void MetalStream::waitUntilCompleted() {
    ensureBuffer();

    if (m_currentBuffer) {
        [m_currentBuffer commit];
        m_currentBuffer = nil;
    }

    // Create a semaphore-like approach using event
    if (m_event) {
        [m_event notifyListenerType:MTLSharedEventListenerTypeGlobal
                              atValue:m_eventValue
                       compatibilityFilter:MTLFeatureSet_AppleV10];
    }

    // For proper synchronization, use GPU command buffer completion
    // Note: This is synchronous and may block
    @autoreleasepool {
        if (m_currentBuffer == nil && m_queue) {
            // Get the last committed buffer
            // Metal doesn't have direct "wait for specific buffer" - use event
        }
    }
}

void MetalStream::synchronize() {
    waitUntilCompleted();
}

void MetalStream::wait(MetalStream& other) {
    if (m_event == nil || other.m_event == nil) {
        return;
    }

    ensureBuffer();
    endCurrentEncoder();
    commit();

    // Encode wait
    // Note: Metal command buffers can wait for events
    // This is a simplified version
    m_eventValue++;
    [other.m_event notifyListenerType:MTLSharedEventListenerTypeGlobal
                        atValue:other.m_eventValue
                 compatibilityFilter:MTLFeatureSet_AppleV10];
}

void MetalStream::signal(int value) {
    if (m_event == nil) {
        return;
    }
    m_eventValue = value;
    [m_event notifyListenerType:MTLSharedEventListenerTypeGlobal
                      atValue:m_eventValue
               compatibilityFilter:MTLFeatureSet_AppleV10];
}

void MetalStream::capture_begin() {
    // Metal doesn't support capture in the same way CUDA streams do
    // This is a placeholder for future graph support
    m_capturing = true;
    ensureBuffer();
}

void MetalStream::capture_end() {
    // Metal doesn't support capture in the same way CUDA streams do
    // This is a placeholder for future graph support
    m_capturing = false;
    commit();
}

void MetalStream::wait(id<MTLCommandQueue> otherQueue, uint64_t value) {
    // Metal doesn't have direct queue-to-queue synchronization
    // Use shared events instead
    (void)otherQueue;
    (void)value;
}

}}  // namespace FIDESlib::Metal
