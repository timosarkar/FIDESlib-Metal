//
// FIDESlib Metal Backend - Device Management Implementation
//

#include "MetalDevice.hpp"
#include <iostream>

namespace FIDESlib {
namespace Metal {

MetalDevice::MetalDevice() : m_device(nil), m_deviceIndex(-1) {}

MetalDevice::MetalDevice(id<MTLDevice> device) : m_device(device), m_deviceIndex(-1) {
    initProperties();
}

MetalDevice::MetalDevice(int deviceIndex) : m_deviceIndex(deviceIndex) {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (deviceIndex < 0 || deviceIndex >= (int)devices.count) {
        m_device = nil;
        return;
    }
    m_device = devices[deviceIndex];
    initProperties();
}

MetalDevice::~MetalDevice() {
    // MTLDevice is Objective-C object, managed by ARC
    m_device = nil;
}

MetalDevice::MetalDevice(MetalDevice&& other) noexcept
    : m_device(other.m_device), m_deviceIndex(other.m_deviceIndex), m_properties(other.m_properties) {
    other.m_device = nil;
    other.m_deviceIndex = -1;
}

MetalDevice& MetalDevice::operator=(MetalDevice&& other) noexcept {
    if (this != &other) {
        m_device = other.m_device;
        m_deviceIndex = other.m_deviceIndex;
        m_properties = other.m_properties;
        other.m_device = nil;
        other.m_deviceIndex = -1;
    }
    return *this;
}

void MetalDevice::initProperties() {
    if (m_device == nil) {
        return;
    }

    m_properties.name = [m_device.name UTF8String] ? [m_device.name UTF8String] : "Unknown Metal Device";
    m_properties.maxThreadsPerThreadgroup = m_device.maxThreadsPerThreadgroup.width *
                                           m_device.maxThreadsPerThreadgroup.height *
                                           m_device.maxThreadsPerThreadgroup.depth;
    m_properties.maxThreadgroupMemory = m_device.maxThreadgroupMemory;
    m_properties.maxBufferSize = m_device.maxBufferLength;
    m_properties.computeUnits = 0;  // Apple Silicon doesn't expose this directly

    // Determine feature set
    m_properties.featureSet = MTLFeatureSet_AppleA11;
    if (@available(macOS 11.0, *)) {
        if ([m_device supportsFamily:MTLGPUFamilyAppleV10]) {
            m_properties.featureSet = MTLFeatureSet_AppleV10;
        } else if ([m_device supportsFamily:MTLGPUFamilyAppleV9]) {
            m_properties.featureSet = MTLFeatureSet_AppleV9;
        } else if ([m_device supportsFamily:MTLGPUFamilyAppleV8]) {
            m_properties.featureSet = MTLFeatureSet_AppleV8;
        }
    }
}

int MetalDevice::getNumDevices() {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        return (int)devices.count;
    }
}

MetalDevice MetalDevice::getDevice(int index) {
    return MetalDevice(index);
}

MetalDevice MetalDevice::getDefaultDevice() {
    @autoreleasepool {
        return MetalDevice(MTLCreateSystemDefaultDevice());
    }
}

// =============================================================================
// MetalDeviceRegistry
// =============================================================================

MetalDeviceRegistry& MetalDeviceRegistry::instance() {
    static MetalDeviceRegistry registry;
    return registry;
}

void MetalDeviceRegistry::initialize() {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        m_devices.reserve(devices.count);

        for (int i = 0; i < (int)devices.count; i++) {
            m_devices.emplace_back(devices[i]);
        }

        // Set default device to discrete GPU if available, otherwise system default
        m_defaultIndex = 0;
        for (int i = 0; i < (int)m_devices.size(); i++) {
            // Prefer discrete GPU (usually index 0 on multi-GPU systems)
            // For Apple Silicon, there's typically only one GPU
            if (i > 0) {
                // Could add logic here to prefer discrete over integrated
            }
            break;
        }

        std::cout << "Metal Devices:" << std::endl;
        for (int i = 0; i < (int)m_devices.size(); i++) {
            const auto& props = m_devices[i].getProperties();
            std::cout << "  [" << i << "] " << props.name
                      << " (maxThreads: " << props.maxThreadsPerThreadgroup
                      << ", threadgroupMem: " << props.maxThreadgroupMemory / 1024 << " KB"
                      << ", maxBuffer: " << props.maxBufferSize / (1024*1024) << " MB)"
                      << std::endl;
        }
    }
}

}}  // namespace FIDESlib::Metal
