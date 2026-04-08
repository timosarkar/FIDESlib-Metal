//
// FIDESlib Metal Backend - Device Management
// Provides device enumeration and properties for Apple Metal GPUs.
//

#ifndef FIDESLIB_METAL_DEVICE_HPP
#define FIDESLIB_METAL_DEVICE_HPP

#include <metal/metal.h>
#include <vector>
#include <string>

namespace FIDESlib {
namespace Metal {

struct MetalDeviceProp {
    std::string name;
    uint32_t maxThreadsPerThreadgroup;
    uint32_t maxThreadgroupMemory;
    uint32_t maxBufferSize;
    uint32_t computeUnits;  // GPU family number
    MTLFeatureFeatureSet featureSet;
};

class MetalDevice {
public:
    MetalDevice();
    MetalDevice(id<MTLDevice> device);
    MetalDevice(int deviceIndex);
    ~MetalDevice();

    // Non-copyable
    MetalDevice(const MetalDevice&) = delete;
    MetalDevice& operator=(const MetalDevice&) = delete;

    // Movable
    MetalDevice(MetalDevice&& other) noexcept;
    MetalDevice& operator=(MetalDevice&& other) noexcept;

    id<MTLDevice> get() const { return m_device; }
    int getIndex() const { return m_deviceIndex; }
    const MetalDeviceProp& getProperties() const { return m_properties; }

    // Factory methods
    static int getNumDevices();
    static MetalDevice getDevice(int index);
    static MetalDevice getDefaultDevice();

private:
    void initProperties();

    id<MTLDevice> m_device = nil;
    int m_deviceIndex = -1;
    MetalDeviceProp m_properties;
};

// Global device registry
class MetalDeviceRegistry {
public:
    static MetalDeviceRegistry& instance();

    void initialize();
    int getDeviceCount() const { return m_devices.size(); }
    MetalDevice getDevice(int index) const { return m_devices[index]; }
    MetalDevice getDefaultDevice() const { return m_devices[m_defaultIndex]; }

private:
    MetalDeviceRegistry() = default;
    std::vector<MetalDevice> m_devices;
    int m_defaultIndex = 0;
};

}}  // namespace FIDESlib::Metal

#endif  // FIDESLIB_METAL_DEVICE_HPP
