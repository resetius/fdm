#pragma once

#include "instance.h"

namespace NVulkan {

class Surface {
public:
    VkSurfaceKHR surface_;
};

class PhyDevice {
public:
    PhyDevice(Instance& instance, int id);
    PhyDevice(Instance& instance, Surface* surface, int id);

    int computeFamily() const {
        return computeFamily_;
    }

    int graphicsFamily() const {
        return graphicsFamily_;
    }

    int presentFamily() const {
        return presentFamily_;
    }

    auto& dev() {
        return dev_;
    }

private:
    void init();
    void initQueueFamilies();
        
    Instance& instance_;
    Surface* surface_ = nullptr;
    int id_;
    VkPhysicalDevice dev_;
    VkPhysicalDeviceProperties properties_;
    VkPhysicalDeviceFeatures features_;
    VkPhysicalDeviceMemoryProperties memoryProperties_;

    int computeFamily_ = -1;
    int graphicsFamily_ = -1;
    int presentFamily_ = -1;
};

class Device {
public:
    Device(PhyDevice& dev);

    VkDevice& dev() {
        return dev_;
    }

private:    
    PhyDevice phyDev_;
    VkDevice dev_;
};

} /* NVulkan */
