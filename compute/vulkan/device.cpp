#include "device.h"

#include <stdexcept>
#include <vector>
#include <set>

namespace NVulkan {

PhyDevice::PhyDevice(Instance& instance, int id)
    : instance_(instance)
    , id_(id)
{
    init();
}

PhyDevice::PhyDevice(Instance& instance, Surface* surface, int id)
    : instance_(instance)
    , surface_(surface)
    , id_(id)
{
    init();
}

void PhyDevice::init()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_.instance(), &deviceCount, NULL);
    if (id_ > deviceCount) {
        throw std::runtime_error("Cannot create phydev");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_.instance(), &deviceCount, &devices[0]);
    dev_ = devices[id_];

    vkGetPhysicalDeviceProperties(dev_, &properties_);
    vkGetPhysicalDeviceFeatures(dev_, &features_);
	vkGetPhysicalDeviceMemoryProperties(dev_, &memoryProperties_);

    initQueueFamilies();
}

void PhyDevice::initQueueFamilies()
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev_, &count, NULL);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev_, &count, &families[0]);

    for (uint32_t i = 0; i < count; i++) {
        VkBool32 flag = 0;
        if (families[i].queueCount > 0 && (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            graphicsFamily_ = i;
        }

        if (families[i].queueCount > 0 && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            computeFamily_ = i;
        }

        if (vkGetPhysicalDeviceSurfaceSupportKHR && surface_) {
            vkGetPhysicalDeviceSurfaceSupportKHR(dev_, i, surface_->surface_, &flag);

            if (families[i].queueCount > 0 && flag) {
                presentFamily_ = i;
            }
        }

        if (presentFamily_ >= 0 && graphicsFamily_ >= 0 && computeFamily_ >= 0) {
            break;
        }
    }
}

Device::Device(PhyDevice& dev)
    : phyDev_(dev) 
{
    std::set<int> f;
    if (phyDev_.computeFamily() >= 0) {
        f.insert(phyDev_.computeFamily());
    }
    if (phyDev_.graphicsFamily() >= 0) {
        f.insert(phyDev_.graphicsFamily());
    }
    if (phyDev_.presentFamily() >= 0) {
        f.insert(phyDev_.presentFamily());
    }
    std::vector<int> families; 
    families.reserve(f.size());
    families.insert(families.end(), f.begin(), f.end());
    
    VkDeviceQueueCreateInfo queueCreateInfo[3];
    float queuePriority = 1.0f;
    for (uint32_t i = 0; i < families.size(); i++) {
        queueCreateInfo[i].queueFamilyIndex = families[i];
        VkDeviceQueueCreateInfo info = {            
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queueCreateInfo[i].queueFamilyIndex,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority
        };
        queueCreateInfo[i] = info;
    }

    VkPhysicalDeviceFeatures features = {
        .largePoints = VK_TRUE,
        .vertexPipelineStoresAndAtomics = VK_TRUE,        
    };

    VkDeviceCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = static_cast<uint32_t>(families.size()),
        .pQueueCreateInfos = &queueCreateInfo[0],
        .enabledLayerCount = 0,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr,
        .pEnabledFeatures = &features,
    };

    if (vkCreateDevice(phyDev_.dev(), &info, NULL, &dev_) != VK_SUCCESS) {
        throw std::runtime_error("Cannot create logical device");
    }

    vk_load_device(dev_);
}

} /* NVulkan */
