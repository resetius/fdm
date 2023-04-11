#include "instance.h"

#include <assert.h>
#include <stdexcept>
#include <vector>

#include <glslang/Include/glslang_c_interface.h>

namespace NVulkan {

Lib::Lib()
{
    glslang_initialize_process();
    vk_init();
    vk_load_global();
}

Lib::~Lib()
{
    vk_destroy();
    glslang_finalize_process();
}

Instance::Instance() {
    uint32_t allExtCount = 0;
    vkEnumerateInstanceExtensionProperties(NULL, &allExtCount, NULL);
    std::vector<VkExtensionProperties> exts(allExtCount);
    vkEnumerateInstanceExtensionProperties(NULL, &allExtCount, &exts[0]);

    int hasPortabilityEnumeration = 0;
    for (int i = 0; i < allExtCount; i++) {
        if (!strcmp(exts[i].extensionName, "VK_KHR_portability_enumeration")) {
            hasPortabilityEnumeration = 1;
        }
    }
    std::vector<const char*> extNames;
    extNames.push_back("VK_KHR_get_physical_device_properties2");
    if (hasPortabilityEnumeration) {
        extNames.push_back("VK_KHR_portability_enumeration");
    }

    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "fdm",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "fdm",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_MAKE_VERSION(1, 3, 0)
    };

    VkInstanceCreateInfo vkInstanceInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = NULL,
        .flags = hasPortabilityEnumeration
            ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
            : static_cast<VkInstanceCreateFlags>(0),
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = 0,
        .enabledExtensionCount = static_cast<uint32_t>(extNames.size()),
        .ppEnabledExtensionNames = &extNames[0]
    };

    if (vkCreateInstance(&vkInstanceInfo, NULL, &instance_) != VK_SUCCESS) {
        throw std::runtime_error("Cannot create instance");
    }

    vk_load_instance(instance_);
}

} /* NVulkan */
