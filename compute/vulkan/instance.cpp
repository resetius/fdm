#include "instance.h"

#include <assert.h>
#include <stdexcept>

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
        .flags = 0,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = 0,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr
    };

    if (vkCreateInstance(&vkInstanceInfo, NULL, &instance_) != VK_SUCCESS) {
        throw std::runtime_error("Cannot create instance");
    }

    vk_load_instance(instance_);
}

} /* NVulkan */
