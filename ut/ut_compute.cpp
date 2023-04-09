#include "compute/vulkan/shader.h"
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <iostream>

extern "C" {
#include <cmocka.h>
#include <compute/vulkan/vk.h>
}

#include <compute/vulkan/instance.h>
#include <compute/vulkan/device.h>
#include <compute/vulkan/shader.h>

const char* sourceDir = "../";

void test_vulkan_load_lib(void** ) {
    assert_true(vkGetInstanceProcAddr == NULL);
    vk_init();
    assert_true(vkGetInstanceProcAddr != NULL);
    vk_destroy();
}

void test_vulkan_load_l0(void** ) {
    vk_init();
    assert_true(vkCreateInstance == NULL);
    assert_true(vkEnumerateInstanceExtensionProperties == NULL);
    vk_load_global();
    assert_true(vkCreateInstance != NULL);
    assert_true(vkEnumerateInstanceExtensionProperties != NULL);
    vk_destroy();
}

void test_vulkan_load_l1(void** ) {
    vk_init();
    vk_load_global();

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

    VkInstance instance;
    assert_true(vkCreateInstance(&vkInstanceInfo, NULL, &instance) == VK_SUCCESS);

    assert_true(vkEnumeratePhysicalDevices == NULL);
    vk_load_instance(instance);
    assert_true(vkEnumeratePhysicalDevices != NULL);

    vk_destroy();
}

void test_vulkan_load_dev(void** ) {
    vk_init();
    vk_load_global();

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

    VkInstance instance;
    assert_true(vkCreateInstance(&vkInstanceInfo, NULL, &instance) == VK_SUCCESS);

    vk_load_instance(instance);

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);

    assert_true(deviceCount > 0);
    VkPhysicalDevice devices[100];
    if (deviceCount > sizeof(devices)/sizeof(VkPhysicalDevice)) {
        deviceCount = sizeof(devices)/sizeof(VkPhysicalDevice);
    }
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
    VkPhysicalDeviceProperties properties;
    for (uint32_t i = 0; i < deviceCount; i++) {
        vkGetPhysicalDeviceProperties(devices[i], &properties);
        std::cerr << "Name: " << i << " '" << properties.deviceName << "'\n";
    }

    VkPhysicalDevice phyDev = devices[0];
    VkPhysicalDeviceFeatures features;
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(phyDev, &memoryProperties);
    vkGetPhysicalDeviceFeatures(phyDev, &features);

    vk_destroy();
}

void test_vulkan_load_libpp(void** ) {
    NVulkan::Lib lib;
}

void test_vulkan_load_instance(void** ) {
    NVulkan::Lib lib;
    NVulkan::Instance instance;
}

void test_vulkan_load_phydev(void** ) {
    NVulkan::Lib lib;
    NVulkan::Instance instance;
    NVulkan::PhyDevice phyDev(instance, 0);
}

void test_vulkan_load_devpp(void** ) {
    NVulkan::Lib lib;
    NVulkan::Instance instance;
    NVulkan::PhyDevice phyDev(instance, 0);
}

void test_vulkan_load_logdev(void** ) {
    NVulkan::Lib lib;
    NVulkan::Instance instance;
    NVulkan::PhyDevice phyDev(instance, 0);
    NVulkan::Device dev(phyDev);
}

void test_shader_load(void** ) {
    NVulkan::Lib lib;
    NVulkan::Instance instance;
    NVulkan::PhyDevice phyDev(instance, 0);
    NVulkan::Device dev(phyDev);

    std::string file = sourceDir;
    file += "/compute/test_shader_1.comp";
    NVulkan::Shader(dev, file);
}

void test_shader_load_with_include(void** ) {
    NVulkan::Lib lib;
    NVulkan::Instance instance;
    NVulkan::PhyDevice phyDev(instance, 0);
    NVulkan::Device dev(phyDev);

    std::string file = sourceDir;
    file += "/compute/test_shader_2.comp";
    NVulkan::Shader(dev, file);
}

int main(int argc, char** argv) {    
    if (argc > 1) {
        sourceDir = argv[1];
    }

    const struct CMUnitTest tests[] = {
        /*cmocka_unit_test(test_vulkan_load_lib),
        cmocka_unit_test(test_vulkan_load_l0),
        cmocka_unit_test(test_vulkan_load_l1),
        cmocka_unit_test(test_vulkan_load_dev),
        cmocka_unit_test(test_vulkan_load_libpp),
        cmocka_unit_test(test_vulkan_load_instance),
        cmocka_unit_test(test_vulkan_load_phydev),
        cmocka_unit_test(test_vulkan_load_devpp),
        cmocka_unit_test(test_vulkan_load_logdev),
        cmocka_unit_test(test_shader_load),*/
        cmocka_unit_test(test_shader_load_with_include)
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
