#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

extern PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;

#define DECL_FUNC(name) \
    extern PFN_##name name;

#define L0_FUNC(name) DECL_FUNC(name)
#define L1_FUNC(name) DECL_FUNC(name)
#define L2_FUNC(name) DECL_FUNC(name)

#include "symbols.h"

#undef DECL_FUNC
#undef L0_FUNC
#undef L1_FUNC
#undef L2_FUNC

void vk_init();
void vk_destroy();
void vk_load_global();
void vk_load_instance(VkInstance instance);
void vk_load_device(VkDevice device);
