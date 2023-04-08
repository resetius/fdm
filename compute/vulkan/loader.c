#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>

#include <assert.h>
#include <stdio.h>

#define DECL_FUNC(name)                         \
    PFN_##name name;                            \
    PFN_##name ktx_##name;


#define L0_FUNC(name) DECL_FUNC(name)
#define L1_FUNC(name) DECL_FUNC(name)
#define L2_FUNC(name) DECL_FUNC(name)

#include "symbols.h"

#undef DECL_FUNC
#undef L0_FUNC
#undef L1_FUNC
#undef L2_FUNC

static void* handle = NULL;
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = NULL;

void vk_init() {
    handle = dlopen("libvulkan.so", RTLD_GLOBAL | RTLD_NOW);
    vkGetInstanceProcAddr = dlsym(handle, "vkGetInstanceProcAddr");
}

void vk_destroy() {
    dlclose(handle);
}

void vk_load_global() {
#define L0_FUNC(name) \
    ktx_##name = name = (PFN_##name)vkGetInstanceProcAddr(NULL, #name); \
    if (!name) { \
        fprintf(stderr, "Warn: cannot load '%s'\n", #name); \
    }

#include "symbols.h"

#undef L0_FUNC
}

void vk_load_instance(VkInstance instance) {
#define L1_FUNC(name) \
    ktx_##name = name = (PFN_##name)vkGetInstanceProcAddr(instance, #name); \
    if (!name) { \
        fprintf(stderr, "Warn: cannot load '%s'\n", #name); \
    }

#include "symbols.h"

#undef L1_FUNC
}

void vk_load_device(VkDevice device) {
#define L2_FUNC(name) \
    ktx_##name = name = (PFN_##name)vkGetDeviceProcAddr(device, #name); \
    if (!name) { \
        fprintf(stderr, "Warn: cannot load '%s'\n", #name); \
    }

#include "symbols.h"

#undef L2_FUNC
}
