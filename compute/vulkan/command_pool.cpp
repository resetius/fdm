#include "command_pool.h"

#include <stdexcept>

namespace NVulkan {

CommandPool::CommandPool(Device& dev, uint32_t family /* compute family*/)
    : dev_(dev)
{
    VkCommandPoolCreateInfo cpInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = family,
    };

    if (vkCreateCommandPool(dev.dev(), &cpInfo, nullptr, &pool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
}

CommandPool::~CommandPool()
{
    vkDestroyCommandPool(dev_.dev(), pool_, NULL);
}

VkCommandBuffer CommandPool::acquire()
{
    VkCommandBuffer buffer;
    VkCommandBufferAllocateInfo info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool_,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    if (vkAllocateCommandBuffers(dev_.dev(), &info, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers");
    }
    return buffer;
}

void CommandPool::reset()
{ }

} // namespace NVulkan
