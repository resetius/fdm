#include "command_pool.h"

#include <stdexcept>

namespace NVulkan {

CommandPool::CommandPool(Device& dev, uint32_t family /* compute family*/)
    : dev_(dev)
{
    VkCommandPoolCreateInfo cpInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = family,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
    };

    if (vkCreateCommandPool(dev.dev(), &cpInfo, nullptr, &pool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }
}

VkCommandBuffer CommandPool::acquire()
{
    return {};
}

void CommandPool::reset()
{ }

} // namespace NVulkan
