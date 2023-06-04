#pragma once

#include "device.h"
#include "command_pool.h"

namespace NVulkan {

class Compute {
public:
    Compute();

private:
    CommandPool commandPool_;
    VkCommandBuffer commandBuffer_;
    VkSemaphore semaphore_;
};

} // namespace NVulkan
