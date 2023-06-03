#pragma once

#include "device.h"

namespace NVulkan {

class Compute {
public:
    Compute();

private:
    VkCommandBuffer commandBuffer_;
    VkSemaphore semaphore_;
};

} // namespace NVulkan
