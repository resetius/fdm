#pragma once

#include "device.h"

namespace NVulkan {

class CommandPool {
public:
    CommandPool(Device& dev, uint32_t family);
    ~CommandPool();

    VkCommandBuffer acquire();
    void reset();

private:
    Device& dev_;
    VkCommandPool pool_;
};

} // namespace NVulkan
