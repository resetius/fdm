#pragma once

extern "C" {
#include "vk.h"
}

namespace NVulkan {

class Lib {
public:
    Lib();
    ~Lib();
};

class Instance {
public:
    Instance();

    auto& instance() {
        return instance_;
    }

private:
    VkInstance instance_;
};

}

