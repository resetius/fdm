#pragma once

#include "device.h"
#include <string>

namespace NVulkan {

enum class EShaderLang {
    GLSL = 0
};

enum class EShaderStage {
    COMPUTE = 0
};

class Shader {
public:
    Shader(Device& dev, const std::string& file, EShaderLang lang = EShaderLang::GLSL, EShaderStage stage = EShaderStage::COMPUTE);

    VkShaderModule get() {
        return shader_;
    }

private:
    VkShaderModule shader_;
};

}
