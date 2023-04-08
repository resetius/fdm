#include "shader.h"

#include <memory>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

namespace NVulkan {

Shader::Shader(Device& dev, const std::string& file, EShaderLang lang, EShaderStage stage) {
    if (lang != EShaderLang::GLSL) {
        throw std::runtime_error("Unsupported shader lang");
    }

    if (stage != EShaderStage::COMPUTE) {
        throw std::runtime_error("Unsupported shader stage");
    }

    std::ifstream f(file);
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string shaderCode = ss.str();

    const glslang_input_t input =
    {
        .language = GLSLANG_SOURCE_GLSL,
        .stage = GLSLANG_STAGE_COMPUTE,
        .client = GLSLANG_CLIENT_VULKAN,
        .client_version = GLSLANG_TARGET_VULKAN_1_1,
        .target_language = GLSLANG_TARGET_SPV,
        .target_language_version = GLSLANG_TARGET_SPV_1_3,
        .code = shaderCode.c_str(),
        .default_version = 460,
        .default_profile = GLSLANG_CORE_PROFILE,
        .force_default_version_and_profile = false,
        .forward_compatible = false,
        .messages = GLSLANG_MSG_DEFAULT_BIT,
        .resource = glslang_default_resource()
    };

    // TODO: raplace with uniq_ptr on C++23
    auto shader = std::shared_ptr<glslang_shader_t>(glslang_shader_create(&input), glslang_shader_delete);
    if (!glslang_shader_preprocess(shader.get(), &input))
    {
        auto log = glslang_shader_get_info_log(shader.get());
        std::cerr << log << "\n";
        throw std::runtime_error("Cannot preprocess shader");
    }

    if (!glslang_shader_parse(shader.get(), &input))
    {
        auto log = glslang_shader_get_info_log(shader.get());
        std::cerr << log << "\n";
        throw std::runtime_error("Cannot parse shader");
    }

    auto program = std::shared_ptr<glslang_program_t>(glslang_program_create(), glslang_program_delete);
    glslang_program_add_shader(program.get(), shader.get());

    if (!glslang_program_link(program.get(), GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT))
    {
        auto log = glslang_program_get_info_log(program.get());
        std::cerr << log << "\n";
        throw std::runtime_error("Cannot link shader");
    }

    glslang_program_SPIRV_generate(program.get(), input.stage);
    if (glslang_program_SPIRV_get_messages(program.get()))
    {
        std::cerr << glslang_program_SPIRV_get_messages(program.get()) << "\n";
    }

    const VkShaderModuleCreateInfo info =
    {
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = glslang_program_SPIRV_get_size(program.get()) * sizeof(unsigned int),
        .pCode    = glslang_program_SPIRV_get_ptr(program.get())
    };

    if (vkCreateShaderModule(dev.dev(), &info, NULL, &shader_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }
}

}
