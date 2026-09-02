#pragma once

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "config.h"

namespace fdm {

struct NSCylCheckpointMetadata {
    int schema_version = 1;
    std::string format_name = "NSCyl nonlinear checkpoint";
    int step_operator_version = 1;
    std::string scalar_type;
    std::string axial_boundary = "periodic";
    std::string state_layout =
        "staggered_component_major_u_v_w_p_v1";
    std::string pressure_gauge = "weighted_volume_zero_mean_v1";
    std::string config_text;

    int nr = 0;
    int nphi = 0;
    int nz = 0;
    int state_size = 0;
    int u_offset = 0;
    int v_offset = 0;
    int w_offset = 0;
    int p_offset = 0;
    int time_index = 0;

    double r = 0;
    double R = 0;
    double h1 = 0;
    double h2 = 0;
    double reynolds = 0;
    double dt = 0;
    double wall_speed = 0;
    double physical_time = 0;
};

template<typename T>
NSCylCheckpointMetadata make_ns_cyl_checkpoint_metadata(
    const Config& config, int time_index) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    NSCylCheckpointMetadata result;
    result.scalar_type = std::is_same_v<T, float> ? "float32" : "float64";
    config.print(result.config_text);

    result.nr = config.get("ns", "nr", 32);
    result.nphi = config.get("ns", "nphi", 32);
    result.nz = config.get("ns", "nz", 31);
    if (result.nr < 2 || result.nphi <= 0 || result.nz <= 0) {
        throw std::invalid_argument("invalid NSCyl checkpoint grid");
    }
    const long long plane_size =
        static_cast<long long>(result.nphi)*result.nz;
    const long long state_size = plane_size*(4LL*result.nr-1);
    if (state_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("NSCyl checkpoint grid is too large");
    }
    result.u_offset = 0;
    result.v_offset = static_cast<int>(plane_size*(result.nr-1));
    result.w_offset = static_cast<int>(
        result.v_offset+plane_size*result.nr);
    result.p_offset = static_cast<int>(
        result.w_offset+plane_size*result.nr);
    result.state_size = static_cast<int>(state_size);
    result.time_index = time_index;

    result.r = config.get("ns", "r", 1.5707963267948966);
    result.R = config.get("ns", "R", 3.1415926535897932);
    result.h1 = config.get("ns", "h1", 0.0);
    result.h2 = config.get("ns", "h2", 10.0);
    result.reynolds = config.get("ns", "Re", 1.0);
    result.dt = config.get("ns", "dt", 0.001);
    result.wall_speed = config.get("ns", "u0", 1.0);
    result.physical_time = time_index*result.dt;
    return result;
}

class NSCylCheckpointStorage {
public:
    explicit NSCylCheckpointStorage(std::string filename)
        : filename_(std::move(filename))
    { }

    void save(const std::vector<float>& state,
              const NSCylCheckpointMetadata& metadata) const;
    void save(const std::vector<double>& state,
              const NSCylCheckpointMetadata& metadata) const;

    void load(std::vector<float>& state,
              NSCylCheckpointMetadata& metadata) const;
    void load(std::vector<double>& state,
              NSCylCheckpointMetadata& metadata) const;

    void load(std::vector<float>& state,
              NSCylCheckpointMetadata& metadata,
              const NSCylCheckpointMetadata& expected) const;
    void load(std::vector<double>& state,
              NSCylCheckpointMetadata& metadata,
              const NSCylCheckpointMetadata& expected) const;

private:
    std::string filename_;

    template<typename T>
    void save_(const std::vector<T>& state,
               const NSCylCheckpointMetadata& metadata) const;

    template<typename T>
    void load_(std::vector<T>& state,
               NSCylCheckpointMetadata& metadata,
               const NSCylCheckpointMetadata* expected) const;
};

} // namespace fdm
