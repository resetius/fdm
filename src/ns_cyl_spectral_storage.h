#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "config.h"
#include "ns_cyl_spectral_modes.h"

namespace fdm {

struct NSCylSpectralMetadata {
    int schema_version = 1;
    std::string operator_name = "NSCyl::L_step";
    int operator_version = 1;
    std::string scalar_type;
    std::string fourier_layout = "samarskii_nikolaev_real_packed_v1";
    std::string state_layout =
        "staggered_radial_component_major_u_v_w_p_v1";
    std::string pressure_gauge =
        "weighted_radial_zero_mean_last_pressure_dependent_v1";
    std::string config_text;

    int nr = 0;
    int nphi = 0;
    int nz = 0;
    int radial_size = 0;
    int u_offset = 0;
    int v_offset = 0;
    int w_offset = 0;
    int p_offset = 0;
    int operator_steps = 0;

    double r = 0;
    double R = 0;
    double h1 = 0;
    double h2 = 0;
    double reynolds = 0;
    double dt = 0;
    double wall_speed = 0;
    double growth_tolerance = 0;
    double residual_tolerance = 0;
    double condition_limit = 0;
};

template<typename T>
NSCylSpectralMetadata make_ns_cyl_spectral_metadata(const Config& config) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    NSCylSpectralMetadata result;
    result.scalar_type = std::is_same_v<T, float> ? "float32" : "float64";
    config.print(result.config_text);

    result.nr = config.get("ns", "nr", 32);
    result.nphi = config.get("ns", "nphi", 32);
    result.nz = config.get("ns", "nz", 32);
    result.radial_size = 4*result.nr-1;
    result.u_offset = 0;
    result.v_offset = result.nr-1;
    result.w_offset = 2*result.nr-1;
    result.p_offset = 3*result.nr-1;
    result.operator_steps = config.get("spectral", "operator_steps", 1);

    result.r = config.get("ns", "r", 1.5707963267948966);
    result.R = config.get("ns", "R", 3.1415926535897932);
    result.h1 = config.get("ns", "h1", 0.0);
    result.h2 = config.get("ns", "h2", 10.0);
    result.reynolds = config.get("ns", "Re", 1.0);
    result.dt = config.get("ns", "dt", 0.001);
    result.wall_speed = config.get("ns", "u0", 1.0);
    result.growth_tolerance = config.get(
        "spectral", "growth_tol", 1e-8);
    result.residual_tolerance = std::max(
        config.get("spectral", "residual_tol", 1e-10),
        64.0*static_cast<double>(std::numeric_limits<T>::epsilon()));
    result.condition_limit = config.get(
        "spectral", "condition_limit", 1e10);
    return result;
}

class NSCylSpectralStorage {
public:
    explicit NSCylSpectralStorage(std::string filename)
        : filename_(std::move(filename))
    { }

    void save(const NSCylSpectralModeSet<float>& modes,
              const NSCylSpectralMetadata& metadata) const;
    void save(const NSCylSpectralModeSet<double>& modes,
              const NSCylSpectralMetadata& metadata) const;

    void load(NSCylSpectralModeSet<float>& modes,
              NSCylSpectralMetadata& metadata) const;
    void load(NSCylSpectralModeSet<double>& modes,
              NSCylSpectralMetadata& metadata) const;

    void load(NSCylSpectralModeSet<float>& modes,
              NSCylSpectralMetadata& metadata,
              const NSCylSpectralMetadata& expected) const;
    void load(NSCylSpectralModeSet<double>& modes,
              NSCylSpectralMetadata& metadata,
              const NSCylSpectralMetadata& expected) const;

private:
    std::string filename_;

    template<typename T>
    void save_(const NSCylSpectralModeSet<T>& modes,
               const NSCylSpectralMetadata& metadata) const;

    template<typename T>
    void load_(NSCylSpectralModeSet<T>& modes,
               NSCylSpectralMetadata& metadata,
               const NSCylSpectralMetadata* expected) const;
};

} // namespace fdm
