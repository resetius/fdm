#pragma once

#include <string>
#include <type_traits>
#include <utility>

#include "config.h"
#include "ns_cyl_boundary_lqr.h"

namespace fdm {

struct NSCylBoundaryLQRGainMetadata {
    int schema_version = 1;
    std::string format_name = "NSCyl physical boundary LQR gain";
    int step_operator_version = 2;
    std::string scalar_type;
    std::string fourier_layout = "samarskii_nikolaev_real_packed_v1";
    std::string state_layout =
        "staggered_radial_component_major_u_v_w_p_v1";
    std::string pressure_gauge =
        "weighted_radial_zero_mean_last_pressure_dependent_v1";
    std::string pressure_boundary = "radial_same_time_neumann_v1";
    std::string control_components;
    std::string config_text;

    int nr = 0;
    int nphi = 0;
    int nz = 0;
    int horizon_intervals = 0;
    int interval_steps = 0;

    double r = 0;
    double R = 0;
    double h1 = 0;
    double h2 = 0;
    double reynolds = 0;
    double dt = 0;
    double wall_speed = 0;
    double control_weight = 0;
    double ridge = 0;
};

template<typename T>
NSCylBoundaryLQRGainMetadata make_ns_cyl_boundary_lqr_gain_metadata(
    const Config& config, int horizon_intervals, int interval_steps,
    double control_weight, double ridge,
    const std::string& control_components) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    NSCylBoundaryLQRGainMetadata result;
    result.scalar_type = std::is_same_v<T, float> ? "float32" : "float64";
    result.control_components = control_components;
    config.print(result.config_text);
    result.nr = config.get("ns", "nr", 32);
    result.nphi = config.get("ns", "nphi", 32);
    result.nz = config.get("ns", "nz", 32);
    result.horizon_intervals = horizon_intervals;
    result.interval_steps = interval_steps;
    result.r = config.get("ns", "r", 1.5707963267948966);
    result.R = config.get("ns", "R", 3.1415926535897932);
    result.h1 = config.get("ns", "h1", 0.0);
    result.h2 = config.get("ns", "h2", 10.0);
    result.reynolds = config.get("ns", "Re", 1.0);
    result.dt = config.get("ns", "dt", 0.001);
    result.wall_speed = config.get("ns", "u0", 1.0);
    result.control_weight = control_weight;
    result.ridge = ridge;
    return result;
}

class NSCylBoundaryLQRGainStorage {
public:
    explicit NSCylBoundaryLQRGainStorage(std::string filename)
        : filename_(std::move(filename))
    { }

    void save(const NSCylBoundaryLQRGainSet<float>& gains,
              const NSCylBoundaryLQRGainMetadata& metadata) const;
    void save(const NSCylBoundaryLQRGainSet<double>& gains,
              const NSCylBoundaryLQRGainMetadata& metadata) const;

    void load(NSCylBoundaryLQRGainSet<float>& gains,
              NSCylBoundaryLQRGainMetadata& metadata) const;
    void load(NSCylBoundaryLQRGainSet<double>& gains,
              NSCylBoundaryLQRGainMetadata& metadata) const;
    void load(NSCylBoundaryLQRGainSet<float>& gains,
              NSCylBoundaryLQRGainMetadata& metadata,
              const NSCylBoundaryLQRGainMetadata& expected) const;
    void load(NSCylBoundaryLQRGainSet<double>& gains,
              NSCylBoundaryLQRGainMetadata& metadata,
              const NSCylBoundaryLQRGainMetadata& expected) const;

private:
    std::string filename_;

    template<typename T>
    void save_(const NSCylBoundaryLQRGainSet<T>& gains,
               const NSCylBoundaryLQRGainMetadata& metadata) const;
    template<typename T>
    void load_(NSCylBoundaryLQRGainSet<T>& gains,
               NSCylBoundaryLQRGainMetadata& metadata,
               const NSCylBoundaryLQRGainMetadata* expected) const;
};

} // namespace fdm
