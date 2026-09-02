#include "ns_cyl_spectral_storage.h"

#include <netcdf.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ns_cyl_spectral_projector.h"

namespace fdm {
namespace {

constexpr int schema_version = 1;

void nc_check(int code, const std::string& operation) {
    if (code != NC_NOERR) {
        throw std::runtime_error(operation+": "+nc_strerror(code));
    }
}

class NcFile {
public:
    explicit NcFile(int id)
        : id_(id)
    { }

    ~NcFile() {
        if (id_ >= 0) {
            nc_close(id_);
        }
    }

    int id() const {
        return id_;
    }

    void close() {
        if (id_ >= 0) {
            const int id = std::exchange(id_, -1);
            nc_check(nc_close(id), "closing NetCDF spectral file");
        }
    }

private:
    int id_;
};

void put_text_attribute(int ncid, const char* name, const std::string& value) {
    nc_check(nc_put_att_text(ncid, NC_GLOBAL, name, value.size(),
                             value.data()),
             std::string("writing attribute ")+name);
}

void put_int_attribute(int ncid, const char* name, int value) {
    nc_check(nc_put_att_int(ncid, NC_GLOBAL, name, NC_INT, 1, &value),
             std::string("writing attribute ")+name);
}

void put_double_attribute(int ncid, const char* name, double value) {
    nc_check(nc_put_att_double(ncid, NC_GLOBAL, name, NC_DOUBLE, 1, &value),
             std::string("writing attribute ")+name);
}

std::string get_text_attribute(int ncid, const char* name) {
    std::size_t size = 0;
    nc_check(nc_inq_attlen(ncid, NC_GLOBAL, name, &size),
             std::string("reading required attribute ")+name);
    std::string result(size, '\0');
    if (size != 0) {
        nc_check(nc_get_att_text(ncid, NC_GLOBAL, name, result.data()),
                 std::string("reading attribute ")+name);
    }
    return result;
}

int get_int_attribute(int ncid, const char* name) {
    int value = 0;
    nc_check(nc_get_att_int(ncid, NC_GLOBAL, name, &value),
             std::string("reading required attribute ")+name);
    return value;
}

double get_double_attribute(int ncid, const char* name) {
    double value = 0;
    nc_check(nc_get_att_double(ncid, NC_GLOBAL, name, &value),
             std::string("reading required attribute ")+name);
    return value;
}

void write_metadata(int ncid, const NSCylSpectralMetadata& metadata,
                    int mode_count, int value_count) {
    put_int_attribute(ncid, "schema_version", metadata.schema_version);
    put_text_attribute(ncid, "operator_name", metadata.operator_name);
    put_int_attribute(ncid, "operator_version", metadata.operator_version);
    put_text_attribute(ncid, "scalar_type", metadata.scalar_type);
    put_text_attribute(ncid, "fourier_layout", metadata.fourier_layout);
    put_text_attribute(ncid, "state_layout", metadata.state_layout);
    put_text_attribute(ncid, "pressure_gauge", metadata.pressure_gauge);
    put_text_attribute(ncid, "config", metadata.config_text);

    put_int_attribute(ncid, "nr", metadata.nr);
    put_int_attribute(ncid, "nphi", metadata.nphi);
    put_int_attribute(ncid, "nz", metadata.nz);
    put_int_attribute(ncid, "radial_size", metadata.radial_size);
    put_int_attribute(ncid, "u_offset", metadata.u_offset);
    put_int_attribute(ncid, "v_offset", metadata.v_offset);
    put_int_attribute(ncid, "w_offset", metadata.w_offset);
    put_int_attribute(ncid, "p_offset", metadata.p_offset);
    put_int_attribute(ncid, "operator_steps", metadata.operator_steps);
    put_int_attribute(ncid, "mode_count", mode_count);
    put_int_attribute(ncid, "column_value_count", value_count);

    put_double_attribute(ncid, "r", metadata.r);
    put_double_attribute(ncid, "R", metadata.R);
    put_double_attribute(ncid, "h1", metadata.h1);
    put_double_attribute(ncid, "h2", metadata.h2);
    put_double_attribute(ncid, "Re", metadata.reynolds);
    put_double_attribute(ncid, "dt", metadata.dt);
    put_double_attribute(ncid, "wall_speed", metadata.wall_speed);
    put_double_attribute(ncid, "growth_tolerance",
                         metadata.growth_tolerance);
    put_double_attribute(ncid, "residual_tolerance",
                         metadata.residual_tolerance);
    put_double_attribute(ncid, "condition_limit", metadata.condition_limit);
}

NSCylSpectralMetadata read_metadata(int ncid) {
    int version = 0;
    const int version_status = nc_get_att_int(
        ncid, NC_GLOBAL, "schema_version", &version);
    if (version_status == NC_ENOTATT) {
        throw std::runtime_error(
            "not an NSCyl spectral file: missing schema_version");
    }
    nc_check(version_status, "reading required attribute schema_version");

    NSCylSpectralMetadata result;
    result.schema_version = version;
    result.operator_name = get_text_attribute(ncid, "operator_name");
    result.operator_version = get_int_attribute(ncid, "operator_version");
    result.scalar_type = get_text_attribute(ncid, "scalar_type");
    result.fourier_layout = get_text_attribute(ncid, "fourier_layout");
    result.state_layout = get_text_attribute(ncid, "state_layout");
    result.pressure_gauge = get_text_attribute(ncid, "pressure_gauge");
    result.config_text = get_text_attribute(ncid, "config");

    result.nr = get_int_attribute(ncid, "nr");
    result.nphi = get_int_attribute(ncid, "nphi");
    result.nz = get_int_attribute(ncid, "nz");
    result.radial_size = get_int_attribute(ncid, "radial_size");
    result.u_offset = get_int_attribute(ncid, "u_offset");
    result.v_offset = get_int_attribute(ncid, "v_offset");
    result.w_offset = get_int_attribute(ncid, "w_offset");
    result.p_offset = get_int_attribute(ncid, "p_offset");
    result.operator_steps = get_int_attribute(ncid, "operator_steps");

    result.r = get_double_attribute(ncid, "r");
    result.R = get_double_attribute(ncid, "R");
    result.h1 = get_double_attribute(ncid, "h1");
    result.h2 = get_double_attribute(ncid, "h2");
    result.reynolds = get_double_attribute(ncid, "Re");
    result.dt = get_double_attribute(ncid, "dt");
    result.wall_speed = get_double_attribute(ncid, "wall_speed");
    result.growth_tolerance = get_double_attribute(
        ncid, "growth_tolerance");
    result.residual_tolerance = get_double_attribute(
        ncid, "residual_tolerance");
    result.condition_limit = get_double_attribute(ncid, "condition_limit");
    return result;
}

template<typename T>
const char* scalar_type_name() {
    return std::is_same_v<T, float> ? "float32" : "float64";
}

template<typename T>
constexpr nc_type scalar_nc_type() {
    return std::is_same_v<T, float> ? NC_FLOAT : NC_DOUBLE;
}

void validate_metadata(const NSCylSpectralMetadata& metadata,
                       const char* scalar_type) {
    if (metadata.schema_version != schema_version) {
        throw std::runtime_error(
            "unsupported NSCyl spectral schema_version="
            +std::to_string(metadata.schema_version));
    }
    if (metadata.operator_name != "NSCyl::L_step"
        || metadata.operator_version != 1) {
        throw std::runtime_error("incompatible NSCyl spectral operator");
    }
    if (metadata.scalar_type != scalar_type) {
        throw std::runtime_error(
            "incompatible NSCyl spectral scalar type: file="
            +metadata.scalar_type+", requested="+scalar_type);
    }
    if (metadata.fourier_layout
            != "samarskii_nikolaev_real_packed_v1"
        || metadata.state_layout
            != "staggered_radial_component_major_u_v_w_p_v1"
        || metadata.pressure_gauge
            != "weighted_radial_zero_mean_last_pressure_dependent_v1") {
        throw std::runtime_error("incompatible NSCyl spectral layout");
    }
    if (metadata.nr <= 0 || metadata.nphi <= 0 || metadata.nz <= 0
        || metadata.nphi%2 != 0 || metadata.nz%2 != 0) {
        throw std::runtime_error("invalid NSCyl spectral grid metadata");
    }
    if (metadata.radial_size != 4*metadata.nr-1
        || metadata.u_offset != 0
        || metadata.v_offset != metadata.nr-1
        || metadata.w_offset != 2*metadata.nr-1
        || metadata.p_offset != 3*metadata.nr-1) {
        throw std::runtime_error("incompatible NSCyl spectral state offsets");
    }
    if (metadata.operator_steps <= 0 || !(metadata.dt > 0)
        || metadata.residual_tolerance < 0
        || !(metadata.condition_limit >= 1)) {
        throw std::runtime_error("invalid NSCyl spectral solver metadata");
    }
    const double finite_values[] = {
        metadata.r, metadata.R, metadata.h1, metadata.h2,
        metadata.reynolds, metadata.dt, metadata.wall_speed,
        metadata.growth_tolerance, metadata.residual_tolerance,
        metadata.condition_limit
    };
    for (double value : finite_values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite NSCyl spectral metadata");
        }
    }
    if (!(metadata.R > metadata.r) || !(metadata.h2 > metadata.h1)) {
        throw std::runtime_error("invalid NSCyl spectral geometry metadata");
    }
}

template<typename Value>
void require_equal(const char* name, const Value& actual,
                   const Value& expected) {
    if (actual != expected) {
        throw std::runtime_error(
            std::string("incompatible NSCyl spectral metadata: ")+name);
    }
}

void validate_compatibility(const NSCylSpectralMetadata& actual,
                            const NSCylSpectralMetadata& expected) {
    require_equal("schema_version", actual.schema_version,
                  expected.schema_version);
    require_equal("operator_name", actual.operator_name,
                  expected.operator_name);
    require_equal("operator_version", actual.operator_version,
                  expected.operator_version);
    require_equal("scalar_type", actual.scalar_type, expected.scalar_type);
    require_equal("fourier_layout", actual.fourier_layout,
                  expected.fourier_layout);
    require_equal("state_layout", actual.state_layout,
                  expected.state_layout);
    require_equal("pressure_gauge", actual.pressure_gauge,
                  expected.pressure_gauge);
    require_equal("nr", actual.nr, expected.nr);
    require_equal("nphi", actual.nphi, expected.nphi);
    require_equal("nz", actual.nz, expected.nz);
    require_equal("radial_size", actual.radial_size, expected.radial_size);
    require_equal("u_offset", actual.u_offset, expected.u_offset);
    require_equal("v_offset", actual.v_offset, expected.v_offset);
    require_equal("w_offset", actual.w_offset, expected.w_offset);
    require_equal("p_offset", actual.p_offset, expected.p_offset);
    require_equal("operator_steps", actual.operator_steps,
                  expected.operator_steps);
    require_equal("r", actual.r, expected.r);
    require_equal("R", actual.R, expected.R);
    require_equal("h1", actual.h1, expected.h1);
    require_equal("h2", actual.h2, expected.h2);
    require_equal("Re", actual.reynolds, expected.reynolds);
    require_equal("dt", actual.dt, expected.dt);
    require_equal("wall_speed", actual.wall_speed, expected.wall_speed);
    require_equal("growth_tolerance", actual.growth_tolerance,
                  expected.growth_tolerance);
    require_equal("residual_tolerance", actual.residual_tolerance,
                  expected.residual_tolerance);
    require_equal("condition_limit", actual.condition_limit,
                  expected.condition_limit);
}

int expected_phase_count(int m, int l,
                         const NSCylSpectralMetadata& metadata) {
    const int phi_phases = (m == 0 || 2*m == metadata.nphi) ? 1 : 2;
    const int z_phases = (l == 0 || 2*l == metadata.nz) ? 1 : 2;
    return phi_phases*z_phases;
}

template<typename T>
void validate_mode(const NSCylSpectralMode<T>& mode,
                   const NSCylSpectralMetadata& metadata) {
    if (!mode.filterable_unstable()) {
        throw std::runtime_error(
            "spectral file contains a non-filterable mode");
    }
    if (mode.m < 0 || mode.m > metadata.nphi/2
        || mode.l < 0 || mode.l > metadata.nz/2) {
        throw std::runtime_error("spectral mode index is outside the grid");
    }
    const int phase_count = expected_phase_count(mode.m, mode.l, metadata);
    const bool pressure_gauge_fixed = mode.m == 0 && mode.l == 0;
    const int block_size = metadata.radial_size*phase_count
        -(pressure_gauge_fixed ? 1 : 0);
    if (mode.phase_count != phase_count
        || mode.radial_size != metadata.radial_size
        || mode.block_size != block_size
        || mode.pressure_gauge_fixed != pressure_gauge_fixed) {
        throw std::runtime_error("incompatible spectral mode block layout");
    }
    const std::size_t value_count =
        static_cast<std::size_t>(mode.column_count)*mode.block_size;
    if (mode.column_count < 1 || mode.column_count > 2
        || mode.right_columns.size() != value_count
        || mode.left_columns.size() != value_count) {
        throw std::runtime_error("invalid spectral mode column storage");
    }
    if ((mode.column_count == 1 && mode.multiplier.imag() != T(0))
        || (mode.column_count == 2 && !(mode.multiplier.imag() > T(0)))) {
        throw std::runtime_error("invalid real/complex spectral mode grouping");
    }
    if (!(mode.growth_rate > metadata.growth_tolerance)
        || mode.right_residual > metadata.residual_tolerance
        || mode.left_residual > metadata.residual_tolerance) {
        throw std::runtime_error("spectral mode contradicts selection thresholds");
    }
    const double scalars[] = {
        static_cast<double>(mode.multiplier.real()),
        static_cast<double>(mode.multiplier.imag()),
        mode.growth_rate, mode.frequency,
        mode.right_residual, mode.left_residual
    };
    for (double value : scalars) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite spectral mode metadata");
        }
    }
    for (T value : mode.right_columns) {
        if (!std::isfinite(static_cast<double>(value))) {
            throw std::runtime_error("non-finite right spectral column");
        }
    }
    for (T value : mode.left_columns) {
        if (!std::isfinite(static_cast<double>(value))) {
            throw std::runtime_error("non-finite left spectral column");
        }
    }
}

int define_variable(int ncid, const char* name, nc_type type, int dimension) {
    int variable = -1;
    nc_check(nc_def_var(ncid, name, type, 1, &dimension, &variable),
             std::string("defining variable ")+name);
    return variable;
}

int require_dimension(int ncid, const char* name, std::size_t expected_size) {
    int dimension = -1;
    nc_check(nc_inq_dimid(ncid, name, &dimension),
             std::string("reading required dimension ")+name);
    std::size_t actual_size = 0;
    nc_check(nc_inq_dimlen(ncid, dimension, &actual_size),
             std::string("reading dimension ")+name);
    if (actual_size != expected_size) {
        throw std::runtime_error(
            std::string("invalid spectral dimension ")+name);
    }
    return dimension;
}

int require_variable(int ncid, const char* name, nc_type expected_type,
                     int expected_dimension) {
    int variable = -1;
    nc_check(nc_inq_varid(ncid, name, &variable),
             std::string("reading required variable ")+name);
    nc_type type = NC_NAT;
    int dimension_count = 0;
    int dimensions[NC_MAX_VAR_DIMS];
    nc_check(nc_inq_var(ncid, variable, nullptr, &type, &dimension_count,
                        dimensions, nullptr),
             std::string("reading variable definition ")+name);
    if (type != expected_type || dimension_count != 1
        || dimensions[0] != expected_dimension) {
        throw std::runtime_error(
            std::string("invalid spectral variable definition: ")+name);
    }
    return variable;
}

template<typename T>
void put_real_values(int ncid, int variable, const std::vector<T>& values) {
    if constexpr (std::is_same_v<T, float>) {
        nc_check(nc_put_var_float(ncid, variable, values.data()),
                 "writing real spectral values");
    } else {
        nc_check(nc_put_var_double(ncid, variable, values.data()),
                 "writing real spectral values");
    }
}

template<typename T>
void get_real_values(int ncid, int variable, std::vector<T>& values) {
    if constexpr (std::is_same_v<T, float>) {
        nc_check(nc_get_var_float(ncid, variable, values.data()),
                 "reading real spectral values");
    } else {
        nc_check(nc_get_var_double(ncid, variable, values.data()),
                 "reading real spectral values");
    }
}

struct Variables {
    int m;
    int l;
    int phase_count;
    int radial_size;
    int block_size;
    int pressure_gauge_fixed;
    int column_count;
    int value_offset;
    int growing;
    int residual_accepted;
    int multiplier_real;
    int multiplier_imag;
    int growth_rate;
    int frequency;
    int right_residual;
    int left_residual;
    int block_condition;
    int right_columns;
    int left_columns;
};

Variables define_variables(int ncid, int mode_dimension, int value_dimension,
                           nc_type real_type) {
    return {
        define_variable(ncid, "m", NC_INT, mode_dimension),
        define_variable(ncid, "l", NC_INT, mode_dimension),
        define_variable(ncid, "phase_count", NC_INT, mode_dimension),
        define_variable(ncid, "radial_size", NC_INT, mode_dimension),
        define_variable(ncid, "block_size", NC_INT, mode_dimension),
        define_variable(ncid, "pressure_gauge_fixed", NC_INT, mode_dimension),
        define_variable(ncid, "column_count", NC_INT, mode_dimension),
        define_variable(ncid, "value_offset", NC_INT, mode_dimension),
        define_variable(ncid, "growing", NC_INT, mode_dimension),
        define_variable(ncid, "residual_accepted", NC_INT, mode_dimension),
        define_variable(ncid, "multiplier_real", NC_DOUBLE, mode_dimension),
        define_variable(ncid, "multiplier_imag", NC_DOUBLE, mode_dimension),
        define_variable(ncid, "growth_rate", NC_DOUBLE, mode_dimension),
        define_variable(ncid, "frequency", NC_DOUBLE, mode_dimension),
        define_variable(ncid, "right_residual", NC_DOUBLE, mode_dimension),
        define_variable(ncid, "left_residual", NC_DOUBLE, mode_dimension),
        define_variable(ncid, "block_condition", NC_DOUBLE, mode_dimension),
        define_variable(ncid, "right_columns", real_type, value_dimension),
        define_variable(ncid, "left_columns", real_type, value_dimension)
    };
}

Variables require_variables(int ncid, int mode_dimension, int value_dimension,
                            nc_type real_type) {
    return {
        require_variable(ncid, "m", NC_INT, mode_dimension),
        require_variable(ncid, "l", NC_INT, mode_dimension),
        require_variable(ncid, "phase_count", NC_INT, mode_dimension),
        require_variable(ncid, "radial_size", NC_INT, mode_dimension),
        require_variable(ncid, "block_size", NC_INT, mode_dimension),
        require_variable(ncid, "pressure_gauge_fixed", NC_INT, mode_dimension),
        require_variable(ncid, "column_count", NC_INT, mode_dimension),
        require_variable(ncid, "value_offset", NC_INT, mode_dimension),
        require_variable(ncid, "growing", NC_INT, mode_dimension),
        require_variable(ncid, "residual_accepted", NC_INT, mode_dimension),
        require_variable(ncid, "multiplier_real", NC_DOUBLE, mode_dimension),
        require_variable(ncid, "multiplier_imag", NC_DOUBLE, mode_dimension),
        require_variable(ncid, "growth_rate", NC_DOUBLE, mode_dimension),
        require_variable(ncid, "frequency", NC_DOUBLE, mode_dimension),
        require_variable(ncid, "right_residual", NC_DOUBLE, mode_dimension),
        require_variable(ncid, "left_residual", NC_DOUBLE, mode_dimension),
        require_variable(ncid, "block_condition", NC_DOUBLE, mode_dimension),
        require_variable(ncid, "right_columns", real_type, value_dimension),
        require_variable(ncid, "left_columns", real_type, value_dimension)
    };
}

} // namespace

template<typename T>
void NSCylSpectralStorage::save_(
    const NSCylSpectralModeSet<T>& input_modes,
    const NSCylSpectralMetadata& metadata) const {
    validate_metadata(metadata, scalar_type_name<T>());
    NSCylSpectralModeSet<T> modes = input_modes;
    modes.sort_by_block_and_growth();
    const NSCylSpectralProjector<T> projector(
        modes, metadata.condition_limit);

    std::map<std::pair<int, int>, double> block_conditions;
    for (const auto& block : projector.blocks()) {
        block_conditions[{block.m(), block.l()}] = block.condition_number();
    }

    std::size_t value_count_size = 0;
    for (const auto& mode : modes.modes()) {
        validate_mode(mode, metadata);
        value_count_size += mode.right_columns.size();
        if (value_count_size
            > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("spectral column storage is too large");
        }
    }
    if (modes.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("too many spectral mode records");
    }
    const int mode_count = static_cast<int>(modes.size());
    const int value_count = static_cast<int>(value_count_size);
    const std::size_t mode_storage = std::max<std::size_t>(1, mode_count);
    const std::size_t value_storage = std::max<std::size_t>(1, value_count);

    int ncid = -1;
    nc_check(nc_create(filename_.c_str(), NC_CLOBBER | NC_64BIT_OFFSET,
                       &ncid),
             "creating NSCyl spectral file");
    NcFile file(ncid);
    write_metadata(ncid, metadata, mode_count, value_count);

    int mode_dimension = -1;
    int value_dimension = -1;
    nc_check(nc_def_dim(ncid, "mode_record", mode_storage, &mode_dimension),
             "defining mode_record dimension");
    nc_check(nc_def_dim(ncid, "column_value", value_storage, &value_dimension),
             "defining column_value dimension");
    const Variables variables = define_variables(
        ncid, mode_dimension, value_dimension, scalar_nc_type<T>());
    nc_check(nc_enddef(ncid), "finishing NSCyl spectral schema");

    std::vector<int> m(mode_storage, 0), l(mode_storage, 0);
    std::vector<int> phase_count(mode_storage, 0);
    std::vector<int> radial_size(mode_storage, 0);
    std::vector<int> block_size(mode_storage, 0);
    std::vector<int> pressure_gauge_fixed(mode_storage, 0);
    std::vector<int> column_count(mode_storage, 0);
    std::vector<int> value_offset(mode_storage, 0);
    std::vector<int> growing(mode_storage, 0);
    std::vector<int> residual_accepted(mode_storage, 0);
    std::vector<double> multiplier_real(mode_storage, 0);
    std::vector<double> multiplier_imag(mode_storage, 0);
    std::vector<double> growth_rate(mode_storage, 0);
    std::vector<double> frequency(mode_storage, 0);
    std::vector<double> right_residual(mode_storage, 0);
    std::vector<double> left_residual(mode_storage, 0);
    std::vector<double> block_condition(mode_storage, 0);
    std::vector<T> right_columns(value_storage, T(0));
    std::vector<T> left_columns(value_storage, T(0));

    int offset = 0;
    for (int i = 0; i < mode_count; ++i) {
        const auto& mode = modes.modes()[i];
        m[i] = mode.m;
        l[i] = mode.l;
        phase_count[i] = mode.phase_count;
        radial_size[i] = mode.radial_size;
        block_size[i] = mode.block_size;
        pressure_gauge_fixed[i] = mode.pressure_gauge_fixed ? 1 : 0;
        column_count[i] = mode.column_count;
        value_offset[i] = offset;
        growing[i] = mode.growing ? 1 : 0;
        residual_accepted[i] = mode.residual_accepted ? 1 : 0;
        multiplier_real[i] = mode.multiplier.real();
        multiplier_imag[i] = mode.multiplier.imag();
        growth_rate[i] = mode.growth_rate;
        frequency[i] = mode.frequency;
        right_residual[i] = mode.right_residual;
        left_residual[i] = mode.left_residual;
        block_condition[i] = block_conditions.at({mode.m, mode.l});
        std::copy(mode.right_columns.begin(), mode.right_columns.end(),
                  right_columns.begin()+offset);
        std::copy(mode.left_columns.begin(), mode.left_columns.end(),
                  left_columns.begin()+offset);
        offset += static_cast<int>(mode.right_columns.size());
    }

    nc_check(nc_put_var_int(ncid, variables.m, m.data()), "writing m");
    nc_check(nc_put_var_int(ncid, variables.l, l.data()), "writing l");
    nc_check(nc_put_var_int(ncid, variables.phase_count, phase_count.data()),
             "writing phase_count");
    nc_check(nc_put_var_int(ncid, variables.radial_size, radial_size.data()),
             "writing radial_size");
    nc_check(nc_put_var_int(ncid, variables.block_size, block_size.data()),
             "writing block_size");
    nc_check(nc_put_var_int(ncid, variables.pressure_gauge_fixed,
                            pressure_gauge_fixed.data()),
             "writing pressure_gauge_fixed");
    nc_check(nc_put_var_int(ncid, variables.column_count, column_count.data()),
             "writing column_count");
    nc_check(nc_put_var_int(ncid, variables.value_offset, value_offset.data()),
             "writing value_offset");
    nc_check(nc_put_var_int(ncid, variables.growing, growing.data()),
             "writing growing");
    nc_check(nc_put_var_int(ncid, variables.residual_accepted,
                            residual_accepted.data()),
             "writing residual_accepted");
    nc_check(nc_put_var_double(ncid, variables.multiplier_real,
                               multiplier_real.data()),
             "writing multiplier_real");
    nc_check(nc_put_var_double(ncid, variables.multiplier_imag,
                               multiplier_imag.data()),
             "writing multiplier_imag");
    nc_check(nc_put_var_double(ncid, variables.growth_rate,
                               growth_rate.data()),
             "writing growth_rate");
    nc_check(nc_put_var_double(ncid, variables.frequency, frequency.data()),
             "writing frequency");
    nc_check(nc_put_var_double(ncid, variables.right_residual,
                               right_residual.data()),
             "writing right_residual");
    nc_check(nc_put_var_double(ncid, variables.left_residual,
                               left_residual.data()),
             "writing left_residual");
    nc_check(nc_put_var_double(ncid, variables.block_condition,
                               block_condition.data()),
             "writing block_condition");
    put_real_values(ncid, variables.right_columns, right_columns);
    put_real_values(ncid, variables.left_columns, left_columns);
    file.close();
}

template<typename T>
void NSCylSpectralStorage::load_(
    NSCylSpectralModeSet<T>& output_modes,
    NSCylSpectralMetadata& output_metadata,
    const NSCylSpectralMetadata* expected) const {
    int ncid = -1;
    nc_check(nc_open(filename_.c_str(), NC_NOWRITE, &ncid),
             "opening NSCyl spectral file");
    NcFile file(ncid);
    NSCylSpectralMetadata metadata = read_metadata(ncid);
    validate_metadata(metadata, scalar_type_name<T>());
    if (expected != nullptr) {
        validate_metadata(*expected, scalar_type_name<T>());
        validate_compatibility(metadata, *expected);
    }

    const int mode_count = get_int_attribute(ncid, "mode_count");
    const int value_count = get_int_attribute(ncid, "column_value_count");
    if (mode_count < 0 || value_count < 0) {
        throw std::runtime_error("negative NSCyl spectral record count");
    }
    const std::size_t mode_storage = std::max<std::size_t>(1, mode_count);
    const std::size_t value_storage = std::max<std::size_t>(1, value_count);
    const int mode_dimension = require_dimension(
        ncid, "mode_record", mode_storage);
    const int value_dimension = require_dimension(
        ncid, "column_value", value_storage);
    const Variables variables = require_variables(
        ncid, mode_dimension, value_dimension, scalar_nc_type<T>());

    std::vector<int> m(mode_storage), l(mode_storage);
    std::vector<int> phase_count(mode_storage), radial_size(mode_storage);
    std::vector<int> block_size(mode_storage), pressure_gauge_fixed(mode_storage);
    std::vector<int> column_count(mode_storage), value_offset(mode_storage);
    std::vector<int> growing(mode_storage), residual_accepted(mode_storage);
    std::vector<double> multiplier_real(mode_storage);
    std::vector<double> multiplier_imag(mode_storage);
    std::vector<double> growth_rate(mode_storage), frequency(mode_storage);
    std::vector<double> right_residual(mode_storage), left_residual(mode_storage);
    std::vector<double> block_condition(mode_storage);
    std::vector<T> right_columns(value_storage), left_columns(value_storage);

    nc_check(nc_get_var_int(ncid, variables.m, m.data()), "reading m");
    nc_check(nc_get_var_int(ncid, variables.l, l.data()), "reading l");
    nc_check(nc_get_var_int(ncid, variables.phase_count, phase_count.data()),
             "reading phase_count");
    nc_check(nc_get_var_int(ncid, variables.radial_size, radial_size.data()),
             "reading radial_size");
    nc_check(nc_get_var_int(ncid, variables.block_size, block_size.data()),
             "reading block_size");
    nc_check(nc_get_var_int(ncid, variables.pressure_gauge_fixed,
                            pressure_gauge_fixed.data()),
             "reading pressure_gauge_fixed");
    nc_check(nc_get_var_int(ncid, variables.column_count, column_count.data()),
             "reading column_count");
    nc_check(nc_get_var_int(ncid, variables.value_offset, value_offset.data()),
             "reading value_offset");
    nc_check(nc_get_var_int(ncid, variables.growing, growing.data()),
             "reading growing");
    nc_check(nc_get_var_int(ncid, variables.residual_accepted,
                            residual_accepted.data()),
             "reading residual_accepted");
    nc_check(nc_get_var_double(ncid, variables.multiplier_real,
                               multiplier_real.data()),
             "reading multiplier_real");
    nc_check(nc_get_var_double(ncid, variables.multiplier_imag,
                               multiplier_imag.data()),
             "reading multiplier_imag");
    nc_check(nc_get_var_double(ncid, variables.growth_rate,
                               growth_rate.data()),
             "reading growth_rate");
    nc_check(nc_get_var_double(ncid, variables.frequency, frequency.data()),
             "reading frequency");
    nc_check(nc_get_var_double(ncid, variables.right_residual,
                               right_residual.data()),
             "reading right_residual");
    nc_check(nc_get_var_double(ncid, variables.left_residual,
                               left_residual.data()),
             "reading left_residual");
    nc_check(nc_get_var_double(ncid, variables.block_condition,
                               block_condition.data()),
             "reading block_condition");
    get_real_values(ncid, variables.right_columns, right_columns);
    get_real_values(ncid, variables.left_columns, left_columns);

    NSCylSpectralModeSet<T> modes;
    int expected_offset = 0;
    for (int i = 0; i < mode_count; ++i) {
        if (growing[i] != 1 || residual_accepted[i] != 1
            || (pressure_gauge_fixed[i] != 0
                && pressure_gauge_fixed[i] != 1)) {
            throw std::runtime_error("invalid spectral mode flags");
        }
        if (column_count[i] < 1 || column_count[i] > 2
            || block_size[i] <= 0
            || value_offset[i] != expected_offset) {
            throw std::runtime_error("invalid spectral mode value offsets");
        }
        const long long next_offset = static_cast<long long>(expected_offset)
            +static_cast<long long>(column_count[i])*block_size[i];
        if (next_offset > value_count) {
            throw std::runtime_error("spectral mode columns exceed storage");
        }

        NSCylSpectralMode<T> mode;
        mode.m = m[i];
        mode.l = l[i];
        mode.phase_count = phase_count[i];
        mode.radial_size = radial_size[i];
        mode.block_size = block_size[i];
        mode.pressure_gauge_fixed = pressure_gauge_fixed[i] != 0;
        mode.multiplier = {
            static_cast<T>(multiplier_real[i]),
            static_cast<T>(multiplier_imag[i])
        };
        mode.growth_rate = growth_rate[i];
        mode.frequency = frequency[i];
        mode.right_residual = right_residual[i];
        mode.left_residual = left_residual[i];
        mode.block_condition_number = block_condition[i];
        mode.growing = true;
        mode.residual_accepted = true;
        mode.column_count = column_count[i];
        mode.right_columns.assign(
            right_columns.begin()+expected_offset,
            right_columns.begin()+next_offset);
        mode.left_columns.assign(
            left_columns.begin()+expected_offset,
            left_columns.begin()+next_offset);
        validate_mode(mode, metadata);
        if (!std::isfinite(mode.block_condition_number)
            || mode.block_condition_number < 1
            || mode.block_condition_number > metadata.condition_limit) {
            throw std::runtime_error("invalid spectral block condition");
        }
        if (!modes.empty()) {
            const auto& previous = modes.modes().back();
            if (std::pair<int, int>{mode.m, mode.l}
                    < std::pair<int, int>{previous.m, previous.l}
                || (mode.m == previous.m && mode.l == previous.l
                    && mode.growth_rate > previous.growth_rate)) {
                throw std::runtime_error("spectral modes are not sorted");
            }
        }
        modes.append_filterable_mode(std::move(mode));
        expected_offset = static_cast<int>(next_offset);
    }
    if (expected_offset != value_count) {
        throw std::runtime_error("unused values in spectral column storage");
    }

    const NSCylSpectralProjector<T> projector(
        modes, metadata.condition_limit);
    const double comparison_tolerance =
        256.0*static_cast<double>(std::numeric_limits<T>::epsilon());
    for (const auto& block : projector.blocks()) {
        const auto first = std::find_if(
            modes.modes().begin(), modes.modes().end(),
            [&](const auto& mode) {
                return mode.m == block.m() && mode.l == block.l();
            });
        if (first == modes.modes().end()) {
            throw std::runtime_error("missing spectral block condition");
        }
        const double stored = first->block_condition_number;
        const double scale = std::max(1.0, block.condition_number());
        if (std::abs(stored-block.condition_number())
            > comparison_tolerance*scale) {
            throw std::runtime_error("spectral block condition mismatch");
        }
        for (const auto& mode : modes.modes()) {
            if (mode.m == block.m() && mode.l == block.l()
                && mode.block_condition_number != stored) {
                throw std::runtime_error(
                    "inconsistent condition within spectral block");
            }
        }
    }

    file.close();
    output_modes = std::move(modes);
    output_metadata = std::move(metadata);
}

void NSCylSpectralStorage::save(
    const NSCylSpectralModeSet<float>& modes,
    const NSCylSpectralMetadata& metadata) const {
    save_(modes, metadata);
}

void NSCylSpectralStorage::save(
    const NSCylSpectralModeSet<double>& modes,
    const NSCylSpectralMetadata& metadata) const {
    save_(modes, metadata);
}

void NSCylSpectralStorage::load(
    NSCylSpectralModeSet<float>& modes,
    NSCylSpectralMetadata& metadata) const {
    load_(modes, metadata, nullptr);
}

void NSCylSpectralStorage::load(
    NSCylSpectralModeSet<double>& modes,
    NSCylSpectralMetadata& metadata) const {
    load_(modes, metadata, nullptr);
}

void NSCylSpectralStorage::load(
    NSCylSpectralModeSet<float>& modes,
    NSCylSpectralMetadata& metadata,
    const NSCylSpectralMetadata& expected) const {
    load_(modes, metadata, &expected);
}

void NSCylSpectralStorage::load(
    NSCylSpectralModeSet<double>& modes,
    NSCylSpectralMetadata& metadata,
    const NSCylSpectralMetadata& expected) const {
    load_(modes, metadata, &expected);
}

} // namespace fdm
