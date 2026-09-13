#include "ns_cyl_boundary_lqr_storage.h"

#include <netcdf.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace fdm {
namespace {

constexpr int gain_schema_version = 1;

void nc_check(int code, const std::string& operation) {
    if (code != NC_NOERR) {
        throw std::runtime_error(operation+": "+nc_strerror(code));
    }
}

class NcFile {
public:
    explicit NcFile(int id) : id_(id) { }
    ~NcFile() {
        if (id_ >= 0) {
            nc_close(id_);
        }
    }
    int id() const { return id_; }
    void close() {
        if (id_ >= 0) {
            const int id = std::exchange(id_, -1);
            nc_check(nc_close(id), "closing NSCyl boundary LQR gain");
        }
    }
private:
    int id_;
};

void put_text(int ncid, const char* name, const std::string& value) {
    nc_check(nc_put_att_text(
        ncid, NC_GLOBAL, name, value.size(), value.data()),
        std::string("writing boundary LQR gain attribute ")+name);
}

void put_int(int ncid, const char* name, int value) {
    nc_check(nc_put_att_int(ncid, NC_GLOBAL, name, NC_INT, 1, &value),
             std::string("writing boundary LQR gain attribute ")+name);
}

void put_double(int ncid, const char* name, double value) {
    nc_check(nc_put_att_double(ncid, NC_GLOBAL, name, NC_DOUBLE, 1, &value),
             std::string("writing boundary LQR gain attribute ")+name);
}

std::string get_text(int ncid, const char* name) {
    std::size_t size = 0;
    nc_check(nc_inq_attlen(ncid, NC_GLOBAL, name, &size),
             std::string("reading boundary LQR gain attribute ")+name);
    std::string result(size, '\0');
    if (size != 0) {
        nc_check(nc_get_att_text(ncid, NC_GLOBAL, name, result.data()),
                 std::string("reading boundary LQR gain attribute ")+name);
    }
    return result;
}

int get_int(int ncid, const char* name) {
    int value = 0;
    nc_check(nc_get_att_int(ncid, NC_GLOBAL, name, &value),
             std::string("reading boundary LQR gain attribute ")+name);
    return value;
}

int get_optional_int(int ncid, const char* name, int fallback) {
    int value = 0;
    const int status = nc_get_att_int(ncid, NC_GLOBAL, name, &value);
    if (status == NC_ENOTATT) {
        return fallback;
    }
    nc_check(status,
             std::string("reading boundary LQR gain attribute ")+name);
    return value;
}

double get_double(int ncid, const char* name) {
    double value = 0;
    nc_check(nc_get_att_double(ncid, NC_GLOBAL, name, &value),
             std::string("reading boundary LQR gain attribute ")+name);
    return value;
}

void write_metadata(int ncid,
                    const NSCylBoundaryLQRGainMetadata& metadata,
                    int block_count, int value_count,
                    int reduced_hessian_value_count) {
    put_int(ncid, "schema_version", metadata.schema_version);
    put_text(ncid, "format_name", metadata.format_name);
    put_int(ncid, "step_operator_version", metadata.step_operator_version);
    put_text(ncid, "scalar_type", metadata.scalar_type);
    put_text(ncid, "fourier_layout", metadata.fourier_layout);
    put_text(ncid, "state_layout", metadata.state_layout);
    put_text(ncid, "pressure_gauge", metadata.pressure_gauge);
    put_text(ncid, "pressure_boundary", metadata.pressure_boundary);
    put_text(ncid, "control_components", metadata.control_components);
    put_text(ncid, "config", metadata.config_text);
    put_int(ncid, "nr", metadata.nr);
    put_int(ncid, "nphi", metadata.nphi);
    put_int(ncid, "nz", metadata.nz);
    put_int(ncid, "horizon_intervals", metadata.horizon_intervals);
    put_int(ncid, "interval_steps", metadata.interval_steps);
    put_int(ncid, "block_count", block_count);
    put_int(ncid, "gain_value_count", value_count);
    put_int(ncid, "reduced_hessian_value_count",
            reduced_hessian_value_count);
    put_double(ncid, "r", metadata.r);
    put_double(ncid, "R", metadata.R);
    put_double(ncid, "h1", metadata.h1);
    put_double(ncid, "h2", metadata.h2);
    put_double(ncid, "Re", metadata.reynolds);
    put_double(ncid, "dt", metadata.dt);
    put_double(ncid, "wall_speed", metadata.wall_speed);
    put_double(ncid, "control_weight", metadata.control_weight);
    put_double(ncid, "ridge", metadata.ridge);
}

NSCylBoundaryLQRGainMetadata read_metadata(int ncid) {
    int version = 0;
    const int status = nc_get_att_int(
        ncid, NC_GLOBAL, "schema_version", &version);
    if (status == NC_ENOTATT) {
        throw std::runtime_error(
            "not an NSCyl boundary LQR gain: missing schema_version");
    }
    nc_check(status, "reading boundary LQR gain schema_version");
    NSCylBoundaryLQRGainMetadata result;
    result.schema_version = version;
    result.format_name = get_text(ncid, "format_name");
    result.step_operator_version = get_int(ncid, "step_operator_version");
    result.scalar_type = get_text(ncid, "scalar_type");
    result.fourier_layout = get_text(ncid, "fourier_layout");
    result.state_layout = get_text(ncid, "state_layout");
    result.pressure_gauge = get_text(ncid, "pressure_gauge");
    result.pressure_boundary = get_text(ncid, "pressure_boundary");
    result.control_components = get_text(ncid, "control_components");
    result.config_text = get_text(ncid, "config");
    result.nr = get_int(ncid, "nr");
    result.nphi = get_int(ncid, "nphi");
    result.nz = get_int(ncid, "nz");
    result.horizon_intervals = get_int(ncid, "horizon_intervals");
    result.interval_steps = get_int(ncid, "interval_steps");
    result.r = get_double(ncid, "r");
    result.R = get_double(ncid, "R");
    result.h1 = get_double(ncid, "h1");
    result.h2 = get_double(ncid, "h2");
    result.reynolds = get_double(ncid, "Re");
    result.dt = get_double(ncid, "dt");
    result.wall_speed = get_double(ncid, "wall_speed");
    result.control_weight = get_double(ncid, "control_weight");
    result.ridge = get_double(ncid, "ridge");
    return result;
}

template<typename T>
const char* scalar_name() {
    return std::is_same_v<T, float> ? "float32" : "float64";
}

template<typename T>
constexpr nc_type scalar_nc_type() {
    return std::is_same_v<T, float> ? NC_FLOAT : NC_DOUBLE;
}

void validate_metadata(const NSCylBoundaryLQRGainMetadata& metadata,
                       const char* scalar_type) {
    if (metadata.schema_version != gain_schema_version
        || metadata.format_name != "NSCyl physical boundary LQR gain"
        || metadata.step_operator_version != 2) {
        throw std::runtime_error("incompatible NSCyl boundary LQR gain format");
    }
    if (metadata.scalar_type != scalar_type) {
        throw std::runtime_error(
            "incompatible NSCyl boundary LQR gain scalar type");
    }
    if (metadata.fourier_layout != "samarskii_nikolaev_real_packed_v1"
        || metadata.state_layout
            != "staggered_radial_component_major_u_v_w_p_v1"
        || metadata.pressure_gauge
            != "weighted_radial_zero_mean_last_pressure_dependent_v1"
        || metadata.pressure_boundary
            != "radial_same_time_neumann_v1") {
        throw std::runtime_error("incompatible NSCyl boundary LQR gain layout");
    }
    if (metadata.control_components != "all"
        && metadata.control_components != "tangential"
        && metadata.control_components != "azimuthal") {
        throw std::runtime_error(
            "invalid NSCyl boundary LQR gain component selection");
    }
    if (metadata.nr < 2 || metadata.nphi <= 0 || metadata.nz <= 0
        || metadata.horizon_intervals <= 0 || metadata.interval_steps <= 0
        || !(metadata.R > metadata.r) || !(metadata.h2 > metadata.h1)
        || !(metadata.dt > 0) || !(metadata.control_weight >= 0)
        || !(metadata.ridge >= 0)) {
        throw std::runtime_error("invalid NSCyl boundary LQR gain metadata");
    }
    const double values[] = {
        metadata.r, metadata.R, metadata.h1, metadata.h2,
        metadata.reynolds, metadata.dt, metadata.wall_speed,
        metadata.control_weight, metadata.ridge
    };
    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "non-finite NSCyl boundary LQR gain metadata");
        }
    }
}

template<typename Value>
void require_equal(const char* name, const Value& actual,
                   const Value& expected) {
    if (actual != expected) {
        throw std::runtime_error(
            std::string("incompatible NSCyl boundary LQR gain metadata: ")
            +name);
    }
}

void validate_compatibility(
    const NSCylBoundaryLQRGainMetadata& actual,
    const NSCylBoundaryLQRGainMetadata& expected) {
    require_equal("schema_version", actual.schema_version,
                  expected.schema_version);
    require_equal("format_name", actual.format_name, expected.format_name);
    require_equal("step_operator_version", actual.step_operator_version,
                  expected.step_operator_version);
    require_equal("scalar_type", actual.scalar_type, expected.scalar_type);
    require_equal("fourier_layout", actual.fourier_layout,
                  expected.fourier_layout);
    require_equal("state_layout", actual.state_layout,
                  expected.state_layout);
    require_equal("pressure_gauge", actual.pressure_gauge,
                  expected.pressure_gauge);
    require_equal("pressure_boundary", actual.pressure_boundary,
                  expected.pressure_boundary);
    require_equal("control_components", actual.control_components,
                  expected.control_components);
    require_equal("nr", actual.nr, expected.nr);
    require_equal("nphi", actual.nphi, expected.nphi);
    require_equal("nz", actual.nz, expected.nz);
    require_equal("horizon_intervals", actual.horizon_intervals,
                  expected.horizon_intervals);
    require_equal("interval_steps", actual.interval_steps,
                  expected.interval_steps);
    require_equal("r", actual.r, expected.r);
    require_equal("R", actual.R, expected.R);
    require_equal("h1", actual.h1, expected.h1);
    require_equal("h2", actual.h2, expected.h2);
    require_equal("Re", actual.reynolds, expected.reynolds);
    require_equal("dt", actual.dt, expected.dt);
    require_equal("wall_speed", actual.wall_speed, expected.wall_speed);
    require_equal("control_weight", actual.control_weight,
                  expected.control_weight);
    require_equal("ridge", actual.ridge, expected.ridge);
}

int phase_count(int m, int l,
                const NSCylBoundaryLQRGainMetadata& metadata) {
    return (m == 0 || 2*m == metadata.nphi ? 1 : 2)
        *(l == 0 || 2*l == metadata.nz ? 1 : 2);
}

int input_size(int m, int l,
               const NSCylBoundaryLQRGainMetadata& metadata) {
    const int phases = phase_count(m, l, metadata);
    if (metadata.control_components == "azimuthal") {
        return phases;
    }
    if (metadata.control_components == "tangential") {
        return 2*phases;
    }
    return (m == 0 && l == 0 ? 2 : 3)*phases;
}

template<typename T>
void validate_gain_block(
    const NSCylBoundaryLQRGainBlock<T>& block,
    const NSCylBoundaryLQRGainMetadata& metadata) {
    if (block.m < 0 || block.m > metadata.nphi/2
        || block.l < 0 || block.l > metadata.nz/2) {
        throw std::runtime_error(
            "boundary LQR gain block index is outside the grid");
    }
    const int phases = phase_count(block.m, block.l, metadata);
    const int augmented_size = phases*(4*metadata.nr+2)
        -(block.m == 0 && block.l == 0 ? 1 : 0);
    const std::size_t expected = static_cast<std::size_t>(
        input_size(block.m, block.l, metadata))*augmented_size;
    if (block.values.size() != expected) {
        throw std::runtime_error("invalid boundary LQR gain block size");
    }
    const std::size_t expected_reduced = static_cast<std::size_t>(
        input_size(block.m, block.l, metadata))
        *input_size(block.m, block.l, metadata);
    if (!block.reduced_hessian.empty()
        && block.reduced_hessian.size() != expected_reduced) {
        throw std::runtime_error(
            "invalid boundary LQR reduced Hessian block size");
    }
    for (T value : block.values) {
        if (!std::isfinite(static_cast<double>(value))) {
            throw std::runtime_error("non-finite boundary LQR gain value");
        }
    }
    for (T value : block.reduced_hessian) {
        if (!std::isfinite(static_cast<double>(value))) {
            throw std::runtime_error(
                "non-finite boundary LQR reduced Hessian value");
        }
    }
}

int define_variable(int ncid, const char* name, nc_type type, int dimension) {
    int variable = -1;
    nc_check(nc_def_var(ncid, name, type, 1, &dimension, &variable),
             std::string("defining boundary LQR gain variable ")+name);
    return variable;
}

int require_dimension(int ncid, const char* name, std::size_t expected) {
    int dimension = -1;
    nc_check(nc_inq_dimid(ncid, name, &dimension),
             std::string("reading boundary LQR gain dimension ")+name);
    std::size_t actual = 0;
    nc_check(nc_inq_dimlen(ncid, dimension, &actual),
             std::string("reading boundary LQR gain dimension ")+name);
    if (actual != expected) {
        throw std::runtime_error(
            std::string("invalid boundary LQR gain dimension ")+name);
    }
    return dimension;
}

int require_variable(int ncid, const char* name, nc_type type,
                     int dimension) {
    int variable = -1;
    nc_check(nc_inq_varid(ncid, name, &variable),
             std::string("reading boundary LQR gain variable ")+name);
    nc_type actual_type = NC_NAT;
    int dimension_count = 0;
    int dimensions[NC_MAX_VAR_DIMS];
    nc_check(nc_inq_var(ncid, variable, nullptr, &actual_type,
                        &dimension_count, dimensions, nullptr),
             std::string("reading boundary LQR gain variable ")+name);
    if (actual_type != type || dimension_count != 1
        || dimensions[0] != dimension) {
        throw std::runtime_error(
            std::string("invalid boundary LQR gain variable ")+name);
    }
    return variable;
}

template<typename T>
void put_values(int ncid, int variable, const std::vector<T>& values) {
    if constexpr (std::is_same_v<T, float>) {
        nc_check(nc_put_var_float(ncid, variable, values.data()),
                 "writing boundary LQR gain values");
    } else {
        nc_check(nc_put_var_double(ncid, variable, values.data()),
                 "writing boundary LQR gain values");
    }
}

template<typename T>
void get_values(int ncid, int variable, std::vector<T>& values) {
    if constexpr (std::is_same_v<T, float>) {
        nc_check(nc_get_var_float(ncid, variable, values.data()),
                 "reading boundary LQR gain values");
    } else {
        nc_check(nc_get_var_double(ncid, variable, values.data()),
                 "reading boundary LQR gain values");
    }
}

} // namespace

template<typename T>
void NSCylBoundaryLQRGainStorage::save_(
    const NSCylBoundaryLQRGainSet<T>& input,
    const NSCylBoundaryLQRGainMetadata& metadata) const {
    validate_metadata(metadata, scalar_name<T>());
    if (input.blocks.empty()) {
        throw std::runtime_error("cannot save an empty boundary LQR gain");
    }
    auto gains = input;
    std::sort(gains.blocks.begin(), gains.blocks.end(),
              [](const auto& first, const auto& second) {
                  return std::pair(first.m, first.l)
                      <std::pair(second.m, second.l);
              });
    std::set<std::pair<int, int>> unique;
    std::size_t value_count_size = 0;
    std::size_t reduced_hessian_count_size = 0;
    bool has_reduced_hessian = false;
    bool lacks_reduced_hessian = false;
    for (const auto& block : gains.blocks) {
        validate_gain_block(block, metadata);
        if (!unique.emplace(block.m, block.l).second) {
            throw std::runtime_error("duplicate boundary LQR gain block");
        }
        value_count_size += block.values.size();
        if (value_count_size
            > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("boundary LQR gain storage is too large");
        }
        if (block.reduced_hessian.empty()) {
            lacks_reduced_hessian = true;
        } else {
            has_reduced_hessian = true;
            reduced_hessian_count_size += block.reduced_hessian.size();
            if (reduced_hessian_count_size
                > static_cast<std::size_t>(
                    std::numeric_limits<int>::max())) {
                throw std::runtime_error(
                    "boundary LQR reduced Hessian storage is too large");
            }
        }
    }
    if (has_reduced_hessian && lacks_reduced_hessian) {
        throw std::runtime_error(
            "boundary LQR gain has incomplete constrained MPC data");
    }
    if (gains.blocks.size()
        > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("too many boundary LQR gain blocks");
    }
    const int block_count = static_cast<int>(gains.blocks.size());
    const int value_count = static_cast<int>(value_count_size);
    const int reduced_hessian_count =
        static_cast<int>(reduced_hessian_count_size);

    int ncid = -1;
    nc_check(nc_create(filename_.c_str(), NC_CLOBBER | NC_64BIT_OFFSET,
                       &ncid),
             "creating NSCyl boundary LQR gain file");
    NcFile file(ncid);
    write_metadata(ncid, metadata, block_count, value_count,
                   reduced_hessian_count);
    int block_dimension = -1, value_dimension = -1;
    nc_check(nc_def_dim(ncid, "block", block_count, &block_dimension),
             "defining boundary LQR gain block dimension");
    nc_check(nc_def_dim(ncid, "gain_value", value_count, &value_dimension),
             "defining boundary LQR gain value dimension");
    const int m_variable = define_variable(
        ncid, "m", NC_INT, block_dimension);
    const int l_variable = define_variable(
        ncid, "l", NC_INT, block_dimension);
    const int rows_variable = define_variable(
        ncid, "gain_rows", NC_INT, block_dimension);
    const int columns_variable = define_variable(
        ncid, "gain_columns", NC_INT, block_dimension);
    const int offset_variable = define_variable(
        ncid, "gain_offset", NC_INT, block_dimension);
    const int values_variable = define_variable(
        ncid, "gain", scalar_nc_type<T>(), value_dimension);
    int reduced_hessian_offset_variable = -1;
    int reduced_hessian_values_variable = -1;
    if (reduced_hessian_count > 0) {
        int reduced_hessian_dimension = -1;
        nc_check(nc_def_dim(
            ncid, "reduced_hessian_value", reduced_hessian_count,
            &reduced_hessian_dimension),
            "defining boundary LQR reduced Hessian dimension");
        reduced_hessian_offset_variable = define_variable(
            ncid, "reduced_hessian_offset", NC_INT, block_dimension);
        reduced_hessian_values_variable = define_variable(
            ncid, "reduced_hessian", scalar_nc_type<T>(),
            reduced_hessian_dimension);
    }
    nc_check(nc_enddef(ncid), "finishing boundary LQR gain schema");

    std::vector<int> m(block_count), l(block_count), rows(block_count);
    std::vector<int> columns(block_count), offsets(block_count);
    std::vector<T> values(value_count);
    std::vector<int> reduced_hessian_offsets(block_count);
    std::vector<T> reduced_hessian_values(reduced_hessian_count);
    int offset = 0;
    int reduced_hessian_offset = 0;
    for (int index = 0; index < block_count; ++index) {
        const auto& block = gains.blocks[index];
        m[index] = block.m;
        l[index] = block.l;
        rows[index] = input_size(block.m, block.l, metadata);
        columns[index] = phase_count(block.m, block.l, metadata)
            *(4*metadata.nr+2)-(block.m == 0 && block.l == 0 ? 1 : 0);
        offsets[index] = offset;
        std::copy(block.values.begin(), block.values.end(),
                  values.begin()+offset);
        offset += static_cast<int>(block.values.size());
        reduced_hessian_offsets[index] = reduced_hessian_offset;
        std::copy(block.reduced_hessian.begin(),
                  block.reduced_hessian.end(),
                  reduced_hessian_values.begin()+reduced_hessian_offset);
        reduced_hessian_offset += static_cast<int>(
            block.reduced_hessian.size());
    }
    nc_check(nc_put_var_int(ncid, m_variable, m.data()), "writing gain m");
    nc_check(nc_put_var_int(ncid, l_variable, l.data()), "writing gain l");
    nc_check(nc_put_var_int(ncid, rows_variable, rows.data()),
             "writing gain rows");
    nc_check(nc_put_var_int(ncid, columns_variable, columns.data()),
             "writing gain columns");
    nc_check(nc_put_var_int(ncid, offset_variable, offsets.data()),
             "writing gain offsets");
    put_values(ncid, values_variable, values);
    if (reduced_hessian_count > 0) {
        nc_check(nc_put_var_int(
            ncid, reduced_hessian_offset_variable,
            reduced_hessian_offsets.data()),
            "writing boundary LQR reduced Hessian offsets");
        put_values(
            ncid, reduced_hessian_values_variable,
            reduced_hessian_values);
    }
    file.close();
}

template<typename T>
void NSCylBoundaryLQRGainStorage::load_(
    NSCylBoundaryLQRGainSet<T>& gains,
    NSCylBoundaryLQRGainMetadata& metadata,
    const NSCylBoundaryLQRGainMetadata* expected) const {
    int ncid = -1;
    nc_check(nc_open(filename_.c_str(), NC_NOWRITE, &ncid),
             "opening NSCyl boundary LQR gain file");
    NcFile file(ncid);
    auto loaded_metadata = read_metadata(ncid);
    validate_metadata(loaded_metadata, scalar_name<T>());
    if (expected) {
        validate_compatibility(loaded_metadata, *expected);
    }
    const int block_count = get_int(ncid, "block_count");
    const int value_count = get_int(ncid, "gain_value_count");
    const int reduced_hessian_count = get_optional_int(
        ncid, "reduced_hessian_value_count", 0);
    if (block_count <= 0 || value_count <= 0
        || reduced_hessian_count < 0) {
        throw std::runtime_error("empty NSCyl boundary LQR gain file");
    }
    const int block_dimension = require_dimension(
        ncid, "block", block_count);
    const int value_dimension = require_dimension(
        ncid, "gain_value", value_count);
    const int m_variable = require_variable(
        ncid, "m", NC_INT, block_dimension);
    const int l_variable = require_variable(
        ncid, "l", NC_INT, block_dimension);
    const int rows_variable = require_variable(
        ncid, "gain_rows", NC_INT, block_dimension);
    const int columns_variable = require_variable(
        ncid, "gain_columns", NC_INT, block_dimension);
    const int offset_variable = require_variable(
        ncid, "gain_offset", NC_INT, block_dimension);
    const int values_variable = require_variable(
        ncid, "gain", scalar_nc_type<T>(), value_dimension);
    int reduced_hessian_offset_variable = -1;
    int reduced_hessian_values_variable = -1;
    if (reduced_hessian_count > 0) {
        const int reduced_hessian_dimension = require_dimension(
            ncid, "reduced_hessian_value", reduced_hessian_count);
        reduced_hessian_offset_variable = require_variable(
            ncid, "reduced_hessian_offset", NC_INT, block_dimension);
        reduced_hessian_values_variable = require_variable(
            ncid, "reduced_hessian", scalar_nc_type<T>(),
            reduced_hessian_dimension);
    }
    std::vector<int> m(block_count), l(block_count), rows(block_count);
    std::vector<int> columns(block_count), offsets(block_count);
    std::vector<T> values(value_count);
    std::vector<int> reduced_hessian_offsets(block_count);
    std::vector<T> reduced_hessian_values(reduced_hessian_count);
    nc_check(nc_get_var_int(ncid, m_variable, m.data()), "reading gain m");
    nc_check(nc_get_var_int(ncid, l_variable, l.data()), "reading gain l");
    nc_check(nc_get_var_int(ncid, rows_variable, rows.data()),
             "reading gain rows");
    nc_check(nc_get_var_int(ncid, columns_variable, columns.data()),
             "reading gain columns");
    nc_check(nc_get_var_int(ncid, offset_variable, offsets.data()),
             "reading gain offsets");
    get_values(ncid, values_variable, values);
    if (reduced_hessian_count > 0) {
        nc_check(nc_get_var_int(
            ncid, reduced_hessian_offset_variable,
            reduced_hessian_offsets.data()),
            "reading boundary LQR reduced Hessian offsets");
        get_values(
            ncid, reduced_hessian_values_variable,
            reduced_hessian_values);
    }

    NSCylBoundaryLQRGainSet<T> loaded;
    loaded.blocks.reserve(block_count);
    std::set<std::pair<int, int>> unique;
    int expected_offset = 0;
    int expected_reduced_hessian_offset = 0;
    for (int index = 0; index < block_count; ++index) {
        const int expected_rows = input_size(
            m[index], l[index], loaded_metadata);
        const int expected_columns = phase_count(
            m[index], l[index], loaded_metadata)
            *(4*loaded_metadata.nr+2)
            -(m[index] == 0 && l[index] == 0 ? 1 : 0);
        const long long count = static_cast<long long>(rows[index])
            *columns[index];
        if (rows[index] != expected_rows
            || columns[index] != expected_columns
            || offsets[index] != expected_offset || count <= 0
            || count > value_count-expected_offset) {
            throw std::runtime_error(
                "invalid NSCyl boundary LQR gain record");
        }
        if (!unique.emplace(m[index], l[index]).second) {
            throw std::runtime_error("duplicate boundary LQR gain block");
        }
        NSCylBoundaryLQRGainBlock<T> block;
        block.m = m[index];
        block.l = l[index];
        block.values.assign(values.begin()+expected_offset,
                            values.begin()+expected_offset+count);
        if (reduced_hessian_count > 0) {
            const long long reduced_count =
                static_cast<long long>(expected_rows)*expected_rows;
            if (reduced_hessian_offsets[index]
                    != expected_reduced_hessian_offset
                || reduced_count <= 0
                || reduced_count > reduced_hessian_count
                    -expected_reduced_hessian_offset) {
                throw std::runtime_error(
                    "invalid NSCyl boundary LQR reduced Hessian record");
            }
            block.reduced_hessian.assign(
                reduced_hessian_values.begin()
                    +expected_reduced_hessian_offset,
                reduced_hessian_values.begin()
                    +expected_reduced_hessian_offset+reduced_count);
            expected_reduced_hessian_offset += static_cast<int>(
                reduced_count);
        }
        validate_gain_block(block, loaded_metadata);
        loaded.blocks.push_back(std::move(block));
        expected_offset += static_cast<int>(count);
    }
    if (expected_offset != value_count) {
        throw std::runtime_error("unused NSCyl boundary LQR gain values");
    }
    if (expected_reduced_hessian_offset != reduced_hessian_count) {
        throw std::runtime_error(
            "unused NSCyl boundary LQR reduced Hessian values");
    }
    file.close();
    gains = std::move(loaded);
    metadata = std::move(loaded_metadata);
}

void NSCylBoundaryLQRGainStorage::save(
    const NSCylBoundaryLQRGainSet<float>& gains,
    const NSCylBoundaryLQRGainMetadata& metadata) const {
    save_(gains, metadata);
}

void NSCylBoundaryLQRGainStorage::save(
    const NSCylBoundaryLQRGainSet<double>& gains,
    const NSCylBoundaryLQRGainMetadata& metadata) const {
    save_(gains, metadata);
}

void NSCylBoundaryLQRGainStorage::load(
    NSCylBoundaryLQRGainSet<float>& gains,
    NSCylBoundaryLQRGainMetadata& metadata) const {
    load_(gains, metadata, nullptr);
}

void NSCylBoundaryLQRGainStorage::load(
    NSCylBoundaryLQRGainSet<double>& gains,
    NSCylBoundaryLQRGainMetadata& metadata) const {
    load_(gains, metadata, nullptr);
}

void NSCylBoundaryLQRGainStorage::load(
    NSCylBoundaryLQRGainSet<float>& gains,
    NSCylBoundaryLQRGainMetadata& metadata,
    const NSCylBoundaryLQRGainMetadata& expected) const {
    load_(gains, metadata, &expected);
}

void NSCylBoundaryLQRGainStorage::load(
    NSCylBoundaryLQRGainSet<double>& gains,
    NSCylBoundaryLQRGainMetadata& metadata,
    const NSCylBoundaryLQRGainMetadata& expected) const {
    load_(gains, metadata, &expected);
}

} // namespace fdm
