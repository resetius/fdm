#include "ns_cyl_checkpoint_storage.h"

#include <netcdf.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace fdm {
namespace {

constexpr int checkpoint_schema_version = 1;

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
            nc_check(nc_close(id), "closing NSCyl checkpoint");
        }
    }

private:
    int id_;
};

void put_text(int ncid, const char* name, const std::string& value) {
    nc_check(nc_put_att_text(
        ncid, NC_GLOBAL, name, value.size(), value.data()),
        std::string("writing checkpoint attribute ")+name);
}

void put_int(int ncid, const char* name, int value) {
    nc_check(nc_put_att_int(ncid, NC_GLOBAL, name, NC_INT, 1, &value),
             std::string("writing checkpoint attribute ")+name);
}

void put_double(int ncid, const char* name, double value) {
    nc_check(nc_put_att_double(ncid, NC_GLOBAL, name, NC_DOUBLE, 1, &value),
             std::string("writing checkpoint attribute ")+name);
}

std::string get_text(int ncid, const char* name) {
    std::size_t size = 0;
    nc_check(nc_inq_attlen(ncid, NC_GLOBAL, name, &size),
             std::string("reading checkpoint attribute ")+name);
    std::string result(size, '\0');
    if (size != 0) {
        nc_check(nc_get_att_text(ncid, NC_GLOBAL, name, result.data()),
                 std::string("reading checkpoint attribute ")+name);
    }
    return result;
}

int get_int(int ncid, const char* name) {
    int value = 0;
    nc_check(nc_get_att_int(ncid, NC_GLOBAL, name, &value),
             std::string("reading checkpoint attribute ")+name);
    return value;
}

double get_double(int ncid, const char* name) {
    double value = 0;
    nc_check(nc_get_att_double(ncid, NC_GLOBAL, name, &value),
             std::string("reading checkpoint attribute ")+name);
    return value;
}

void write_metadata(int ncid, const NSCylCheckpointMetadata& metadata) {
    put_int(ncid, "schema_version", metadata.schema_version);
    put_text(ncid, "format_name", metadata.format_name);
    put_int(ncid, "step_operator_version", metadata.step_operator_version);
    put_text(ncid, "scalar_type", metadata.scalar_type);
    put_text(ncid, "axial_boundary", metadata.axial_boundary);
    put_text(ncid, "state_layout", metadata.state_layout);
    put_text(ncid, "pressure_gauge", metadata.pressure_gauge);
    put_text(ncid, "config", metadata.config_text);

    put_int(ncid, "nr", metadata.nr);
    put_int(ncid, "nphi", metadata.nphi);
    put_int(ncid, "nz", metadata.nz);
    put_int(ncid, "state_size", metadata.state_size);
    put_int(ncid, "u_offset", metadata.u_offset);
    put_int(ncid, "v_offset", metadata.v_offset);
    put_int(ncid, "w_offset", metadata.w_offset);
    put_int(ncid, "p_offset", metadata.p_offset);
    put_int(ncid, "time_index", metadata.time_index);

    put_double(ncid, "r", metadata.r);
    put_double(ncid, "R", metadata.R);
    put_double(ncid, "h1", metadata.h1);
    put_double(ncid, "h2", metadata.h2);
    put_double(ncid, "Re", metadata.reynolds);
    put_double(ncid, "dt", metadata.dt);
    put_double(ncid, "wall_speed", metadata.wall_speed);
    put_double(ncid, "physical_time", metadata.physical_time);
}

NSCylCheckpointMetadata read_metadata(int ncid) {
    int version = 0;
    const int status = nc_get_att_int(
        ncid, NC_GLOBAL, "schema_version", &version);
    if (status == NC_ENOTATT) {
        throw std::runtime_error(
            "not an NSCyl checkpoint: missing schema_version");
    }
    nc_check(status, "reading checkpoint attribute schema_version");

    NSCylCheckpointMetadata result;
    result.schema_version = version;
    result.format_name = get_text(ncid, "format_name");
    result.step_operator_version = get_int(ncid, "step_operator_version");
    result.scalar_type = get_text(ncid, "scalar_type");
    result.axial_boundary = get_text(ncid, "axial_boundary");
    result.state_layout = get_text(ncid, "state_layout");
    result.pressure_gauge = get_text(ncid, "pressure_gauge");
    result.config_text = get_text(ncid, "config");

    result.nr = get_int(ncid, "nr");
    result.nphi = get_int(ncid, "nphi");
    result.nz = get_int(ncid, "nz");
    result.state_size = get_int(ncid, "state_size");
    result.u_offset = get_int(ncid, "u_offset");
    result.v_offset = get_int(ncid, "v_offset");
    result.w_offset = get_int(ncid, "w_offset");
    result.p_offset = get_int(ncid, "p_offset");
    result.time_index = get_int(ncid, "time_index");

    result.r = get_double(ncid, "r");
    result.R = get_double(ncid, "R");
    result.h1 = get_double(ncid, "h1");
    result.h2 = get_double(ncid, "h2");
    result.reynolds = get_double(ncid, "Re");
    result.dt = get_double(ncid, "dt");
    result.wall_speed = get_double(ncid, "wall_speed");
    result.physical_time = get_double(ncid, "physical_time");
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

void validate_metadata(const NSCylCheckpointMetadata& metadata,
                       const char* scalar_type) {
    if (metadata.schema_version != checkpoint_schema_version) {
        throw std::runtime_error(
            "unsupported NSCyl checkpoint schema_version="
            +std::to_string(metadata.schema_version));
    }
    if (metadata.format_name != "NSCyl nonlinear checkpoint"
        || metadata.step_operator_version != 1) {
        throw std::runtime_error("incompatible NSCyl checkpoint operator");
    }
    if (metadata.scalar_type != scalar_type) {
        throw std::runtime_error(
            "incompatible NSCyl checkpoint scalar type: file="
            +metadata.scalar_type+", requested="+scalar_type);
    }
    if (metadata.axial_boundary != "periodic"
        || metadata.state_layout
            != "staggered_component_major_u_v_w_p_v1"
        || metadata.pressure_gauge != "weighted_volume_zero_mean_v1") {
        throw std::runtime_error("incompatible NSCyl checkpoint layout");
    }
    if (metadata.nr < 2 || metadata.nphi <= 0 || metadata.nz <= 0
        || metadata.time_index < 0) {
        throw std::runtime_error("invalid NSCyl checkpoint grid or time");
    }
    const long long plane = static_cast<long long>(metadata.nphi)*metadata.nz;
    const long long v_offset = plane*(metadata.nr-1);
    const long long w_offset = v_offset+plane*metadata.nr;
    const long long p_offset = w_offset+plane*metadata.nr;
    const long long state_size = p_offset+plane*metadata.nr;
    if (state_size > std::numeric_limits<int>::max()
        || metadata.u_offset != 0 || metadata.v_offset != v_offset
        || metadata.w_offset != w_offset || metadata.p_offset != p_offset
        || metadata.state_size != state_size) {
        throw std::runtime_error("incompatible NSCyl checkpoint offsets");
    }
    const double values[] = {
        metadata.r, metadata.R, metadata.h1, metadata.h2,
        metadata.reynolds, metadata.dt, metadata.wall_speed,
        metadata.physical_time
    };
    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite NSCyl checkpoint metadata");
        }
    }
    if (!(metadata.R > metadata.r) || !(metadata.h2 > metadata.h1)
        || !(metadata.dt > 0)) {
        throw std::runtime_error("invalid NSCyl checkpoint geometry");
    }
    const double expected_time = metadata.time_index*metadata.dt;
    const double tolerance = 8*std::numeric_limits<double>::epsilon()
        *std::max(1.0, std::abs(expected_time));
    if (std::abs(metadata.physical_time-expected_time) > tolerance) {
        throw std::runtime_error("inconsistent NSCyl checkpoint time");
    }
}

template<typename Value>
void require_equal(const char* name, const Value& actual,
                   const Value& expected) {
    if (actual != expected) {
        throw std::runtime_error(
            std::string("incompatible NSCyl checkpoint metadata: ")+name);
    }
}

void validate_compatibility(const NSCylCheckpointMetadata& actual,
                            const NSCylCheckpointMetadata& expected) {
    require_equal("schema_version", actual.schema_version,
                  expected.schema_version);
    require_equal("format_name", actual.format_name, expected.format_name);
    require_equal("step_operator_version", actual.step_operator_version,
                  expected.step_operator_version);
    require_equal("scalar_type", actual.scalar_type, expected.scalar_type);
    require_equal("axial_boundary", actual.axial_boundary,
                  expected.axial_boundary);
    require_equal("state_layout", actual.state_layout, expected.state_layout);
    require_equal("pressure_gauge", actual.pressure_gauge,
                  expected.pressure_gauge);
    require_equal("nr", actual.nr, expected.nr);
    require_equal("nphi", actual.nphi, expected.nphi);
    require_equal("nz", actual.nz, expected.nz);
    require_equal("state_size", actual.state_size, expected.state_size);
    require_equal("u_offset", actual.u_offset, expected.u_offset);
    require_equal("v_offset", actual.v_offset, expected.v_offset);
    require_equal("w_offset", actual.w_offset, expected.w_offset);
    require_equal("p_offset", actual.p_offset, expected.p_offset);
    require_equal("r", actual.r, expected.r);
    require_equal("R", actual.R, expected.R);
    require_equal("h1", actual.h1, expected.h1);
    require_equal("h2", actual.h2, expected.h2);
    require_equal("Re", actual.reynolds, expected.reynolds);
    require_equal("dt", actual.dt, expected.dt);
    require_equal("wall_speed", actual.wall_speed, expected.wall_speed);
}

int require_state_variable(int ncid, int dimension, nc_type type) {
    int variable = -1;
    nc_check(nc_inq_varid(ncid, "state", &variable),
             "reading checkpoint state variable");
    nc_type actual_type = NC_NAT;
    int dimension_count = 0;
    int dimensions[NC_MAX_VAR_DIMS];
    nc_check(nc_inq_var(ncid, variable, nullptr, &actual_type,
                        &dimension_count, dimensions, nullptr),
             "reading checkpoint state definition");
    if (actual_type != type || dimension_count != 1
        || dimensions[0] != dimension) {
        throw std::runtime_error("invalid NSCyl checkpoint state variable");
    }
    return variable;
}

template<typename T>
void put_state(int ncid, int variable, const std::vector<T>& state) {
    if constexpr (std::is_same_v<T, float>) {
        nc_check(nc_put_var_float(ncid, variable, state.data()),
                 "writing checkpoint state");
    } else {
        nc_check(nc_put_var_double(ncid, variable, state.data()),
                 "writing checkpoint state");
    }
}

template<typename T>
void get_state(int ncid, int variable, std::vector<T>& state) {
    if constexpr (std::is_same_v<T, float>) {
        nc_check(nc_get_var_float(ncid, variable, state.data()),
                 "reading checkpoint state");
    } else {
        nc_check(nc_get_var_double(ncid, variable, state.data()),
                 "reading checkpoint state");
    }
}

template<typename T>
void validate_state(const std::vector<T>& state,
                    const NSCylCheckpointMetadata& metadata) {
    long double weighted_pressure = 0;
    long double pressure_weight = 0;
    long double pressure_scale = 0;
    const long double dr = (metadata.R-metadata.r)/metadata.nr;
    for (int index = 0; index < metadata.state_size; ++index) {
        const T value = state[index];
        if (!std::isfinite(static_cast<double>(value))) {
            throw std::runtime_error("non-finite NSCyl checkpoint state");
        }
        if (index >= metadata.p_offset) {
            const int j = (index-metadata.p_offset)%metadata.nr;
            const long double radius = metadata.r+(j+0.5L)*dr;
            weighted_pressure += radius*static_cast<long double>(value);
            pressure_weight += radius;
            pressure_scale = std::max(
                pressure_scale,
                std::abs(static_cast<long double>(value)));
        }
    }

    const long double pressure_mean = weighted_pressure/pressure_weight;
    const long double tolerance = 256
        *static_cast<long double>(std::numeric_limits<T>::epsilon())
        *std::max(1.0L, pressure_scale);
    if (std::abs(pressure_mean) > tolerance) {
        throw std::runtime_error(
            "NSCyl checkpoint pressure does not satisfy its zero-mean gauge");
    }
}

} // namespace

template<typename T>
void NSCylCheckpointStorage::save_(
    const std::vector<T>& state,
    const NSCylCheckpointMetadata& metadata) const {
    validate_metadata(metadata, scalar_name<T>());
    if (state.size() != static_cast<std::size_t>(metadata.state_size)) {
        throw std::invalid_argument("NSCyl checkpoint state has the wrong size");
    }
    validate_state(state, metadata);

    int ncid = -1;
    nc_check(nc_create(filename_.c_str(), NC_CLOBBER | NC_64BIT_OFFSET,
                       &ncid),
             "creating NSCyl checkpoint");
    NcFile file(ncid);
    write_metadata(ncid, metadata);
    int dimension = -1;
    nc_check(nc_def_dim(ncid, "state_value", metadata.state_size, &dimension),
             "defining checkpoint state dimension");
    int variable = -1;
    nc_check(nc_def_var(ncid, "state", scalar_nc_type<T>(), 1, &dimension,
                        &variable),
             "defining checkpoint state variable");
    nc_check(nc_enddef(ncid), "finishing NSCyl checkpoint schema");
    put_state(ncid, variable, state);
    file.close();
}

template<typename T>
void NSCylCheckpointStorage::load_(
    std::vector<T>& output_state,
    NSCylCheckpointMetadata& output_metadata,
    const NSCylCheckpointMetadata* expected) const {
    int ncid = -1;
    nc_check(nc_open(filename_.c_str(), NC_NOWRITE, &ncid),
             "opening NSCyl checkpoint");
    NcFile file(ncid);
    NSCylCheckpointMetadata metadata = read_metadata(ncid);
    validate_metadata(metadata, scalar_name<T>());
    if (expected != nullptr) {
        validate_metadata(*expected, scalar_name<T>());
        validate_compatibility(metadata, *expected);
    }

    int dimension = -1;
    nc_check(nc_inq_dimid(ncid, "state_value", &dimension),
             "reading checkpoint state dimension");
    std::size_t state_size = 0;
    nc_check(nc_inq_dimlen(ncid, dimension, &state_size),
             "reading checkpoint state size");
    if (state_size != static_cast<std::size_t>(metadata.state_size)) {
        throw std::runtime_error("invalid NSCyl checkpoint state dimension");
    }
    const int variable = require_state_variable(
        ncid, dimension, scalar_nc_type<T>());
    std::vector<T> state(state_size);
    get_state(ncid, variable, state);
    validate_state(state, metadata);
    file.close();
    output_state = std::move(state);
    output_metadata = std::move(metadata);
}

void NSCylCheckpointStorage::save(
    const std::vector<float>& state,
    const NSCylCheckpointMetadata& metadata) const {
    save_(state, metadata);
}

void NSCylCheckpointStorage::save(
    const std::vector<double>& state,
    const NSCylCheckpointMetadata& metadata) const {
    save_(state, metadata);
}

void NSCylCheckpointStorage::load(
    std::vector<float>& state, NSCylCheckpointMetadata& metadata) const {
    load_(state, metadata, nullptr);
}

void NSCylCheckpointStorage::load(
    std::vector<double>& state, NSCylCheckpointMetadata& metadata) const {
    load_(state, metadata, nullptr);
}

void NSCylCheckpointStorage::load(
    std::vector<float>& state, NSCylCheckpointMetadata& metadata,
    const NSCylCheckpointMetadata& expected) const {
    load_(state, metadata, &expected);
}

void NSCylCheckpointStorage::load(
    std::vector<double>& state, NSCylCheckpointMetadata& metadata,
    const NSCylCheckpointMetadata& expected) const {
    load_(state, metadata, &expected);
}

} // namespace fdm
