#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mgsch.h"
#include "ns_cyl_spectral_modes.h"
#include "ns_cyl_spectral_projector.h"
#include "ns_cyl_spectral_storage.h"
#include "ns_cyl_state.h"

namespace fdm {

enum class NSCylSpectralComparisonStatus {
    comparable,
    phase_layout_change,
    unstable_dimension_change,
    coarse_only,
    fine_only
};

inline const char* ns_cyl_spectral_comparison_status_name(
    NSCylSpectralComparisonStatus status) {
    switch (status) {
    case NSCylSpectralComparisonStatus::comparable:
        return "comparable";
    case NSCylSpectralComparisonStatus::phase_layout_change:
        return "phase_layout_change";
    case NSCylSpectralComparisonStatus::unstable_dimension_change:
        return "unstable_dimension_change";
    case NSCylSpectralComparisonStatus::coarse_only:
        return "coarse_only";
    case NSCylSpectralComparisonStatus::fine_only:
        return "fine_only";
    }
    throw std::logic_error("unknown spectral comparison status");
}

struct NSCylSpectralBlockComparison {
    int m = -1;
    int l = -1;
    NSCylSpectralComparisonStatus status =
        NSCylSpectralComparisonStatus::comparable;
    int coarse_phase_count = 0;
    int fine_phase_count = 0;
    int coarse_group_count = 0;
    int fine_group_count = 0;
    int coarse_dimension = 0;
    int fine_dimension = 0;
    double coarse_leading_growth =
        std::numeric_limits<double>::quiet_NaN();
    double fine_leading_growth =
        std::numeric_limits<double>::quiet_NaN();
    double max_growth_change =
        std::numeric_limits<double>::quiet_NaN();
    double max_frequency_change =
        std::numeric_limits<double>::quiet_NaN();
    double right_subspace_sine =
        std::numeric_limits<double>::quiet_NaN();
    double right_velocity_subspace_sine =
        std::numeric_limits<double>::quiet_NaN();
    double left_subspace_sine =
        std::numeric_limits<double>::quiet_NaN();
};

struct NSCylSpectralComparison {
    std::vector<NSCylSpectralBlockComparison> blocks;
    int common_blocks = 0;
    int comparable_blocks = 0;
    int phase_layout_changes = 0;
    int unstable_dimension_changes = 0;
    int coarse_only_blocks = 0;
    int fine_only_blocks = 0;
};

namespace ns_cyl_spectral_compare_detail {

inline bool nearly_equal(double a, double b) {
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    return std::abs(a-b) <= 64*std::numeric_limits<double>::epsilon()*scale;
}

inline void validate_problem_compatibility(
    const NSCylSpectralMetadata& coarse,
    const NSCylSpectralMetadata& fine) {
    if (coarse.schema_version != fine.schema_version
        || coarse.operator_name != fine.operator_name
        || coarse.operator_version != fine.operator_version
        || coarse.scalar_type != fine.scalar_type
        || coarse.fourier_layout != fine.fourier_layout
        || coarse.state_layout != fine.state_layout
        || coarse.pressure_gauge != fine.pressure_gauge) {
        throw std::invalid_argument(
            "spectral files use different operators or layouts");
    }
    if (!nearly_equal(coarse.r, fine.r)
        || !nearly_equal(coarse.R, fine.R)
        || !nearly_equal(coarse.h1, fine.h1)
        || !nearly_equal(coarse.h2, fine.h2)
        || !nearly_equal(coarse.reynolds, fine.reynolds)
        || !nearly_equal(coarse.wall_speed, fine.wall_speed)
        || !nearly_equal(coarse.base_outer_radius,
                         fine.base_outer_radius)) {
        throw std::invalid_argument(
            "spectral files describe different physical problems");
    }
    if (coarse.nr > fine.nr || coarse.nphi > fine.nphi
        || coarse.nz > fine.nz) {
        throw std::invalid_argument(
            "the first spectral file must use the coarser grid");
    }
}

inline double linear_sample(const std::vector<double>& points,
                            const std::vector<double>& values,
                            double point) {
    if (points.size() != values.size() || points.size() < 2) {
        throw std::invalid_argument("invalid radial interpolation samples");
    }
    auto upper = std::upper_bound(points.begin(), points.end(), point);
    std::size_t right = 1;
    if (upper == points.begin()) {
        right = 1;
    } else if (upper == points.end()) {
        right = points.size()-1;
    } else {
        right = static_cast<std::size_t>(upper-points.begin());
    }
    const std::size_t left = right-1;
    const double fraction = (point-points[left])
        /(points[right]-points[left]);
    return values[left]+fraction*(values[right]-values[left]);
}

inline void orthonormalize(std::vector<std::vector<double>>& columns) {
    if (columns.empty()) {
        return;
    }
    const int size = static_cast<int>(columns.front().size());
    for (const auto& column : columns) {
        if (column.size() != static_cast<std::size_t>(size)) {
            throw std::invalid_argument("inconsistent subspace vector sizes");
        }
    }
    if (mgsch_checked<double>(
            columns, static_cast<int>(columns.size()), size,
            256*std::numeric_limits<double>::epsilon()) == 0) {
        throw std::runtime_error("rank-deficient comparison subspace");
    }
}

inline double smallest_symmetric_eigenvalue(std::vector<double> matrix,
                                             int n) {
    if (n <= 0 || matrix.size() != static_cast<std::size_t>(n)*n) {
        throw std::invalid_argument("invalid symmetric matrix size");
    }
    const int max_sweeps = std::max(32, 16*n*n);
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        int p = 0;
        int q = 1;
        double maximum = 0;
        for (int row = 0; row < n; ++row) {
            for (int column = row+1; column < n; ++column) {
                const double value = std::abs(
                    matrix[static_cast<std::size_t>(row)*n+column]);
                if (value > maximum) {
                    maximum = value;
                    p = row;
                    q = column;
                }
            }
        }
        double diagonal_scale = 1;
        for (int i = 0; i < n; ++i) {
            diagonal_scale = std::max(diagonal_scale, std::abs(
                matrix[static_cast<std::size_t>(i)*n+i]));
        }
        if (maximum <= 64*std::numeric_limits<double>::epsilon()
                       *diagonal_scale) {
            break;
        }

        const double app = matrix[static_cast<std::size_t>(p)*n+p];
        const double aqq = matrix[static_cast<std::size_t>(q)*n+q];
        const double apq = matrix[static_cast<std::size_t>(p)*n+q];
        const double angle = 0.5*std::atan2(2*apq, aqq-app);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (int i = 0; i < n; ++i) {
            if (i == p || i == q) {
                continue;
            }
            const double aip = matrix[static_cast<std::size_t>(i)*n+p];
            const double aiq = matrix[static_cast<std::size_t>(i)*n+q];
            const double new_ip = cosine*aip-sine*aiq;
            const double new_iq = sine*aip+cosine*aiq;
            matrix[static_cast<std::size_t>(i)*n+p] = new_ip;
            matrix[static_cast<std::size_t>(p)*n+i] = new_ip;
            matrix[static_cast<std::size_t>(i)*n+q] = new_iq;
            matrix[static_cast<std::size_t>(q)*n+i] = new_iq;
        }
        matrix[static_cast<std::size_t>(p)*n+p] =
            cosine*cosine*app-2*sine*cosine*apq+sine*sine*aqq;
        matrix[static_cast<std::size_t>(q)*n+q] =
            sine*sine*app+2*sine*cosine*apq+cosine*cosine*aqq;
        matrix[static_cast<std::size_t>(p)*n+q] = 0;
        matrix[static_cast<std::size_t>(q)*n+p] = 0;
    }
    double result = matrix[0];
    for (int i = 1; i < n; ++i) {
        result = std::min(result,
            matrix[static_cast<std::size_t>(i)*n+i]);
    }
    return result;
}

inline std::vector<int> minimum_cost_matching(
    const std::vector<std::pair<double, double>>& coarse,
    const std::vector<std::pair<double, double>>& fine) {
    if (coarse.size() != fine.size()) {
        throw std::invalid_argument("matching needs equal group counts");
    }
    const int n = static_cast<int>(coarse.size());
    std::vector<double> row_potential(n+1), column_potential(n+1);
    std::vector<int> matched_row(n+1), previous_column(n+1);
    for (int row = 1; row <= n; ++row) {
        matched_row[0] = row;
        int column0 = 0;
        std::vector<double> minimum(n+1,
            std::numeric_limits<double>::infinity());
        std::vector<bool> used(n+1, false);
        do {
            used[column0] = true;
            const int row0 = matched_row[column0];
            double delta = std::numeric_limits<double>::infinity();
            int column1 = 0;
            for (int column = 1; column <= n; ++column) {
                if (used[column]) {
                    continue;
                }
                const double dg = coarse[row0-1].first-fine[column-1].first;
                const double df = coarse[row0-1].second-fine[column-1].second;
                const double cost = std::hypot(dg, df)
                    -row_potential[row0]-column_potential[column];
                if (cost < minimum[column]) {
                    minimum[column] = cost;
                    previous_column[column] = column0;
                }
                if (minimum[column] < delta) {
                    delta = minimum[column];
                    column1 = column;
                }
            }
            for (int column = 0; column <= n; ++column) {
                if (used[column]) {
                    row_potential[matched_row[column]] += delta;
                    column_potential[column] -= delta;
                } else {
                    minimum[column] -= delta;
                }
            }
            column0 = column1;
        } while (matched_row[column0] != 0);
        do {
            const int column1 = previous_column[column0];
            matched_row[column0] = matched_row[column1];
            column0 = column1;
        } while (column0 != 0);
    }
    std::vector<int> result(n, -1);
    for (int column = 1; column <= n; ++column) {
        result[matched_row[column]-1] = column-1;
    }
    return result;
}

template<typename T>
std::vector<std::pair<double, double>> spectral_values(
    const std::vector<const NSCylSpectralMode<T>*>& modes) {
    std::vector<std::pair<double, double>> result;
    result.reserve(modes.size());
    for (const auto* mode : modes) {
        result.emplace_back(mode->growth_rate, mode->frequency);
    }
    return result;
}

enum class SubspaceMetric {
    full_euclidean,
    velocity_cylindrical
};

inline void apply_metric(std::vector<double>& column, int phase_count,
                         bool pressure_gauge_fixed,
                         const NSCylSpectralMetadata& metadata,
                         SubspaceMetric metric) {
    if (metric == SubspaceMetric::full_euclidean) {
        return;
    }
    const NSCylStateLayout<double> layout(
        metadata.nr, metadata.nz, metadata.nphi);
    const int full_size = phase_count*layout.radial_size;
    if (pressure_gauge_fixed) {
        if (phase_count != 1
            || column.size() != static_cast<std::size_t>(full_size-1)) {
            throw std::invalid_argument("invalid zero-gauge comparison vector");
        }
        column.resize(full_size, 0);
    } else if (column.size() != static_cast<std::size_t>(full_size)) {
        throw std::invalid_argument("invalid comparison vector size");
    }
    const double dr = (metadata.R-metadata.r)/metadata.nr;
    for (int phase = 0; phase < phase_count; ++phase) {
        const int offset = phase*layout.radial_size;
        for (int j = 1; j < metadata.nr; ++j) {
            const double radius = metadata.r+j*dr;
            column[offset+layout.radial_index(
                NSCylStateLayout<double>::Component::u, j)] *=
                std::sqrt(radius*dr);
        }
        for (auto component : {
                NSCylStateLayout<double>::Component::v,
                NSCylStateLayout<double>::Component::w}) {
            for (int j = 1; j <= metadata.nr; ++j) {
                const double radius = metadata.r+(j-0.5)*dr;
                column[offset+layout.radial_index(component, j)] *=
                    std::sqrt(radius*dr);
            }
        }
        for (int j = 1; j <= metadata.nr; ++j) {
            column[offset+layout.radial_index(
                NSCylStateLayout<double>::Component::p, j)] = 0;
        }
    }
    if (pressure_gauge_fixed) {
        column.resize(full_size-1);
    }
}

template<typename T>
std::vector<std::vector<double>> convert_basis(
    const std::vector<std::vector<T>>& input, int phase_count,
    bool pressure_gauge_fixed, const NSCylSpectralMetadata& metadata,
    SubspaceMetric metric) {
    std::vector<std::vector<double>> result;
    result.reserve(input.size());
    for (const auto& source : input) {
        result.emplace_back(source.begin(), source.end());
        apply_metric(result.back(), phase_count, pressure_gauge_fixed,
                     metadata, metric);
    }
    orthonormalize(result);
    return result;
}

} // namespace ns_cyl_spectral_compare_detail

// Prolong one real Fourier-block column between staggered radial grids.
// Velocity perturbations use their homogeneous wall values; pressure is
// extrapolated from cell centres.  The zero block is returned in the same
// weighted-zero-mean pressure gauge used by NSCylStateLayout.
template<typename T>
std::vector<double> prolong_ns_cyl_spectral_column(
    const T* source, int source_size, int phase_count,
    bool pressure_gauge_fixed,
    const NSCylSpectralMetadata& coarse,
    const NSCylSpectralMetadata& fine) {
    using Component = NSCylStateLayout<double>::Component;
    const NSCylStateLayout<double> coarse_layout(
        coarse.nr, coarse.nz, coarse.nphi);
    const NSCylStateLayout<double> fine_layout(
        fine.nr, fine.nz, fine.nphi);
    const int coarse_full_size = phase_count*coarse_layout.radial_size;
    const int fine_full_size = phase_count*fine_layout.radial_size;
    const int expected_source_size = coarse_full_size
        -(pressure_gauge_fixed ? 1 : 0);
    if (source == nullptr || source_size != expected_source_size
        || phase_count <= 0 || (pressure_gauge_fixed && phase_count != 1)) {
        throw std::invalid_argument("invalid coarse Fourier-block column");
    }
    if (!ns_cyl_spectral_compare_detail::nearly_equal(coarse.r, fine.r)
        || !ns_cyl_spectral_compare_detail::nearly_equal(coarse.R, fine.R)) {
        throw std::invalid_argument("radial domains do not match");
    }

    const double coarse_dr = (coarse.R-coarse.r)/coarse.nr;
    const double fine_dr = (fine.R-fine.r)/fine.nr;
    std::vector<double> coarse_full(coarse_full_size);
    std::copy(source, source+source_size, coarse_full.begin());
    if (pressure_gauge_fixed) {
        long double weighted_sum = 0;
        for (int j = 1; j < coarse.nr; ++j) {
            const double radius = coarse.r+(j-0.5)*coarse_dr;
            weighted_sum += radius*coarse_full[
                coarse_layout.radial_index(Component::p, j)];
        }
        const double last_radius = coarse.r+(coarse.nr-0.5)*coarse_dr;
        coarse_full[coarse_layout.radial_index(Component::p, coarse.nr)] =
            static_cast<double>(-weighted_sum/last_radius);
    }

    std::vector<double> fine_full(fine_full_size, 0);
    for (int phase = 0; phase < phase_count; ++phase) {
        const int coarse_phase = phase*coarse_layout.radial_size;
        const int fine_phase = phase*fine_layout.radial_size;

        std::vector<double> points(coarse.nr+1);
        std::vector<double> values(coarse.nr+1, 0);
        for (int j = 0; j <= coarse.nr; ++j) {
            points[j] = coarse.r+j*coarse_dr;
        }
        for (int j = 1; j < coarse.nr; ++j) {
            values[j] = coarse_full[coarse_phase
                +coarse_layout.radial_index(Component::u, j)];
        }
        for (int j = 1; j < fine.nr; ++j) {
            const double radius = fine.r+j*fine_dr;
            fine_full[fine_phase+fine_layout.radial_index(Component::u, j)] =
                ns_cyl_spectral_compare_detail::linear_sample(
                    points, values, radius);
        }

        for (Component component : {Component::v, Component::w}) {
            points.resize(coarse.nr+2);
            values.assign(coarse.nr+2, 0);
            points.front() = coarse.r;
            points.back() = coarse.R;
            for (int j = 1; j <= coarse.nr; ++j) {
                points[j] = coarse.r+(j-0.5)*coarse_dr;
                values[j] = coarse_full[coarse_phase
                    +coarse_layout.radial_index(component, j)];
            }
            for (int j = 1; j <= fine.nr; ++j) {
                const double radius = fine.r+(j-0.5)*fine_dr;
                fine_full[fine_phase+fine_layout.radial_index(component, j)] =
                    ns_cyl_spectral_compare_detail::linear_sample(
                        points, values, radius);
            }
        }

        points.resize(coarse.nr);
        values.resize(coarse.nr);
        for (int j = 1; j <= coarse.nr; ++j) {
            points[j-1] = coarse.r+(j-0.5)*coarse_dr;
            values[j-1] = coarse_full[coarse_phase
                +coarse_layout.radial_index(Component::p, j)];
        }
        for (int j = 1; j <= fine.nr; ++j) {
            const double radius = fine.r+(j-0.5)*fine_dr;
            fine_full[fine_phase+fine_layout.radial_index(Component::p, j)] =
                ns_cyl_spectral_compare_detail::linear_sample(
                    points, values, radius);
        }
    }

    if (!pressure_gauge_fixed) {
        return fine_full;
    }
    long double weighted_sum = 0;
    long double weight = 0;
    for (int j = 1; j <= fine.nr; ++j) {
        const double radius = fine.r+(j-0.5)*fine_dr;
        weighted_sum += radius*fine_full[
            fine_layout.radial_index(Component::p, j)];
        weight += radius;
    }
    const double pressure_mean = static_cast<double>(weighted_sum/weight);
    for (int j = 1; j <= fine.nr; ++j) {
        fine_full[fine_layout.radial_index(Component::p, j)] -= pressure_mean;
    }
    fine_full.resize(fine_full_size-1);
    return fine_full;
}

inline double ns_cyl_subspace_maximum_sine(
    std::vector<std::vector<double>> first,
    std::vector<std::vector<double>> second) {
    if (first.size() != second.size() || first.empty()) {
        throw std::invalid_argument(
            "principal angles need non-empty equal-dimensional subspaces");
    }
    ns_cyl_spectral_compare_detail::orthonormalize(first);
    ns_cyl_spectral_compare_detail::orthonormalize(second);
    const int dimension = static_cast<int>(first.size());
    const std::size_t vector_size = first.front().size();
    if (second.front().size() != vector_size) {
        throw std::invalid_argument("subspaces use different ambient spaces");
    }
    std::vector<double> cross(static_cast<std::size_t>(dimension)*dimension);
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            long double value = 0;
            for (std::size_t row = 0; row < vector_size; ++row) {
                value += static_cast<long double>(first[i][row])*second[j][row];
            }
            cross[static_cast<std::size_t>(i)*dimension+j] =
                static_cast<double>(value);
        }
    }
    std::vector<double> gram(static_cast<std::size_t>(dimension)*dimension, 0);
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            long double value = 0;
            for (int k = 0; k < dimension; ++k) {
                value += static_cast<long double>(
                    cross[static_cast<std::size_t>(k)*dimension+i])
                    *cross[static_cast<std::size_t>(k)*dimension+j];
            }
            gram[static_cast<std::size_t>(i)*dimension+j] =
                static_cast<double>(value);
        }
    }
    double smallest_squared_cosine = std::clamp(
        ns_cyl_spectral_compare_detail::smallest_symmetric_eigenvalue(
            std::move(gram), dimension), 0.0, 1.0);
    if (1-smallest_squared_cosine
        < 512*std::numeric_limits<double>::epsilon()) {
        smallest_squared_cosine = 1;
    }
    return std::sqrt(std::max(0.0, 1-smallest_squared_cosine));
}

template<typename T>
NSCylSpectralComparison compare_ns_cyl_spectral_mode_sets(
    const NSCylSpectralModeSet<T>& coarse_modes,
    const NSCylSpectralMetadata& coarse_metadata,
    const NSCylSpectralModeSet<T>& fine_modes,
    const NSCylSpectralMetadata& fine_metadata) {
    using namespace ns_cyl_spectral_compare_detail;
    validate_problem_compatibility(coarse_metadata, fine_metadata);

    using Index = std::pair<int, int>;
    std::map<Index, std::vector<const NSCylSpectralMode<T>*>> coarse_groups;
    std::map<Index, std::vector<const NSCylSpectralMode<T>*>> fine_groups;
    for (const auto& mode : coarse_modes.modes()) {
        coarse_groups[{mode.m, mode.l}].push_back(&mode);
    }
    for (const auto& mode : fine_modes.modes()) {
        fine_groups[{mode.m, mode.l}].push_back(&mode);
    }
    const double condition_limit = std::max(
        coarse_metadata.condition_limit, fine_metadata.condition_limit);
    const NSCylSpectralProjector<T> coarse_projector(
        coarse_modes, condition_limit);
    const NSCylSpectralProjector<T> fine_projector(
        fine_modes, condition_limit);

    std::set<Index> indices;
    for (const auto& [index, modes] : coarse_groups) {
        indices.insert(index);
    }
    for (const auto& [index, modes] : fine_groups) {
        indices.insert(index);
    }

    NSCylSpectralComparison result;
    for (const auto& [m, l] : indices) {
        NSCylSpectralBlockComparison row;
        row.m = m;
        row.l = l;
        const auto coarse_iterator = coarse_groups.find({m, l});
        const auto fine_iterator = fine_groups.find({m, l});
        if (coarse_iterator == coarse_groups.end()) {
            row.status = NSCylSpectralComparisonStatus::fine_only;
            row.fine_group_count = static_cast<int>(fine_iterator->second.size());
            for (const auto* mode : fine_iterator->second) {
                row.fine_dimension += mode->column_count;
                row.fine_phase_count = mode->phase_count;
                row.fine_leading_growth = std::isnan(row.fine_leading_growth)
                    ? mode->growth_rate
                    : std::max(row.fine_leading_growth, mode->growth_rate);
            }
            ++result.fine_only_blocks;
            result.blocks.push_back(row);
            continue;
        }
        if (fine_iterator == fine_groups.end()) {
            row.status = NSCylSpectralComparisonStatus::coarse_only;
            row.coarse_group_count =
                static_cast<int>(coarse_iterator->second.size());
            for (const auto* mode : coarse_iterator->second) {
                row.coarse_dimension += mode->column_count;
                row.coarse_phase_count = mode->phase_count;
                row.coarse_leading_growth =
                    std::isnan(row.coarse_leading_growth)
                    ? mode->growth_rate
                    : std::max(row.coarse_leading_growth, mode->growth_rate);
            }
            ++result.coarse_only_blocks;
            result.blocks.push_back(row);
            continue;
        }

        ++result.common_blocks;
        row.coarse_group_count =
            static_cast<int>(coarse_iterator->second.size());
        row.fine_group_count = static_cast<int>(fine_iterator->second.size());
        for (const auto* mode : coarse_iterator->second) {
            row.coarse_dimension += mode->column_count;
            row.coarse_phase_count = mode->phase_count;
            row.coarse_leading_growth = std::isnan(row.coarse_leading_growth)
                ? mode->growth_rate
                : std::max(row.coarse_leading_growth, mode->growth_rate);
        }
        for (const auto* mode : fine_iterator->second) {
            row.fine_dimension += mode->column_count;
            row.fine_phase_count = mode->phase_count;
            row.fine_leading_growth = std::isnan(row.fine_leading_growth)
                ? mode->growth_rate
                : std::max(row.fine_leading_growth, mode->growth_rate);
        }

        if (row.coarse_group_count == row.fine_group_count) {
            const auto coarse_values = spectral_values(coarse_iterator->second);
            const auto fine_values = spectral_values(fine_iterator->second);
            const auto matching = minimum_cost_matching(coarse_values, fine_values);
            row.max_growth_change = 0;
            row.max_frequency_change = 0;
            for (int i = 0; i < row.coarse_group_count; ++i) {
                row.max_growth_change = std::max(row.max_growth_change,
                    std::abs(coarse_values[i].first
                             -fine_values[matching[i]].first));
                row.max_frequency_change = std::max(row.max_frequency_change,
                    std::abs(coarse_values[i].second
                             -fine_values[matching[i]].second));
            }
        }

        if (row.coarse_phase_count != row.fine_phase_count) {
            row.status = NSCylSpectralComparisonStatus::phase_layout_change;
            ++result.phase_layout_changes;
            result.blocks.push_back(row);
            continue;
        }
        if (row.coarse_dimension != row.fine_dimension
            || row.coarse_group_count != row.fine_group_count) {
            row.status =
                NSCylSpectralComparisonStatus::unstable_dimension_change;
            ++result.unstable_dimension_changes;
            result.blocks.push_back(row);
            continue;
        }

        const auto* coarse_block = coarse_projector.find_block(m, l);
        const auto* fine_block = fine_projector.find_block(m, l);
        if (coarse_block == nullptr || fine_block == nullptr) {
            throw std::logic_error("missing common spectral projector block");
        }
        auto prolong_basis = [&](const auto& basis, SubspaceMetric metric) {
            std::vector<std::vector<double>> prolonged;
            prolonged.reserve(basis.size());
            for (const auto& column : basis) {
                prolonged.push_back(prolong_ns_cyl_spectral_column(
                    column.data(), static_cast<int>(column.size()),
                    coarse_block->phase_count(),
                    coarse_block->pressure_gauge_fixed(),
                    coarse_metadata, fine_metadata));
                apply_metric(prolonged.back(), fine_block->phase_count(),
                             fine_block->pressure_gauge_fixed(), fine_metadata,
                             metric);
            }
            orthonormalize(prolonged);
            return prolonged;
        };
        const auto prolonged_right = prolong_basis(
            coarse_block->right_basis(), SubspaceMetric::full_euclidean);
        const auto fine_right = convert_basis(
            fine_block->right_basis(), fine_block->phase_count(),
            fine_block->pressure_gauge_fixed(), fine_metadata,
            SubspaceMetric::full_euclidean);
        row.right_subspace_sine = ns_cyl_subspace_maximum_sine(
            prolonged_right, fine_right);

        const auto prolonged_velocity = prolong_basis(
            coarse_block->right_basis(), SubspaceMetric::velocity_cylindrical);
        const auto fine_velocity = convert_basis(
            fine_block->right_basis(), fine_block->phase_count(),
            fine_block->pressure_gauge_fixed(), fine_metadata,
            SubspaceMetric::velocity_cylindrical);
        row.right_velocity_subspace_sine = ns_cyl_subspace_maximum_sine(
            prolonged_velocity, fine_velocity);

        const auto prolonged_left = prolong_basis(
            coarse_block->left_basis(), SubspaceMetric::full_euclidean);
        const auto fine_left = convert_basis(
            fine_block->left_basis(), fine_block->phase_count(),
            fine_block->pressure_gauge_fixed(), fine_metadata,
            SubspaceMetric::full_euclidean);
        row.left_subspace_sine = ns_cyl_subspace_maximum_sine(
            prolonged_left, fine_left);
        ++result.comparable_blocks;
        result.blocks.push_back(row);
    }
    return result;
}

} // namespace fdm
