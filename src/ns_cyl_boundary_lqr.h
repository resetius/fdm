#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "config.h"
#include "fft.h"
#include "ns_cyl_fourier_block.h"
#include "ns_cyl_fourier_native.h"
#include "ns_cyl_state.h"
#include "projection.h"

namespace fdm {

template<typename T>
struct NSCylBoundaryLQRBlockDiagnostics {
    int m = -1;
    int l = -1;
    int input_size = 0;
    double predicted_cost_before = 0;
    double predicted_cost_after = 0;
    double control_norm = 0;
};

template<typename T>
struct NSCylBoundaryLQRResult {
    int nphi = 0;
    int nz = 0;
    std::vector<T> radial;
    std::vector<T> axial;
    std::vector<T> azimuthal;
    double predicted_cost_before = 0;
    double predicted_cost_after = 0;
    std::vector<NSCylBoundaryLQRBlockDiagnostics<T>> blocks;

    double rms_norm() const {
        if (radial.empty()) {
            return 0;
        }
        long double sum = 0;
        for (std::size_t index = 0; index < radial.size(); ++index) {
            const long double ur = radial[index];
            const long double uz = axial[index];
            const long double uphi = azimuthal[index];
            sum += ur*ur+uz*uz+uphi*uphi;
        }
        return std::sqrt(static_cast<double>(sum/radial.size()));
    }

    double maximum_norm() const {
        double result = 0;
        for (std::size_t index = 0; index < radial.size(); ++index) {
            result = std::max(result, std::sqrt(
                static_cast<double>(radial[index])*radial[index]
                +static_cast<double>(axial[index])*axial[index]
                +static_cast<double>(azimuthal[index])*azimuthal[index]));
        }
        return result;
    }
};

// Condensed finite-horizon controller for one exact discrete Fourier-block
// plant.  The sampled state is augmented by the wall velocity from the
// preceding interval:
//
//   z_j=(q_j,b_j),  q_{j+1}=A q_j+D b_j+E b_{j+1}.
//
// A physical NSCyl step uses b_j in the momentum boundary stencil and
// b_{j+1} in the same-time pressure Neumann condition.  Treating b_{j+1} as
// an impulse added directly to q would therefore be a different plant.
template<typename T>
class NSCylFourierBoundaryLQR {
public:
    using Native = NSCylFourierBlockNative<T>;
    using Layout = NSCylStateLayout<T>;
    using Component = typename Layout::Component;
    using BlockDiagnostics = NSCylBoundaryLQRBlockDiagnostics<T>;

    NSCylFourierBoundaryLQR(const Config& config, int m, int l,
                            int horizon_intervals, int interval_steps,
                            double control_weight, double ridge,
                            const std::string& components = "all")
        : dynamics_(config, m, l, interval_steps)
        , horizon_(horizon_intervals)
        , control_weight_(control_weight)
        , ridge_(ridge)
        , phase_gram_(build_phase_gram()) {
        if (horizon_ <= 0 || !(control_weight_ >= 0)
            || !std::isfinite(control_weight_) || !(ridge_ >= 0)
            || !std::isfinite(ridge_)) {
            throw std::invalid_argument(
                "invalid physical boundary LQR settings");
        }
        if (components != "all" && components != "tangential"
            && components != "azimuthal") {
            throw std::invalid_argument(
                "physical boundary LQR components must be 'all', "
                "'tangential', or 'azimuthal'");
        }
        for (int component = 0; component < 3; ++component) {
            const bool enabled = components == "all"
                || (components == "tangential" && component != 0)
                || (components == "azimuthal" && component == 2);
            if (!enabled) {
                continue;
            }
            for (int phase = 0; phase < dynamics_.phase_count(); ++phase) {
                // A spatially constant radial wall velocity has nonzero net
                // flux and is incompatible with the closed periodic cylinder.
                if (component == 0 && dynamics_.m() == 0
                    && dynamics_.l() == 0) {
                    continue;
                }
                input_indices_.push_back(
                    component*dynamics_.phase_count()+phase);
            }
        }
        build_impulse_responses();
        build_inverse_hessian();
    }

    int m() const { return dynamics_.m(); }
    int l() const { return dynamics_.l(); }
    int state_size() const { return dynamics_.size(); }
    int boundary_size() const { return dynamics_.outer_boundary_size(); }
    int input_size() const { return static_cast<int>(input_indices_.size()); }
    int phase_count() const { return dynamics_.phase_count(); }
    int horizon_intervals() const { return horizon_; }
    int interval_steps() const { return dynamics_.operator_steps(); }

    const Native& dynamics() const { return dynamics_; }

    std::vector<T> control(const T* state, const T* current_boundary,
                           BlockDiagnostics* diagnostics = nullptr) {
        if (!state || !current_boundary) {
            throw std::invalid_argument(
                "null physical boundary LQR state");
        }
        const int problem_size = horizon_*input_size();
        std::vector<T> free_state(state, state+state_size());
        std::vector<T> old_boundary(
            current_boundary, current_boundary+boundary_size());
        std::vector<T> zero_boundary(boundary_size(), T(0));
        std::vector<T> next_state(state_size(), T(0));
        std::vector<T> linear(problem_size, T(0));
        long double free_cost = 0;

        for (int sample = 1; sample <= horizon_; ++sample) {
            dynamics_.apply_with_outer_boundary(
                next_state.data(), free_state.data(), old_boundary.data(),
                zero_boundary.data());
            free_state.swap(next_state);
            old_boundary = zero_boundary;
            free_cost += velocity_inner_product(
                free_state.data(), free_state.data());
            for (int stage = 0; stage < sample; ++stage) {
                const int lag = sample-stage;
                for (int input = 0; input < input_size(); ++input) {
                    linear[stage*input_size()+input] += static_cast<T>(
                        velocity_inner_product(
                            impulse_response(lag, input).data(),
                            free_state.data()));
                }
            }
        }

        std::vector<T> plan(problem_size, T(0));
        for (int row = 0; row < problem_size; ++row) {
            for (int column = 0; column < problem_size; ++column) {
                plan[row] -= inverse_hessian_[
                    static_cast<std::size_t>(row)*problem_size+column]
                    *linear[column];
            }
        }

        long double reduction = 0;
        for (int coordinate = 0; coordinate < problem_size; ++coordinate) {
            reduction += static_cast<long double>(plan[coordinate])
                *linear[coordinate];
        }
        const double predicted_after = static_cast<double>(
            free_cost+reduction);
        const double tolerance = 256*std::numeric_limits<T>::epsilon()
            *std::max(1.0, static_cast<double>(free_cost));
        if (!std::isfinite(predicted_after)
            || predicted_after > static_cast<double>(free_cost)+tolerance
            || predicted_after < -tolerance) {
            throw std::runtime_error(
                "physical boundary LQR failed to reduce its quadratic cost "
                "in Fourier block (m="+std::to_string(m())+",l="
                +std::to_string(l())+")");
        }

        std::vector<T> result(boundary_size(), T(0));
        for (int input = 0; input < input_size(); ++input) {
            result[input_indices_[input]] = plan[input];
        }

        if (diagnostics) {
            diagnostics->m = m();
            diagnostics->l = l();
            diagnostics->input_size = input_size();
            diagnostics->predicted_cost_before =
                static_cast<double>(free_cost);
            long double norm2 = 0;
            for (int input = 0; input < input_size(); ++input) {
                norm2 += static_cast<long double>(plan[input])*plan[input];
            }
            diagnostics->predicted_cost_after = std::max(
                0.0, predicted_after);
            diagnostics->control_norm = std::sqrt(
                static_cast<double>(norm2));
        }
        return result;
    }

    double velocity_inner_product(const T* first, const T* second) const {
        const int phases = phase_count();
        const long double cell_measure = dynamics_.dr*dynamics_.dphi
            *dynamics_.dz;
        long double result = 0;
        for (Component component : {
                 Component::u, Component::v, Component::w}) {
            const int radial_end = component == Component::u
                ? dynamics_.nr-1 : dynamics_.nr;
            for (int j = 1; j <= radial_end; ++j) {
                const long double radius = component == Component::u
                    ? dynamics_.r0+j*dynamics_.dr
                    : dynamics_.r0+(j-0.5L)*dynamics_.dr;
                const int radial = dynamics_.state_layout().radial_index(
                    component, j);
                for (int row_phase = 0; row_phase < phases; ++row_phase) {
                    const T first_value = first[
                        static_cast<std::size_t>(row_phase)
                            *dynamics_.radial_size()+radial];
                    for (int column_phase = 0;
                         column_phase < phases; ++column_phase) {
                        const T second_value = second[
                            static_cast<std::size_t>(column_phase)
                                *dynamics_.radial_size()+radial];
                        result += radius*static_cast<long double>(first_value)
                            *phase_gram_[static_cast<std::size_t>(row_phase)
                                *phases+column_phase]*second_value;
                    }
                }
            }
        }
        return static_cast<double>(cell_measure*result);
    }

private:
    Native dynamics_;
    int horizon_;
    double control_weight_;
    double ridge_;
    std::vector<T> phase_gram_;
    std::vector<int> input_indices_;
    std::vector<std::vector<T>> impulse_responses_;
    std::vector<T> inverse_hessian_;

    std::vector<T> build_phase_gram() const {
        const int phases = phase_count();
        std::vector<T> result(
            static_cast<std::size_t>(phases)*phases, T(0));
        for (int row = 0; row < phases; ++row) {
            for (int column = 0; column < phases; ++column) {
                long double entry = 0;
                for (int i = 0; i < dynamics_.nphi; ++i) {
                    for (int k = 0; k < dynamics_.nz; ++k) {
                        entry += static_cast<long double>(
                            dynamics_.phase_value(row, i, k))
                            *dynamics_.phase_value(column, i, k);
                    }
                }
                result[static_cast<std::size_t>(row)*phases+column] =
                    static_cast<T>(entry);
            }
        }
        return result;
    }

    const std::vector<T>& impulse_response(int lag, int input) const {
        return impulse_responses_[static_cast<std::size_t>(lag-1)
            *input_size()+input];
    }

    void build_impulse_responses() {
        impulse_responses_.assign(
            static_cast<std::size_t>(horizon_)*input_size(),
            std::vector<T>(state_size(), T(0)));
        std::vector<T> zero_state(state_size(), T(0));
        std::vector<T> zero_boundary(boundary_size(), T(0));
        std::vector<T> unit_boundary(boundary_size(), T(0));
        std::vector<T> state(state_size(), T(0));
        std::vector<T> next_state(state_size(), T(0));
        for (int input = 0; input < input_size(); ++input) {
            std::fill(unit_boundary.begin(), unit_boundary.end(), T(0));
            unit_boundary[input_indices_[input]] = T(1);
            dynamics_.apply_with_outer_boundary(
                state.data(), zero_state.data(), zero_boundary.data(),
                unit_boundary.data());
            impulse_response_mutable(1, input) = state;
            for (int lag = 2; lag <= horizon_; ++lag) {
                dynamics_.apply_with_outer_boundary(
                    next_state.data(), state.data(),
                    lag == 2 ? unit_boundary.data() : zero_boundary.data(),
                    zero_boundary.data());
                state.swap(next_state);
                impulse_response_mutable(lag, input) = state;
            }
        }
    }

    std::vector<T>& impulse_response_mutable(int lag, int input) {
        return impulse_responses_[static_cast<std::size_t>(lag-1)
            *input_size()+input];
    }

    double boundary_inner_product(int first_input, int second_input) const {
        const int phases = phase_count();
        const int first = input_indices_[first_input];
        const int second = input_indices_[second_input];
        if (first/phases != second/phases) {
            return 0;
        }
        return static_cast<double>(phase_gram_[
            static_cast<std::size_t>(first%phases)*phases+second%phases])
            /(dynamics_.nphi*dynamics_.nz);
    }

    void build_inverse_hessian() {
        const int inputs = input_size();
        const int problem_size = horizon_*inputs;
        std::vector<T> hessian(
            static_cast<std::size_t>(problem_size)*problem_size, T(0));
        for (int first_stage = 0; first_stage < horizon_; ++first_stage) {
            for (int second_stage = first_stage;
                 second_stage < horizon_; ++second_stage) {
                for (int first_input = 0; first_input < inputs;
                     ++first_input) {
                    for (int second_input = 0; second_input < inputs;
                         ++second_input) {
                        long double entry = 0;
                        for (int sample = second_stage+1;
                             sample <= horizon_; ++sample) {
                            entry += velocity_inner_product(
                                impulse_response(
                                    sample-first_stage, first_input).data(),
                                impulse_response(
                                    sample-second_stage,
                                    second_input).data());
                        }
                        if (first_stage == second_stage
                            && control_weight_ != 0) {
                            entry += control_weight_*boundary_inner_product(
                                first_input, second_input);
                        }
                        const int row = first_stage*inputs+first_input;
                        const int column = second_stage*inputs+second_input;
                        const T value = static_cast<T>(entry);
                        hessian[static_cast<std::size_t>(row)*problem_size
                                +column] = value;
                        hessian[static_cast<std::size_t>(column)*problem_size
                                +row] = value;
                    }
                }
            }
        }

        if (ridge_ != 0) {
            long double diagonal_sum = 0;
            for (int row = 0; row < problem_size; ++row) {
                diagonal_sum += std::abs(static_cast<long double>(
                    hessian[static_cast<std::size_t>(row)*problem_size+row]));
            }
            const long double scale = diagonal_sum > 0
                ? diagonal_sum/problem_size : 1;
            for (int row = 0; row < problem_size; ++row) {
                hessian[static_cast<std::size_t>(row)*problem_size+row] +=
                    static_cast<T>(ridge_*scale);
            }
        }

        inverse_hessian_.resize(hessian.size());
        const T pivot = inverse_general_matrix(
            inverse_hessian_.data(), hessian.data(), problem_size);
        if (!(pivot > T(0))) {
            throw std::runtime_error(
                "singular physical boundary LQR problem in Fourier block (m="
                +std::to_string(m())+",l="+std::to_string(l())+")");
        }
    }
};

// Transform a physical packed NSCyl perturbation once, run independent exact
// block controllers, and synthesize their wall commands on the (phi,z) grid.
template<typename T>
class NSCylBoundaryLQR {
public:
    using Layout = NSCylStateLayout<T>;
    using Component = typename Layout::Component;
    using Result = NSCylBoundaryLQRResult<T>;

    NSCylBoundaryLQR(const Config& config,
                     const std::vector<std::pair<int, int>>& blocks,
                     int horizon_intervals, int interval_steps,
                     double control_weight, double ridge,
                     const std::string& components = "all")
        : nr_(config.get("ns", "nr", 32))
        , nphi_(config.get("ns", "nphi", 32))
        , nz_(config.get("ns", "nz", 32))
        , layout_(nr_, nz_, nphi_)
        , fft_(nphi_, nz_)
        , packed_state_(layout_.state_size)
        , plane_values_(fft_.size())
        , plane_coefficients_(fft_.size()) {
        std::set<std::pair<int, int>> unique;
        for (const auto& index : blocks) {
            if (!unique.insert(index).second) {
                continue;
            }
            controllers_.push_back(std::make_unique<
                NSCylFourierBoundaryLQR<T>>(
                    config, index.first, index.second,
                    horizon_intervals, interval_steps,
                    control_weight, ridge, components));
        }
        if (controllers_.empty()) {
            throw std::invalid_argument(
                "physical boundary LQR needs at least one Fourier block");
        }
    }

    std::size_t block_count() const { return controllers_.size(); }

    Result control(const std::vector<T>& state,
                   const std::vector<T>& current_radial,
                   const std::vector<T>& current_axial,
                   const std::vector<T>& current_azimuthal) {
        if (static_cast<int>(state.size()) != layout_.state_size) {
            throw std::invalid_argument(
                "physical boundary LQR state has the wrong size");
        }
        const std::size_t plane_size = fft_.size();
        if (current_radial.size() != plane_size
            || current_axial.size() != plane_size
            || current_azimuthal.size() != plane_size) {
            throw std::invalid_argument(
                "physical boundary LQR wall plane has the wrong size");
        }
        analyze_state(state);
        std::vector<std::vector<T>> boundary_coefficients(
            3, std::vector<T>(plane_size, T(0)));
        analyze_plane(current_radial, boundary_coefficients[0]);
        analyze_plane(current_axial, boundary_coefficients[1]);
        analyze_plane(current_azimuthal, boundary_coefficients[2]);
        std::vector<std::vector<T>> next_coefficients(
            3, std::vector<T>(plane_size, T(0)));

        Result result;
        result.nphi = nphi_;
        result.nz = nz_;
        for (auto& controller : controllers_) {
            std::vector<T> block_state = gather_state(*controller);
            std::vector<T> block_boundary = gather_boundary(
                *controller, boundary_coefficients);
            typename NSCylFourierBoundaryLQR<T>::BlockDiagnostics diagnostic;
            const auto next = controller->control(
                block_state.data(), block_boundary.data(), &diagnostic);
            scatter_boundary(*controller, next, next_coefficients);
            result.predicted_cost_before += diagnostic.predicted_cost_before;
            result.predicted_cost_after += diagnostic.predicted_cost_after;
            result.blocks.push_back(diagnostic);
        }

        result.radial.resize(plane_size);
        result.axial.resize(plane_size);
        result.azimuthal.resize(plane_size);
        fft_.synthesis(next_coefficients[0].data(), result.radial.data());
        fft_.synthesis(next_coefficients[1].data(), result.axial.data());
        fft_.synthesis(
            next_coefficients[2].data(), result.azimuthal.data());
        return result;
    }

private:
    int nr_;
    int nphi_;
    int nz_;
    Layout layout_;
    PeriodicPackedFFT2<T> fft_;
    std::vector<std::unique_ptr<NSCylFourierBoundaryLQR<T>>> controllers_;
    std::vector<T> packed_state_;
    std::vector<T> plane_values_;
    std::vector<T> plane_coefficients_;

    std::size_t plane_index(int i, int k) const {
        return static_cast<std::size_t>(i)*nz_+k;
    }

    int state_index(Component component, int i, int k, int j) const {
        int offset = 0;
        int radial_size = nr_;
        switch (component) {
        case Component::u:
            offset = layout_.u_offset;
            radial_size = nr_-1;
            break;
        case Component::v: offset = layout_.v_offset; break;
        case Component::w: offset = layout_.w_offset; break;
        case Component::p: offset = layout_.p_offset; break;
        }
        return offset+(i*nz_+k)*radial_size+j-1;
    }

    void analyze_state(const std::vector<T>& state) {
        layout_.for_each_radial([&](Component component, int j, int) {
            for (int i = 0; i < nphi_; ++i) {
                for (int k = 0; k < nz_; ++k) {
                    plane_values_[plane_index(i, k)] = state[
                        state_index(component, i, k, j)];
                }
            }
            fft_.analysis(plane_values_.data(), plane_coefficients_.data());
            for (int i = 0; i < nphi_; ++i) {
                for (int k = 0; k < nz_; ++k) {
                    packed_state_[state_index(component, i, k, j)] =
                        plane_coefficients_[plane_index(i, k)];
                }
            }
        });
    }

    void analyze_plane(const std::vector<T>& physical,
                       std::vector<T>& coefficients) {
        fft_.analysis(physical.data(), coefficients.data());
    }

    template<typename Controller>
    std::vector<T> gather_state(const Controller& controller) const {
        std::vector<T> full(controller.phase_count()*layout_.radial_size);
        int phase = 0;
        for (int i : packed_indices(controller.m(), nphi_)) {
            for (int k : packed_indices(controller.l(), nz_)) {
                layout_.for_each_radial(
                    [&](Component component, int j, int radial) {
                        full[static_cast<std::size_t>(phase)
                                 *layout_.radial_size+radial] =
                            packed_state_[state_index(component, i, k, j)];
                    });
                ++phase;
            }
        }
        full.resize(controller.state_size());
        return full;
    }

    template<typename Controller>
    std::vector<T> gather_boundary(
        const Controller& controller,
        const std::vector<std::vector<T>>& coefficients) const {
        std::vector<T> result(controller.boundary_size());
        for (int component = 0; component < 3; ++component) {
            int phase = 0;
            for (int i : packed_indices(controller.m(), nphi_)) {
                for (int k : packed_indices(controller.l(), nz_)) {
                    result[component*controller.phase_count()+phase] =
                        coefficients[component][plane_index(i, k)];
                    ++phase;
                }
            }
        }
        return result;
    }

    template<typename Controller>
    void scatter_boundary(
        const Controller& controller, const std::vector<T>& block,
        std::vector<std::vector<T>>& coefficients) const {
        for (int component = 0; component < 3; ++component) {
            int phase = 0;
            for (int i : packed_indices(controller.m(), nphi_)) {
                for (int k : packed_indices(controller.l(), nz_)) {
                    coefficients[component][plane_index(i, k)] =
                        block[component*controller.phase_count()+phase];
                    ++phase;
                }
            }
        }
    }

    static std::vector<int> packed_indices(int frequency, int size) {
        if (frequency == 0 || 2*frequency == size) {
            return {frequency};
        }
        return {frequency, size-frequency};
    }
};

} // namespace fdm
