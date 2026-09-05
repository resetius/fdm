#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "config.h"
#include "ns_cyl.h"
#include "ns_cyl_fourier_block.h"
#include "ns_cyl_state.h"

namespace fdm {

template<typename T>
struct NSCylFourierBatchRequest {
    int m = 0;
    int l = 0;
    const T* input = nullptr;
    T* output = nullptr;
    int size = 0;
};

namespace ns_cyl_fourier_batch_detail {

inline std::vector<int> packed_indices(int q, int n) {
    if (q < 0 || q > n/2) {
        throw std::invalid_argument(
            "batched Fourier frequency is outside [0,N/2]");
    }
    if (q == 0 || 2*q == n) {
        return {q};
    }
    return {q, n-q};
}

template<typename T>
class Transform {
public:
    using Request = NSCylFourierBatchRequest<T>;
    using Layout = NSCylStateLayout<T>;
    using Component = typename Layout::Component;

    Transform(int nr, int nz, int nphi)
        : layout_(nr, nz, nphi)
        , fft_(nphi, nz)
        , coefficients_(fft_.size())
        , values_(fft_.size())
    { }

    const Layout& layout() const { return layout_; }

    template<typename Geometry>
    void lift(const Geometry& geometry, const std::vector<Request>& requests,
              std::vector<T>& physical) {
        prepare(geometry, requests);
        physical.assign(layout_.state_size, T(0));
        layout_.for_each_radial(
            [&](Component component, int, int radial_index) {
                std::fill(coefficients_.begin(), coefficients_.end(), T(0));
                for (std::size_t request_index = 0;
                     request_index < requests.size(); ++request_index) {
                    const Prepared& item = prepared_[request_index];
                    if (item.norm == 0) {
                        continue;
                    }
                    const T* input = item.expanded.empty()
                        ? requests[request_index].input
                        : item.expanded.data();
                    for (std::size_t pi = 0; pi < item.phi.size(); ++pi) {
                        for (std::size_t zi = 0; zi < item.z.size(); ++zi) {
                            const int phase = static_cast<int>(pi*item.z.size()+zi);
                            coefficients_[plane_index(item.phi[pi], item.z[zi])] =
                                input[phase*layout_.radial_size+radial_index]
                                /item.norm;
                        }
                    }
                }
                fft_.synthesis(coefficients_.data(), values_.data());
                copy_values_to_physical(component, radial_index, physical);
            });
    }

    template<typename Geometry>
    void extract(const Geometry& geometry, const std::vector<Request>& requests,
                 const std::vector<T>& physical) {
        if (requests.size() != prepared_.size()) {
            throw std::logic_error("batched Fourier request set changed");
        }
        for (std::size_t request_index = 0;
             request_index < requests.size(); ++request_index) {
            if (prepared_[request_index].gauge) {
                prepared_[request_index].expanded.assign(
                    prepared_[request_index].full_size, T(0));
            } else {
                std::fill(requests[request_index].output,
                          requests[request_index].output+requests[request_index].size,
                          T(0));
            }
        }

        layout_.for_each_radial(
            [&](Component component, int, int radial_index) {
                copy_physical_to_values(component, radial_index, physical);
                fft_.analysis(values_.data(), coefficients_.data());
                for (std::size_t request_index = 0;
                     request_index < requests.size(); ++request_index) {
                    Prepared& item = prepared_[request_index];
                    if (item.norm == 0) {
                        continue;
                    }
                    T* output = item.gauge
                        ? item.expanded.data()
                        : requests[request_index].output;
                    for (std::size_t pi = 0; pi < item.phi.size(); ++pi) {
                        for (std::size_t zi = 0; zi < item.z.size(); ++zi) {
                            const int phase = static_cast<int>(pi*item.z.size()+zi);
                            output[phase*layout_.radial_size+radial_index] =
                                coefficients_[plane_index(item.phi[pi], item.z[zi])]
                                *item.norm;
                        }
                    }
                }
            });

        for (std::size_t request_index = 0;
             request_index < requests.size(); ++request_index) {
            Prepared& item = prepared_[request_index];
            if (item.norm == 0) {
                std::fill(requests[request_index].output,
                          requests[request_index].output+requests[request_index].size,
                          T(0));
            } else if (item.gauge) {
                layout_.reduce_zero_gauge_block(
                    geometry, item.expanded.data(),
                    requests[request_index].output);
            }
        }
    }

private:
    struct Prepared {
        std::vector<int> phi;
        std::vector<int> z;
        double norm = 0;
        bool gauge = false;
        int full_size = 0;
        std::vector<T> expanded;
    };

    Layout layout_;
    PeriodicPackedFFT2<T> fft_;
    std::vector<T> coefficients_;
    std::vector<T> values_;
    std::vector<Prepared> prepared_;

    template<typename Geometry>
    void prepare(const Geometry& geometry,
                 const std::vector<Request>& requests) {
        if (requests.empty()) {
            throw std::invalid_argument("empty Fourier batch");
        }
        std::set<std::pair<int, int>> indices;
        prepared_.clear();
        prepared_.reserve(requests.size());
        for (const Request& request : requests) {
            if (!request.input || !request.output) {
                throw std::invalid_argument("null batched Fourier vector");
            }
            if (!indices.emplace(request.m, request.l).second) {
                throw std::invalid_argument("duplicate block in Fourier batch");
            }
            Prepared item;
            item.phi = packed_indices(request.m, layout_.nphi);
            item.z = packed_indices(request.l, layout_.nz);
            item.gauge = request.m == 0 && request.l == 0;
            item.full_size = static_cast<int>(
                item.phi.size()*item.z.size())*layout_.radial_size;
            const int expected_size = item.full_size-(item.gauge ? 1 : 0);
            if (request.size != expected_size) {
                throw std::invalid_argument(
                    "batched Fourier vector has the wrong size");
            }
            long double norm2 = 0;
            for (int i = 0; i < request.size; ++i) {
                const long double value = request.input[i];
                norm2 += value*value;
            }
            item.norm = std::sqrt(static_cast<double>(norm2));
            if (item.gauge && item.norm != 0) {
                item.expanded.resize(item.full_size);
                layout_.expand_zero_gauge_block(
                    geometry, request.input, item.expanded.data());
            }
            prepared_.push_back(std::move(item));
        }
    }

    std::size_t plane_index(int i, int k) const {
        return static_cast<std::size_t>(i)*layout_.nz+k;
    }

    int component_offset(Component component) const {
        switch (component) {
        case Component::u: return layout_.u_offset;
        case Component::v: return layout_.v_offset;
        case Component::w: return layout_.w_offset;
        case Component::p: return layout_.p_offset;
        }
        throw std::logic_error("unknown NSCyl component");
    }

    int component_radial_offset(Component component) const {
        switch (component) {
        case Component::u: return layout_.u_radial_offset;
        case Component::v: return layout_.v_radial_offset;
        case Component::w: return layout_.w_radial_offset;
        case Component::p: return layout_.p_radial_offset;
        }
        throw std::logic_error("unknown NSCyl component");
    }

    int component_radial_size(Component component) const {
        return component == Component::u ? layout_.nr-1 : layout_.nr;
    }

    std::size_t physical_index(Component component, int radial_index,
                               int i, int k) const {
        const int local = radial_index-component_radial_offset(component);
        return component_offset(component)
            +(static_cast<std::size_t>(i)*layout_.nz+k)
                *component_radial_size(component)+local;
    }

    void copy_values_to_physical(Component component, int radial_index,
                                 std::vector<T>& physical) const {
        for (int i = 0; i < layout_.nphi; ++i) {
            for (int k = 0; k < layout_.nz; ++k) {
                physical[physical_index(component, radial_index, i, k)] =
                    values_[plane_index(i, k)];
            }
        }
    }

    void copy_physical_to_values(Component component, int radial_index,
                                 const std::vector<T>& physical) {
        for (int i = 0; i < layout_.nphi; ++i) {
            for (int k = 0; k < layout_.nz; ++k) {
                values_[plane_index(i, k)] =
                    physical[physical_index(component, radial_index, i, k)];
            }
        }
    }
};

} // namespace ns_cyl_fourier_batch_detail

template<typename T, bool check=false>
class NSCylFourierBlockBatchReference {
public:
    using value_type = T;
    using Request = NSCylFourierBatchRequest<T>;
    using Task = NSCyl<T, check, tensor_flag::periodic>;

    NSCylFourierBlockBatchReference(const Config& config,
                                    int operator_steps=1)
        : ns_(config)
        , transform_(ns_.nr, ns_.nz, ns_.nphi)
        , operator_steps_(operator_steps)
        , physical_(transform_.layout().state_size)
    {
        if (operator_steps_ <= 0) {
            throw std::invalid_argument("operator_steps must be positive");
        }
        transform_.layout().initialize_couette_linearization(ns_);
        ns_.U0 = 0;
    }

    int operator_steps() const { return operator_steps_; }
    const Task& geometry() const { return ns_; }

    void apply(const std::vector<Request>& requests) {
        transform_.lift(ns_, requests, physical_);
        transform_.layout().unpack(ns_, physical_.data());
        for (int step = 0; step < operator_steps_; ++step) {
            ns_.L_step();
        }
        transform_.layout().pack(ns_, physical_.data());
        transform_.extract(ns_, requests, physical_);
    }

private:
    Task ns_;
    ns_cyl_fourier_batch_detail::Transform<T> transform_;
    int operator_steps_;
    std::vector<T> physical_;
};

} // namespace fdm
