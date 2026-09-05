#pragma once

#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

#include "ns_cyl_fourier_batch.h"
#include "ns_cyl_sycl.h"

namespace fdm {

template<typename T>
class NSCylSyclFourierBlockBatchReference {
public:
    using value_type = T;
    using Request = NSCylFourierBatchRequest<T>;

    NSCylSyclFourierBlockBatchReference(
        sycl::queue& queue, const Config& config, int operator_steps=1)
        : queue_(queue)
        , ns_(queue,
              config.get("ns", "nr", 32),
              config.get("ns", "nz", 32),
              config.get("ns", "nphi", 32),
              static_cast<T>(config.get("ns", "r", M_PI/2)),
              static_cast<T>(config.get("ns", "R", M_PI)),
              static_cast<T>(config.get("ns", "h2", 10.0)
                             -config.get("ns", "h1", 0.0)),
              T(0),
              static_cast<T>(config.get("ns", "Re", 1.0)),
              static_cast<T>(config.get("ns", "dt", 0.001)))
        , transform_(ns_.nr, ns_.nz, ns_.nphi)
        , operator_steps_(operator_steps)
        , physical_(transform_.layout().state_size)
    {
        if (operator_steps_ <= 0) {
            throw std::invalid_argument("operator_steps must be positive");
        }
        ns_.initialize_couette_linearization(
            static_cast<T>(config.get("ns", "u0", 1.0)),
            static_cast<T>(config.get(
                "spectral", "base_outer_radius",
                config.get("ns", "R", M_PI))));
    }

    int operator_steps() const { return operator_steps_; }
    const NSCylSycl<T>& geometry() const { return ns_; }

    void apply(const std::vector<Request>& requests) {
        queue_.wait();
        transform_.lift(ns_, requests, physical_);
        unpack();
        for (int step = 0; step < operator_steps_; ++step) {
            ns_.L_step();
        }
        queue_.wait();
        pack();
        transform_.extract(ns_, requests, physical_);
    }

private:
    sycl::queue& queue_;
    NSCylSycl<T> ns_;
    ns_cyl_fourier_batch_detail::Transform<T> transform_;
    int operator_steps_;
    std::vector<T> physical_;

    void unpack() {
        auto u = ns_.ua();
        auto v = ns_.va();
        auto w = ns_.wa();
        auto p = ns_.pa();
        const auto& layout = transform_.layout();
        int index = 0;
        for (int i = 0; i < ns_.nphi; ++i) {
            for (int k = 0; k < ns_.nz; ++k) {
                for (int j = 1; j < ns_.nr; ++j) {
                    u(i, k, j) = physical_[index++];
                }
            }
        }
        if (index != layout.v_offset) {
            throw std::logic_error("invalid batched SYCL u layout");
        }
        for (auto field : {v, w, p}) {
            for (int i = 0; i < ns_.nphi; ++i) {
                for (int k = 0; k < ns_.nz; ++k) {
                    for (int j = 1; j <= ns_.nr; ++j) {
                        field(i, k, j) = physical_[index++];
                    }
                }
            }
        }
        if (index != layout.state_size) {
            throw std::logic_error("invalid batched SYCL state layout");
        }
    }

    void pack() {
        auto u = ns_.ua();
        auto v = ns_.va();
        auto w = ns_.wa();
        auto p = ns_.pa();
        const auto& layout = transform_.layout();
        int index = 0;
        for (int i = 0; i < ns_.nphi; ++i) {
            for (int k = 0; k < ns_.nz; ++k) {
                for (int j = 1; j < ns_.nr; ++j) {
                    physical_[index++] = u(i, k, j);
                }
            }
        }
        if (index != layout.v_offset) {
            throw std::logic_error("invalid batched SYCL u layout");
        }
        for (auto field : {v, w, p}) {
            for (int i = 0; i < ns_.nphi; ++i) {
                for (int k = 0; k < ns_.nz; ++k) {
                    for (int j = 1; j <= ns_.nr; ++j) {
                        physical_[index++] = field(i, k, j);
                    }
                }
            }
        }
        if (index != layout.state_size) {
            throw std::logic_error("invalid batched SYCL state layout");
        }
    }
};

} // namespace fdm
