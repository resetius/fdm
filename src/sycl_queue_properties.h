#pragma once

#include <sycl/sycl.hpp>

namespace fdm {

inline constexpr bool sycl_queue_uses_coarse_grained_events =
#if defined(__ACPP__) || defined(ACPP_EXT_COARSE_GRAINED_EVENTS)
    true;
#else
    false;
#endif

inline sycl::property_list sycl_in_order_queue_properties() {
#if defined(__ACPP__) || defined(ACPP_EXT_COARSE_GRAINED_EVENTS)
    return sycl::property_list{
        sycl::property::queue::in_order{},
        sycl::property::queue::AdaptiveCpp_coarse_grained_events{}};
#else
    return sycl::property_list{sycl::property::queue::in_order{}};
#endif
}

} // namespace fdm
