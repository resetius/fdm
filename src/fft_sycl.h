#pragma once
#include <sycl/sycl.hpp>
#include <oneapi/math.hpp>

namespace fdm {

template<typename T>
class sycl_allocator {
public:
    using value_type      = T;
    using pointer         = T*;
    using const_pointer   = const T*;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    explicit sycl_allocator(sycl::queue &q) noexcept
        : queue_(&q) {}

    template<typename U>
    sycl_allocator(const sycl_allocator<U> &other) noexcept
        : queue_(other.queue_) {}

    pointer allocate(size_type n) {
        if (n == 0)
            return nullptr;
        void *p = sycl::malloc_shared(n * sizeof(T), *queue_);
        if (!p)
            throw std::bad_alloc();
        return static_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type /*n*/) noexcept {
        sycl::free(p, *queue_);
    }

    template<typename U>
    struct rebind { using other = sycl_allocator<U>; };

    bool operator==(const sycl_allocator &other) const noexcept {
        return queue_ == other.queue_;
    }
    bool operator!=(const sycl_allocator &other) const noexcept {
        return !(*this == other);
    }

private:
    template<typename U> friend class sycl_allocator;
    sycl::queue *queue_;
};

template<typename T>
class FFTSyclDescriptorType { };

template<>
class FFTSyclDescriptorType<float> {
public:
    using value = oneapi::math::dft::descriptor<oneapi::math::dft::precision::SINGLE, oneapi::math::dft::domain::REAL>;
};

template<>
class FFTSyclDescriptorType<double> {
public:
    using value = oneapi::math::dft::descriptor<oneapi::math::dft::precision::DOUBLE, oneapi::math::dft::domain::REAL>;
};

template<typename T>
class FFTSycl {
    using FFTSyclDescriptor = typename FFTSyclDescriptorType<T>::value;

    sycl::queue& q;
    int N;
    FFTSyclDescriptor desc;
    std::complex<T>* tmp;

public:
    FFTSycl(sycl::queue& q, int N)
        : q(q)
        , N(N)
        , desc(N)
        , tmp(sycl::malloc_device<std::complex<T>>(N/2 + 1, q))
    {
        desc.set_value(oneapi::math::dft::config_param::PLACEMENT, oneapi::math::dft::config_value::NOT_INPLACE);
        desc.set_value(oneapi::math::dft::config_param::CONJUGATE_EVEN_STORAGE,
            oneapi::math::dft::config_value::COMPLEX_COMPLEX );
        desc.commit(q);
    }

    ~FFTSycl() {
        sycl::free(tmp, q);
    }

    // r2c
    sycl::event pFFT_1(T *S, T* s, T dx) {
        auto* out = tmp;
        auto fft_event = oneapi::math::dft::compute_forward(desc, s, out);

        return q.submit([&](sycl::handler& h) {
            h.depends_on(fft_event);
            h.parallel_for(sycl::range<1>(N/2 + 1), [=,N=N](sycl::id<1> idx) {
                int k = idx[0];
                if (k == 0) {
                    S[0] = out[0].real() * dx;
                } else if (k == N/2) {
                    S[N/2] = out[N/2].real() * dx;
                } else {
                    S[k] = out[k].real() * dx;
                    S[N - k] = -out[k].imag() * dx;
                }
            });
        });
    }

    // c2r
    sycl::event pFFT(T *S, const T* s, T dx) {
        auto* in = tmp;

        auto preproc_event = q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N/2 + 1), [=,N=N](sycl::id<1> idx) {
                int k = idx[0];
                if (k == 0) {
                    in[0] = { s[0], T(0) };
                } else if (k == N/2) {
                    in[N/2] = { s[N/2], T(0) };
                } else {
                    in[k] = { s[k], -s[N - k] };
                }
            });
        });

        auto ifft_event = oneapi::math::dft::compute_backward(desc, in, S, {preproc_event});

        return q.submit([&](sycl::handler& h) {
            h.depends_on(ifft_event);
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                int i = idx[0];
                S[i] *= dx * 0.5;
            });
        });
    }
};

} // namespace fdm
