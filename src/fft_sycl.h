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

    explicit sycl_allocator(sycl::queue &q, sycl::usm::alloc kind = sycl::usm::alloc::shared) noexcept
        : queue_(&q)
        , kind_(kind)
    {}

    template<typename U>
    sycl_allocator(const sycl_allocator<U> &other) noexcept
        : queue_(other.queue_)
        , kind_(other.kind_)
    {}

    pointer allocate(size_type n) {
        if (n == 0)
            return nullptr;
        void *p = sycl::malloc(n * sizeof(T), *queue_, kind_);
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
    sycl::usm::alloc kind_ = sycl::usm::alloc::device;
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
    FFTSyclDescriptor desc_batch;
    std::complex<T>* tmp;
    std::complex<T>* tmp_3d;

public:
    FFTSycl(sycl::queue& q, int N)
        : q(q)
        , N(N)
        , desc(N)
        , desc_batch(N)
        , tmp(sycl::malloc_device<std::complex<T>>(N/2 + 1, q))
        , tmp_3d(sycl::malloc_device<std::complex<T>>(N*N*(N/2+1), q))
    {
        desc.set_value(oneapi::math::dft::config_param::PLACEMENT, oneapi::math::dft::config_value::NOT_INPLACE);
        desc.set_value(oneapi::math::dft::config_param::CONJUGATE_EVEN_STORAGE,
            oneapi::math::dft::config_value::COMPLEX_COMPLEX );
        desc.commit(q);

        desc_batch.set_value(oneapi::math::dft::config_param::PLACEMENT, oneapi::math::dft::config_value::NOT_INPLACE);
        desc_batch.set_value(oneapi::math::dft::config_param::NUMBER_OF_TRANSFORMS, N*N);
        desc_batch.set_value(oneapi::math::dft::config_param::BWD_DISTANCE, N/2+1);
        desc_batch.set_value(oneapi::math::dft::config_param::FWD_DISTANCE, N);
        desc_batch.commit(q);
    }

    ~FFTSycl() {
        sycl::free(tmp, q);
        sycl::free(tmp_3d, q);
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

    // r2c 3d
    sycl::event pFFT_1_3d(T *S, T* s, T dx) {
        auto* out = tmp_3d;

        auto event1 = oneapi::math::dft::compute_forward(desc_batch, s, out);
        auto post_event1 = q.submit([&](sycl::handler& h) {
            h.depends_on(event1);
            h.parallel_for(
                sycl::range<3>(N, N, N/2 + 1),
                [=,N=N](sycl::id<3> idx) {
                    int i = idx[0], j = idx[1], k = idx[2];
                    int in_idx = (i*N + j)*(N/2+1) + k;
                    auto val = out[in_idx];
                    if (k == 0) {
                        S[(i*N + 0)*N + j] = val.real() * dx;
                    } else if (k == N/2) {
                        S[(i*N + N/2)*N + j] = val.real() * dx;
                    } else {
                        S[(i*N + k)*N + j] = val.real() * dx;
                        S[(i*N + (N - k))*N + j] = -val.imag() * dx;
                    }
                });
        });

        auto event2 = oneapi::math::dft::compute_forward(desc_batch, S, out, {post_event1});
        auto post_event2 = q.submit([&](sycl::handler& h) {
            h.depends_on(event2);
            h.parallel_for(
                sycl::range<3>(N, N, N/2 + 1),
                [=,N=N](sycl::id<3> idx) {
                    int i = idx[0], j = idx[1], k = idx[2];
                    int in_idx = (i*N + j)*(N/2+1) + k;
                    auto val = out[in_idx];
                    if (k == 0) {
                        S[(0*N + j)*N + i] = val.real() * dx;
                    } else if (k == N/2) {
                        S[(N/2*N + j)*N + i] = val.real() * dx;
                    } else {
                        S[(k*N + j)*N + i] = val.real() * dx;
                        S[((N - k)*N + j)*N + i] = -val.imag() * dx;
                    }
                });
        });

        auto event3 = oneapi::math::dft::compute_forward(desc_batch, S, out, {post_event2});
        return q.submit([&](sycl::handler& h) {
            h.depends_on(event3);
            h.parallel_for(
                sycl::range<3>(N, N, N/2 + 1),
                [=,N=N](sycl::id<3> idx) {
                    int i = idx[0], j = idx[1], k = idx[2];
                    int in_idx = (i*N + j)*(N/2+1) + k;
                    auto val = out[in_idx];
                    if (k == 0) {
                        S[(0*N + i)*N + j] = val.real() * dx;
                    } else if (k == N/2) {
                        S[(N/2*N + i)*N + j] = val.real() * dx;
                    } else {
                        S[(k*N + i)*N + j] = val.real() * dx;
                        S[((N - k)*N + i)*N + j] = -val.imag() * dx;
                    }
                });
        });
    }

    // c2r 3d
    sycl::event pFFT_3d(T *S, const T* s, T dx) {
        auto* in = tmp_3d;

        auto pre_event1 = q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::range<3>(N, N, N/2 + 1),
                    [=,N=N](sycl::id<3> idx) {
                    int i = idx[0], j = idx[1], k = idx[2];
                    int out_idx = (i*N + j)*(N/2+1) + k;
                    if (k == 0) {
                        in[out_idx] = { s[(i*N + j)*N + 0], T(0) };
                    } else if (k == N/2) {
                        in[out_idx] = { s[(i*N + j)*N + N/2], T(0) };
                    } else {
                        in[out_idx] = { s[(i*N + j)*N + k], -s[(i*N + j)*N + (N - k)] };
                    }
                });
        });

        auto event1 = oneapi::math::dft::compute_backward(desc_batch, in, S, {pre_event1});
        auto pre_event2 = q.submit([&](sycl::handler& h) {
            h.depends_on(event1);
            h.parallel_for(
                sycl::range<3>(N, N, N/2 + 1),
                    [=,N=N](sycl::id<3> idx) {
                    int i = idx[0], k = idx[1], j = idx[2];
                    int out_idx = (i*N + k)*(N/2+1) + j;
                    if (j == 0) {
                        in[out_idx] = { S[(i*N + j)*N + k], T(0) };
                    } else if (j == N/2) {
                        in[out_idx] = { S[(i*N + j)*N + k], T(0) };
                    } else {
                        in[out_idx] = { S[(i*N + j)*N + k], -S[(i*N + (N - j))*N + k] };
                    }
                });
        });

        auto event2 = oneapi::math::dft::compute_backward(desc_batch, in, S, {pre_event2});
        auto pre_event3 = q.submit([&](sycl::handler& h) {
            h.depends_on(event2);
            h.parallel_for(
                sycl::range<3>(N, N, N/2 + 1),
                    [=,N=N](sycl::id<3> idx) {
                    int j = idx[0], k = idx[1], i = idx[2];
                    int  out_idx = (j*N + k)*(N/2+1) + i;
                    if (i == 0) {
                        in[out_idx] = { S[(i*N + j)*N + k], T(0) };
                    } else if (i == N/2) {
                        in[out_idx] = { S[(i*N + j)*N + k], T(0) };
                    } else {
                        in[out_idx] = { S[(i*N + j)*N + k], -S[((N-i)*N + j)*N + k] };
                    }
                });
        });
        auto event3 = oneapi::math::dft::compute_backward(desc_batch, in, S, {pre_event3});

        return q.submit([&](sycl::handler& h) {
            h.depends_on(event3);
            T scale = dx * T(0.5);
            scale = scale*scale*scale;
            h.parallel_for(
                sycl::range<3>(N, N, N),
                [=,N=N](sycl::id<3> idx) {
                    int i = idx[0], j = idx[1], k = idx[2];
                    if (i < k) {
                        int idx1 = (i*N + j)*N + k;
                        int idx2 = (k*N + j)*N + i;
                        auto tmp = S[idx1];
                        S[idx1] = S[idx2]*scale;
                        S[idx2] = tmp*scale;
                    } else if (i == k) {
                        int idx1 = (i*N + j)*N + k;
                        S[idx1] = S[idx1]*scale;
                    }
                });
        });
    }

};

} // namespace fdm
