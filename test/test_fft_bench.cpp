#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <type_traits>
#include <utility>
#include <functional>

#include "fft.h"
#include "unixbench_score.h"

#ifdef HAVE_ONEMATH
#include "fft_sycl.h"
#endif

using namespace fdm;

struct CombinedBenchmarkStats {
    double pFFT_1;
    double pFFT;
};

#ifdef HAVE_ONEMATH
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
#endif

template <typename F, typename... Args>
void call_and_wait(F&& f, Args&&... args) {
    using Ret = std::invoke_result_t<F, Args...>;

    if constexpr (std::is_void_v<Ret>) {
        std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    } else {
        Ret ev = std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
        ev.wait();
    }
}

template <typename T, typename FFTClass, typename Allocator>
CombinedBenchmarkStats benchmark_fft(FFTClass& fft, int N, int iterations, Allocator alloc) {
    std::vector<T, Allocator> S(N, 0.0, alloc);
    std::vector<T, Allocator> s(N, 0.0, alloc);

    for(int i = 0; i < N; ++i) {
        S[i] = sin(2.0 * M_PI * i / N) + 0.5 * cos(4.0 * M_PI * i / N);
    }

    std::vector<double> times_pFFT_1;
    std::vector<double> times_pFFT;
    times_pFFT_1.reserve(iterations);
    times_pFFT.reserve(iterations);

    for(int it = 0; it < iterations; ++it) {
        std::copy(S.begin(), S.end(), s.begin());

        auto start = std::chrono::high_resolution_clock::now();
        call_and_wait([&](auto* S, auto* s, auto dx){
            return fft.pFFT_1(S, s, dx);
        }, S.data(), s.data(), 1.0);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        times_pFFT_1.push_back(elapsed.count());
    }

    for(int it = 0; it < iterations; ++it) {
        std::copy(S.begin(), S.end(), s.begin());

        auto start = std::chrono::high_resolution_clock::now();
        call_and_wait([&](auto* S, auto* s, auto dx){
            return fft.pFFT(S, s, dx);
        }, S.data(), s.data(), 1.0);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        times_pFFT.push_back(elapsed.count());
    }

    auto stats_pFFT_1 = unixbench_score(times_pFFT_1);
    auto stats_pFFT = unixbench_score(times_pFFT);

    return CombinedBenchmarkStats{stats_pFFT_1, stats_pFFT};
}


int main() {
#ifdef HAVE_ONEMATH
    sycl::queue q{ sycl::default_selector_v };
    std::cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
    sycl_allocator<double> sycl_d_alloc(q);
    sycl_allocator<float> sycl_f_alloc(q);
#endif
    const int min_power = 4;
    const int max_power = 16;
    const int iterations = 2000;

    std::cout << std::setw(10) << "N"
              << std::setw(12) << "Class"
              << std::setw(15) << "pFFT_1(ms)"
              << std::setw(15) << "pFFT(ms)"
              << std::endl;

    auto output = [&](int N, CombinedBenchmarkStats stats, std::string name) {
        std::cout << std::setw(10) << N
                    << std::setw(12) << name
                    << std::setw(15) << stats.pFFT_1
                    << std::setw(15) << stats.pFFT
                    << std::endl;
    };

    for(int power = min_power; power <= max_power; ++power) {
        int N = 1 << power;
        {
            FFTTable<double> table(N);
            FFT<double> ft1(table, N);
            auto stats = benchmark_fft<double>(ft1, N, iterations, std::allocator<double>());
            output(N, stats, "FFT(d)");
        }
        {
#ifdef HAVE_FFTW3
            FFT_fftw3<double> ft2(N);
            auto stats = benchmark_fft<double>(ft2, N, iterations, std::allocator<double>());
            output(N, stats, "fftw3(d)");
#endif
        }
        {
#ifdef HAVE_ONEMATH
            FFTSycl<double> ft1(q, N);
            auto stats = benchmark_fft<double>(ft1, N, iterations, sycl_d_alloc);
            output(N, stats, "Sycl(d)");
#endif
        }

        {
            FFTTable<float> table(N);
            FFT<float> ft1(table, N);
            auto stats = benchmark_fft<float>(ft1, N, iterations, std::allocator<float>());
            output(N, stats, "FFT(f)");
        }
        {
#ifdef HAVE_FFTW3
            FFT_fftw3<float> ft2(N);
            auto stats = benchmark_fft<float>(ft2, N, iterations, std::allocator<float>());
            output(N, stats, "fftw3(f)");
#endif
        }
        {
#ifdef HAVE_ONEMATH
            FFTSycl<float> ft1(q, N);
            auto stats = benchmark_fft<float>(ft1, N, iterations, sycl_f_alloc);
            output(N, stats, "Sycl(f)");
#endif
        }
    }
}
