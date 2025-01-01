#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <math.h>

#include "asp_gauss.h"
#include "blas.h"
#include "cyclic_reduction.h"
#include "unixbench_score.h"

using namespace fdm;

template<typename T, typename Func1, typename Func2>
double benchmark_tdiag(int N, int iterations, Func1 prep, Func2 f)
{
    static constexpr bool debug = false;
    std::vector<T> A1(N-1);
    std::vector<T> A2(N);
    std::vector<T> A3(N-1);
    std::vector<T> B(N);

    auto init = [&] () {
        for (int i = 0; i < N-1; i++) {
            A1[i] = 1.0;
            A2[i] = 2.0;
            A3[i] = 1.0;
            B[i] = 1.0;
        }
        A2[N-1] = 2.0;
        B[N-1] = 1.0;
    };

    auto data = prep(A1.data(), A2.data(), A3.data(), N);

    std::vector<double> times;
    times.reserve(iterations);

    for (int it = 0; it < iterations; ++it) {
        init();

        auto start = std::chrono::high_resolution_clock::now();
        f(data, A1.data(), A2.data(), A3.data(), B.data(), N);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }

    if constexpr(debug) {
        for (int i = 0; i < N; i++) {
            std::cout << B[i] << " ";
        }
        std::cout << std::endl;
    }

    return unixbench_score(times);
}

int main() {
    const int min_power = 4;
    const int max_power = 16;
    const int iterations = 2000;

    std::cout << std::setw(10) << "N"
              << std::setw(12) << "Class"
              << std::setw(15) << "Score(ms)"
              << std::endl;

    auto output = [](int N, double result, const char *name) {
        std::cout << std::setw(10) << N
                  << std::setw(12) << name
                  << std::setw(15) << result
                  << std::endl;
    };

    for(int power = min_power; power <= max_power; ++power) {
        int N = (1 << power) - 1;
        auto stats = benchmark_tdiag<double>(N, iterations,
            [](double *A1, double *A2, double *A3, int N) -> void* {
                return nullptr;
            },
            [](void*, double *A1, double *A2, double *A3, double *B, int N) {
                solve_tdiag_linear_my(B, A1, A2, A3, N);
            }
        );
        output(N, stats, "my(d)");

        stats = benchmark_tdiag<double>(N, iterations,
            [](double *A1, double *A2, double *A3, int N) -> void* {
                return nullptr;
            },
            [](void*, double *A1, double *A2, double *A3, double *B, int N) {
                int info = 0;
                lapack::gtsv(N, 1, A1, A2, A3, B, N, &info);
            }
        );
        output(N, stats, "gtsv(d)");

        struct gttrf_data {
            std::vector<double> DU2;
            std::vector<int> ipiv;
        };

        stats = benchmark_tdiag<double>(N, iterations,
            [](double *A1, double *A2, double *A3, int N) {
                gttrf_data data;
                data.DU2.resize(N);
                data.ipiv.resize(N);
                auto& DU2 = data.DU2;
                auto& ipiv = data.ipiv;
                int info = 0;
                lapack::gttrf(N, A1, A2, A3, DU2.data(), ipiv.data(), &info);
                return data;
            },
            [](gttrf_data& data, double *A1, double *A2, double *A3, double *B, int N) {
                int info = 0;
                auto& DU2 = data.DU2;
                auto& ipiv = data.ipiv;
                lapack::gttrs("N", N, 1, A1, A2, A3, DU2.data(), ipiv.data(), B, N, &info);
            }
        );
        output(N, stats, "gttrs(d)");

        stats = benchmark_tdiag<double>(N, iterations,
            [](double *A1, double *A2, double *A3, int N) -> void* {
                return nullptr;
            },
            [&](void*, double *A1, double *A2, double *A3, double *B, int N) {
                cyclic_reduction(A2, A1, A3, B, power, N);
            }
        );
        output(N, stats, "cr(d)");

        stats = benchmark_tdiag<double>(N, iterations,
            [](double *A1, double *A2, double *A3, int N) -> void* {
                return nullptr;
            },
            [&](void*, double *A1, double *A2, double *A3, double *B, int N) {
                cyclic_reduction_general(A2, A1, A3, B, power, N);
            }
        );
        output(N, stats, "crg(d)");

        stats = benchmark_tdiag<float>(N, iterations,
            [](float *A1, float *A2, float *A3, int N) -> void* {
                return nullptr;
            },
            [](void*, float *A1, float *A2, float *A3, float *B, int N) {
                solve_tdiag_linearf_my(B, A1, A2, A3, N);
            }
        );
        output(N, stats, "my(f)");

        stats = benchmark_tdiag<float>(N, iterations,
            [](float *A1, float *A2, float *A3, int N) -> void* {
                return nullptr;
            },
            [](void*, float *A1, float *A2, float *A3, float *B, int N) {
                int info = 0;
                lapack::gtsv(N, 1, A1, A2, A3, B, N, &info);
            }
        );
        output(N, stats, "gtsv(f)");

        struct gttrf_dataf {
            std::vector<float> DU2;
            std::vector<int> ipiv;
        };

        stats = benchmark_tdiag<float>(N, iterations,
            [](float *A1, float *A2, float *A3, int N) {
                gttrf_dataf data;
                data.DU2.resize(N);
                data.ipiv.resize(N);
                auto& DU2 = data.DU2;
                auto& ipiv = data.ipiv;
                int info = 0;
                lapack::gttrf(N, A1, A2, A3, DU2.data(), ipiv.data(), &info);
                return data;
            },
            [](gttrf_dataf& data, float *A1, float *A2, float *A3, float *B, int N) {
                int info = 0;
                auto& DU2 = data.DU2;
                auto& ipiv = data.ipiv;
                lapack::gttrs("N", N, 1, A1, A2, A3, DU2.data(), ipiv.data(), B, N, &info);
            }
        );
        output(N, stats, "gttrs(f)");

        stats = benchmark_tdiag<float>(N, iterations,
            [](float *A1, float *A2, float *A3, int N) -> void* {
                return nullptr;
            },
            [&](void*, float *A1, float *A2, float *A3, float *B, int N) {
                cyclic_reduction(A2, A1, A3, B, power, N);
            }
        );
        output(N, stats, "cr(f)");

        stats = benchmark_tdiag<float>(N, iterations,
            [](float *A1, float *A2, float *A3, int N) -> void* {
                return nullptr;
            },
            [&](void*, float *A1, float *A2, float *A3, float *B, int N) {
                cyclic_reduction_general(A2, A1, A3, B, power, N);
            }
        );
        output(N, stats, "crg(f)");
    }
    return 0;
}