#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <math.h>

#include "asp_gauss.h"
#include "blas.h"
#include "cyclic_reduction.h"

using namespace fdm;

double unixbench_score(std::vector<double>& data) {
    if (data.empty()) return 0.0;
    std::sort(data.begin(), data.end(), std::greater<double>());
    int keep = (2 * data.size()) / 3;
    data.resize(keep);
    double score = 0.0;
    for (auto d : data) {
        score += log(d);
    }
    score = exp(score / keep);
    return score;
}

template<typename T, typename Func>
double benchmark_tdiag(int N, int iterations, Func f)
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

    std::vector<double> times;
    times.reserve(iterations);

    for (int it = 0; it < iterations; ++it) {
        init();

        auto start = std::chrono::high_resolution_clock::now();
        f(A1.data(), A2.data(), A3.data(), B.data(), N);
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
            [](double *A1, double *A2, double *A3, double *B, int N) {
                solve_tdiag_linear_my(B, A1, A2, A3, N);
            }
        );
        output(N, stats, "my(d)");

        stats = benchmark_tdiag<double>(N, iterations,
            [](double *A1, double *A2, double *A3, double *B, int N) {
                int info = 0;
                lapack::gtsv(N, 1, A1, A2, A3, B, N, &info);
            }
        );
        output(N, stats, "gtsv(d)");

        stats = benchmark_tdiag<double>(N, iterations,
            [&](double *A1, double *A2, double *A3, double *B, int N) {
                cyclic_reduction(A2, A1, A3, B, power, N);
            }
        );
        output(N, stats, "cr(d)");

        stats = benchmark_tdiag<double>(N, iterations,
            [&](double *A1, double *A2, double *A3, double *B, int N) {
                cyclic_reduction_general(A2, A1, A3, B, power, N);
            }
        );
        output(N, stats, "crg(d)");

        stats = benchmark_tdiag<float>(N, iterations,
            [](float *A1, float *A2, float *A3, float *B, int N) {
                solve_tdiag_linearf_my(B, A1, A2, A3, N);
            }
        );
        output(N, stats, "my(f)");

        stats = benchmark_tdiag<float>(N, iterations,
            [](float *A1, float *A2, float *A3, float *B, int N) {
                int info = 0;
                lapack::gtsv(N, 1, A1, A2, A3, B, N, &info);
            }
        );
        output(N, stats, "gtsv(f)");

        stats = benchmark_tdiag<float>(N, iterations,
            [&](float *A1, float *A2, float *A3, float *B, int N) {
                cyclic_reduction(A2, A1, A3, B, power, N);
            }
        );
        output(N, stats, "cr(f)");

        stats = benchmark_tdiag<float>(N, iterations,
            [&](float *A1, float *A2, float *A3, float *B, int N) {
                cyclic_reduction_general(A2, A1, A3, B, power, N);
            }
        );
        output(N, stats, "crg(f)");
    }
    return 0;
}