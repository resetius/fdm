#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <math.h>

#include "asp_gauss.h"
#include "blas.h"

using namespace fdm;

double compute_percentile(std::vector<double> &data, double percentile) {
    if (data.empty()) return 0.0;
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    double index = (percentile / 100.0) * (sorted.size() - 1);
    int lower = static_cast<int>(floor(index));
    int upper = static_cast<int>(ceil(index));
    double weight = index - lower;
    if (upper >= sorted.size()) return sorted[lower];
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

struct BenchmarkStats {
    double min_time;
    double max_time;
    double p50;
    double p80;
    double p90;
    double p99;
};

template<typename T, typename Func>
BenchmarkStats benchmark_tdiag(int N, int iterations, Func f)
{
    std::vector<T> A1(N-1);
    std::vector<T> A2(N);
    std::vector<T> A3(N-1);
    std::vector<T> B(N);

    for (int i = 0; i < N-1; i++) {
        A1[i] = 1.0;
        A2[i] = 2.0;
        A3[i] = 1.0;
        B[i] = 1.0;
    }
    A2[N-1] = 2.0;
    B[N-1] = 1.0;

    std::vector<double> times;
    times.reserve(iterations);

    for (int it = 0; it < iterations; ++it) {
        for (int i = 0; i < N; i++) {
            B[i] = 1.0;
        }
        auto start = std::chrono::high_resolution_clock::now();
        f(A1.data(), A2.data(), A3.data(), B.data(), N);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }

    BenchmarkStats stats;
    stats.min_time = *std::min_element(times.begin(), times.end());
    stats.max_time = *std::max_element(times.begin(), times.end());
    stats.p50 = compute_percentile(times, 50.0);
    stats.p80 = compute_percentile(times, 80.0);
    stats.p90 = compute_percentile(times, 90.0);
    stats.p99 = compute_percentile(times, 99.0);

    return stats;
}

int main() {
    const int min_power = 4;
    const int max_power = 16;
    const int iterations = 2000;
    std::cout << std::setw(10) << "N"
              << std::setw(12) << "Class"
              << std::setw(15) << "Min(ms)"
              << std::setw(15) << "Max(ms)"
              << std::setw(15) << "P50(ms)"
              << std::setw(15) << "P90(ms)"
              << std::endl;

    auto output = [](int N, BenchmarkStats stats, const char *name) {
        std::cout << std::setw(10) << N
                  << std::setw(12) << name
                  << std::setw(15) << stats.min_time
                  << std::setw(15) << stats.max_time
                  << std::setw(15) << stats.p50
                  << std::setw(15) << stats.p90
                  << std::endl;
    };

    for(int power = min_power; power <= max_power; ++power) {
        int N = 1 << power;
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
    }
    return 0;
}