#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <math.h>

#include "asp_gauss.h"
#include "blas.h"

using namespace fdm;

template<typename T>
void cyclic_reduction(
    T *d, T *e, T *f, T *b,
    int q, int n)
{
    T alpha, gamma;
    int l, j, s, h;

    for (l = 1; l < q; l++) {
        s = 1 << l;
        h = 1 << (l-1);

        for (j = s-1; j < n-h; j += s) {
            alpha = - e[j] / d[j-h];
            gamma = - f[j] / d[j+h];
            d[j] += alpha * f[j-h] + gamma * e[j+h];
            b[j] += alpha * b[j-h] + gamma * b[j+h];
            e[j] = alpha * e[j-h];
            f[j] = gamma * f[j+h];
        }
    }

    j = (1<<(q-1)) - 1;
    b[j] = b[j] / d[j];

    for (l = q-1; l > 0; l--) {
        s = 1 << l;
        h = 1 << (l-1);
        j = h-1;

        b[j] = (b[j] - f[j] * b[j+h]) / d[j];
        for (j = h + s - 1; j < n - h; j += s) {
            b[j] = (b[j] - e[j] * b[j-h] - f[j] * b[j+h]) / d[j];
        }
        b[j] = (b[j] - e[j] * b[j-h]) / d[j];
    }
    return;
}

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
    }
    return 0;
}