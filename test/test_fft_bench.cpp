#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>

#include "fft.h"

using namespace fdm;

double compute_percentile(std::vector<double> &data, double percentile) {
    if (data.empty()) return 0.0;
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    double index = (percentile / 100.0) * (sorted.size() - 1);
    int lower = static_cast<int>(std::floor(index));
    int upper = static_cast<int>(std::ceil(index));
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

struct CombinedBenchmarkStats {
    BenchmarkStats pFFT_1;
    BenchmarkStats pFFT;
};

template <typename T, typename FFTClass>
CombinedBenchmarkStats benchmark_fft(FFTClass& fft, int N, int iterations) {
    std::vector<T> S(N, 0.0);
    std::vector<T> s(N, 0.0);

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
        fft.pFFT_1(S.data(), s.data(), 1.0);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        times_pFFT_1.push_back(elapsed.count());
    }

    for(int it = 0; it < iterations; ++it) {
        std::copy(S.begin(), S.end(), s.begin());

        auto start = std::chrono::high_resolution_clock::now();
        fft.pFFT(S.data(), s.data(), 1.0);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        times_pFFT.push_back(elapsed.count());
    }

    BenchmarkStats stats_pFFT_1;
    stats_pFFT_1.min_time = *std::min_element(times_pFFT_1.begin(), times_pFFT_1.end());
    stats_pFFT_1.max_time = *std::max_element(times_pFFT_1.begin(), times_pFFT_1.end());
    stats_pFFT_1.p50 = compute_percentile(times_pFFT_1, 50.0);
    stats_pFFT_1.p80 = compute_percentile(times_pFFT_1, 80.0);
    stats_pFFT_1.p90 = compute_percentile(times_pFFT_1, 90.0);
    stats_pFFT_1.p99 = compute_percentile(times_pFFT_1, 99.0);

    BenchmarkStats stats_pFFT;
    stats_pFFT.min_time = *std::min_element(times_pFFT.begin(), times_pFFT.end());
    stats_pFFT.max_time = *std::max_element(times_pFFT.begin(), times_pFFT.end());
    stats_pFFT.p50 = compute_percentile(times_pFFT, 50.0);
    stats_pFFT.p80 = compute_percentile(times_pFFT, 80.0);
    stats_pFFT.p90 = compute_percentile(times_pFFT, 90.0);
    stats_pFFT.p99 = compute_percentile(times_pFFT, 99.0);

    return CombinedBenchmarkStats{stats_pFFT_1, stats_pFFT};
}


int main() {
    const int min_power = 4;
    const int max_power = 16;
    const int iterations = 2000;

    std::cout << std::setw(10) << "N"
              << std::setw(12) << "Class"
              << std::setw(15) << "pFFT_1_Min(ms)"
              << std::setw(15) << "pFFT_1_Max(ms)"
              //<< std::setw(15) << "pFFT_1_P50(ms)"
              //<< std::setw(15) << "pFFT_1_P80(ms)"
              << std::setw(15) << "pFFT_1_P90(ms)"
              //<< std::setw(15) << "pFFT_1_P99(ms)"
              << std::setw(15) << "pFFT_Min(ms)"
              << std::setw(15) << "pFFT_Max(ms)"
              //<< std::setw(15) << "pFFT_P50(ms)"
              //<< std::setw(15) << "pFFT_P80(ms)"
              << std::setw(15) << "pFFT_P90(ms)"
              //<< std::setw(15) << "pFFT_P99(ms)"
              << std::endl;

    auto output = [&](int N, CombinedBenchmarkStats stats, std::string name) {
        std::cout << std::setw(10) << N
                    << std::setw(12) << name
                    << std::setw(15) << stats.pFFT_1.min_time
                    << std::setw(15) << stats.pFFT_1.max_time
                    //<< std::setw(15) << stats.pFFT_1.p50
                    //<< std::setw(15) << stats.pFFT_1.p80
                    << std::setw(15) << stats.pFFT_1.p90
                    //<< std::setw(15) << stats.pFFT_1.p99
                    << std::setw(15) << stats.pFFT.min_time
                    << std::setw(15) << stats.pFFT.max_time
                    //<< std::setw(15) << stats.pFFT.p50
                    //<< std::setw(15) << stats.pFFT.p80
                    << std::setw(15) << stats.pFFT.p90
                    //<< std::setw(15) << stats.pFFT.p99
                    << std::endl;
    };

    for(int power = min_power; power <= max_power; ++power) {
        int N = 1 << power;
        FFTTable<double> table(N);
        FFT<double> ft1(table, N);
        auto stats = benchmark_fft<double>(ft1, N, iterations);
        output(N, stats, "FFT");
#ifdef HAVE_FFTW3
        FFT_fftw3<double> ft2(N);
        stats = benchmark_fft<double>(ft2, N, iterations);
        output(N, stats, "fftw3");
#endif
    }
}
