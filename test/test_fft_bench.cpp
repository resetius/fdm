#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <math.h>

#include "fft.h"

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

struct CombinedBenchmarkStats {
    double pFFT_1;
    double pFFT;
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

    auto stats_pFFT_1 = unixbench_score(times_pFFT_1);
    auto stats_pFFT = unixbench_score(times_pFFT);

    return CombinedBenchmarkStats{stats_pFFT_1, stats_pFFT};
}


int main() {
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
