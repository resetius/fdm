// ~/opt/adaptivecpp/bin/acpp -std=c++20 -O2 -I../src test_tile_bench.cpp -o test_tile_bench
//
// SYCL micro-benchmarks:
//   1. Memory bandwidth (write / read / copy)
//   2. u32 ALU throughput (independent adds)
//   3. u32/u64 carry chain latency (models BigFloat<N> add)
//   4. Divergence cost (uniform vs scattered early-exit)
//   5. Tile shape sweep (nd_range local size effect)
//   6. Tiled GEMM (different tile sizes, local memory)
//   7. FFT butterfly with local memory (different work-group sizes)
//
// stdout: "label: score ms" per benchmark (unixbench: geomean of best 2/3)
// stderr: per-iteration timings + section headers
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

// ---- helpers ----------------------------------------------------------------

static double unixbench_score(std::vector<double> data) {
    if (data.empty()) return 0.0;
    std::sort(data.begin(), data.end());
    int keep = std::max(1, (int)(2 * (int)data.size() / 3));
    data.resize(keep);
    double s = 0.0;
    for (double d : data) s += std::log(d);
    return std::exp(s / keep);
}

struct Bench {
    sycl::queue& q;
    int iters;

    template<typename F>
    double run(const std::string& label, F kernel) {
        kernel(); q.wait(); // warmup

        std::vector<double> times;
        times.reserve(iters);
        for (int i = 0; i < iters; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            kernel(); q.wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        double score = unixbench_score(times);
        std::cout << label << ": " << score << " ms\n" << std::flush;
        return score;
    }
};

// ============================================================================
// 1. Memory bandwidth
// ============================================================================

void bench_memory(Bench& b, int N) {
    std::cout << "\n=== Memory bandwidth (N=" << N << " uint32, "
              << N * 4 / 1024 / 1024 << " MB) ===\n";
    uint32_t* src = sycl::malloc_device<uint32_t>(N, b.q);
    uint32_t* dst = sycl::malloc_device<uint32_t>(N, b.q);

    b.run("mem write ", [&] {
        b.q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                dst[i[0]] = (uint32_t)i[0];
            });
        });
    });

    b.run("mem read  ", [&] {
        b.q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                // avoid dead-code elimination
                dst[i[0]] = src[i[0]] ^ (uint32_t)(i[0] >> 16);
            });
        });
    });

    b.run("mem copy  ", [&] {
        b.q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                dst[i[0]] = src[i[0]];
            });
        });
    });

    sycl::free(src, b.q); sycl::free(dst, b.q);
}

// ============================================================================
// 2. u32 ALU throughput (independent adds)
// ============================================================================

void bench_u32_throughput(Bench& b, int N) {
    std::cout << "\n=== u32 independent adds (N=" << N << ", 256 reps) ===\n";
    uint32_t* out = sycl::malloc_device<uint32_t>(N, b.q);

    b.run("u32 throughput 256 reps", [&] {
        b.q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                uint32_t v = (uint32_t)i[0];
                // 8 independent accumulators — ILP-friendly
                uint32_t a0=v,a1=v+1,a2=v+2,a3=v+3,a4=v+4,a5=v+5,a6=v+6,a7=v+7;
                for (int k = 0; k < 32; k++) {
                    a0+=v; a1+=v; a2+=v; a3+=v; a4+=v; a5+=v; a6+=v; a7+=v;
                }
                out[i[0]] = a0^a1^a2^a3^a4^a5^a6^a7;
            });
        });
    });

    sycl::free(out, b.q);
}

// ============================================================================
// 3. Carry chain latency (u32 / u64)
// ============================================================================

template<typename T, int DEPTH>
void bench_carry_chain(Bench& b, int N, const std::string& label) {
    std::cout << "\n=== " << label << " carry chain depth=" << DEPTH
              << " (N=" << N << ") ===\n";
    T* out = sycl::malloc_device<T>(N, b.q);

    b.run(label + " depth=" + std::to_string(DEPTH), [&] {
        b.q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                T v = (T)i[0], carry = 0;
                for (int k = 0; k < DEPTH; k++) {
                    T sum = v + v + carry;
                    carry = (sum < v) ? T(1) : T(0);
                    v = sum;
                }
                out[i[0]] = v ^ carry;
            });
        });
    });

    sycl::free(out, b.q);
}

void bench_carry_chains(Bench& b, int N) {
    // short chains — fast baseline
    bench_carry_chain<uint32_t,  4>(b, N, "u32");
    bench_carry_chain<uint32_t,  8>(b, N, "u32");
    bench_carry_chain<uint64_t,  4>(b, N, "u64");
    bench_carry_chain<uint64_t,  8>(b, N, "u64");
    // longer chains — should take seconds, model deeper BigFloat
    bench_carry_chain<uint32_t, 256>(b, N, "u32");
    bench_carry_chain<uint32_t, 512>(b, N, "u32");
    bench_carry_chain<uint64_t, 256>(b, N, "u64");
    bench_carry_chain<uint64_t, 512>(b, N, "u64");
}

// ============================================================================
// 4. Divergence
// ============================================================================

void bench_divergence(Bench& b, int N) {
    std::cout << "\n=== Divergence (N=" << N << ", 128 iters) ===\n";
    uint32_t* out = sycl::malloc_device<uint32_t>(N, b.q);
    constexpr int MAX = 128;

    // All threads do exactly MAX iterations
    b.run("diverge uniform    ", [&] {
        b.q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                uint32_t v = (uint32_t)i[0];
                for (int k = 0; k < MAX; k++) v = v*1664525u + 1013904223u;
                out[i[0]] = v;
            });
        });
    });

    // Alternating threads: odd=1 iter, even=MAX iters (worst divergence)
    b.run("diverge alternating", [&] {
        b.q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                uint32_t v = (uint32_t)i[0];
                int its = (i[0] & 1) ? 1 : MAX;
                for (int k = 0; k < its; k++) v = v*1664525u + 1013904223u;
                out[i[0]] = v;
            });
        });
    });

    // First 16 threads of each 32-group: 1 iter; second 16: MAX iters
    // (half-warp coherent — medium divergence)
    b.run("diverge half-group ", [&] {
        b.q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                uint32_t v = (uint32_t)i[0];
                int its = ((i[0] % 32) < 16) ? 1 : MAX;
                for (int k = 0; k < its; k++) v = v*1664525u + 1013904223u;
                out[i[0]] = v;
            });
        });
    });

    sycl::free(out, b.q);
}

// ============================================================================
// 5. Tile shape sweep
// ============================================================================

template<int TR, int TC, int REPS>
void bench_one_tile(Bench& b, int W, int H, const std::string& label) {
    int rh = ((H + TR - 1) / TR) * TR;
    int rw = ((W + TC - 1) / TC) * TC;
    uint32_t* out = sycl::malloc_device<uint32_t>(W * H, b.q);

    b.run(label, [&] {
        b.q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(rh, rw), sycl::range<2>(TR, TC)),
                [=](sycl::nd_item<2> item) {
                    int row = (int)item.get_global_id(0);
                    int col = (int)item.get_global_id(1);
                    if (row >= H || col >= W) return;
                    uint32_t v = (uint32_t)(row * W + col);
                    for (int k = 0; k < REPS; k++) v = v*1664525u + 1013904223u;
                    out[row * W + col] = v;
                }
            );
        });
    });

    sycl::free(out, b.q);
}

template<int REPS>
void bench_tiles_impl(Bench& b, int W, int H) {
    std::cout << "\n=== Tile shape sweep (W=" << W << " H=" << H
              << ", " << REPS << " LCG iters/pixel) ===\n";

    uint32_t* out = sycl::malloc_device<uint32_t>(W * H, b.q);
    b.run("tile 1D  ", [&] {
        b.q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(W * H), [=](sycl::id<1> id) {
                uint32_t v = (uint32_t)id[0];
                for (int k = 0; k < REPS; k++) v = v*1664525u + 1013904223u;
                out[id[0]] = v;
            });
        });
    });
    sycl::free(out, b.q);

    bench_one_tile< 1, 32, REPS>(b, W, H, "tile 1x32");
    bench_one_tile< 2, 16, REPS>(b, W, H, "tile 2x16");
    bench_one_tile< 4,  8, REPS>(b, W, H, "tile 4x8 ");
    bench_one_tile< 8,  4, REPS>(b, W, H, "tile 8x4 ");
    bench_one_tile<16,  2, REPS>(b, W, H, "tile 16x2");
    bench_one_tile<32,  1, REPS>(b, W, H, "tile 32x1");
}

void bench_tiles(Bench& b, int W, int H) {
    bench_tiles_impl<256>(b, W, H);        // fast baseline
    bench_tiles_impl<4096>(b, 4096, 4096); // large — should take seconds
}

// ============================================================================
// 6. Tiled GEMM (C = A * B, N×N matrices, float)
// ============================================================================

template<int TILE>
void bench_gemm_tile(Bench& b, int N, const std::string& label) {
    size_t sz = (size_t)N * N;
    float* A = sycl::malloc_device<float>(sz, b.q);
    float* B = sycl::malloc_device<float>(sz, b.q);
    float* C = sycl::malloc_device<float>(sz, b.q);

    // Init A and B with something non-zero
    b.q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(sz), [=](sycl::id<1> i) {
            A[i[0]] = (float)(i[0] % 17 + 1);
            B[i[0]] = (float)(i[0] % 13 + 1);
        });
    }).wait();

    int groups = (N + TILE - 1) / TILE;

    b.run(label, [&] {
        b.q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 2> lA({TILE, TILE}, h);
            sycl::local_accessor<float, 2> lB({TILE, TILE}, h);

            h.parallel_for(
                sycl::nd_range<2>(
                    sycl::range<2>(groups * TILE, groups * TILE),
                    sycl::range<2>(TILE, TILE)
                ),
                [=](sycl::nd_item<2> item) {
                    int row = (int)item.get_global_id(0);
                    int col = (int)item.get_global_id(1);
                    int lr  = (int)item.get_local_id(0);
                    int lc  = (int)item.get_local_id(1);

                    float acc = 0.0f;
                    for (int t = 0; t < groups; t++) {
                        // Load tile into local memory
                        int ar = row, ac = t * TILE + lc;
                        lA[lr][lc] = (ar < N && ac < N) ? A[ar * N + ac] : 0.0f;
                        int br = t * TILE + lr, bc = col;
                        lB[lr][lc] = (br < N && bc < N) ? B[br * N + bc] : 0.0f;
                        item.barrier(sycl::access::fence_space::local_space);

                        for (int k = 0; k < TILE; k++) acc += lA[lr][k] * lB[k][lc];
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (row < N && col < N) C[row * N + col] = acc;
                }
            );
        });
    });

    sycl::free(A, b.q); sycl::free(B, b.q); sycl::free(C, b.q);
}

void bench_gemm_naive(Bench& b, int N) {
    size_t sz = (size_t)N * N;
    float* A = sycl::malloc_device<float>(sz, b.q);
    float* B = sycl::malloc_device<float>(sz, b.q);
    float* C = sycl::malloc_device<float>(sz, b.q);

    b.q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(sz), [=](sycl::id<1> i) {
            A[i[0]] = (float)(i[0] % 17 + 1);
            B[i[0]] = (float)(i[0] % 13 + 1);
        });
    }).wait();

    b.run("gemm naive N=" + std::to_string(N), [&] {
        b.q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(N, N), [=](sycl::id<2> id) {
                int row = (int)id[0], col = (int)id[1];
                float acc = 0.0f;
                for (int k = 0; k < N; k++) acc += A[row*N+k] * B[k*N+col];
                C[row*N+col] = acc;
            });
        });
    });

    sycl::free(A, b.q); sycl::free(B, b.q); sycl::free(C, b.q);
}

void bench_gemm(Bench& b) {
    // small N — fast baseline
    {
        const int N = 1024;
        std::cout << "\n=== Tiled GEMM " << N << "x" << N << " float ===\n";
        bench_gemm_naive(b, N);
        bench_gemm_tile< 8>(b, N, "gemm tile=8  N=" + std::to_string(N));
        bench_gemm_tile<16>(b, N, "gemm tile=16 N=" + std::to_string(N));
        bench_gemm_tile<32>(b, N, "gemm tile=32 N=" + std::to_string(N));
    }
    // large N — should take seconds
    {
        const int N = 3072;
        std::cout << "\n=== Tiled GEMM " << N << "x" << N << " float ===\n";
        bench_gemm_tile< 8>(b, N, "gemm tile=8  N=" + std::to_string(N));
        bench_gemm_tile<16>(b, N, "gemm tile=16 N=" + std::to_string(N));
        bench_gemm_tile<32>(b, N, "gemm tile=32 N=" + std::to_string(N));
    }
}

// ============================================================================
// 7. FFT butterfly with local memory
//    Each work-group of size WG computes a WG-point radix-2 DIT FFT
//    over WG consecutive complex floats.
//    Benchmark: N/WG independent FFTs launched in parallel.
// ============================================================================

// DIF (Decimation In Frequency) FFT in local memory.
// Uses item.get_local_range(0) instead of template WG so LLVM never sees
// a compile-time constant range → loop/index variables stay i32, avoiding
// "Unsupported integer bit width" errors in the Metal backend.
// Output is bit-reversed (fine for a benchmark).
template<int WG>
void bench_fft_local(Bench& b, int N_FFTS, const std::string& label) {
    int total = N_FFTS * WG * 2;
    float* buf = sycl::malloc_device<float>(total, b.q);

    b.q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i) {
            buf[i[0]] = (float)(i[0] % 7 - 3);
        });
    }).wait();

    b.run(label, [&] {
        b.q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> re(WG, h);
            sycl::local_accessor<float, 1> im(WG, h);

            h.parallel_for(
                sycl::nd_range<1>(N_FFTS * WG, WG),
                [=](sycl::nd_item<1> item) {
                    int tid  = (int)item.get_local_id(0);
                    int base = (int)item.get_group(0) * WG * 2;

                    re[tid] = buf[base + 2*tid];
                    im[tid] = buf[base + 2*tid+1];
                    item.barrier(sycl::access::fence_space::local_space);

                    // DIF butterfly stages: half = WG/2, WG/4, ..., 1
                    // Use get_local_range so LLVM treats 'wg' as runtime → i32 throughout
                    int wg = (int)item.get_local_range(0);
                    for (int half = wg >> 1; half >= 1; half >>= 1) {
                        // pos_in_group = tid mod (2*half), stays i32 because half is runtime
                        int pos = tid & ((half << 1) - 1);
                        if (pos < half) {
                            int u = tid;
                            int v = tid + half;
                            float ang = -3.14159265358979323846f
                                        * (float)pos / (float)half;
                            float wr = sycl::cos(ang);
                            float wi = sycl::sin(ang);
                            float ur = re[u], ui = im[u];
                            float vr = re[v], vi = im[v];
                            re[u] = ur + vr;
                            im[u] = ui + vi;
                            re[v] = (ur - vr) * wr - (ui - vi) * wi;
                            im[v] = (ur - vr) * wi + (ui - vi) * wr;
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    buf[base + 2*tid]   = re[tid];
                    buf[base + 2*tid+1] = im[tid];
                }
            );
        });
    });

    sycl::free(buf, b.q);
}

void bench_fft(Bench& b) {
    // small — fast baseline
    {
        const int TOTAL = 1 << 22; // 4M complex
        std::cout << "\n=== FFT DIF local memory (total=" << TOTAL << " complex) ===\n";
        bench_fft_local< 32>(b, TOTAL /  32, "fft small wg=32  ");
        bench_fft_local< 64>(b, TOTAL /  64, "fft small wg=64  ");
        bench_fft_local<128>(b, TOTAL / 128, "fft small wg=128 ");
        bench_fft_local<256>(b, TOTAL / 256, "fft small wg=256 ");
    }
    // large — should take seconds
    {
        const int TOTAL = 1 << 26; // 64M complex (~1.5 GB floats)
        std::cout << "\n=== FFT DIF local memory (total=" << TOTAL << " complex) ===\n";
        bench_fft_local< 32>(b, TOTAL /  32, "fft large wg=32  ");
        bench_fft_local< 64>(b, TOTAL /  64, "fft large wg=64  ");
        bench_fft_local<128>(b, TOTAL / 128, "fft large wg=128 ");
        bench_fft_local<256>(b, TOTAL / 256, "fft large wg=256 ");
    }
}

// ============================================================================
// main
// ============================================================================

int main() {
    sycl::queue q{sycl::default_selector_v};
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    const int ITERS  = 8;
    const int N32M   = 32 * 1024 * 1024; // 128 MB
    Bench b{q, ITERS};

    bench_memory(b, N32M);
    bench_u32_throughput(b, N32M);
    bench_carry_chains(b, N32M);
    bench_divergence(b, N32M);
    bench_tiles(b, 1024, 1024);
    bench_gemm(b);
    bench_fft(b);

    return 0;
}
