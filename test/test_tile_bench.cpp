// ~/opt/adaptivecpp/bin/acpp -std=c++20 -O2 -I../src test_tile_bench.cpp -o test_tile_bench
//
// SYCL micro-benchmarks:
//   1. Memory bandwidth (write / read / copy)
//   2. u32 ALU throughput (independent adds)
//   3. u32/u64 carry chain latency (models BigFloat<N> add)
//   4. Divergence cost (uniform vs scattered early-exit)
//   5. Tile shape sweep (nd_range local size effect)
//   6. Tiled GEMM (different tile sizes, local memory)
//   6b. Register-tiled GEMM (higher arithmetic intensity)
//   7. FFT butterfly with local memory (different work-group sizes)
//
// stdout: "label: score ms" per benchmark (unixbench: geomean of best 2/3)
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
// Correctness verification (small N, runs once before benchmarks)
// ============================================================================

static bool verify_memory(sycl::queue& q) {
    const int N = 1024;
    uint32_t* dev = sycl::malloc_device<uint32_t>(N, q);
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            dev[i[0]] = (uint32_t)i[0] * 2654435761u;
        });
    }).wait();
    std::vector<uint32_t> host(N);
    q.memcpy(host.data(), dev, N * sizeof(uint32_t)).wait();
    sycl::free(dev, q);
    for (int i = 0; i < N; i++) {
        if (host[i] != (uint32_t)i * 2654435761u) {
            std::cout << "FAIL verify_memory at i=" << i
                      << ": got " << host[i]
                      << " expected " << (uint32_t)i * 2654435761u << "\n";
            return false;
        }
    }
    return true;
}

static bool verify_gemm(sycl::queue& q) {
    const int N = 64;
    size_t sz = N * N;
    float* A = sycl::malloc_device<float>(sz, q);
    float* B = sycl::malloc_device<float>(sz, q);
    float* C = sycl::malloc_device<float>(sz, q);

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(sz), [=](sycl::id<1> i) {
            A[i[0]] = (float)(i[0] % 17 + 1);
            B[i[0]] = (float)(i[0] % 13 + 1);
        });
    }).wait();

    constexpr int TILE = 16;
    int groups = (N + TILE - 1) / TILE;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 2> lA({TILE, TILE}, h);
        sycl::local_accessor<float, 2> lB({TILE, TILE}, h);
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(groups*TILE, groups*TILE),
                              sycl::range<2>(TILE, TILE)),
            [=](sycl::nd_item<2> item) {
                int row = (int)item.get_global_id(0);
                int col = (int)item.get_global_id(1);
                int lr  = (int)item.get_local_id(0);
                int lc  = (int)item.get_local_id(1);
                float acc = 0.0f;
                for (int t = 0; t < groups; t++) {
                    int ar = row, ac = t * TILE + lc;
                    lA[lr][lc] = (ar < N && ac < N) ? A[ar*N+ac] : 0.0f;
                    int br = t * TILE + lr, bc = col;
                    lB[lr][lc] = (br < N && bc < N) ? B[br*N+bc] : 0.0f;
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int k = 0; k < TILE; k++) acc += lA[lr][k] * lB[k][lc];
                    item.barrier(sycl::access::fence_space::local_space);
                }
                if (row < N && col < N) C[row*N+col] = acc;
            }
        );
    }).wait();

    std::vector<float> hA(sz), hB(sz), hC(sz);
    q.memcpy(hA.data(), A, sz*sizeof(float)).wait();
    q.memcpy(hB.data(), B, sz*sizeof(float)).wait();
    q.memcpy(hC.data(), C, sz*sizeof(float)).wait();
    sycl::free(A, q); sycl::free(B, q); sycl::free(C, q);

    bool ok = true;
    for (int i = 0; i < N && ok; i++) {
        for (int j = 0; j < N && ok; j++) {
            float ref = 0.0f;
            for (int k = 0; k < N; k++) ref += hA[i*N+k] * hB[k*N+j];
            float diff = std::abs(hC[i*N+j] - ref);
            if (diff > 1e-3f * std::abs(ref) + 0.1f) {
                std::cout << "FAIL verify_gemm at (" << i << "," << j
                          << "): gpu=" << hC[i*N+j] << " ref=" << ref << "\n";
                ok = false;
            }
        }
    }
    return ok;
}

static bool verify_fft(sycl::queue& q) {
    const int WG = 16;
    int total = WG * 2; // interleaved re/im
    float* buf = sycl::malloc_device<float>(total, q);

    // Known input: real ramp, imaginary zero
    std::vector<float> h_in(total);
    for (int i = 0; i < WG; i++) { h_in[2*i] = (float)i; h_in[2*i+1] = 0.0f; }
    q.memcpy(buf, h_in.data(), total*sizeof(float)).wait();

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> re(WG, h);
        sycl::local_accessor<float, 1> im(WG, h);
        h.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> item) {
            int tid = (int)item.get_local_id(0);
            re[tid] = buf[2*tid]; im[tid] = buf[2*tid+1];
            item.barrier(sycl::access::fence_space::local_space);
            int wg = (int)item.get_local_range(0);
            for (int half = wg >> 1; half >= 1; half >>= 1) {
                int pos = tid & ((half << 1) - 1);
                if (pos < half) {
                    int u = tid, v = tid + half;
                    float ang = -3.14159265358979323846f * (float)pos / (float)half;
                    float wr = sycl::cos(ang), wi = sycl::sin(ang);
                    float ur = re[u], ui = im[u];
                    float vr = re[v], vi = im[v];
                    re[u] = ur + vr; im[u] = ui + vi;
                    re[v] = (ur-vr)*wr - (ui-vi)*wi;
                    im[v] = (ur-vr)*wi + (ui-vi)*wr;
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
            buf[2*tid] = re[tid]; buf[2*tid+1] = im[tid];
        });
    }).wait();

    std::vector<float> h_out(total);
    q.memcpy(h_out.data(), buf, total*sizeof(float)).wait();
    sycl::free(buf, q);

    // CPU DFT reference
    std::vector<float> ref_re(WG, 0.0f), ref_im(WG, 0.0f);
    for (int k = 0; k < WG; k++)
        for (int n = 0; n < WG; n++) {
            float ang = -2.0f * 3.14159265f * (float)k * (float)n / (float)WG;
            ref_re[k] += h_in[2*n] * std::cos(ang);
            ref_im[k] += h_in[2*n] * std::sin(ang);
        }

    // DIF produces bit-reversed output: position i holds DFT[bitrev(i)]
    auto bitrev = [](int x, int bits) {
        int r = 0;
        for (int b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; }
        return r;
    };
    const int BITS = 4; // log2(16)

    bool ok = true;
    for (int i = 0; i < WG && ok; i++) {
        int k  = bitrev(i, BITS);
        float gr = h_out[2*i], gi = h_out[2*i+1];
        float rr = ref_re[k],  ri = ref_im[k];
        float diff = std::sqrt((gr-rr)*(gr-rr) + (gi-ri)*(gi-ri));
        float mag  = std::sqrt(rr*rr + ri*ri) + 1.0f;
        if (diff > 0.01f * mag + 0.1f) {
            std::cout << "FAIL verify_fft at i=" << i << " (k=" << k
                      << "): gpu=(" << gr << "," << gi
                      << ") ref=(" << rr << "," << ri << ")\n";
            ok = false;
        }
    }
    return ok;
}

static void verify_all(sycl::queue& q) {
    std::cout << "\n=== Correctness verification ===\n";
    std::cout << "memory : " << (verify_memory(q) ? "OK" : "FAIL") << "\n";
    std::cout << "gemm   : " << (verify_gemm(q)   ? "OK" : "FAIL") << "\n";
    std::cout << "fft    : " << (verify_fft(q)    ? "OK" : "FAIL") << "\n";
}

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

// ============================================================================
// 6b. Register-tiled GEMM
//
// Each thread computes a TM×TN block of C, dramatically increasing
// arithmetic intensity vs the 1-output-per-thread tile approach.
//
// Arithmetic intensity comparison (FMAs per local memory read):
//   tile=16 (1 output/thread): TILE/2 = 8 FMAs/read
//   reg BM=BN=64, BK=8, TM=TN=8: TM*TN*BK/(TM*BK + TN*BK) = 512/128 = 4 FMAs/read
//   ... actually 8× more FMAs per global memory byte due to BM/BN being 4× larger
// ============================================================================

template<int BM, int BN, int BK, int TM, int TN>
void bench_gemm_reg(Bench& b, int N, const std::string& label) {
    constexpr int THREADS_M = BM / TM;
    constexpr int THREADS_N = BN / TN;
    constexpr int THREADS   = THREADS_M * THREADS_N;

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

    int gm = (N + BM - 1) / BM;
    int gn = (N + BN - 1) / BN;

    b.run(label, [&] {
        b.q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 2> lA({BM, BK}, h);
            sycl::local_accessor<float, 2> lB({BK, BN}, h);

            h.parallel_for(
                sycl::nd_range<2>(
                    sycl::range<2>(gm * THREADS_M, gn * THREADS_N),
                    sycl::range<2>(THREADS_M, THREADS_N)
                ),
                [=](sycl::nd_item<2> item) {
                    int tr  = (int)item.get_local_id(0);
                    int tc  = (int)item.get_local_id(1);
                    int tid = tr * THREADS_N + tc;

                    int bm = (int)item.get_group(0) * BM; // block row start in C
                    int bn = (int)item.get_group(1) * BN; // block col start in C

                    float acc[TM][TN] = {};

                    int nk = (N + BK - 1) / BK;
                    for (int kt = 0; kt < nk; kt++) {
                        int bk = kt * BK; // K block start

                        // Cooperatively load lA[BM][BK] — stride by THREADS
                        for (int s = tid; s < BM * BK; s += THREADS) {
                            int sm = s / BK, sk = s % BK;
                            int row = bm + sm, col = bk + sk;
                            lA[sm][sk] = (row < N && col < N) ? A[row * N + col] : 0.0f;
                        }
                        // Cooperatively load lB[BK][BN] — stride by THREADS
                        for (int s = tid; s < BK * BN; s += THREADS) {
                            int sk = s / BN, sn = s % BN;
                            int row = bk + sk, col = bn + sn;
                            lB[sk][sn] = (row < N && col < N) ? B[row * N + col] : 0.0f;
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        // Accumulate into TM×TN register tile
                        for (int k = 0; k < BK; k++) {
                            float aReg[TM], bReg[TN];
                            for (int m = 0; m < TM; m++) aReg[m] = lA[tr*TM+m][k];
                            for (int n = 0; n < TN; n++) bReg[n] = lB[k][tc*TN+n];
                            for (int m = 0; m < TM; m++)
                                for (int n = 0; n < TN; n++)
                                    acc[m][n] += aReg[m] * bReg[n];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    // Write back TM×TN outputs
                    for (int m = 0; m < TM; m++)
                        for (int n = 0; n < TN; n++) {
                            int row = bm + tr*TM + m;
                            int col = bn + tc*TN + n;
                            if (row < N && col < N) C[row * N + col] = acc[m][n];
                        }
                }
            );
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
        // Register tiling: BM=BN=64, BK=8, TM=TN=8 → 64 threads, 64 outputs each
        bench_gemm_reg<64, 64, 8, 8, 8>(b, N, "gemm reg64 TM=TN=8 N=" + std::to_string(N));
    }
    // large N — should take seconds
    {
        const int N = 3072;
        std::cout << "\n=== Tiled GEMM " << N << "x" << N << " float ===\n";
        bench_gemm_tile< 8>(b, N, "gemm tile=8  N=" + std::to_string(N));
        bench_gemm_tile<16>(b, N, "gemm tile=16 N=" + std::to_string(N));
        bench_gemm_tile<32>(b, N, "gemm tile=32 N=" + std::to_string(N));
        bench_gemm_reg<64, 64, 8, 8, 8>(b, N, "gemm reg64 TM=TN=8 N=" + std::to_string(N));
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

    verify_all(q);

    bench_memory(b, N32M);
    bench_u32_throughput(b, N32M);
    bench_carry_chains(b, N32M);
    bench_divergence(b, N32M);
    bench_tiles(b, 1024, 1024);
    bench_gemm(b);
    bench_fft(b);

    return 0;
}
