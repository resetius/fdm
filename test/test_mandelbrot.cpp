#include "big_float.h"
#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>

#include "unixbench_score.h"

// Simple HSV → RGB conversion
inline void hsv2rgb(float h, float s, float v,
                    unsigned char &r, unsigned char &g, unsigned char &b) {
    float c = v * s;
    float hp = h / 60.0f;
    float x = c * (1 - std::fabs(std::fmod(hp, 2.0f) - 1));
    float m = v - c;
    float rp, gp, bp;
    if      (0 <= hp && hp < 1)  { rp = c;  gp = x;  bp = 0; }
    else if (1 <= hp && hp < 2)  { rp = x;  gp = c;  bp = 0; }
    else if (2 <= hp && hp < 3)  { rp = 0;  gp = c;  bp = x; }
    else if (3 <= hp && hp < 4)  { rp = 0;  gp = x;  bp = c; }
    else if (4 <= hp && hp < 5)  { rp = x;  gp = 0;  bp = c; }
    else                          { rp = c;  gp = 0;  bp = x; }
    r = static_cast<unsigned char>((rp + m) * 255);
    g = static_cast<unsigned char>((gp + m) * 255);
    b = static_cast<unsigned char>((bp + m) * 255);
}

/**
 * Save a Mandelbrot iteration buffer as a color PPM image.
 *
 * @tparam BF          BigFloat specialization, must have static constexpr int blocks.
 * @param device_buf   Pointer to device memory containing int[width*height] iterations.
 * @param width        Image width in pixels.
 * @param height       Image height in pixels.
 * @param max_iter     Number of iterations used (for color normalization).
 * @param q            SYCL queue (used for device→host copy).
 * @param filename     Output filename; if empty, defaults to "mandelbrot_<blocks>.ppm".
 */
void save_mandelbrot_ppm(const int *device_buf,
                         int width, int height,
                         int max_iter,
                         sycl::queue &q,
                         std::string filename) {
    // Copy iteration counts from device to host
    std::vector<int> host_buf(width * height);
    q.memcpy(host_buf.data(), device_buf,
             sizeof(int) * host_buf.size()).wait();

    // Open output file in binary mode
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Write PPM header (P6 = binary RGB)
    ofs << "P6\n" << width << " " << height << "\n255\n";

    // Write pixel data
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int iter = host_buf[y * width + x];
            unsigned char r, g, b;
            if (iter >= max_iter) {
                // Points inside the set → black
                r = g = b = 0;
            } else {
                // Map iteration count to hue [0,360)
                float t   = float(iter) / float(max_iter);
                float hue = 360.0f * t;
                hsv2rgb(hue, 1.0f, 1.0f, r, g, b);
            }
            ofs.put(r).put(g).put(b);
        }
    }
    ofs.close();
}

template<typename T>
int get_iteration_mandelbrot(T x0, T y0, int max_iterations) {
    T x = 0;
    T y = 0;
    T xn;
    T x2=x*x, y2=y*y;
    T _4 = T(4);
    int i;
    for (i = 1; i < max_iterations && x2+y2 < _4; i=i+1) {
        x2=x*x; y2=y*y;
        xn = x2 - y2 + x0;
        if constexpr(std::is_same_v<T,double> || std::is_same_v<T,float>) {
            y = T(2.0)*x*y + y0;
        } else {
            y = (x*y).Mul2() + y0;
        }
        x = xn;
    }

    return i;
}

template<int bs>
struct TypeSelector {
    static constexpr int blocks = bs;
    using T = BigFloat<blocks, uint32_t>;
};

template<>
struct TypeSelector<0> {
    static constexpr int blocks = 0;
    using T = float;
};

template<>
struct TypeSelector<1> {
    static constexpr int blocks = 1;
    using T = double;
};

template<int blocks>
void mandelbrot()
{
    sycl::queue q{ sycl::default_selector_v };
    int height = 1000;
    int width = 1000;
    using T = typename TypeSelector<blocks>::T;

    int* buffer = sycl::malloc_device<int>(height * width, q);

    T view_width = 4.0;
    T center_x = -0.75;
    T center_y = 0.0;
    double _1_w = 1.0 / width;
    T pixel_size = view_width * T(_1_w);
    int tries = 50;
    int max_iterations = 400;

    std::vector<double> times;
    for (int i = 0; i < tries; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
                int x = idx[0];
                int y = idx[1];

                T x0 = center_x + T(x - width / 2.0) * pixel_size;
                T y0 = center_y + T(y - height / 2.0) * pixel_size;

                buffer[y*width+x] = get_iteration_mandelbrot(x0, y0, max_iterations);
            });
        }).wait();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }
    auto score = fdm::unixbench_score(times);
    std::cerr << "Blocks: " << blocks << ", elapsed time: " << score << " ms" << std::endl;

    std::string filename = "mandelbrot_" + std::to_string(blocks) + ".ppm";
    save_mandelbrot_ppm(buffer, width, height, max_iterations, q, filename);

    sycl::free(buffer, q);
}

void test_mandelbrot() {
    mandelbrot<0>();
    mandelbrot<1>();
    mandelbrot<2>();
    mandelbrot<4>();
    mandelbrot<8>();
    mandelbrot<16>();
    mandelbrot<32>();
}

int main() {
    test_mandelbrot();
    return 0;
}
