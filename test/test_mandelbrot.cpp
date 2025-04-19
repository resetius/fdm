#include "big_float.h"
#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>

#include "unixbench_score.h"

template<typename T>
int get_iteration_mandelbrot(T x0, T y0, int max_iterations = 100) {
    T x = 0;
    T y = 0;
    T xn;
    T x2=x*x, y2=y*y;
    T _4 = T(4);
    int i;
    for (i = 1; i < max_iterations && x2+y2 < _4; i=i+1) {
        x2=x*x; y2=y*y;
        xn = x2 - y2 + x0;
        if constexpr(std::is_same_v<T,double>) {
            y = T(2.0)*x*y + y0;
        } else {
            y = (x*y).Mul2() + y0;
        }
        x = xn;
    }

    return i;
}

void mandelbrot()
{
    sycl::queue q{ sycl::default_selector_v };
    int height = 1000;
    int width = 1000;
    using T = BigFloat<4, uint32_t>;
    int* buffer = sycl::malloc_device<int>(height * width, q);

    T view_width = 4.0;
    T center_x = -0.75;
    T center_y = 0.0;
    double _1_w = 1.0 / width;
    T pixel_size = view_width * T(_1_w);
    int tries = 50;

    std::vector<double> times;
    for (int i = 0; i < tries; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
                int x = idx[0];
                int y = idx[1];

                T x0 = center_x + T(x - width / 2.0) * pixel_size;
                T y0 = center_y + T(y - height / 2.0) * pixel_size;

                buffer[y*width+x] = get_iteration_mandelbrot(x0, y0);
            });
        }).wait();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }
    auto score = fdm::unixbench_score(times);
    std::cerr << "Elapsed time: " << score << " ms" << std::endl;

    sycl::free(buffer, q);
}

void test_mandelbrot() {
    mandelbrot();
}

int main() {
    test_mandelbrot();
    return 0;
}
