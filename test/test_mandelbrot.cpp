#include "big_float.h"
#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <random>
#include <complex>

#include <boost/multiprecision/cpp_bin_float.hpp>

#include "unixbench_score.h"

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dist;

using namespace boost::multiprecision;
using boost_float256 = number<cpp_bin_float<256, digit_base_2, void, long long, false>>;

template<typename T>
std::pair<T, T> find_zoom_point(const int* iterations_buffer, int width, int height, int max_iterations, const T& view_width, const T& cx, const T& cy) {
    int center_x = width / 2;
    int center_y = height / 2;
    int search_radius = width / 4;
    double _1_w = 1.0 / width;
    double _1_h = 1.0 / height;

    std::multimap<double,std::pair<int,int>> top_scores;
    int scores = 10;

    T pixel_width = view_width * T(_1_w);
    T pixel_height = (view_width * T(height) * T(_1_w)) * T(_1_h);

    for (int y = center_y - search_radius; y < center_y + search_radius; y++) {
        for (int x = center_x - search_radius; x < center_x + search_radius; x++) {
            if (x < 2 || x >= width-2 || y < 2 || y >= height-2) continue;

            int black_count = 0;
            int colored_count = 0;
            //int total_count = 0;

            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    int idx = ((y + dy) * width + (x + dx));
                    if (iterations_buffer[idx] == max_iterations) {
                        black_count++;
                    } else {
                        colored_count++;
                    }
                    //total_count ++;
                }
            }

            // double score = black_count * (total_count-colored_count);
            //double score = colored_count == 0
            //    ? 0
            //    : (double)black_count / (double)colored_count;
            double score = black_count == 0
                ? 0
                : (double)colored_count / (double)black_count;

            double distance_penalty = std::sqrt(
                std::pow(x - center_x, 2) +
                std::pow(y - center_y, 2)
            ) / search_radius;
            score *= (1.0 - 0.3 * distance_penalty);

            top_scores.insert(std::make_pair(score, std::make_pair(x, y)));

            if ((int)top_scores.size() > scores) {
                top_scores.erase(top_scores.begin());
            }
        }
    }

    double r = dist(gen);
    double exponent = 2.0;
    int n = (int)top_scores.size();

    size_t index = static_cast<size_t>( std::floor( (n - 1) - std::pow(r, exponent) * (n - 1) ) );

    auto it = top_scores.begin();
    std::advance(it, index);
    auto [best_x, best_y] = it->second;

    T new_x = cx + T(best_x - width/2) * pixel_width;
    T new_y = cy + T(best_y - height/2) * pixel_height;

    return {new_x, new_y};
}

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
void save_mandelbrot_ppm(const int *host_buf,
                         int width, int height,
                         int max_iter,
                         std::string filename) {

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

void save_mandelbrot_ppm(const unsigned char *host_buf,
                         int width, int height,
                         std::string filename) {

    // Open output file in binary mode
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Write PPM header (P6 = binary RGB)
    ofs << "P6\n" << width << " " << height << "\n255\n";

    const size_t data_size = static_cast<size_t>(width) * height * 3;
    ofs.write(reinterpret_cast<const char*>(host_buf), data_size);
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
        if constexpr(std::is_same_v<T,double> || std::is_same_v<T,float> || std::is_same_v<T,boost_float256>) {
            y = T(2.0)*x*y + y0;
        } else {
            y = (x*y).Mul2() + y0;
        }
        x = xn;
    }

    return i;
}

template<typename T>
unsigned char get_iteration_mandelbrot2(T x0, T y0, int max_iterations) {
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

    double mu = i;
    if (i < max_iterations) {
        double log_zn = std::log((x2 + y2).ToDouble()) / 2.0;
        double nu     = std::log(log_zn / std::log(2.0)) / std::log(2.0);
        mu += 1.0 - nu;
    }

    double t = mu / max_iterations;
    double shade = std::sqrt(1.0 - t);
    std::cerr << "shade: " << shade << std::endl;
    unsigned char c = static_cast<unsigned char>(shade * 255.0);

    // std::cerr << "c: " << (int)c << std::endl;
    return c;
}

template<typename T>
void get_iteration_mandelbrot3(std::pair<T,T>& c, std::vector<std::complex<double>>& control, T x0, T y0, int max_iterations) {
    T x = 0;
    T y = 0;
    T xn;
    T _4 = T(4);
    T x2=x*x, y2=y*y;
    int i;

    c = {x0, y0};
    control.emplace_back(0.0, 0.0);
    for (i = 1; i < max_iterations && x2+y2 < _4; i=i+1) {
        x2=x*x; y2=y*y;
        xn = x2 - y2 + x0;
        if constexpr(std::is_same_v<T,double> || std::is_same_v<T,float> || std::is_same_v<T,boost_float256>) {
            y = T(2.0)*x*y + y0;
        } else {
            y = (x*y).Mul2() + y0;
        }
        x = xn;
        control.emplace_back((double)x, (double)y);
    }
}

template<typename T>
int get_iteration_mandelbrot4(const std::pair<T,T> c, const std::complex<double>* z, T x0, T y0, int max_iterations) {
    std::complex<double> d = {(x0 - c.first).ToDouble(), (y0 - c.second).ToDouble()};
    std::complex<double> e = {0, 0};
    std::complex<double> en = {0, 0};

    int i;
    //e = d;
    // en = 2z e + e e + delta
    for (i = 1; i < max_iterations; i=i+1) {
        const std::complex<double>& zp = z[i-1];
        en = 2.0 * e * zp + e * e + d;
        const std::complex<double>& ze = z[i];
        if (std::norm(en+ze) > 4.0) {
            break;
        }
        e = en;
    }
    //std::cerr << "i: " << i << ", e: " << e.real() << ", " << e.imag() << std::endl;

    return i;
}


template<int bs, typename BlockType = uint32_t>
struct TypeSelector {
    static constexpr int blocks = bs;
    using T = BigFloat<blocks, BlockType, GenericPlatformSpec<BlockType>>;
};

template<typename BlockType>
struct TypeSelector<0, BlockType> {
    static constexpr int blocks = 0;
    using T = float;
};

template<typename BlockType>
struct TypeSelector<1, BlockType> {
    static constexpr int blocks = 1;
    using T = double;
};

template<int blocks, typename BlockType = uint32_t>
void mandelbrot()
{
    sycl::queue q{ sycl::default_selector_v };
    int height = 1000;
    int width = 1000;
    using T = typename TypeSelector<blocks, BlockType>::T;

    int* buffer = sycl::malloc_device<int>(height * width, q);

    T view_width = 4.0;
    T center_x = -0.75;
    T center_y = 0.0;
    double _1_w = 1.0 / width;
    T pixel_size = view_width * T(_1_w);
    int tries = 10;
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
    // Copy iteration counts from device to host
    std::vector<int> host_buf(width * height);
    q.memcpy(host_buf.data(), buffer, sizeof(int) * host_buf.size()).wait();
    save_mandelbrot_ppm(host_buf.data(), width, height, max_iterations, filename);

    sycl::free(buffer, q);
}

template<int blocks, typename BlockType = uint32_t>
void mandelbrot_omp()
{
    int height = 1000;
    int width = 1000;
    using T = typename TypeSelector<blocks, BlockType>::T;

    std::vector<int> host_buf(width * height);
    int* buffer = host_buf.data();

    T view_width = 4.0;
    T center_x = -0.75;
    T center_y = 0.0;
    double _1_w = 1.0 / width;
    T pixel_size = view_width * T(_1_w);
    int tries = 10;
    int max_iterations = 400;

    std::vector<double> times;
    for (int i = 0; i < tries; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                T x0 = center_x + T(x - width / 2.0) * pixel_size;
                T y0 = center_y + T(y - height / 2.0) * pixel_size;

                buffer[y*width+x] = get_iteration_mandelbrot(x0, y0, max_iterations);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }
    auto score = fdm::unixbench_score(times);
    std::cerr << "Blocks: " << blocks << ", elapsed time: " << score << " ms" << std::endl;

    std::string filename = "mandelbrot_" + std::to_string(blocks) + ".ppm";
    save_mandelbrot_ppm(host_buf.data(), width, height, max_iterations, filename);
}

void test_mandelbrot() {
    std::cerr << "Testing mandelbrot uint32_t ..." << std::endl;
    mandelbrot<0>();
    mandelbrot<1>();
    mandelbrot<2>();
    mandelbrot<4>();
    mandelbrot<8>();
    mandelbrot<16>();
    mandelbrot<32>();

    std::cerr << "Testing mandelbrot uint64_t ..." << std::endl;
    mandelbrot<0, uint64_t>();
    mandelbrot<1, uint64_t>();
    mandelbrot<2, uint64_t>();
    mandelbrot<4, uint64_t>();
    mandelbrot<8, uint64_t>();
    mandelbrot<16, uint64_t>();

    std::cerr << "Testing mandelbrot uint32_t (omp)..." << std::endl;
    mandelbrot_omp<0>();
    mandelbrot_omp<1>();
    mandelbrot_omp<2>();
    mandelbrot_omp<4>();
    mandelbrot_omp<8>();

    std::cerr << "Testing mandelbrot uint64_t (omp)..." << std::endl;
    mandelbrot_omp<0, uint64_t>();
    mandelbrot_omp<1, uint64_t>();
    mandelbrot_omp<2, uint64_t>();
    mandelbrot_omp<4, uint64_t>();

    //mandelbrot_omp<4, uint64_t>();
    //mandelbrot_shade_omp<4, uint64_t>();
}

template<typename T>
bool search_control(std::pair<T,T>& c, std::vector<std::complex<double>>& control, T x0, T y0, int max_iterations) {
    control.clear();
    get_iteration_mandelbrot3(c, control, x0, y0, max_iterations);
    return control.size() >= max_iterations;
}

template<typename T, typename F>
void spiral_search(
    F lambda,
    const T& center_xx, const T& center_yy,
    const T& view_width, int width, int height)
{
    double _1_w = 1.0 / width;
    T pixel_size = view_width * T(_1_w);

    int center_x = width / 2;
    int center_y = height / 2;
    int max_radius = std::max(width, height) / 2;
    bool found = false;

    // Start at the center point
    {
        T x0 = center_xx;
        T y0 = center_yy;
        found = lambda(x0, y0);
    }

    // If not found at center, expand outward in squares of increasing size
    for (int radius = 1; !found && radius <= max_radius; ++radius) {
        // Top and bottom edges of the square
        for (int x = center_x - radius; !found && x <= center_x + radius; ++x) {
            // Skip points outside the image boundaries
            if (x < 0 || x >= width) continue;

            // Top edge
            int y = center_y - radius;
            if (y >= 0 && y < height) {
                T x0 = center_xx + T(x - width / 2.0) * pixel_size;
                T y0 = center_yy + T(y - height / 2.0) * pixel_size;
                if ((found = lambda(x0, y0))) {
                    break;
                }
            }

            // Bottom edge
            y = center_y + radius;
            if (y >= 0 && y < height) {
                T x0 = center_xx + T(x - width / 2.0) * pixel_size;
                T y0 = center_yy + T(y - height / 2.0) * pixel_size;
                if ((found = lambda(x0, y0))) {
                    break;
                }
            }
        }

        // Left and right edges of the square (excluding corners which were handled above)
        for (int y = center_y - radius + 1; !found && y <= center_y + radius - 1; ++y) {
            // Skip points outside the image boundaries
            if (y < 0 || y >= height) continue;

            // Left edge
            int x = center_x - radius;
            if (x >= 0 && x < width) {
                T x0 = center_xx + T(x - width / 2.0) * pixel_size;
                T y0 = center_yy + T(y - height / 2.0) * pixel_size;
                if ((found = lambda(x0, y0))) {
                    break;
                }
            }

            // Right edge
            x = center_x + radius;
            if (x >= 0 && x < width) {
                T x0 = center_xx + T(x - width / 2.0) * pixel_size;
                T y0 = center_yy + T(y - height / 2.0) * pixel_size;
                if ((found = lambda(x0, y0))) {
                    break;
                }
            }
        }
    }

    if (!found) {
        std::cerr << "Parameter not found!" << std::endl;
        abort();
    }
}

float fract(float x) {
    return x - std::floor(x);
}

sycl::event color(unsigned char* output, const int* in, int width, int height, int max_iterations, sycl::queue& q) {
    const float z_factor = 10;
    const float kernelsize = 8;
    const float azimuth = 135 * M_PI/180;
    const float altitude = 45 * M_PI/180;
    const float zenith = M_PI/2 - altitude;

    const float P = 180.0; // tune this to change the color map

#define off(x,y) ((y)*width+(x))

    return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
            int x = idx[0];
            int y = idx[1];
            if (x < 1 || x >= width-1 || y < 1 || y >= height-1 || in[off(x,y)] == max_iterations) {
                output[3*(y * width + x) + 0] = 0;
                output[3*(y * width + x) + 1] = 0;
                output[3*(y * width + x) + 2] = 0;
                return;
            }

            int mu = in[off(x,y)];
            unsigned char cr, cg, cb;
            float lb = in[off(x-1,y-1)];
            float b = in[off(x,y-1)];
            float rb = in[off(x+1,y-1)];
            float l = in[off(x-1,y)];
            float r = in[off(x+1,y)];
            float lt = in[off(x-1,y+1)];
            float t = in[off(x,y+1)];
            float rt = in[off(x+1,y+1)];
            float dzdx = ((rb + 2*r + rt) - (lb + 2*l + lt)) / kernelsize;
            float dzdy = ((lt + 2*t + rt) - (lb + 2*b + rb)) / kernelsize;

            float slope = atan(z_factor * sqrt(dzdx*dzdx + dzdy*dzdy));
            float aspect = atan2(dzdy, -dzdx);
            float shade = ((cos(zenith)*cos(slope)) + (sin(zenith)*sin(slope)*cos(azimuth-aspect)));
            shade = std::max<float>(0, shade);

            float perc =  fract(mu / P);
            float h = 360.0 * perc;
            float s = 1;
            float v = shade;

            hsv2rgb(h, s, v, cr, cg, cb);

            output[3*(y * width + x) + 0] = cr;
            output[3*(y * width + x) + 1] = cg;
            output[3*(y * width + x) + 2] = cb;
        });
    });

#undef off
}

template<int blocks, typename BlockType = uint32_t>
void calc_mandelbrot() {
    sycl::queue q{ sycl::default_selector_v };
    int height = 1000;
    int width = 1000;
    using T = typename TypeSelector<blocks, BlockType>::T;

    int* buffer = sycl::malloc_device<int>(height * width, q);
    std::vector<int> host_buf(width * height);

    unsigned char* rgb_data = sycl::malloc_device<unsigned char>(3 * height * width, q);
    std::vector<unsigned char> rgb(3 * width * height);

    T view_width = 4.0;
    T center_x = -0.75;
    T center_y = 0.0;
    double _1_w = 1.0 / width;
    int max_iterations = 50;
    int max_frames = 2000;
    static constexpr bool perturb = true;

    std::vector<std::complex<double>> control;
    std::pair<T,T> c;
    std::complex<double>* control_data = sycl::malloc_device<std::complex<double>>(max_iterations + max_frames + 1, q);

    auto lambda = [&](T x0, T y0) {
        return search_control(c, control, x0, y0, max_iterations);
    };

    for (int frame = 0; frame < max_frames; frame++) {
        T pixel_size = view_width * T(_1_w);
        auto start = std::chrono::high_resolution_clock::now();
        if (view_width < T(1e-1) && perturb) {
            spiral_search(lambda, center_x, center_y, view_width, width, height);
            q.memcpy(control_data, control.data(), sizeof(std::complex<double>)*control.size());

            q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
                    int x = idx[0];
                    int y = idx[1];

                    T x0 = center_x + T(x - width / 2.0) * pixel_size;
                    T y0 = center_y + T(y - height / 2.0) * pixel_size;

                    buffer[y*width+x] = get_iteration_mandelbrot4(c, control_data, x0, y0, max_iterations);
                });
            }).wait();

        } else {
            q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
                    int x = idx[0];
                    int y = idx[1];

                    T x0 = center_x + T(x - width / 2.0) * pixel_size;
                    T y0 = center_y + T(y - height / 2.0) * pixel_size;

                    buffer[y*width+x] = get_iteration_mandelbrot(x0, y0, max_iterations);
                });
            }).wait();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cerr << "Blocks: " << blocks << ", frame: " << frame << ", elapsed time: " << elapsed.count() << " ms, eps: " << view_width.ToDouble() << std::endl;

        auto event1 = q.memcpy(host_buf.data(), buffer, sizeof(int) * host_buf.size());
        auto event2 = color(rgb_data, buffer, width, height, max_iterations, q);
        auto event3 = q.memcpy(rgb.data(), rgb_data, sizeof(unsigned char) * rgb.size(), {event2});

        {
            char buf[256];
            snprintf(buf, sizeof(buf), "%03d:%.2le", frame,view_width.ToDouble());
            std::string filename = "mandelbrot_frame_" + std::string(buf) + ".ppm";

            event3.wait();
            save_mandelbrot_ppm(rgb.data(), width, height, filename);
        }

        event1.wait();
        auto [new_x, new_y] = find_zoom_point(host_buf.data(), width, height, max_iterations, view_width, center_x, center_y);
        center_x = (center_x+new_x)*0.5;
        center_y = (center_y+new_y)*0.5;
        // view_width = view_width * T(1.0/1.5);
        view_width = view_width * T(1.0/1.2);
        max_iterations += 1;
    }
//    std::cerr << "X: " << center_x.ToString() << "\n";
//    std::cerr << "Y: " << center_y.ToString() << "\n";
//    std::cerr << "W: " << view_width.ToString() << "\n";
    sycl::free(buffer, q);
    sycl::free(control_data, q);
}

template<int blocks, typename BlockType = uint32_t, typename T = typename TypeSelector<blocks, BlockType>::T>
double calc_mandelbrot_perturb_gpu() {
    sycl::queue q{ sycl::default_selector_v };
    int height = 1000;
    int width = 1000;

    T center_x("-0.7476180759505704135946970713481386103624764298397722545821903892128864105376320896311400072644479937");
    T center_y("-0.0974709816484987892699819856482762443302570416103311139092943008409339789097694332141201683941661361");
    T view_width("0.0000000000000000000000000000000850279233328961660578925450907037419478065610364650578428219078049755");

    view_width *= T(1./16);
    int max_iterations = 500;
    double _1_w = 1.0 / width;
    T pixel_size = view_width * T(_1_w);

    std::vector<std::complex<double>> control;
    std::pair<T,T> c;
    std::complex<double>* control_data = sycl::malloc_device<std::complex<double>>(max_iterations + 10 + 1, q);
    auto lambda = [&](T x0, T y0) {
        return search_control(c, control, x0, y0, max_iterations);
    };

    spiral_search(lambda, center_x, center_y, view_width, width, height);
    q.memcpy(control_data, control.data(), sizeof(std::complex<double>)*control.size()).wait();

    std::vector<int> buffer_host(width * height);
    int* buffer = sycl::malloc_device<int>(width * height, q);

    auto start = std::chrono::high_resolution_clock::now();

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
            int x = idx[0];
            int y = idx[1];

            T x0 = center_x + T(x - width / 2.0) * pixel_size;
            T y0 = center_y + T(y - height / 2.0) * pixel_size;

            //buffer[y*width+x] = get_iteration_mandelbrot4(c, control.data(), x0, y0, max_iterations);
            buffer[y*width+x] = get_iteration_mandelbrot(x0, y0, max_iterations);
        });
    }).wait();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cerr << "elapsed time: " << elapsed.count() << " ms " << "\n";

    // save_mandelbrot_ppm(buffer.data(), width, height, max_iterations, "mandelbrot_perturb.ppm");

    sycl::free(buffer, q);
    sycl::free(control_data, q);

    return elapsed.count();
}

template<int blocks, typename BlockType = uint32_t, typename T = typename TypeSelector<blocks, BlockType>::T>
double calc_mandelbrot_perturb_cpu() {
    int height = 1000;
    int width = 1000;

    T center_x("-0.7476180759505704135946970713481386103624764298397722545821903892128864105376320896311400072644479937");
    T center_y("-0.0974709816484987892699819856482762443302570416103311139092943008409339789097694332141201683941661361");
    T view_width("0.0000000000000000000000000000000850279233328961660578925450907037419478065610364650578428219078049755");

    view_width *= T(1./16);
    int max_iterations = 500;
    double _1_w = 1.0 / width;
    T pixel_size = view_width * T(_1_w);

    std::vector<std::complex<double>> control;
    std::pair<T,T> c;
    //std::complex<double>* control_data = sycl::malloc_device<std::complex<double>>(max_iterations + 10 + 1, q);
    //std::complex<double>* control_data = sycl::malloc_device<std::complex<double>>(max_iterations + 10 + 1, q);
    std::vector<std::complex<double>> control_data(max_iterations + 10 + 1);
    auto lambda = [&](T x0, T y0) {
        return search_control(c, control, x0, y0, max_iterations);
    };

    spiral_search(lambda, center_x, center_y, view_width, width, height);
    memcpy(control_data.data(), control.data(), sizeof(std::complex<double>)*control.size());

    std::vector<int> buffer(width * height);

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            T x0 = center_x + T(x - width / 2.0) * pixel_size;
            T y0 = center_y + T(y - height / 2.0) * pixel_size;
            //buffer[y*width+x] = get_iteration_mandelbrot4(c, control.data(), x0, y0, max_iterations);
            buffer[y*width+x] = get_iteration_mandelbrot(x0, y0, max_iterations);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cerr << "elapsed time: " << elapsed.count() << " ms " << "\n";

    save_mandelbrot_ppm(buffer.data(), width, height, max_iterations, "mandelbrot_perturb.ppm");

    return elapsed.count();
}

int main() {
    //test_mandelbrot();
    //calc_mandelbrot<4,uint64_t>();
    //calc_mandelbrot<8>();
    //calc_mandelbrot<8>();
    //calc_mandelbrot<8,uint64_t>();
    //calc_mandelbrot_perturb_cpu<8>();

    int its = 8;
    double score;
    std::vector<double> times; times.reserve(its);

    // 32 bit gpu
    std::cerr << "Testing mandelbrot... (256bit, 32bit block, gpu, generic arithmetics)" << std::endl;
    times.clear();
    for (int i = 0; i < its; i++)
    {
        std::cerr << "Iteration: " << i << "/" << its << "... " << std::flush;
        times.emplace_back(calc_mandelbrot_perturb_gpu<8, uint32_t, BigFloat<8, uint32_t, GenericPlatformSpec<uint32_t>>>());
    }
    score = fdm::unixbench_score(times);
    std::cerr << "Score: " << score << " ms" << std::endl;

    // 64 bit gpu
    std::cerr << "Testing mandelbrot... (256bit, 64bit block, gpu, generic arithmetics)" << std::endl;
    times.clear();
    for (int i = 0; i < its; i++)
    {
        std::cerr << "Iteration: " << i << "/" << its << "... " << std::flush;
        times.emplace_back(calc_mandelbrot_perturb_gpu<4, uint64_t, BigFloat<4, uint64_t, GenericPlatformSpec<uint64_t>>>());
    }
    score = fdm::unixbench_score(times);
    std::cerr << "Score: " << score << " ms" << std::endl;

    // boost float256
    std::cerr << "Testing mandelbrot... (256bit, omp, boost_float256)" << std::endl;
    times.clear();
    for (int i = 0; i < 8; i++)
    {
        std::cerr << "Iteration: " << i << "/" << its << "... " << std::flush;
        times.emplace_back(calc_mandelbrot_perturb_cpu<4, uint64_t, boost_float256>());
    }
    score = fdm::unixbench_score(times);
    std::cerr << "Score: " << score << " ms" << std::endl;

    // 32 bit
    std::cerr << "Testing mandelbrot... (256bit, 32bit block, omp, optimized arithmetics)" << std::endl;
    times.clear();
    for (int i = 0; i < its; i++)
    {
        std::cerr << "Iteration: " << i << "/" << its << "... " << std::flush;
        times.emplace_back(calc_mandelbrot_perturb_cpu<8, uint32_t, BigFloat<8, uint32_t, AMD64PlatformSpec<uint32_t>>>());
    }
    score = fdm::unixbench_score(times);
    std::cerr << "Score: " << score << " ms" << std::endl;

    std::cerr << "Testing mandelbrot... (256bit, 32bit block, omp, generic arithmetics)" << std::endl;
    times.clear();
    for (int i = 0; i < its; i++)
    {
        std::cerr << "Iteration: " << i << "/" << its << "... " << std::flush;
        times.emplace_back(calc_mandelbrot_perturb_cpu<8, uint32_t, BigFloat<8, uint32_t, GenericPlatformSpec<uint32_t>>>());
    }
    score = fdm::unixbench_score(times);
    std::cerr << "Score: " << score << " ms" << std::endl;

    std::cerr << "Testing mandelbrot... (256bit, 32bit block, omp, naive artithmetics)" << std::endl;
    times.clear();
    for (int i = 0; i < 8; i++)
    {
        std::cerr << "Iteration: " << i << "/" << its << "... " << std::flush;
        times.emplace_back(calc_mandelbrot_perturb_cpu<8, uint32_t, BigFloat<8, uint32_t, NaivePlatformSpec<uint32_t>>>());
    }
    score = fdm::unixbench_score(times);
    std::cerr << "Score: " << score << " ms" << std::endl;

    // 64 bit
    std::cerr << "Testing mandelbrot... (256bit, 64bit block, omp, optimized arithmetics)" << std::endl;
    times.clear();
    for (int i = 0; i < its; i++)
    {
        std::cerr << "Iteration: " << i << "/" << its << "... " << std::flush;
        times.emplace_back(calc_mandelbrot_perturb_cpu<4, uint64_t, BigFloat<4, uint64_t, AMD64PlatformSpec<uint64_t>>>());
    }
    score = fdm::unixbench_score(times);
    std::cerr << "Score: " << score << " ms" << std::endl;

    std::cerr << "Testing mandelbrot... (256bit, 64bit block, omp, generic arithmetics)" << std::endl;
    times.clear();
    for (int i = 0; i < its; i++)
    {
        std::cerr << "Iteration: " << i << "/" << its << "... " << std::flush;
        times.emplace_back(calc_mandelbrot_perturb_cpu<4, uint64_t, BigFloat<4, uint64_t, GenericPlatformSpec<uint64_t>>>());
    }
    score = fdm::unixbench_score(times);
    std::cerr << "Score: " << score << " ms" << std::endl;

    std::cerr << "Testing mandelbrot... (256bit, 64bit block, omp, naive artithmetics)" << std::endl;
    times.clear();
    for (int i = 0; i < 8; i++)
    {
        std::cerr << "Iteration: " << i << "/" << its << "... " << std::flush;
        times.emplace_back(calc_mandelbrot_perturb_cpu<4, uint64_t, BigFloat<4, uint64_t, NaivePlatformSpec<uint64_t>>>());
    }
    score = fdm::unixbench_score(times);
    std::cerr << "Score: " << score << " ms" << std::endl;

    std::cerr << "Testing mandelbrot... (256bit, 64bit block, omp, naive artithmetics (int128))" << std::endl;
    times.clear();
    for (int i = 0; i < 8; i++)
    {
        std::cerr << "Iteration: " << i << "/" << its << "... " << std::flush;
        times.emplace_back(calc_mandelbrot_perturb_cpu<4, uint64_t, BigFloat<4, uint64_t, NaivePlatformSpec<uint64_t, true>>>());
    }
    score = fdm::unixbench_score(times);
    std::cerr << "Score: " << score << " ms" << std::endl;

    return 0;
}
