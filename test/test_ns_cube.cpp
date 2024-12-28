#include <string>
#include <vector>
#include <climits>
#include <cmath>
#include <chrono>
#include <cstdio>

#include "tensor.h"
#include "velocity_plot.h"
#include "config.h"
#include "sparse.h"
#include "asp_misc.h"
#include "ns_cube.h"

using namespace fdm;
using namespace asp;
using std::vector;
using std::string;

template<typename T, bool check>
void calc(const Config& c) {
    using namespace std::chrono;

    NSCube<T, true> ns(c);
    velocity_plotter<T,true> plot(
        ns.dx, ns.dy, ns.dz,
        ns.nx, ns.ny, ns.nz,
        ns.x1, ns.x2,
        ns.y1, ns.y2,
        ns.z1, ns.z2);

    const int steps = c.get("ns", "steps", 1);
    const int plot_interval = c.get("plot", "interval", 100);
    const int png = c.get("plot", "png", 1);
    const int vtk = c.get("plot", "vtk", 0);
    int i;

    if (png || vtk) {
        plot.use(ns.u.vec, ns.v.vec, ns.w.vec);
        plot.update();
    }

    if (png) {
        plot.plot(format("step_%07d.png", ns.time_index), ns.time_index*ns.dt);
    }
    if (vtk) {
        plot.vtk_out(format("step_%07d.vtk", ns.time_index), ns.time_index);
    }

    auto t1 = steady_clock::now();

    for (i = 0; i < steps; i++) {
        ns.step();

        if ((i+1) % plot_interval == 0 && (png || vtk)) {
            plot.update();
            if (png) {
                plot.plot(format("step_%07d.png", ns.time_index), ns.time_index*ns.dt);
            }
            if (vtk) {
                plot.vtk_out(format("step_%07d.vtk", ns.time_index), ns.time_index);
            }
        }
    }

    auto t2 = steady_clock::now();
    auto interval = duration_cast<duration<double>>(t2 - t1);
    printf("It took me '%f' seconds\n", interval.count());
}

template<typename T>
void calc1(const Config& c) {
    bool check = c.get("other", "check", 0) == 1;
    if (check) {
        calc<T, true>(c);
    } else {
        calc<T, false>(c);
    }
}

// Флетчер, том 2, страница 398
int main(int argc, char** argv) {
    string config_fn = "ns_cube.ini";

    Config c;

    c.open(config_fn);
    c.rewrite(argc, argv);

    string datatype = c.get("solver", "datatype", "double");

    if (datatype == "float") {
        using T = float;
        calc1<T>(c);
    } else {
        using T = double;
        calc1<T>(c);
    }

    return 0;
}
