#include "ns_cyl.h"

#include "umfpack_solver.h"
#include "velocity_plot.h"
#include "eigenvectors_storage.h"
#include "mgsch.h"
#include "projection.h"

using namespace fdm;
using namespace std;
using namespace asp;

template<typename T, bool check, tensor_flag zflag>
void calc(const Config& c) {
    using namespace std::chrono;

    Config c1;
    using Task = NSCyl<T, true, zflag>;
    using tensor = typename Task::tensor;
    Task ns(c);

    const int steps = c.get("ns", "steps", 1);
    const int plot_interval = c.get("plot", "interval", 100);
    const int png = c.get("plot", "png", 1);
    const int vtk = c.get("plot", "vtk", 0);
    const int stabilize = c.get("st", "enable", 0) && zflag == tensor_flag::periodic;
    string fn = c.get("st", "input", "input.nc");
    const int ststep = c.get("st", "step", 100);
    vector<vector<T>> eigenvectors;
    int i;

    // for stabilization
    int nphi = ns.nphi, nz = ns.nz, nr = ns.nr;
    tensor u{{0, nphi-1, 0, nz-1, 1, nr-1}, (T*)0xff};
    tensor v{{0, nphi-1, 0, nz-1, 1, nr}, (T*)0xff};
    tensor w{{0, nphi-1, 0, nz-1, 1, nr}, (T*)0xff};
    tensor p({0, nphi-1, 0, nz-1, 1, nr}, (T*)0xff);
    int eig_size = u.size+v.size+w.size+p.size;
    vector<T> vec(eig_size);
    int off = 0;
    u.use(&vec[off]); off += u.size;
    v.use(&vec[off]); off += v.size;
    w.use(&vec[off]); off += w.size;
    p.use(&vec[off]); off += p.size;

    if (stabilize) {
        eigenvectors_storage s(fn);
        s.load(eigenvectors, c1);
        mgsch<T>(eigenvectors, (int) eigenvectors.size(), (int) eigenvectors[0].size());
    }

    velocity_plotter<T,true,typename NSCyl<T, true, zflag>::tensor_flags> plot(
        ns.dr, ns.dz, ns.dphi,
        ns.nr, ns.nz, ns.nphi,
        ns.r0, ns.R,
        ns.h1, ns.h2,
        0, 2*M_PI, true);

    plot.set_labels("R", "Z", "PHI");

    if (png || vtk) {
        plot.use(ns.u.vec, ns.v.vec, ns.w.vec);
        plot.update();
    }

    if (png) {
        //ns.plot();
        plot.plot(format("step_%07d.png", ns.time_index), ns.time_index*ns.dt);
    }
    if (vtk) {
        //ns.vtk_out();
        plot.vtk_out(format("step_%07d.vtk", ns.time_index), ns.time_index);
    }

    auto t1 = steady_clock::now();
    for (i = 0; i < steps; i++) {
        ns.step();

        if ((i+1) % plot_interval == 0 && (png || vtk)) {
            plot.update();
            if (png) {
                //ns.plot();
                plot.plot(format("step_%07d.png", ns.time_index), ns.time_index*ns.dt);
            }
            if (vtk) {
                //ns.vtk_out();
                plot.vtk_out(format("step_%07d.vtk", ns.time_index), ns.time_index);
            }
        }
        if ((i+1) % ststep == 0) {
            u = ns.u; v = ns.v; w = ns.w; p = ns.p;
            ortoproj_along(&vec[0], eigenvectors, eigenvectors.size(), eig_size);
            ns.u = u; ns.v = v; ns.w = w; ns.p = p;
        }
    }

    auto t2 = steady_clock::now();
    auto interval = duration_cast<duration<double>>(t2 - t1);
    printf("It took me '%f' seconds\n", interval.count());
}

template<typename T, tensor_flag zflag>
void calc1(const Config& c) {
    bool check = c.get("other", "check", 0) == 1;
    if (check) {
        calc<T, true, zflag>(c);
    } else {
        calc<T, false, zflag>(c);
    }
}

template<typename T>
void calc2(const Config& c) {
    bool periodic = c.get("ns", "zperiod", 0) == 1;
    if (periodic) {
        calc1<T, tensor_flag::periodic>(c);
    } else {
        calc1<T, tensor_flag::none>(c);
    }
}

// Флетчер, том 2, страница 398
int main(int argc, char** argv) {
    string config_fn = "ns_rect.ini";

    Config c;

    c.open(config_fn);
    c.rewrite(argc, argv);

    string solver = c.get("solver", "name", "umfpack");
    string datatype = c.get("solver", "datatype", "double");

    if (datatype == "float") {
        using T = float;
        calc2<T>(c);
    } else {
        using T = double;
        calc2<T>(c);
    }

    return 0;
}
