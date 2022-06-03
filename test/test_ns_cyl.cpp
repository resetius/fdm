#include "ns_cyl.h"

#include "umfpack_solver.h"
#include "gmres_solver.h"
#include "superlu_solver.h"
#include "jacobi_solver.h"

template<typename T, template<typename> class Solver, bool check>
void calc(const Config& c) {
    using namespace std::chrono;

    NSCyl<T, Solver, true> ns(c);

    const int steps = c.get("ns", "steps", 1);
    const int plot_interval = c.get("plot", "interval", 100);
    const int png = c.get("plot", "png", 1);
    const int vtk = c.get("plot", "vtk", 0);
    int i;

    if (png) {
        ns.plot();
    }
    if (vtk) {
        ns.vtk_out();
    }

    auto t1 = steady_clock::now();
    for (i = 0; i < steps; i++) {
        ns.step();

        if ((i+1) % plot_interval == 0) {
            if (png) {
                ns.plot();
            }
            if (vtk) {
                ns.vtk_out();
            }
        }
    }

    auto t2 = steady_clock::now();
    auto interval = duration_cast<duration<double>>(t2 - t1);
    printf("It took me '%f' seconds\n", interval.count());
}

template<typename T, template<typename> class Solver>
void calc1(const Config& c) {
    bool check = c.get("other", "check", 0) == 1;
    if (check) {
        calc<T, Solver, true>(c);
    } else {
        calc<T, Solver, false>(c);
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
        if (solver == "gmres") {
            calc1<T, gmres_solver>(c);
        } else if (solver == "superlu") {
            calc1<T, superlu_solver>(c);
        } else if (solver == "jacobi") {
            calc1<T, jacobi_solver>(c);
        } else {
            calc1<T, superlu_solver>(c);
        }
    } else {
        using T = double;
        if (solver == "gmres") {
            calc1<T, gmres_solver>(c);
        } else if (solver == "superlu") {
            calc1<T, superlu_solver>(c);
        } else if (solver == "jacobi") {
            calc1<T, jacobi_solver>(c);
        } else {
            calc1<T, umfpack_solver>(c);
        }
    }

    return 0;
}
