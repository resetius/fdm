#include "matrix_plot.h"

namespace fdm {

matrix_plotter::matrix_plotter(const matrix_plotter::settings& s)
    : levels(nullptr)
    , s(s)
{
    plspage(300, 300, 2000, 2000, 0, 0);
    plsdev(s.dname.c_str());
    plsfnam(s.fname_.c_str());
    plscolbga(255,255,255,1);
    plscol0(1,0,0,0); // foreground
    plssub(s.x, s.y);
    plinit();
}

matrix_plotter::~matrix_plotter() {
    clear();
    plend();
}

void matrix_plotter::clear() {
    free(levels);
    levels = nullptr;
}

void matrix_plotter::data::clear() {
    if (d) {
        for (int i = 0; i < rows; i++) {
            free(d[i]);
        }
    }
    free(d);

    d = nullptr;
}

matrix_plotter::data::~data() {
    clear();
}

void matrix_plotter::plot_internal(const page& p) {
    plenv(p.y1, p.y2, p.x1, p.x2, 1, 0);
    pllab(p.xlab.c_str(), p.ylab.c_str(), p.tlab.c_str());
    pllsty(2);

    if (!p.v.d) {
        pl_setcontlabelparam(0.006, 0.3, 0.1, 1);
        plcont(p.u.d, p.u.rows, p.u.rs, 1, p.u.rows, 1, p.u.rs,
               levels, p.nlevels, transform,
               const_cast<matrix_plotter::page*>(&p));
    } else {
        plvect(p.v.d, p.u.d, p.u.rs, p.u.rows, 0,
               transform,
               const_cast<matrix_plotter::page*>(&p));
    }
}

void matrix_plotter::transform(double y, double x, double* ty, double* tx, void* data) {
    matrix_plotter::page* p = static_cast<matrix_plotter::page*>(data);
    *tx = p->x1 + x/(p->u.rs-1) * (p->x2-p->x1);
    *ty = p->y1 + y/(p->u.rows-1) * (p->y2-p->y1);
}

} // namespace fdm
