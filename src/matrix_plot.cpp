#include "matrix_plot.h"

namespace fdm {

matrix_plotter::matrix_plotter(const matrix_plotter::settings& s)
    : levels(nullptr)
    , s(s)
{
    plspage(300, 300, 1000, 1000, 0, 0);
    plsdev(s.dname.c_str());
    plsfnam(s.fname_.c_str());
    plscolor(0);
    //plsdiori(3);
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

matrix_plotter::page::~page() {
    if (data) {
        for (int i = 0; i < rows; i++) {
            free(data[i]);
        }
    }
    free(data);

    data = nullptr;
}

void matrix_plotter::plot_internal(const page& p) {
    plenv(p.y1, p.y2, p.x1, p.x2, 0, 0);
    pllab(p.xlab.c_str(), p.ylab.c_str(), p.tlab.c_str());
    pllsty(2);
    pl_setcontlabelparam(0.006, 0.6, 0.1, 1);
    plcont(p.data, p.rows, p.rs, 1, p.rows, 1, p.rs, levels, p.nlevels, transform,
           const_cast<matrix_plotter::page*>(&p));
    //pladv(0);
}

void matrix_plotter::transform(double x, double y, double* tx, double* ty, void* data) {
    matrix_plotter::page* p = static_cast<matrix_plotter::page*>(data);
    *tx = p->x1 + x/p->rs * (p->x2-p->x1+1);
    *ty = p->y1 + y/p->rows * (p->y2-p->y1+1);
}

} // namespace fdm
