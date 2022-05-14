#include "matrix_plot.h"

namespace fdm {

matrix_plotter::matrix_plotter()
    : levels(nullptr)
{ }

matrix_plotter::~matrix_plotter() {
    clear();
}

void matrix_plotter::clear() {
    free(levels);
    levels = nullptr;
}

matrix_plotter::settings::~settings() {
    if (data) {
        for (int i = 0; i < rows; i++) {
            free(data[i]);
        }
    }
    free(data);

    data = nullptr;
}

void matrix_plotter::plot_internal(const settings& s) {
    plspage(300, 300, 1000, 1000, 0, 0);
    plsdev(s.dname.c_str());
    plsfnam(s.fname_.c_str());
    plscolor(0);
    //plsdiori(3);

    plinit();
    plenv(s.y1, s.y2, s.x1, s.x2, 0, 0);
    pllab("Y", "X", "");
    pllsty(2);
    pl_setcontlabelparam(0.006, 0.6, 0.1, 1);
    plcont(s.data, s.rows, s.rs, 1, s.rows, 1, s.rs, levels, s.nlevels, transform,
           const_cast<matrix_plotter::settings*>(&s));
    plend();
}

void matrix_plotter::transform(double x, double y, double* tx, double* ty, void* data) {
    matrix_plotter::settings* s = static_cast<matrix_plotter::settings*>(data);
    *tx = s->x1 + x/s->rs * (s->x2-s->x1+1);
    *ty = s->y1 + y/s->rows * (s->y2-s->y1+1);
}

} // namespace fdm
