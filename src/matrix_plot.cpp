#include "matrix_plot.h"

namespace fdm {

matrix_plotter::~matrix_plotter() {
    for (int i = 0; i < rows; i++) {
        free(data[i]);
    }
    free(data);
    free(levels);
}

void matrix_plotter::plot() {
    plspage(300, 300, 1000, 1000, 0, 0);

    plinit();
    plenv(0, rs-1, 0, rows-1, 0, 0);
    pllsty(2);
    pl_setcontlabelparam(0.006, 0.6, 0.1, 1);
    plcont(data, rs, rows, 1, rs, 1, rows, levels, nlevels, transform, this);
    plend();
}

void matrix_plotter::transform(double x, double y, double* tx, double* ty, void* data) {
    *tx = x;
    *ty = y;
}

} // namespace fdm
