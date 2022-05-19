#include <cmath>
#include "tensor.h"

using namespace fdm;

template<bool check>
void run() {
    int x1 = -1, x2 = 10;
    int y1 = -2, y2 = 5;
    int z1 = -5, z2 = 6;
    tensor<double, 3, check> t({z1, z2, y1, y2, x1, x2});
    for (int i = z1; i <= z2; i++) {
        for (int j = y1; j <= y2; j++) {
            for (int k = x1; k <= x2; k++) {
                t[i][j][k] = (i+0.5)*(j+0.5)*(k+0.5);
            }
        }
    }
    for (int i = z1; i <= z2; i++) {
        for (int j = y1; j <= y2; j++) {
            for (int k = x1; k <= x2; k++) {
                printf("%f ", t[i][j][k]);
                verify(
                    fabs((i+0.5)*(j+0.5)*(k+0.5)-t[i][j][k]) < 1e-15
                    );
            }
            printf("\n");
        }
        printf(":\n");
    }
}

int main() {
    run<false>();
    run<true>();

    tensor<double, 2, true, tensor_flags<tensor_flag::periodic>> t({0, 4, -1, 1});
    t[0][1] = 10;
    t[4][1] = 11;
    verify(t[0][1] == t[5][1]);
    verify(t[-1][1] == t[4][1]);

    tensor<double, 2, true, tensor_flags<tensor_flag::periodic>> t2({-1, 4, -1, 2});
    t2[-1][2] = 20;
    verify(t2[5][2] == t2[-1][2]);

    return 0;
}
