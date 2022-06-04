#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <type_traits>

#include "umfpack_solver.h"
#include "lapl.h"

extern "C" {
#include <cmocka.h>
}

using namespace fdm;
using namespace std;

void test_lapl_cyl(void** ) {
    LaplCyl3<double, umfpack_solver, true> lapl(
        M_PI, M_PI/2, 0, 10,
        32, 32, 32
        );
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_lapl_cyl)
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
