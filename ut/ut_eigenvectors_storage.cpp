#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <type_traits>
#include <chrono>

#include "umfpack_solver.h"
#include "superlu_solver.h"
#include "lapl_cyl.h"
#include "config.h"
#include "eigenvectors_storage.h"

extern "C" {
#include <cmocka.h>
}

using namespace fdm;
using namespace std;
using namespace std::chrono;
using namespace asp;

template<typename T>
void test_eig_storage() {
    Config c;
    eigenvectors_storage s("test.nc");

    vector<vector<T>> vecs = {
        {1.0, 3.1, 4.3},
        {5.4, 3.4, 5.4}
    };
    vector<vector<T>> vecs2;
    vector<int> indices = {1,0};

    s.save(vecs, indices, c);
    s.load(vecs2, c);
    for (int i = 0; i < 3; i++) {
        assert_float_equal(vecs2[1][i], vecs[0][i], 1e-15);
        assert_float_equal(vecs2[0][i], vecs[1][i], 1e-15);
    }

}

void test_eig_storage_double(void** ) {
    test_eig_storage<double>();
}

void test_eig_storage_float(void** ) {
    test_eig_storage<float>();
}

int main(int argc, char** argv) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate(test_eig_storage_double, nullptr),
        cmocka_unit_test_prestate(test_eig_storage_float, nullptr),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
