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
    Config c, c2;
    eigenvectors_storage s("test.nc");

    const char* args[] = {"main", "--nc:test=2", "--nc:var=1.2", nullptr};
    c.rewrite(3, const_cast<char**>(args));
    string s1, s2;
    c.print(s1);

    vector<vector<T>> vecs = {
        {1.0, 3.1, 4.3},
        {5.4, 3.4, 5.4}
    };
    vector<vector<T>> vecs2;
    vector<int> indices = {1,0};

    s.save(vecs, indices, c);
    s.load(vecs2, c2);
    c2.print(s2);
    for (int i = 0; i < 3; i++) {
        assert_float_equal(vecs2[1][i], vecs[0][i], 1e-15);
        assert_float_equal(vecs2[0][i], vecs[1][i], 1e-15);
    }

    assert_string_equal(s1.c_str(), s2.c_str());
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
