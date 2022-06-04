#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <type_traits>

#include "lapl.h"

extern "C" {
#include <cmocka.h>
}

void test_lapl_cyl(void** ) {

}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_lapl_cyl)
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
