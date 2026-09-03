#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>

#include "verify.h"

namespace fdm {

template<typename T>
class OmpSafeTmpVector {
public:
    OmpSafeTmpVector(int size)
#ifdef _OPENMP
        : thread_count(omp_get_max_threads())
#else
        : thread_count(1)
#endif
        , size(size)
        , vec(size*thread_count)
    { }

    T& operator[](int i) {
        return vec[offset() + i];
    }

    const T operator[](int i) const {
        return vec[offset() + i];
    }

    T* data() {
        return vec.data() + offset();
    }

private:
    // The slot count is fixed at construction; the slot is selected on use.
    int offset() const {
        const int id = thread_id();
        verify(id < thread_count);
        return id*size;
    }

    const int thread_count;
    const int size;
    std::vector<T> vec;

    static int thread_id() {
#ifdef _OPENMP
        return omp_get_thread_num();
#else
        return 0;
#endif
    }
};

} // namespace fdm
