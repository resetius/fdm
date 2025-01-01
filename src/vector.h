#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

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
        return vec[thread_id()*size + i];
    }

    const T operator[](int i) const {
        return vec[thread_id()*size + i];
    }

    T* data() {
        return vec.data() + thread_id()*size;
    }

private:
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
