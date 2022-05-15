#pragma once

#include <vector>
#include "verify.h"

namespace fdm {

template<typename T, int rank, bool check>
class tensor_accessor {
    T* vec;
    const std::vector<int>& sizes;
    const std::vector<int>& offsets;
    int index;

public:
    tensor_accessor(T* v,
                    const std::vector<int>& sizes,
                    const std::vector<int>& offsets, int index)
        : vec(v)
        , sizes(sizes)
        , offsets(offsets)
        , index(index)
    { }

    tensor_accessor<T,rank-1,check> operator[](int y) {
        if constexpr(check == true) {
            verify(offsets[2*index] <= y && y <= offsets[2*index+1]);
        }
        return tensor_accessor<T,rank-1,check>(
            &vec[(y-offsets[2*index])*sizes[index]],
            sizes, offsets, index+1
            );
    }

    tensor_accessor<const T,rank-1,check> operator[](int y) const {
        if constexpr(check == true) {
            verify(offsets[2*index] <= y && y <= offsets[2*index+1]);
        }
        return tensor_accessor<const T,rank-1,check>(
            &vec[(y-offsets[2*index])*sizes[index]],
            sizes, offsets, index+1
            );
    }
};

// vector
template<typename T>
class tensor_accessor<T, 1, false>
{
    T* vec;

public:
    tensor_accessor(T* v,
                    const std::vector<int>& sizes,
                    const std::vector<int>& offsets, int index)
        : vec(v-offsets[2*index])
    { }

    T& operator[](int x) {
        return vec[x];
    }

    T operator[](int x) const {
        return vec[x];
    }
};

template<typename T>
class tensor_accessor<T, 1, true>
{
    T* vec;
    const std::vector<int>& offsets;
    int index;

public:
    tensor_accessor(T* v,
                    const std::vector<int>& sizes,
                    const std::vector<int>& offsets, int index)
        : vec(v-offsets[2*index])
        , offsets(offsets)
        , index(index)
    { }

    T& operator[](int x) {
        verify(offsets[2*index] <= x && x <= offsets[2*index+1]);
        return vec[x];
    }

    T operator[](int x) const {
        verify(offsets[2*index] <= x && x <= offsets[2*index+1]);
        return vec[x];
    }
};

template<typename T, int rank, bool check>
class tensor {
    std::vector<int> offsets;
    std::vector<int> sizes;
    std::vector<T> vec;
    tensor_accessor<T, rank, check> acc;

public:
    // z1,z2 y1,y2 x1,x2
    tensor(const std::vector<int>& offsets_)
        : offsets(offsets_)
        , sizes(calc_sizes(offsets))
        , vec((offsets[1]-offsets[0]+1)*sizes[0])
        , acc(&vec[0], sizes, offsets, 0)
    {
        verify(offsets.size() == rank*2);
        verify(sizes.size() == rank-1);
    }

    auto operator[](int y) {
        return acc[y];
    }

private:
    std::vector<int> calc_sizes(const std::vector<int>& offsets) {
        std::vector<int> s;
        int prev = 1;
        for (int i = rank-1; i >= 1; i--) {
            int size = prev*(offsets[2*i+1] - offsets[2*i] + 1);
            s.push_back(size);
            prev = size;
        }
        std::reverse(s.begin(), s.end());
        return s;
    }
};

} // namespace fdm
