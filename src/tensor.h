#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include "verify.h"

namespace fdm {

enum struct tensor_flag : int {
    none = 0,
    periodic = 1
};

template<tensor_flag... Args>
struct tensor_flags;

template<tensor_flag Head, tensor_flag... Tail>
struct tensor_flags<Head,Tail...>
{
    using tail = tensor_flags<Tail...>;
    static constexpr tensor_flag head = Head;
};

template<>
struct tensor_flags<>
{
    using tail = tensor_flags<>;
    static constexpr tensor_flag head = tensor_flag::none;
};

constexpr bool has_tensor_flag(tensor_flag flags, tensor_flag flag) {
    return (static_cast<int>(flags) & static_cast<int>(flag)) != 0;
}

template<typename T, int rank, bool check, typename F>
class tensor_accessor {
    T* vec;
    const std::vector<int>& sizes;
    const std::vector<int>& offsets;
    int index;

public:
    tensor_accessor(T* v,
                    const std::vector<int>& sizes,
                    const std::vector<int>& offsets,
                    int index)
        : vec(v)
        , sizes(sizes)
        , offsets(offsets)
        , index(index)
    { }

    auto operator[](int y) {
        y = adjust_and_check(y);
        return tensor_accessor<T,rank-1,check,typename F::tail>(
            &vec[(y-offsets[2*index])*sizes[index]],
            sizes, offsets, index+1
            );
    }

    auto operator[](int y) const {
        y = adjust_and_check(y);
        return tensor_accessor<const T,rank-1,check,typename F::tail>(
            &vec[(y-offsets[2*index])*sizes[index]],
            sizes, offsets, index+1
            );
    }

    auto off(int* indices, int i) {
        return (*this)[indices[i]].off(indices, i+1);
    }

private:
    int adjust_and_check(int y) const {
        if constexpr(has_tensor_flag(F::head, tensor_flag::periodic)) {
            y -= offsets[2*index];
            y = (y+offsets[2*index+1]-offsets[2*index]+1) % (offsets[2*index+1]-offsets[2*index]+1);
            y += offsets[2*index];
        }
        if constexpr(check == true) {
            verify(offsets[2*index] <= y && y <= offsets[2*index+1]);
        }
        return y;
    }
};

// vector
template<typename T, typename F>
class tensor_accessor<T, 1, false, F>
{
    T* vec;

public:
    tensor_accessor(T* v,
                    const std::vector<int>& sizes,
                    const std::vector<int>& offsets,
                    int index)
        : vec(v-offsets[2*index])
    { }

    T& operator[](int x) {
        return vec[x];
    }

    T operator[](int x) const {
        return vec[x];
    }

    T* off(int* indices, int i) {
        return &(*this)[indices[i]];
    }
};

template<typename T, typename F>
class tensor_accessor<T, 1, true, F>
{
    T* vec;
    const std::vector<int>& offsets;
    int index;

public:
    tensor_accessor(T* v,
                    const std::vector<int>& sizes,
                    const std::vector<int>& offsets,
                    int index)
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

    T* off(int* indices, int i) {
        return &(*this)[indices[i]];
    }
};

template<typename T, int rank, bool check=true, typename F = tensor_flags<>>
class tensor {
public:
    std::vector<int> offsets;
    std::vector<int> sizes;
    std::vector<T> vec;

private:
    tensor_accessor<T, rank, check, F> acc;

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

    int index(const std::array<int,rank>& indices /*z,y,x*/) {
        return acc.off(&indices[0], 0) - &vec[0];
    }

    T maxabs() const {
        return std::accumulate(vec.begin(), vec.end(), 0.0, [](T a, T b) {
            a = std::abs(a); b = std::abs(b);
            return a<b?b:a;
        });
    }

private:
    static std::vector<int> calc_sizes(const std::vector<int>& offsets) {
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
