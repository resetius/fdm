#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include "blas.h"
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

template<tensor_flag... Args>
struct short_flags
{
    using value = tensor_flags<Args...>;
};

template<>
struct short_flags<tensor_flag::periodic, tensor_flag::none> {
    using value = tensor_flags<tensor_flag::periodic>;
};

template<>
struct short_flags<tensor_flag::none, tensor_flag::none> {
    using value = tensor_flags<>;
};

template<>
struct short_flags<tensor_flag::none> {
    using value = tensor_flags<>;
};

constexpr bool has_tensor_flag(tensor_flag flags, tensor_flag flag) {
    return (static_cast<int>(flags) & static_cast<int>(flag)) != 0;
}

template<typename T, int rank, bool check, typename F>
class tensor_accessor {
    T* vec;
    const int* sizes;

public:
    const int* offsets;
    const int index;

    tensor_accessor(T* v,
                    const int* sizes,
                    const int* offsets,
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

    auto off(const int* indices, int i) {
        return (*this)[indices[i]].off(indices, i+1);
    }

    template<typename T2>
    void assign(const tensor_accessor<T2,rank,check,F>& other) {
        auto from = std::max(offsets[2*index], other.offsets[2*other.index]);
        auto to = std::min(offsets[2*index+1], other.offsets[2*other.index+1]);

        for (int i = from; i <= to; i++) {
            (*this)[i].assign(other[i]);
        }
    }

    void use(T* vec) {
        this->vec = vec;
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
template<typename T, bool check, typename F>
class tensor_accessor<T, 1, check, F>
{
    T* vec;

public:
    const int* offsets;
    const int index;

    tensor_accessor(T* v,
                    const int* sizes,
                    const int* offsets,
                    int index)
        : vec(v-offsets[2*index])
        , offsets(offsets)
        , index(index)
    { }

    T& operator[](int x) {
        x = adjust_and_check(x);
        return vec[x];
    }

    T operator[](int x) const {
        x = adjust_and_check(x);
        return vec[x];
    }

    T* off(const int* indices, int i) {
        return &(*this)[indices[i]];
    }

    template<typename T2>
    void assign(const tensor_accessor<T2,1,check,F>& other) {
        auto from = std::max(offsets[2*index], other.offsets[2*other.index]);
        auto to = std::min(offsets[2*index+1], other.offsets[2*other.index+1]);
        for (int i = from; i <= to; i++) {
            (*this)[i] = other[i];
        }
    }

    void use(T* vec) {
        this->vec = vec;
    }

private:
    // TODO: remove copy-paste
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

template<typename T, int rank, bool check=true, typename F = tensor_flags<>>
class tensor {
public:
    const std::array<int,rank*2> offsets;
    const std::array<int,rank-1> sizes;
    const int size;
    std::vector<T> storage;
    T* vec;
    tensor_accessor<T, rank, check, F> acc;

public:
    // z1,z2 y1,y2 x1,x2
    tensor(const std::array<int,rank*2>& offsets_, T* data = nullptr)
        : offsets(offsets_)
        , sizes{}
        , size(calc_size())
        , storage(data ? 0 : size)
        , vec(data ? data : &storage[0])
        , acc(&vec[0], &sizes[0], &offsets[0], 0)
    {
    }

    tensor(const tensor& other)
        : offsets(other.offsets)
        , sizes(other.sizes)
        , size(other.size)
        , storage(other.storage)
        , vec(storage.empty() ? other.vec: &storage[0])
        , acc(&vec[0], &sizes[0], &offsets[0], 0)
    {
    }

    auto operator[](int y) {
        return acc[y];
    }

    int index(const std::array<int,rank>& indices /*z,y,x*/) {
        return acc.off(&indices[0], 0) - &vec[0];
    }

    T maxabs() const {
        return std::accumulate(vec, vec+size, 0.0, [](T a, T b) {
            a = std::abs(a); b = std::abs(b);
            return a<b?b:a;
        });
    }

    T norm2() const {
        return blas::nrm2(size, vec, 1);
    }

    tensor<T,rank,check,F>& operator=(const tensor<T,rank,check,F>& other) {
        acc.assign(other.acc);
        return *this;
    }

    void use(T* vec) {
        this->vec = vec;
        acc.use(vec);
    }

private:
    int calc_size() {
        int* s = const_cast<int*>(&sizes[0]);
        int prev = 1; int j = 0;
        for (int i = rank-1; i >= 1; i--) {
            int size = prev*(offsets[2*i+1] - offsets[2*i] + 1);
            s[j++] = size;
            prev = size;
        }
        std::reverse(&s[0], &s[0]+j);

        return (offsets[1]-offsets[0]+1)*sizes[0];
    }
};

} // namespace fdm
