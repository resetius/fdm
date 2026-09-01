#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fdm {

// The sampled continuum profile A*r+B/r is not an exact null vector of the
// staggered viscous stencil. Solve that stencil with the same ghost conditions
// as NSCyl so the base state is stationary for the discrete time step.
template<typename T, typename Geometry>
std::vector<T> make_discrete_couette_velocity(const Geometry& geometry) {
    const int nr = geometry.nr;
    if (nr < 2) {
        throw std::invalid_argument("discrete Couette profile needs nr >= 2");
    }

    std::vector<long double> lower(nr, 0);
    std::vector<long double> diagonal(nr, 0);
    std::vector<long double> upper(nr, 0);
    std::vector<long double> right_hand_side(nr, 0);

    const long double dr = geometry.dr;
    const long double dr2 = dr*dr;
    for (int j = 1; j <= nr; ++j) {
        const int row = j-1;
        const long double r = geometry.r0+(j-0.5L)*dr;
        const long double inner = (r-0.5L*dr)/r;
        const long double outer = (r+0.5L*dr)/r;
        lower[row] = inner;
        diagonal[row] = -2-dr2/(r*r);
        upper[row] = outer;
    }

    // w[0] = 2*U0-w[1], w[nr+1] = -w[nr].
    diagonal[0] -= lower[0];
    right_hand_side[0] = -2*lower[0]*geometry.U0;
    lower[0] = 0;
    diagonal[nr-1] -= upper[nr-1];
    upper[nr-1] = 0;

    for (int row = 1; row < nr; ++row) {
        if (diagonal[row-1] == 0) {
            throw std::runtime_error("singular discrete Couette system");
        }
        const long double factor = lower[row]/diagonal[row-1];
        diagonal[row] -= factor*upper[row-1];
        right_hand_side[row] -= factor*right_hand_side[row-1];
    }

    std::vector<long double> solution(nr);
    if (diagonal[nr-1] == 0) {
        throw std::runtime_error("singular discrete Couette system");
    }
    solution[nr-1] = right_hand_side[nr-1]/diagonal[nr-1];
    for (int row = nr-2; row >= 0; --row) {
        solution[row] = (right_hand_side[row]
                         -upper[row]*solution[row+1])/diagonal[row];
    }

    std::vector<T> profile(nr+2);
    for (int j = 1; j <= nr; ++j) {
        profile[j] = static_cast<T>(solution[j-1]);
    }
    profile[0] = T(2)*static_cast<T>(geometry.U0)-profile[1];
    profile[nr+1] = -profile[nr];
    return profile;
}

template<typename T, typename Geometry>
std::vector<T> make_discrete_couette_pressure(
    const Geometry& geometry, const std::vector<T>& velocity) {
    const int nr = geometry.nr;
    if (static_cast<int>(velocity.size()) != nr+2) {
        throw std::invalid_argument("discrete Couette velocity has the wrong size");
    }

    std::vector<T> pressure(nr+2);
    for (int j = 1; j < nr; ++j) {
        const long double r_face = geometry.r0+j*geometry.dr;
        const long double w_face = 0.5L*(velocity[j]+velocity[j+1]);
        pressure[j+1] = pressure[j]+static_cast<T>(
            geometry.dr*w_face*w_face/r_face);
    }

    const T inner_velocity = T(0.5)*(velocity[0]+velocity[1]);
    const T outer_velocity = T(0.5)*(velocity[nr]+velocity[nr+1]);
    pressure[0] = pressure[1]-static_cast<T>(
        geometry.dr*inner_velocity*inner_velocity/geometry.r0);
    pressure[nr+1] = pressure[nr]+static_cast<T>(
        geometry.dr*outer_velocity*outer_velocity/geometry.R);

    long double weighted_sum = 0;
    long double weight = 0;
    for (int j = 1; j <= nr; ++j) {
        const long double r = geometry.r0+(j-0.5L)*geometry.dr;
        weighted_sum += r*pressure[j];
        weight += r;
    }
    const T mean = static_cast<T>(weighted_sum/weight);
    for (T& value : pressure) {
        value -= mean;
    }
    return pressure;
}

// Layout of the independent NSCyl unknowns for periodic z. The full packed
// state is component-major, while one Fourier phase uses the radial offsets.
template<typename T>
class NSCylStateLayout {
public:
    enum class Component {
        u,
        v,
        w,
        p
    };

    const int nr;
    const int nz;
    const int nphi;

    const int u_radial_offset;
    const int v_radial_offset;
    const int w_radial_offset;
    const int p_radial_offset;
    const int radial_size;

    const int u_offset;
    const int v_offset;
    const int w_offset;
    const int p_offset;
    const int u_size;
    const int v_size;
    const int w_size;
    const int p_size;
    const int state_size;

    NSCylStateLayout(int nr, int nz, int nphi)
        : nr(nr)
        , nz(nz)
        , nphi(nphi)
        , u_radial_offset(0)
        , v_radial_offset(nr-1)
        , w_radial_offset(2*nr-1)
        , p_radial_offset(3*nr-1)
        , radial_size(4*nr-1)
        , u_offset(0)
        , v_offset(nphi*nz*(nr-1))
        , w_offset(v_offset+nphi*nz*nr)
        , p_offset(w_offset+nphi*nz*nr)
        , u_size(v_offset-u_offset)
        , v_size(w_offset-v_offset)
        , w_size(p_offset-w_offset)
        , p_size(nphi*nz*nr)
        , state_size(p_offset+p_size)
    {
        if (nr < 2 || nz <= 0 || nphi <= 0) {
            throw std::invalid_argument("invalid NSCyl state dimensions");
        }
    }

    template<typename Task>
    explicit NSCylStateLayout(const Task& task)
        : NSCylStateLayout(task.nr, task.nz, task.nphi)
    { }

    int radial_offset(Component component) const {
        switch (component) {
        case Component::u: return u_radial_offset;
        case Component::v: return v_radial_offset;
        case Component::w: return w_radial_offset;
        case Component::p: return p_radial_offset;
        }
        throw std::logic_error("unknown NSCyl state component");
    }

    int radial_index(Component component, int j) const {
        if (component == Component::u) {
            if (j < 1 || j >= nr) {
                throw std::out_of_range("u radial index is outside [1,nr-1]");
            }
            return u_radial_offset+j-1;
        }
        if (j < 1 || j > nr) {
            throw std::out_of_range("radial index is outside [1,nr]");
        }
        return radial_offset(component)+j-1;
    }

    template<typename Function>
    void for_each_radial(Function&& function) const {
        for (int j = 1; j < nr; ++j) {
            function(Component::u, j, radial_index(Component::u, j));
        }
        for (int j = 1; j <= nr; ++j) {
            function(Component::v, j, radial_index(Component::v, j));
        }
        for (int j = 1; j <= nr; ++j) {
            function(Component::w, j, radial_index(Component::w, j));
        }
        for (int j = 1; j <= nr; ++j) {
            function(Component::p, j, radial_index(Component::p, j));
        }
    }

    template<typename Task>
    void pack(Task& state, T* destination) const {
        int index = u_offset;
        for (int i = 0; i < nphi; ++i) {
            for (int k = 0; k < nz; ++k) {
                for (int j = 1; j < nr; ++j) {
                    destination[index++] = state.u[i][k][j];
                }
            }
        }
        if (index != v_offset) {
            throw std::logic_error("invalid packed u size");
        }

        pack_cell_field(state.v, destination, index);
        if (index != w_offset) {
            throw std::logic_error("invalid packed v size");
        }
        pack_cell_field(state.w, destination, index);
        if (index != p_offset) {
            throw std::logic_error("invalid packed w size");
        }
        pack_cell_field(state.p, destination, index);
        if (index != state_size) {
            throw std::logic_error("invalid packed p size");
        }
    }

    template<typename Task>
    std::vector<T> pack(Task& state) const {
        std::vector<T> result(state_size);
        pack(state, result.data());
        return result;
    }

    template<typename Task>
    void unpack(Task& state, const T* source) const {
        clear_state(state);

        int index = u_offset;
        for (int i = 0; i < nphi; ++i) {
            for (int k = 0; k < nz; ++k) {
                for (int j = 1; j < nr; ++j) {
                    state.u[i][k][j] = source[index++];
                }
            }
        }
        if (index != v_offset) {
            throw std::logic_error("invalid unpacked u size");
        }

        unpack_cell_field(state.v, source, index);
        if (index != w_offset) {
            throw std::logic_error("invalid unpacked v size");
        }
        unpack_cell_field(state.w, source, index);
        if (index != p_offset) {
            throw std::logic_error("invalid unpacked w size");
        }
        unpack_cell_field(state.p, source, index);
        if (index != state_size) {
            throw std::logic_error("invalid unpacked p size");
        }

        state.apply_boundary_conditions();
    }

    template<typename Task>
    void pack_difference(Task& state, const std::vector<T>& reference,
                         T* destination) const {
        require_size(reference);
        pack(state, destination);
        for (int i = 0; i < state_size; ++i) {
            destination[i] -= reference[i];
        }
    }

    template<typename Task>
    void unpack_sum(Task& state, const std::vector<T>& reference,
                    const T* perturbation) const {
        require_size(reference);
        std::vector<T> sum(state_size);
        for (int i = 0; i < state_size; ++i) {
            sum[i] = reference[i]+perturbation[i];
        }
        unpack(state, sum.data());
    }

    template<typename Task>
    void initialize_couette_state(Task& state) const {
        clear_state(state);
        const auto velocity = make_discrete_couette_velocity<T>(state);
        const auto pressure = make_discrete_couette_pressure(state, velocity);

        for (int i = 0; i < nphi; ++i) {
            for (int k = 0; k < nz; ++k) {
                for (int j = 0; j <= nr+1; ++j) {
                    state.w[i][k][j] = velocity[j];
                    state.p[i][k][j] = pressure[j];
                }
            }
        }
        normalize_pressure(state);
    }

    template<typename Task>
    void initialize_couette_linearization(Task& state) const {
        std::fill(state.u0.vec, state.u0.vec+state.u0.size, T(0));
        std::fill(state.v0.vec, state.v0.vec+state.v0.size, T(0));
        std::fill(state.w0.vec, state.w0.vec+state.w0.size, T(0));
        const auto profile = make_discrete_couette_velocity<T>(state);

        for (int i = 0; i < nphi; ++i) {
            for (int k = 0; k < nz; ++k) {
                for (int j = 0; j <= nr+1; ++j) {
                    state.w0[i][k][j] = profile[j];
                }
            }
        }
    }

    template<typename Task>
    double pressure_mean(Task& state) const {
        long double sum = 0;
        long double weight = 0;
        for (int i = 0; i < nphi; ++i) {
            for (int k = 0; k < nz; ++k) {
                for (int j = 1; j <= nr; ++j) {
                    const long double r = state.r0+(j-0.5L)*state.dr;
                    sum += r*static_cast<long double>(state.p[i][k][j]);
                    weight += r;
                }
            }
        }
        return static_cast<double>(sum/weight);
    }

    template<typename Task>
    void normalize_pressure(Task& state) const {
        const T mean = static_cast<T>(pressure_mean(state));
        for (int i = 0; i < state.p.size; ++i) {
            state.p.vec[i] -= mean;
        }
    }

private:
    void require_size(const std::vector<T>& state) const {
        if (static_cast<int>(state.size()) != state_size) {
            throw std::invalid_argument("packed NSCyl state has the wrong size");
        }
    }

    template<typename Field>
    void pack_cell_field(Field& field, T* destination, int& index) const {
        for (int i = 0; i < nphi; ++i) {
            for (int k = 0; k < nz; ++k) {
                for (int j = 1; j <= nr; ++j) {
                    destination[index++] = field[i][k][j];
                }
            }
        }
    }

    template<typename Field>
    void unpack_cell_field(Field& field, const T* source, int& index) const {
        for (int i = 0; i < nphi; ++i) {
            for (int k = 0; k < nz; ++k) {
                for (int j = 1; j <= nr; ++j) {
                    field[i][k][j] = source[index++];
                }
            }
        }
    }

    template<typename Task>
    void clear_state(Task& state) const {
        std::fill(state.u.vec, state.u.vec+state.u.size, T(0));
        std::fill(state.v.vec, state.v.vec+state.v.size, T(0));
        std::fill(state.w.vec, state.w.vec+state.w.size, T(0));
        std::fill(state.p.vec, state.p.vec+state.p.size, T(0));
    }

};

} // namespace fdm
