#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.h"
#include "cyclic_reduction.h"
#include "lapl_cyl.h"
#include "ns_cyl_state.h"
#include "tensor.h"

namespace fdm {

// Direct radial realization of one real-packed Fourier block of NSCyl::L_step.
// The phi/z shifts act on at most four tensor-product phase coefficients; no
// full three-dimensional state or FFT is used by apply().
template<typename T>
class NSCylFourierBlockNative {
public:
    using value_type = T;
    using StateLayout = NSCylStateLayout<T>;
    using Component = typename StateLayout::Component;

    const double R;
    const double r0;
    const double h1;
    const double h2;
    const double U0;
    const double base_outer_radius;
    const double Re;
    const double dt;
    const int nr;
    const int nz;
    const int nphi;
    const double dr;
    const double dz;
    const double dphi;
    const double dr2;
    const double dz2;
    const double dphi2;

    NSCylFourierBlockNative(const Config& config, int m, int l,
                            int operator_steps=1)
        : R(config.get("ns", "R", M_PI))
        , r0(config.get("ns", "r", M_PI/2))
        , h1(config.get("ns", "h1", 0.0))
        , h2(config.get("ns", "h2", 10.0))
        , U0(config.get("ns", "u0", 1.0))
        , base_outer_radius(config.get(
              "spectral", "base_outer_radius",
              config.get("ns", "R", M_PI)))
        , Re(config.get("ns", "Re", 1.0))
        , dt(config.get("ns", "dt", 0.001))
        , nr(config.get("ns", "nr", 32))
        , nz(config.get("ns", "nz", 32))
        , nphi(config.get("ns", "nphi", 32))
        , dr((R-r0)/nr)
        , dz((h2-h1)/nz)
        , dphi(2*M_PI/nphi)
        , dr2(dr*dr)
        , dz2(dz*dz)
        , dphi2(dphi*dphi)
        , layout_(nr, nz, nphi)
        , m_(m)
        , l_(l)
        , phi_phases_(endpoint(m, nphi) ? 1 : 2)
        , z_phases_(endpoint(l, nz) ? 1 : 2)
        , phases_(phi_phases_*z_phases_)
        , full_size_(phases_*layout_.radial_size)
        , pressure_gauge_fixed_(m == 0 && l == 0)
        , size_(full_size_-(pressure_gauge_fixed_ ? 1 : 0))
        , operator_steps_(operator_steps)
        , radial_stride_(nr+3)
        , state_(full_size_)

        , u_({0, phases_-1, -1, nr+1})
        , v_({0, phases_-1, -1, nr+1})
        , w_({0, phases_-1, -1, nr+1})
        , p_({0, phases_-1, -1, nr+1})
        , F_({0, phases_-1, -1, nr+1})
        , G_({0, phases_-1, -1, nr+1})
        , H_({0, phases_-1, -1, nr+1})

        , rhs_({0, phases_-1, -1, nr+1})
        , x_({0, phases_-1, -1, nr+1})

        , w0_(make_discrete_couette_velocity<T>(*this, base_outer_radius))
        , phi_plus_(make_shift(true, +1))
        , phi_minus_(make_shift(true, -1))
        , z_plus_(make_shift(false, +1))
        , z_minus_(make_shift(false, -1))
        , identity_(make_identity())
        , lap_phi_(-4*std::pow(std::sin(M_PI*m/nphi), 2)) // = P+ + P- - 2I = (2cos(theta) - 2)
        , lap_z_(-4*std::pow(std::sin(M_PI*l/nz), 2)) // = Z+ + Z- - 2I = (2cos(theta) - 2)
        , centered_phi_(combine(phi_plus_, T(1), phi_minus_, T(-1)))
        , backward_phi_(combine(identity_, T(1), phi_minus_, T(-1)))
        , forward_phi_(combine(phi_plus_, T(1), identity_, T(-1)))
        , backward_z_(combine(identity_, T(1), z_minus_, T(-1)))
        , forward_z_(combine(z_plus_, T(1), identity_, T(-1)))
        , phi_plus_sum_(combine(identity_, T(1), phi_plus_, T(1)))
        , phi_plus_backward_z_(multiply(phi_plus_sum_, backward_z_))
        , poisson_lower_(nr)
        , poisson_diagonal_(nr)
        , poisson_upper_(nr)
        , poisson_upper2_(nr)
        , poisson_pivots_(nr)
        , poisson_work_(nr)
        , cyclic_reduction_(config.get("spectral", "tridiagonal", std::string("lapack")) == "cr")
        , cr_diagonal_(nr)
        , cr_lower_(nr)
        , cr_upper_(nr)
        , cr_(nr)
    {
        if (nr < 2 || nz <= 0 || nphi <= 0) {
            throw std::invalid_argument("invalid native NSCyl dimensions");
        }
        if (m < 0 || m > nphi/2 || l < 0 || l > nz/2) {
            throw std::invalid_argument(
                "native Fourier frequency is outside [0,N/2]");
        }
        if (operator_steps <= 0) {
            throw std::invalid_argument("operator_steps must be positive");
        }
        factor_poisson();
    }

    int radial_size() const { return layout_.radial_size; }
    int phase_count() const { return phases_; }
    int size() const { return size_; }
    int full_size() const { return full_size_; }
    int operator_steps() const { return operator_steps_; }
    int m() const { return m_; }
    int l() const { return l_; }
    bool pressure_gauge_fixed() const { return pressure_gauge_fixed_; }
    double last_fourier_leakage() const { return 0; }
    const StateLayout& state_layout() const { return layout_; }

    void apply(T* output, const T* input) {
        if (!output || !input) {
            throw std::invalid_argument("null native Fourier block vector");
        }
        if (pressure_gauge_fixed_) {
            layout_.expand_zero_gauge_block(*this, input, state_.data());
        } else {
            std::copy(input, input+size_, state_.begin());
        }
        unpack();
        for (int step = 0; step < operator_steps_; ++step) {
            apply_boundary_conditions();
            compute_fgh();
            project();
            update();
        }
        pack();
        if (pressure_gauge_fixed_) {
            layout_.reduce_zero_gauge_block(
                *this, state_.data(), output);
        } else {
            std::copy(state_.begin(), state_.end(), output);
        }
    }

private:
    using Matrix = std::vector<T>;

    StateLayout layout_;
    int m_;
    int l_;
    int phi_phases_;
    int z_phases_;
    int phases_;
    int full_size_;
    bool pressure_gauge_fixed_;
    int size_;
    int operator_steps_;
    int radial_stride_;

    std::vector<T> state_;

    tensor<T, 2, false> u_;
    tensor<T, 2, false> v_;
    tensor<T, 2, false> w_;
    tensor<T, 2, false> p_;
    tensor<T, 2, false> F_;
    tensor<T, 2, false> G_;
    tensor<T, 2, false> H_;

    tensor<T, 2, false> rhs_;
    tensor<T, 2, false> x_;

    std::vector<T> w0_;

    Matrix phi_plus_;
    Matrix phi_minus_;
    Matrix z_plus_;
    Matrix z_minus_;
    Matrix identity_;
    double lap_phi_;
    double lap_z_;
    Matrix centered_phi_;
    Matrix backward_phi_;
    Matrix forward_phi_;
    Matrix backward_z_;
    Matrix forward_z_;
    Matrix phi_plus_sum_;
    Matrix phi_plus_backward_z_;

    std::vector<T> poisson_lower_;
    std::vector<T> poisson_diagonal_;
    std::vector<T> poisson_upper_;
    std::vector<T> poisson_upper2_;
    std::vector<int> poisson_pivots_;
    std::vector<T> poisson_work_;

    bool cyclic_reduction_;
    std::vector<T> cr_diagonal_;
    std::vector<T> cr_lower_;
    std::vector<T> cr_upper_;
    CyclicReduction<T> cr_;

    static bool endpoint(int q, int n) {
        return q == 0 || 2*q == n;
    }

    std::size_t field_storage_size() const {
        return static_cast<std::size_t>(phases_)*radial_stride_;
    }

    T transformed(const Matrix& matrix, tensor<T, 2, false>& field, int phase, int j) const {
        T value = 0;
        for (int column = 0; column < phases_; ++column) {
            value += matrix[phase*phases_+column]*field[column][j];
        }
        return value;
    }

    Matrix make_identity() const {
        Matrix result(static_cast<std::size_t>(phases_)*phases_, T(0));
        for (int phase = 0; phase < phases_; ++phase) {
            result[phase*phases_+phase] = T(1);
        }
        return result;
    }

    Matrix make_shift(bool azimuthal, int direction) const {
        const int count = azimuthal ? phi_phases_ : z_phases_;
        const int q = azimuthal ? m_ : l_;
        const int n = azimuthal ? nphi : nz;
        const T angle = static_cast<T>(
            direction*2*M_PI*static_cast<double>(q)/n);
        std::vector<T> one(static_cast<std::size_t>(count)*count, T(0));
        if (count == 1) {
            one[0] = std::cos(angle);
        } else {
            const T cosine = std::cos(angle);
            const T sine = std::sin(angle);
            one[0] = cosine;
            one[1] = sine;
            one[2] = -sine;
            one[3] = cosine;
        }

        Matrix result(static_cast<std::size_t>(phases_)*phases_, T(0));
        for (int phi_row = 0; phi_row < phi_phases_; ++phi_row) {
            for (int z_row = 0; z_row < z_phases_; ++z_row) {
                const int row = phi_row*z_phases_+z_row;
                for (int phi_column = 0; phi_column < phi_phases_; ++phi_column) {
                    for (int z_column = 0; z_column < z_phases_; ++z_column) {
                        const int column = phi_column*z_phases_+z_column;
                        if (azimuthal && z_row == z_column) {
                            result[row*phases_+column] = one[phi_row*phi_phases_+phi_column];
                        } else if (!azimuthal && phi_row == phi_column) {
                            result[row*phases_+column] = one[z_row*z_phases_+z_column];
                        }
                    }
                }
            }
        }
        return result;
    }

    Matrix combine(const Matrix& a, T a_scale, const Matrix& b, T b_scale) const {
        Matrix result(a.size());
        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i] = a_scale*a[i]+b_scale*b[i];
        }
        return result;
    }

    Matrix multiply(const Matrix& a, const Matrix& b) const {
        Matrix result(static_cast<std::size_t>(phases_)*phases_, T(0));
        for (int row = 0; row < phases_; ++row) {
            for (int column = 0; column < phases_; ++column) {
                for (int inner = 0; inner < phases_; ++inner) {
                    result[row*phases_+column] +=
                        a[row*phases_+inner]*b[inner*phases_+column];
                }
            }
        }
        return result;
    }

    void factor_poisson() {
        const double lambda_phi = 4.0/dphi2
            *std::pow(std::sin(0.5*m_*dphi), 2);
        const double lambda_z = 4.0/dz2
            *std::pow(std::sin(M_PI*static_cast<double>(l_)/nz), 2);
        int lower = 0;
        int diagonal = 0;
        int upper = 0;
        for (int j = 1; j <= nr; ++j) {
            const double radius = r0+(j-0.5)*dr;
            poisson_diagonal_[diagonal++] = static_cast<T>(-2/dr2-lambda_phi/(radius*radius)-lambda_z);
            if (j > 1) {
                poisson_lower_[lower++] = static_cast<T>((radius-0.5*dr)/(dr2*radius));
            }
            if (j < nr) {
                poisson_upper_[upper++] = static_cast<T>((radius+0.5*dr)/(dr2*radius));
            }
        }
        if (cyclic_reduction_) {
            for (int i = 0; i < nr; ++i) {
                cr_diagonal_[i] = poisson_diagonal_[i];
                cr_lower_[i] = (i > 0) ? poisson_lower_[i-1] : T(0);
                cr_upper_[i] = (i < nr-1) ? poisson_upper_[i] : T(0);
            }
            cr_.prepare(cr_diagonal_.data(), cr_lower_.data(), cr_upper_.data());
            return;
        }

        int info = 0;
        lapack::gttrf(
            nr, poisson_lower_.data(), poisson_diagonal_.data(),
            poisson_upper_.data(), poisson_upper2_.data(),
            poisson_pivots_.data(), &info);
        if (info != 0) {
            throw std::runtime_error("native Fourier Poisson factorization failed");
        }
    }

    void unpack() {
        u_.fill(T(0));
        v_.fill(T(0));
        w_.fill(T(0));
        p_.fill(T(0));
        for (int phase = 0; phase < phases_; ++phase) {
            const T* source = state_.data()+phase*layout_.radial_size;
            for (int j = 1; j < nr; ++j) {
                u_[phase][j] = source[layout_.radial_index(Component::u, j)];
            }
            for (int j = 1; j <= nr; ++j) {
                v_[phase][j] = source[layout_.radial_index(Component::v, j)];
                w_[phase][j] = source[layout_.radial_index(Component::w, j)];
                p_[phase][j] = source[layout_.radial_index(Component::p, j)];
            }
        }
    }

    void pack() {
        for (int phase = 0; phase < phases_; ++phase) {
            T* destination = state_.data()+phase*layout_.radial_size;
            for (int j = 1; j < nr; ++j) {
                destination[layout_.radial_index(Component::u, j)] = u_[phase][j];
            }
            for (int j = 1; j <= nr; ++j) {
                destination[layout_.radial_index(Component::v, j)] = v_[phase][j];
                destination[layout_.radial_index(Component::w, j)] = w_[phase][j];
                destination[layout_.radial_index(Component::p, j)] = p_[phase][j];
            }
        }
    }

    void apply_boundary_conditions() {
        for (int phase = 0; phase < phases_; ++phase) {
            u_[phase][0] = T(0);
            u_[phase][nr] = T(0);
            u_[phase][-1] = u_[phase][1];
            u_[phase][nr+1] = u_[phase][nr-1];
            v_[phase][0] = -v_[phase][1];
            v_[phase][nr+1] = -v_[phase][nr];
            w_[phase][0] = -w_[phase][1];
            w_[phase][nr+1] = -w_[phase][nr];
        }
    }

    void compute_fgh() {
        for (int phase = 0; phase < phases_; ++phase) {
            for (int j = 0; j <= nr; ++j) {
                const double radius = r0+j*dr;
                const double outer = (radius+0.5*dr)/radius;
                const double inner = (radius-0.5*dr)/radius;
                const double radius2 = radius*radius;
                const double base_sum = w0_[j+1]+w0_[j];
                double increment =
                    (outer*u_[phase][j+1]-2*u_[phase][j]
                     +inner*u_[phase][j-1])/(Re*dr2)
                    +lap_z_*u_[phase][j]/(Re*dz2) // = lap_z_ * u =
                    +lap_phi_*u_[phase][j]/(Re*dphi2*radius2) // = lap_phi_ * u =
                    -0.25*base_sum*transformed(centered_phi_, u_, phase, j)/(dphi*radius)
                    +0.5*base_sum*(w_[phase][j+1]+w_[phase][j])/radius
                    -u_[phase][j]/(Re*radius2)
                    -transformed(backward_phi_, w_, phase, j+1)/(Re*dphi*radius2)
                    -transformed(backward_phi_, w_, phase, j)/(Re*dphi*radius2);
                F_[phase][j] = static_cast<T>(u_[phase][j]+dt*increment);
            }

            for (int j = 1; j <= nr; ++j) {
                const double radius = r0+(j-0.5)*dr;
                const double outer = (radius+0.5*dr)/radius;
                const double inner = (radius-0.5*dr)/radius;
                const double radius2 = radius*radius;
                double increment =
                    (outer*v_[phase][j+1]-2*v_[phase][j]
                    +inner*v_[phase][j-1])/(Re*dr2)
                    +lap_z_*v_[phase][j]/(Re*dz2) // = lap_z_ * v =
                    +lap_phi_*v_[phase][j]/(Re*dphi2*radius2) // = lap_phi_ * v =
                    -0.5*w0_[j]*transformed(centered_phi_, v_, phase, j)/(dphi*radius);
                G_[phase][j] = static_cast<T>(
                    v_[phase][j]+dt*increment);

                const double base_outer = w0_[j+1]+w0_[j];
                const double base_inner = w0_[j]+w0_[j-1];
                increment =
                    (outer*w_[phase][j+1]-2*w_[phase][j]
                     +inner*w_[phase][j-1])/(Re*dr2)
                    +lap_z_*w_[phase][j]/(Re*dz2) // = lap_z_ * w =
                    +lap_phi_*w_[phase][j]/(Re*dphi2*radius2) // = lap_phi_ * w =
                    -w0_[j]*transformed(centered_phi_, w_, phase, j)/(dphi*radius)
                    -0.25*(outer*base_outer*transformed(phi_plus_sum_, u_, phase, j)
                          -inner*base_inner*transformed(phi_plus_sum_, u_, phase, j-1))/dr
                    -0.5*w0_[j]*transformed(phi_plus_backward_z_, v_, phase, j)/dz
                    -0.5*w0_[j]*transformed(phi_plus_sum_, u_, phase, j)/radius
                    -w_[phase][j]/(Re*radius2)
                    +transformed(centered_phi_, u_, phase, j)/(Re*dphi*radius2);
                H_[phase][j] = static_cast<T>(w_[phase][j]+dt*increment);
            }
        }
    }

    void project() {
        for (int phase = 0; phase < phases_; ++phase) {
            p_[phase][0] = p_[phase][1] - static_cast<T>(dr/dt) * F_[phase][0];
            p_[phase][nr+1] = p_[phase][nr] + static_cast<T>(dr/dt) * F_[phase][nr];
        }

        for (int phase = 0; phase < phases_; ++phase) {
            for (int j = 1; j <= nr; ++j) {
                const double radius = r0+(j-0.5)*dr;
                double value = (
                    ((radius+0.5*dr)*F_[phase][j]
                     -(radius-0.5*dr)*F_[phase][j-1])/(radius*dr)
                    +transformed(backward_z_, G_, phase, j)/dz
                    +transformed(backward_phi_, H_, phase, j)/(dphi*radius))/dt;
                if (j == 1) {
                    value -= (radius-0.5*dr)/radius
                        *p_[phase][0]/dr2;
                }
                if (j == nr) {
                    value -= (radius+0.5*dr)/radius
                        *p_[phase][nr+1]/dr2;
                }
                rhs_[phase][j] = static_cast<T>(value);
            }
        }

        for (int phase = 0; phase < phases_; ++phase) {
            for (int j = 1; j <= nr; ++j) {
                poisson_work_[j-1] = rhs_[phase][j];
            }
            if (cyclic_reduction_) {
                cr_.execute(cr_diagonal_.data(), cr_lower_.data(), cr_upper_.data(), poisson_work_.data());
            } else {
                int info = 0;
                lapack::gttrs(
                    "N", nr, 1, poisson_lower_.data(),
                    poisson_diagonal_.data(), poisson_upper_.data(),
                    poisson_upper2_.data(), poisson_pivots_.data(),
                    poisson_work_.data(), nr, &info);
                if (info != 0) {
                    throw std::runtime_error(
                        "native Fourier Poisson solve failed");
                }
            }
            for (int j = 1; j <= nr; ++j) {
                x_[phase][j] = poisson_work_[j-1];
            }
        }
    }

    void update() {
        for (int phase = 0; phase < phases_; ++phase) {
            for (int j = 1; j < nr; ++j) {
                u_[phase][j] = F_[phase][j]
                    -static_cast<T>(dt/dr)
                        *(x_[phase][j+1]-x_[phase][j]);
            }
            for (int j = 1; j <= nr; ++j) {
                const double radius = r0+(j-0.5)*dr;
                v_[phase][j] = G_[phase][j]
                    -static_cast<T>(dt/dz)
                        *transformed(forward_z_, x_, phase, j);
                w_[phase][j] = H_[phase][j]
                    -static_cast<T>(dt/(dphi*radius))
                        *transformed(forward_phi_, x_, phase, j);
                p_[phase][j] = x_[phase][j];
            }
        }
    }
};

} // namespace fdm
