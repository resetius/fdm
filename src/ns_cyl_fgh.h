#pragma once

namespace fdm {

template<typename T>
struct NSCylFGHParams {
    using value_type = T;

    T r0;
    T dr, dz, dphi;
    T dr2, dz2, dphi2;
    T dt, Re;

    NSCylFGHParams(
        T r0_, T dr_, T dz_, T dphi_,
        T dr2_, T dz2_, T dphi2_, T dt_, T Re_)
        : r0(r0_)
        , dr(dr_), dz(dz_), dphi(dphi_)
        , dr2(dr2_), dz2(dz2_), dphi2(dphi2_)
        , dt(dt_), Re(Re_) {}
};

template<typename Field, typename Result, typename Params>
inline void ns_cyl_f_node(
    Field& u, Field& v, Field& w, Result& F,
    int i, int k, int j, const Params& p) {
    using T = typename Params::value_type;
    const T r = p.r0+p.dr*T(j);
    const T rp = (r+T(0.5)*p.dr)/r;
    const T rm = (r-T(0.5)*p.dr)/r;
    const T rr = r*r;
    const auto sq = [](T x) { return x*x; };

    F(i,k,j) = u(i,k,j)+p.dt*(
        (rp*u(i,k,j+1)-T(2)*u(i,k,j)+rm*u(i,k,j-1))/p.Re/p.dr2+
        (u(i,k+1,j)-T(2)*u(i,k,j)+u(i,k-1,j))/p.Re/p.dz2+
        (u(i+1,k,j)-T(2)*u(i,k,j)+u(i-1,k,j))/p.Re/p.dphi2/rr-
        (rp*sq(T(0.5)*(u(i,k,j)+u(i,k,j+1)))-
         rm*sq(T(0.5)*(u(i,k,j-1)+u(i,k,j))))/p.dr-
        T(0.25)*((u(i,k,j)+u(i,k+1,j))*(v(i,k,j+1)+v(i,k,j))-
                 (u(i,k-1,j)+u(i,k,j))*(v(i,k-1,j+1)+v(i,k-1,j)))/p.dz-
        T(0.25)*((u(i,k,j)+u(i+1,k,j))*(w(i,k,j+1)+w(i,k,j))-
                 (u(i-1,k,j)+u(i,k,j))*(w(i-1,k,j+1)+w(i-1,k,j)))/p.dphi/r+
        sq(T(0.5)*(w(i,k,j+1)+w(i,k,j)))/r-u(i,k,j)/rr/p.Re-
        T(2)*(T(0.5)*(w(i,k,j+1)+w(i,k,j))-
              T(0.5)*(w(i-1,k,j+1)+w(i-1,k,j)))/rr/p.dphi/p.Re);
}

template<typename Field, typename Result, typename Params>
inline void ns_cyl_g_node(
    Field& u, Field& v, Field& w, Result& G,
    int i, int k, int j, const Params& p) {
    using T = typename Params::value_type;
    const T r = p.r0+p.dr*T(j)-T(0.5)*p.dr;
    const T rp = (r+T(0.5)*p.dr)/r;
    const T rm = (r-T(0.5)*p.dr)/r;
    const T rr = r*r;
    const auto sq = [](T x) { return x*x; };

    G(i,k,j) = v(i,k,j)+p.dt*(
        (rp*v(i,k,j+1)-T(2)*v(i,k,j)+rm*v(i,k,j-1))/p.Re/p.dr2+
        (v(i,k+1,j)-T(2)*v(i,k,j)+v(i,k-1,j))/p.Re/p.dz2+
        (v(i+1,k,j)-T(2)*v(i,k,j)+v(i-1,k,j))/p.Re/p.dphi2/rr-
        (sq(T(0.5)*(v(i,k,j)+v(i,k+1,j)))-
         sq(T(0.5)*(v(i,k-1,j)+v(i,k,j))))/p.dz-
        T(0.25)*(rp*(u(i,k,j)+u(i,k+1,j))*(v(i,k,j+1)+v(i,k,j))-
                 rm*(u(i,k,j-1)+u(i,k+1,j-1))*(v(i,k,j)+v(i,k,j-1)))/p.dr-
        T(0.25)*((w(i,k,j)+w(i,k+1,j))*(v(i,k,j)+v(i+1,k,j))-
                 (w(i-1,k,j)+w(i-1,k+1,j))*(v(i-1,k,j)+v(i,k,j)))/p.dphi/r);
}

template<typename Field, typename Result, typename Params>
inline void ns_cyl_h_node(
    Field& u, Field& v, Field& w, Result& H,
    int i, int k, int j, const Params& p) {
    using T = typename Params::value_type;
    const T r = p.r0+p.dr*T(j)-T(0.5)*p.dr;
    const T rp = (r+T(0.5)*p.dr)/r;
    const T rm = (r-T(0.5)*p.dr)/r;
    const T rr = r*r;
    const auto sq = [](T x) { return x*x; };

    H(i,k,j) = w(i,k,j)+p.dt*(
        (rp*w(i,k,j+1)-T(2)*w(i,k,j)+rm*w(i,k,j-1))/p.Re/p.dr2+
        (w(i,k+1,j)-T(2)*w(i,k,j)+w(i,k-1,j))/p.Re/p.dz2+
        (w(i+1,k,j)-T(2)*w(i,k,j)+w(i-1,k,j))/p.Re/p.dphi2/rr-
        (sq(T(0.5)*(w(i+1,k,j)+w(i,k,j)))-
         sq(T(0.5)*(w(i-1,k,j)+w(i,k,j))))/p.dphi/r-
        T(0.25)*(rp*(u(i+1,k,j)+u(i,k,j))*(w(i,k,j+1)+w(i,k,j))-
                 rm*(u(i+1,k,j-1)+u(i,k,j-1))*(w(i,k,j)+w(i,k,j-1)))/p.dr-
        T(0.25)*((w(i,k,j)+w(i,k+1,j))*(v(i,k,j)+v(i+1,k,j))-
                 (w(i,k-1,j)+w(i,k,j))*(v(i,k-1,j)+v(i+1,k-1,j)))/p.dz-
        w(i,k,j)*T(0.5)*(u(i+1,k,j)+u(i,k,j))/r-w(i,k,j)/rr/p.Re+
        T(2)*(T(0.5)*(u(i+1,k,j)+u(i,k,j))-
              T(0.5)*(u(i,k,j)+u(i-1,k,j)))/rr/p.dphi/p.Re);
}

template<typename Field, typename Result, typename Params>
inline void ns_cyl_fgh_node(
    Field& u, Field& v, Field& w,
    Result& F, Result& G, Result& H,
    int i, int k, int face, int nr, const Params& p) {
    ns_cyl_f_node(u, v, w, F, i, k, face, p);
    if (face == nr) {
        return;
    }
    const int center = face+1;
    ns_cyl_g_node(u, v, w, G, i, k, center, p);
    ns_cyl_h_node(u, v, w, H, i, k, center, p);
}

} // namespace fdm
