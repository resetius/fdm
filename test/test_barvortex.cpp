/**/

#define _USE_MATH_DEFINES
#include <string>
#include <vector>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "sds_bar.h"
#include "asp_misc.h"
#include "asp_sphere_lapl.h"
#include "asp_sphere_jac.h"
#include "statistics.h"
#include "srtm_rel.h"

using namespace std;
using namespace asp;
using namespace SDS;

#undef max

#ifdef WIN32
#include <float.h>
#define isnan _isnan
inline bool isinf(double x)
{
	int c = _fpclass(x);
	return (c == _FPCLASS_NINF || c == _FPCLASS_PINF);
}
#endif

double zero_rp(double phi, double lambda, BarVortexConf * conf)
{
	return 0;
}

double rp(double phi, double lambda, BarVortexConf * conf)
{
	double omg = 2.*M_PI/24./60./60.; // ?
	double T0  = 1./omg;
	double R   = 6.371e+6;
	double c   = T0*T0/R/R;
	double x   = phi;

	double pt1 = -0.5 * (sin(x)*M_PI*x-2*sin(x)*x*x);
	if (fabs(pt1) > 1e-14) {
		pt1 /= cos(x);
	}

	double pt2 = -0.5*(-M_PI+4*x);
	return -T0/R * 16.0 / M_PI / M_PI * 30.0 * (pt1 + pt2);
}

double zero_cor(double phi, double lambda, BarVortexConf * conf)
{
	return 0.0;
}

double cor(double phi, double lambda, BarVortexConf * conf)
{
	return 2.*sin(phi) +  // l
		0.5 * cos(2*lambda)*sin(2*phi)*sin(2*phi); //h
}

double cor2(double phi, double lambda, BarVortexConf * conf)
{
	return 2.*sin(phi);
}

double u0(double phi, double lambda)
{
	double omg = 2.*M_PI/24./60./60.; // ?
	double T0  = 1./omg;
	double R   = 6.371e+6;

	return -T0/R * 16.0 / M_PI / M_PI * 30.0 *
		(M_PI/4 * phi * phi - phi * phi * phi / 3);
}

double u1(double phi, double lambda)
{
	double omg = 2.*M_PI/24./60./60.; // ?
	double T0  = 1./omg;
	double R   = 6.371e+6;

	return sin(2. * phi);
}

#define  pOff(i, j) ( i ) * conf.n_la + ( j )

#ifdef WIN32
#include <windows.h>
#include <float.h>
void set_fpe_except()
{
	int cw = _controlfp(0, 0);
	cw &=~(EM_OVERFLOW|EM_UNDERFLOW|EM_ZERODIVIDE|EM_DENORMAL);
	_controlfp(cw, MCW_EM);
}
#endif

void calc_barvortex_right_part()
{
	BarVortexConf conf;
	conf.steps = 1;
	double R   = 6.371e+6;
	double H   = 5000;
	conf.omg   = 2.*M_PI/24./60./60.; // ?
	double T0  = 1./conf.omg;
	conf.k1    = 1.0;
	conf.k2    = 1.0;
	conf.tau   = 0.001;
	conf.sigma = 1.14e-2;
	conf.mu    = 6.77e-5;
	conf.n_phi = 24;
	conf.n_la  = 32;
	conf.full  = 0;
	conf.rho   = 1;
	conf.theta = 0.5;

	conf.cor    = cor2;
	conf.rp     = rp;
	conf.filter = 1;

	int n = conf.n_phi * conf.n_la;

	double t = 0;
	double T = 30 * 2.0 * M_PI;;
	double nr;
	int i = 0;


	double * u0;
	double * rel;
	int n_phi;
	int n_la;
	asp::load_matrix_from_txtfile(&u0, &n_phi, &n_la, "u0.txt");
	asp::load_matrix_from_txtfile(&rel, &n_phi, &n_la, "rel.txt");
	conf.cor_add = rel;
	vector < double > rp(n_la * n_phi);

	BarVortex bv(conf);
	bv.calc_rp(&rp[0], &u0[0]);

	_fprintfwmatrix("forc.txt", &rp[0], conf.n_phi, conf.n_la, conf.n_la, "%.16lf ");
}

double rnd_u(double phi, double lambda)
{
	return (phi - M_PI / 2) *  phi * sin(lambda) * rand() / (double)RAND_MAX;
}

void LOp(double * u, const double * v, vector < vector < double > > & z, BarVortex * bv, BarVortexConf * conf)
{
	int nn = conf->n_la * conf->n_phi;

	vector < double > tmp1(nn);
	vector < double > tmp2(nn);

	memcpy(&tmp1[0], v, nn * sizeof(double));

	for (int i = 0; i < (int)z.size(); ++i)
	{
		double * z1 = &z[i][0];
		bv->L_step(&tmp2[0], &tmp1[0], z1);
		tmp1.swap(tmp2);
	}

	memcpy(u, &tmp1[0], nn * sizeof(double));
}

void LTOp(double * u, const double * v, vector < vector < double > > & z, BarVortex * bv, BarVortexConf * conf)
{
	int nn = conf->n_la * conf->n_phi;

	vector < double > tmp1(nn);
	vector < double > tmp2(nn);

	memcpy(&tmp1[0], v, nn * sizeof(double));

	for (int i = (int)z.size() - 1; i >= 0; --i)
	{
		double * z1 = &z[i][0];
		bv->LT_step(&tmp2[0], &tmp1[0], z1);
		tmp1.swap(tmp2);
	}

	memcpy(u, &tmp1[0], nn * sizeof(double));
}

void test_barvortex_linear()
{
	BarVortexConf conf;
	conf.steps = 1;
	double R   = 6.371e+6;
	double H   = 5000;
	conf.omg   = 2.*M_PI/24./60./60.; // ?
	double T0  = 1./conf.omg;
	conf.k1    = 1.0;
	conf.k2    = 1.0;
	conf.tau   = 0.001;
	conf.sigma = 1.14e-2;
	conf.mu    = 6.77e-5;
	conf.n_phi = 24;
	conf.n_la  = 32;
	conf.full  = 0;
	conf.rho   = 1;
	conf.theta = 0.5;

	conf.cor    = cor;
	conf.rp     = rp;
	conf.filter = 1;
	conf.cor_add= 0;

	int n = conf.n_phi * conf.n_la;

	double t = 0;
	double T = 10;
	double nr;
	int i = 0;

	BarVortex bv(conf);
	vector < double > u(n);
	vector < double > v(n);

	vector < double > lu(n);
	vector < double > ltv(n);

	vector < double > z1(n);
	vector < double > z0(n);
	vector < vector < double > > z;

	double nr1, nr2;

	z.push_back(z0);

#if 1
	while (t < T)
	{
		bv.S_step(&z1[0], &z0[0]);
		z1.swap(z0);
		z.push_back(z0);
		t += conf.tau;
	}
#endif

	for (int i = 0; i < conf.n_phi; ++i) {
		for (int j = 0; j < conf.n_la; ++j) {
			u[pOff(i, j)] = rnd_u(bv.phi(i), bv.lambda(j));
			v[pOff(i, j)] = rnd_u(bv.phi(i), bv.lambda(j));
		}
	}

	LOp(&lu[0],   &u[0], z, &bv, &conf);
	LTOp(&ltv[0], &v[0], z, &bv, &conf);

	nr1 = bv.scalar(&lu[0], &v[0], n);
	nr2 = bv.scalar(&ltv[0], &u[0], n);
	fprintf(stderr, "%le\n%le\n%le\n", nr1, nr2, fabs(nr1 - nr2));
}

void test_barvortex()
{
	BarVortexConf conf;
	conf.steps = 1;
	double R   = 6.371e+6;
	double H   = 5000;
	conf.omg   = 2.*M_PI/24./60./60.; // ?
	double T0  = 1./conf.omg;
	conf.k1    = 1.0;
	conf.k2    = 1.0;
	conf.tau   = 0.001;
	conf.sigma = 1.14e-2;
	conf.mu    = 6.77e-5;
	conf.n_phi = 24;
	conf.n_la  = 32;
	conf.full  = 0;
	conf.rho   = 1;
	conf.theta = 0.5;

	conf.cor    = cor;
	conf.rp     = rp;
	conf.filter = 1;
	conf.cor_add= 0;

	int n = conf.n_phi * conf.n_la;

	double t = 0;
	double T = 30 * 2.0 * M_PI;;
	double nr;
	int i = 0;

	BarVortex bv(conf);
	vector < double > u(n);
	vector < double > u1(n);

	fprintf(stderr, "#domain:sphere half\n");
	fprintf(stderr, "#mesh_w:%d\n", conf.n_la);
	fprintf(stderr, "#mesh_h:%d\n", conf.n_phi);
	fprintf(stderr, "#filter:%d\n", conf.filter);
	fprintf(stderr, "#tau:%.16lf\n", conf.tau);
	fprintf(stderr, "#sigma:%.16lf\n", conf.sigma);
	fprintf(stderr, "#mu:%.16lf\n", conf.mu);
	fprintf(stderr, "#k1:%.16lf\n", conf.k1);
	fprintf(stderr, "#k2:%.16lf\n", conf.k2);
	fprintf(stderr, "#theta:%.16lf\n", conf.theta);
	fprintf(stderr, "#rp:kornev1\n");
	fprintf(stderr, "#coriolis:kornev1\n");
	fprintf(stderr, "#initial:kornev1\n");
	fprintf(stderr, "#build:$Id$\n");

	for (int i = 0; i < conf.n_phi; ++i) {
		for (int j = 0; j < conf.n_la; ++j) {
			u[pOff(i, j)] = u0(bv.phi(i), bv.lambda(j));
		}
	}

//	{
//		_fprintfwmatrix("kornev1_u0_1.txt", &u[0], conf.n_phi, conf.n_la, conf.n_la, "%.16lf ");
//	}

	while (t < T) {
		bv.S_step(&u1[0], &u[0]);
		t += conf.tau;

//		if (i % 10000 == 0) {
			nr = bv.norm(&u1[0], n);
			fprintf(stderr, "t=%le; nr=%le; min=%le; max=%le;\n",
					t, nr,
					find_min(&u1[0], n),
					find_max(&u1[0], n));
//			char buf[1024];
//			sprintf(buf, "u_%05d.txt", i);
//			_fprintfwmatrix(buf, &u1[0], conf.n_phi, conf.n_la, conf.n_la, "%.16lf ");
			//exit(1);

			fprintfwmatrix(stdout, &u1[0], conf.n_phi, conf.n_la, conf.n_la, "%.16lf ");
			fprintf(stdout, "\n"); fflush(stdout);

			if (isnan(nr)) {
				return;
			}
//		}

		u1.swap(u);
		i ++;
	}
}

double cor3(double phi, double lambda, BarVortexConf * conf)
{
	double T0   = 1./conf->omg;
	double OMGE = 2.*M_PI / (24.*60.*60.);
	double H0   = OMGE * 0.5;

	double l = T0 * 2. * OMGE * sin (phi);
	double h = T0 * H0 * cos (2.0 * lambda) * sin (2.0 * phi) * sin (2.0 * phi);
	return (l + h);
}

double one_rp(double phi, double lambda, BarVortexConf * conf)
{
	return 1.0;
}

void output_psi(const char * prefix, const char * suffix,
                                const double * psi, long nlat, long nlon,
                                double U0, double PSI0
                                /*SphereGrad & grad*/)
{
//        vector < double > u (nlat * nlon);
//        vector < double > v (nlat * nlon);
        vector < double > Psi (nlat * nlon);

//        grad.calc(&u[0], &v[0], &psi[0]);

//        vec_mult_scalar(&u[0], &u[0], -1.0, nlat * nlon);

        char ubuf[1024]; char vbuf[1024]; char psibuf[1024];
        char Ubuf[1024]; char Vbuf[1024]; char Psibuf[1024];

        snprintf(ubuf, 1024,   "out/%snorm_u%s.txt", prefix, suffix);
        snprintf(vbuf, 1024,   "out/%snorm_v%s.txt", prefix, suffix);
        snprintf(psibuf, 1024, "out/%snorm_psi%s.txt", prefix, suffix);

        snprintf(Ubuf, 1024,   "out/%sorig_u%s.txt", prefix, suffix);
        snprintf(Vbuf, 1024,   "out/%sorig_v%s.txt", prefix, suffix);
        snprintf(Psibuf, 1024, "out/%sorig_psi%s.txt", prefix, suffix);

//        fprintfwmatrix(ubuf,   &u[0], nlat, nlon, "%23.16lf ");
//        fprintfwmatrix(vbuf,   &v[0], nlat, nlon, "%23.16lf ");
        _fprintfwmatrix(psibuf, &psi[0], nlat, nlon, std::max(nlat, nlon), "%23.16lf ");

//        vec_mult_scalar(&u[0], &u[0], U0, nlon * nlat);
//        vec_mult_scalar(&v[0], &v[0], U0, nlon * nlat);
        vector_mult_scalar(&Psi[0],  &psi[0],  PSI0, nlon * nlat);

//        fprintfwmatrix(Ubuf,   &u[0], nlat, nlon, "%23.16le ");
//        fprintfwmatrix(Vbuf,   &v[0], nlat, nlon, "%23.16le ");
        _fprintfwmatrix(Psibuf, &Psi[0], nlat, nlon, std::max(nlat, nlon), "%23.16le ");
}

void test_barvortex_plan(const char * srtm)
{
	BarVortexConf conf;
	conf.steps = 1;
	double R   = 6.371e+6;
	double H   = 5000;
	conf.omg   = 2.*M_PI/24./60./60.; // ?
	double T0  = 1./conf.omg;
	conf.k1    = 1.0;
	conf.k2    = 1.0;
	int part_of_the_day = 192;
	conf.tau   = 2 * M_PI / (double) part_of_the_day;
	conf.sigma = 1.14e-2;
	conf.mu    = 1.77e-6;
	conf.n_phi = 24;
	conf.n_la  = 32;
	conf.full  = 0;
	conf.rho   = 1;
	conf.theta = 0.5;

	conf.cor    = 0;//cor3;//zero_cor;
	conf.rp     = 0;//one_rp;//zero_rp;
	conf.filter = 0;
	conf.cor_add= 0;

	int n = conf.n_phi * conf.n_la;

	double t = 0;
	double T = 2 * M_PI * 1000.0; //1445;//300 * 2.0 * M_PI;;
	double nr;
	int i, j;

	vector < double > cor(n);
	vector < double > rel(n);
	vector < double > u(n);
	vector < double > v(n);
	vector < double > f(n);
	vector < double > u1(n);
	int nlat     = conf.n_phi;
	int nlon     = conf.n_la;

	const char * fn = srtm ? srtm : "";
	ReliefLoader rel_loader(fn);
	rel_loader.get(&rel[0], 24, 32, conf.full, true);

	double rel_max = 0.0;
        for (i = 0; i < nlat * nlon; ++i) {
                if (rel_max < rel[i]) rel_max = rel[i];
                //if (rel_max < fabs(rel[i])) rel_max = fabs(rel[i]);
        }

	conf.cor_add = &cor[0];
	conf.rp_add  = &f[0];

	SLaplacian lapl(conf.n_phi, conf.n_la, false);
	SJacobian jac(conf.n_phi, conf.n_la, false);

	fprintf(stderr, "#domain:sphere half\n");
	fprintf(stderr, "#mesh_w:%d\n", conf.n_la);
	fprintf(stderr, "#mesh_h:%d\n", conf.n_phi);
	fprintf(stderr, "#filter:%d\n", conf.filter);
	fprintf(stderr, "#tau:%.16lf\n", conf.tau);
	fprintf(stderr, "#sigma:%.16lf\n", conf.sigma);
	fprintf(stderr, "#mu:%.16lf\n", conf.mu);
	fprintf(stderr, "#k1:%.16lf\n", conf.k1);
	fprintf(stderr, "#k2:%.16lf\n", conf.k2);
	fprintf(stderr, "#theta:%.16lf\n", conf.theta);
	fprintf(stderr, "#rp:kornev1\n");
	fprintf(stderr, "#coriolis:kornev1\n");
	fprintf(stderr, "#initial:kornev1\n");
	fprintf(stderr, "#build:$Id$\n");

	double U0max = 30.;
	double omg = 2.*M_PI/24./60./60.;
	double TE  = 1./omg;
	double RE  = 6.371e+6;
        double PSI0 = RE * RE / TE;
	double U0    = R / T0;
	double Ly    = M_PI / 2.0;
	double Lx    = 2.0 * M_PI;

        for (i = 0; i < nlat; ++i)
        {
                for (j = 0; j < nlon; ++j)
                {
                        double phi    = lapl.phi(i);
                        double lambda = lapl.lambda(j);

                        if (phi > 0) {
                                u[i * nlon + j] = (phi * (M_PI / 2. - phi) * 16 / M_PI / M_PI * 100.0 / U0);
                        } else {
                                u[i * nlon + j] = (phi * (M_PI / 2. + phi) * 16 / M_PI / M_PI * 100.0 / U0);
                        }
                        v[i * nlon + j] = 0;

                        if (rel[i * nlon + j] > 0) {
                                rel[i * nlon + j] = 1.0 * rel[i * nlon + j] / rel_max;
                        } else {
                                rel[i * nlon + j] = 0.0;
                        }
                        cor[i * nlon + j] = rel[i * nlon + j] + 2 * sin(phi);
                }
        }

	lapl.make_psi(&f[0], &u[0], &v[0]);
	lapl.lapl_1(&u[0], &f[0]);
	vector_mult_scalar(&f[0], &f[0], conf.sigma, n);

	BarVortex bv(conf);

	Variance < double > var(n);
	_fprintfwmatrix("out/cor.txt", &cor[0], nlat, nlon, std::max(nlat, nlon), "%23.16lf ");
        _fprintfwmatrix("out/rel.txt", &rel[0], nlat, nlon, std::max(nlat, nlon), "%23.16lf ");
        _fprintfwmatrix("out/rp.txt", &f[0], nlat, nlon, std::max(nlat, nlon), "%23.16lf ");
        _fprintfwmatrix("out/u0.txt", &u[0], nlat, nlon, std::max(nlat, nlon), "%23.16lf ");
        _fprintfwmatrix("out/v0.txt", &v[0], nlat, nlon, std::max(nlat, nlon), "%23.16lf ");

	i = 0;
	while (t < T) {
		if (i % part_of_the_day == 0) {
                        char buf[1024];
                        nr = bv.norm(&u[0], n);
                        fprintf(stderr, "nr=%.16lf, t=%.16lf of %.16lf\n", nr, t, T);
                        snprintf(buf, 1024, "%06d", i);
                        output_psi("", buf, &u[0], nlat, nlon, U0, PSI0/*, grad*/);

                        vector < double > m = var.m_current();
                        vector < double > d = var.current();

                        output_psi("m_", "", &m[0], nlat, nlon, U0, PSI0/*, grad*/);
                        output_psi("d_", "", &d[0], nlat, nlon, U0, PSI0/*, grad*/);
                }

		bv.S_step(&u1[0], &u[0]);
		t += conf.tau;

		u1.swap(u);
		i ++;
	}
}

void test_barvortex_real()
{
	BarVortexConf conf;
	conf.steps = 1;
	double R   = 6.371e+6;
	double H   = 5000;
	conf.omg   = 2.*M_PI/24./60./60.; // ?
	double T0  = 1./conf.omg;
	conf.k1    = 1.0;
	conf.k2    = 1.0;
	conf.tau   = 0.001;
	conf.sigma = 1.14e-2;
	conf.mu    = 6.77e-5;
	conf.n_phi = 24;
	conf.n_la  = 32;
	conf.full  = 0;
	conf.rho   = 1;
	conf.theta = 0.5;

	conf.cor    = cor2;
	conf.rp     = 0;
	conf.filter = 1;

	int n_phi, n_la;

	int n = conf.n_phi * conf.n_la;

	double t = 0;
	double T = 30 * 2.0 * M_PI;;
	double nr;
	double * rel, * forc, *uu0;
	int i = 0;

	asp::load_matrix_from_txtfile(&rel, &n_phi, &n_la, "rel.txt");
	asp::load_matrix_from_txtfile(&forc, &n_phi, &n_la, "forc.txt");

	conf.cor_add = rel;
	conf.rp_add  = forc;

	BarVortex bv(conf);
	vector < double > u(n);
	vector < double > u1(n);

	fprintf(stderr, "#domain:sphere half\n");
	fprintf(stderr, "#mesh_w:%d\n", conf.n_la);
	fprintf(stderr, "#mesh_h:%d\n", conf.n_phi);
	fprintf(stderr, "#filter:%d\n", conf.filter);
	fprintf(stderr, "#tau:%.16lf\n", conf.tau);
	fprintf(stderr, "#sigma:%.16lf\n", conf.sigma);
	fprintf(stderr, "#mu:%.16lf\n", conf.mu);
	fprintf(stderr, "#k1:%.16lf\n", conf.k1);
	fprintf(stderr, "#k2:%.16lf\n", conf.k2);
	fprintf(stderr, "#theta:%.16lf\n", conf.theta);
	fprintf(stderr, "#rp:real1\n");
	fprintf(stderr, "#coriolis:real1\n");
	fprintf(stderr, "#initial:real1\n");
	fprintf(stderr, "#build:$Id$\n");


	asp::load_matrix_from_txtfile(&uu0, &n_phi, &n_la, "u0.txt");
	memcpy(&u[0], uu0, n_phi * n_la * sizeof(double));

	while (t < T) {
		bv.S_step(&u1[0], &u[0]);
		t += conf.tau;

//		if (i % 10000 == 0) {
			nr = bv.norm(&u1[0], n);
			fprintf(stderr, "t=%le; nr=%le; min=%le; max=%le;\n",
					t, nr,
					find_min(&u1[0], n),
					find_max(&u1[0], n));
//			char buf[1024];
//			sprintf(buf, "u_%05d.txt", i);
//			_fprintfwmatrix(buf, &u1[0], conf.n_phi, conf.n_la, conf.n_la, "%.16lf ");
			//exit(1);

			fprintfwmatrix(stdout, &u1[0], conf.n_phi, conf.n_la, conf.n_la, "%.16lf ");
			fprintf(stdout, "\n"); fflush(stdout);

			if (isnan(nr)) {
				return;
			}
//		}

		u1.swap(u);
		i ++;
	}
}

int main(int argc, char ** argv)
{
	fprintf(stderr, "#cmd:");
	for (int i = 0; i < argc; ++i) {
		fprintf(stderr, "%s ", argv[i]);
	}
	fprintf(stderr, "\n");
	//set_fpe_except();
//	if (argc < 3) {
//		fprintf(stderr, "calc\n");
		//test_barvortex();
		//test_barvortex_real();
		//test_barvortex_linear();
		test_barvortex_plan(argv[1]);
//	} else {
//		calc_barvortex_right_part();
//	}
}

