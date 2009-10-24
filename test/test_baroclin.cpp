/**/

#define _USE_MATH_DEFINES
#include <string>
#include <vector>
#include <math.h>

#include "sds_bar2.h"
#include "asp_misc.h"

using namespace std;
using namespace asp;
using namespace SDS;

#ifdef WIN32
#include <float.h>
#define isnan _isnan
inline bool isinf(double x)
{
	int c = _fpclass(x);
	return (c == _FPCLASS_NINF || c == _FPCLASS_PINF);
}
#endif

double rp1(double phi, double lambda, BaroclinConf * conf)
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

double rp2(double phi, double lambda, BaroclinConf * conf)
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

double cor(double phi, double lambda, BaroclinConf * conf)
{
	return 2.*sin(phi) +  // l
		0.5 * cos(2*lambda)*sin(2*phi)*sin(2*phi); //h
}

double u0(double phi, double lambda)
{
	double omg = 2.*M_PI/24./60./60.; // ?
	double T0  = 1./omg;
	double R   = 6.371e+6;

	return -T0/R * 16.0 / M_PI / M_PI * 30.0 * 
		(M_PI/4 * phi * phi - phi * phi * phi / 3);
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

void test_barvortex()
{
	BaroclinConf conf;
	conf.steps = 1;
	double R   = 6.371e+6;
	double H   = 5000;
	conf.omg   = 2.*M_PI/24./60./60.; // ?
	double T0  = 1./conf.omg;
	conf.k1    = 1.0;
	conf.k2    = 1.0;
	conf.tau   = 0.0001;
	conf.sigma = 1.14e-2;
	conf.mu    = 6.77e-5;

	conf.sigma1= 1.14e-2;
	conf.mu1   = 6.77e-5;

	conf.n_phi = 24;
	conf.n_la  = 32;
	conf.full  = 0;
	conf.rho   = 1;
	conf.theta = 0.5;
	conf.alpha = 1.0;//1.0

	conf.cor   = cor;
	conf.rp1   = 0;
	conf.rp2   = 0;
	conf.filter = 0;

	int n = conf.n_phi * conf.n_la;

	double t = 0;
	double T = 30 * 2.0 * M_PI;;
	double nr1;
	double nr2;
	int i = 0;

	Baroclin bv(conf);
	vector < double > u1(n);
	vector < double > u2(n);
	vector < double > u11(n);
	vector < double > u21(n);

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
			u1[pOff(i, j)] = u0(bv.phi(i), bv.lambda(j));
			u2[pOff(i, j)] = u0(bv.phi(i), bv.lambda(j));
		}
	}

//	{
//		_fprintfwmatrix("kornev1_u0_1.txt", &u[0], conf.n_phi, conf.n_la, conf.n_la, "%.16lf ");
//	}

	while (t < T) {
		bv.S_step(&u11[0], &u21[0], &u1[0], &u2[0]);
		t += conf.tau;

//		if (i % 10000 == 0) {
			nr1 = bv.norm(&u11[0], n);
			nr2 = bv.norm(&u21[0], n);
			fprintf(stderr, "t=%le; nr=%le; nr=%le; min=%le; max=%le;\n", 
					t, nr1, nr2, 
					find_min(&u11[0], n),
					find_max(&u11[0], n));
//			char buf[1024];
//			sprintf(buf, "u_%05d.txt", i);
//			_fprintfwmatrix(buf, &u1[0], conf.n_phi, conf.n_la, conf.n_la, "%.16lf ");
			//exit(1);

//			fprintfwmatrix(stdout, &u1[0], conf.n_phi, conf.n_la, conf.n_la, "%.16lf ");
//			fprintf(stdout, "\n"); fflush(stdout);

			if (isnan(nr1) || isnan(nr2)) {
				return;
			}
//		}

		u11.swap(u1);
		u21.swap(u2);
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
	test_barvortex();
}

