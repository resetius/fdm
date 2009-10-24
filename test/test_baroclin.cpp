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

double u1_t (double x, double y, double t)
{
	return x*sin(y+t)*ipow(cos(x),4);
}

double u2_t (double x, double y, double t)
{
	return x*cos(y+t)*ipow(cos(x),4);
}

double rp_f1(double x, double y, double t, BaroclinConf * conf)
{
	double sigma = conf->sigma;
	double mu    = conf->mu;
	double sigma1= conf->sigma;
	double mu1   = conf->mu;
	double alpha = conf->alpha;
	
	return -45*mu*sin(y+t)*x-(9./2.)*sigma*
		ipow(cos(x),3)*sin(y+t)*sin(x)+(9./2.)*sigma*
		ipow(cos(x),3)*sin(x)*cos(y+t)-10*sigma*
		ipow(cos(x),4)*x*sin(y+t)+10*sigma*
		ipow(cos(x),4)*x*cos(y+t)+(15./2.)*sigma*
		ipow(cos(x),2)*x*sin(y+t)-(15./2.)*sigma*
		ipow(cos(x),2)*x*cos(y+t)-360*mu*sin(y+t)*sin(x)*
		ipow(cos(x),3)+147*mu*sin(y+t)*sin(x)*cos(x)-400*mu*sin(y+t)*x*
		ipow(cos(x),4)+390*mu*sin(y+t)*x*
		ipow(cos(x),2)-20*x*cos(y+t)*
		ipow(cos(x),4)-9*cos(y+t)*
		ipow(cos(x),3)*sin(x)+15*x*cos(y+t)*
		ipow(cos(x),2);
}

double rp_g1(double x, double y, double t, BaroclinConf * conf)
{
	double sigma = conf->sigma;
	double mu    = conf->mu;
	double sigma1= conf->sigma;
	double mu1   = conf->mu;
	double alpha = conf->alpha;

	double alpha2 = alpha * alpha;
	double r = 20*x*sin(y+t)*
		ipow(cos(x),4)+9*sin(y+t)*
		ipow(cos(x),3)*sin(x)-15*x*sin(y+t)*
		ipow(cos(x),2)+390*mu*cos(y+t)*x*
		ipow(cos(x),2)-10*sigma*
		ipow(cos(x),4)*x*cos(y+t)-(9./2.)*sigma*
		ipow(cos(x),3)*sin(y+t)*sin(x)-alpha2*
		ipow(cos(x),7)*x-45*mu*cos(y+t)*x+18*
		ipow(cos(x),6)*
		ipow(cos(y+t),2)*sin(x)-9*
		ipow(cos(x),6)*sin(x)+9*x*
		ipow(cos(x),5)-(9./2.)*sigma*
		ipow(cos(x),3)*sin(x)*cos(y+t)-30*x*x*
		ipow(cos(x),4)*sin(x)-18*x*
		ipow(cos(x),5)*
		ipow(cos(y+t),2)+alpha2*
		ipow(cos(x),4)*x*sin(y+t)+(15./2.)*sigma*
		ipow(cos(x),2)*x*sin(y+t)+(15./2.)*sigma*
		ipow(cos(x),2)*x*cos(y+t)-400*mu*cos(y+t)*x*
		ipow(cos(x),4)+147*mu*cos(y+t)*sin(x)*cos(x)+60*x*x*
		ipow(cos(x),4)*
		ipow(cos(y+t),2)*sin(x)-360*mu*cos(y+t)*sin(x)*
		ipow(cos(x),3)-10*sigma*
		ipow(cos(x),4)*x*sin(y+t)+4*alpha2*
		ipow(cos(x),6)*x*x*sin(x)-9*alpha2*
		ipow(cos(x),3)*mu1*cos(y+t)*sin(x)-20*alpha2*
		ipow(cos(x),4)*mu1*cos(y+t)*x+15*alpha2*
		ipow(cos(x),2)*mu1*cos(y+t)*x-alpha2*
		ipow(cos(x),4)*sigma1*x*cos(y+t);
	r /= alpha2;
	return r;
}

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

double zero_cor(double phi, double lambda, BaroclinConf * conf)
{
	return 0;
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

	conf.sigma1= conf.sigma;
	conf.mu1   = conf.mu;

	conf.n_phi = 24;
	conf.n_la  = 32;
	conf.full  = 0;
	conf.rho   = 1;
	conf.theta = 0.5;
	conf.alpha = 1.0;//1.0

	conf.cor    = cor;
	conf.rp1    = rp_f1;
	conf.rp2    = rp_g1;
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
	fprintf(stderr, "#sigma1:%.16lf\n", conf.sigma1);
	fprintf(stderr, "#mu1:%.16lf\n", conf.mu1);
	fprintf(stderr, "#alpha:%.16lf\n", conf.alpha);
	fprintf(stderr, "#k1:%.16lf\n", conf.k1);
	fprintf(stderr, "#k2:%.16lf\n", conf.k2);
	fprintf(stderr, "#theta:%.16lf\n", conf.theta);
	fprintf(stderr, "#rp:kornev1\n");
	fprintf(stderr, "#coriolis:kornev1\n");
	fprintf(stderr, "#initial:kornev1\n");
	fprintf(stderr, "#build:$Id$\n");

	for (int i = 0; i < conf.n_phi; ++i) {
		for (int j = 0; j < conf.n_la; ++j) {
			u1[pOff(i, j)] = u1_t(bv.phi(i), bv.lambda(j), 0);
			u2[pOff(i, j)] = u2_t(bv.phi(i), bv.lambda(j), 0);
		}
	}

	while (t < T) {
		bv.S_step(&u11[0], &u21[0], &u1[0], &u2[0], t);
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

