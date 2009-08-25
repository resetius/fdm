/**/

#define _USE_MATH_DEFINES
#include <string>
#include <vector>
#include <math.h>

#include "sds_bar.h"

using namespace std;

double rp(double phi, double lambda, BarVortexConf * conf)
{
	return -conf->sigma * 180/1.15 * (6*(2*cos(phi)*cos(phi)-1)*sin(phi));
}

double cor(double phi, double lambda, BarVortexConf * conf)
{
	return 2.*conf->omg * sin(phi) +  // l
		0.1 /** 5000*/ * cos(2*lambda)*sin(2*phi)*sin(2*phi); //h
}

double u0(double phi, double lambda)
{
	double s = sin(phi);
	return - /*180/1.15 * */ s * s * s;
}

#define  pOff(i, j) ( i ) * conf.n_la + ( j )

void test_barvortex()
{
	BarVortexConf conf;
	conf.steps = 1;
	conf.omg   = 2.*M_PI/24./60./60.; // ?
	conf.k1    = 1.;
	conf.k2    = 1./conf.omg; //T0
	conf.tau   = 0.001;
	conf.sigma = 1./20./24./60./60.;
	conf.sigma = conf.sigma * conf.k2;
	conf.mu    = conf.sigma/100.;
	conf.n_phi = 24;
	conf.n_la  = 32;
	conf.full  = 0;
	conf.rho   = 1;
	conf.theta = 0.5;

	conf.cor   = cor;
	conf.rp    = rp;

	int n = conf.n_phi * conf.n_la;

	double t = 0;
	double T = 10;

	BarVortex bv(conf);
	vector < double > u(n);
	vector < double > u1(n);

	for (int i = 0; i < conf.n_phi; ++i) {
		for (int j = 0; j < conf.n_la; ++j) {
			u[pOff(i, j)] = u0(bv.phi(i), bv.lambda(j));
		}
	}

	while (t < T) {
		bv.S_step(&u1[0], &u[0]);
		t += conf.tau;

		fprintf(stderr, "t=%le/nr=%le\n", t, bv.norm(&u1[0], n));
		u1.swap(u);
	}
}

int main()
{
	test_barvortex();
}
