/**/

#define _USE_MATH_DEFINES
#include <string>
#include <vector>
#include <math.h>

#include "sds_bar.h"
#include "asp_misc.h"

using namespace std;
using namespace asp;

double rp(double phi, double lambda, BarVortexConf * conf)
{
	return -conf->sigma * 180/1.15 * (6*(2*cos(phi)*cos(phi)-1)*sin(phi))/5000;
}

double cor(double phi, double lambda, BarVortexConf * conf)
{
	return 2.*conf->omg * sin(phi) +  // l
		0.1 * cos(2*lambda)*sin(2*phi)*sin(2*phi); //h
}

double u0(double phi, double lambda)
{
	double s = sin(phi);
	return - 180/1.15 * s * s * s;
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
	BarVortexConf conf;
	conf.steps = 1;
	double H   = 5000;
	conf.omg   = 2.*M_PI/24./60./60.; // ?
	double T0  = 1./conf.omg;
	conf.k1    = 1.0/H;//1.;
	conf.k2    = T0; //T0
	conf.tau   = 0.001;
	conf.sigma = 1./20./2./M_PI/H;
	conf.mu    = conf.sigma/100.;
	conf.n_phi = 96;
	conf.n_la  = 128;
	conf.full  = 0;
	conf.rho   = 1;
	conf.theta = 0.5;

	conf.cor   = cor;
	conf.rp    = rp;

	int n = conf.n_phi * conf.n_la;

	double t = 0;
	double T = 10;
	int i = 0;

	BarVortex bv(conf);
	vector < double > u(n);
	vector < double > u1(n);

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
			fprintf(stderr, "t=%le/nr=%le\n", t, bv.norm(&u1[0], n));
//			char buf[1024];
//			sprintf(buf, "u_%05d.txt", i);
//			_fprintfwmatrix(buf, &u1[0], conf.n_phi, conf.n_la, conf.n_la, "%.16lf ");
			//exit(1);

			fprintfwmatrix(stdout, &u1[0], conf.n_phi, conf.n_la, conf.n_la, "%.16lf ");
			fprintf(stdout, "\n");
//		}

		u1.swap(u);
		i ++;
	}
}

int main()
{
	//set_fpe_except();
	test_barvortex();
}
