/*$Id$*/

/* Copyright (c) 2011 Alexey Ozeritsky
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define _USE_MATH_DEFINES
#include <string>
#include <vector>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "sds_bar.h"
#include "asp_misc.h"
#include "asp_sphere_lapl.h"
#include "asp_sphere_jac.h"
#include "statistics.h"
#include "config.h"

using namespace std;
using namespace asp;
using namespace SDS;

#undef max

void output_psi(const char * prefix, const char * suffix,
                const double * psi, long nlat, long nlon,
                double U0, double PSI0)
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
    _fprintfwmatrix(psibuf, &psi[0], nlat, nlon, max(nlat, nlon), "%23.16lf ");

    //        vec_mult_scalar(&u[0], &u[0], U0, nlon * nlat);
    //        vec_mult_scalar(&v[0], &v[0], U0, nlon * nlat);
    vector_mult_scalar(&Psi[0],  &psi[0],  PSI0, nlon * nlat);

    //        fprintfwmatrix(Ubuf,   &u[0], nlat, nlon, "%23.16le ");
    //        fprintfwmatrix(Vbuf,   &v[0], nlat, nlon, "%23.16le ");
    _fprintfwmatrix(Psibuf, &Psi[0], nlat, nlon, max(nlat, nlon), "%23.16le ");
}

void load_relief(double * cor, double *rel, long nlat, long nlon,
                 int full, int offset,
				 const string & fn)
{
	fprintf(stderr, "loading relief\n");

	FILE * f = fopen(fn.c_str(), "rb");
	if (f) {
		int n1, n2;
        double * tmp;
        asp::load_matrix_from_txtfile(&tmp, &n1, &n2, fn.c_str());
		if (n1 != nlat || n2 != nlon)
		{
			fprintf(stderr, "bad relief file format\n");
			exit(1);
		}
		memcpy(rel, tmp, nlat * nlon * sizeof(double));
		fclose(f);
        free(tmp);
	} else {
		fprintf(stderr, "file not found !\n");
		exit(1);
	}

	double rel_max = 0.0;
	for (int i = 0; i < nlat * nlon; ++i)
	{
		if (rel_max < rel[i]) rel_max = rel[i];
		//if (rel_max < fabs(rel[i])) rel_max = fabs(rel[i]);
	}

	double dlat = -1, dlon = -1, phi, lambda[[maybe_unused]];

	if (full && !offset) {
		dlat = M_PI / (nlat - 1);
		dlon = 2. * M_PI / nlon;
	} else if (!full && !offset) {
		dlat = M_PI / (nlat - 1) / 2;
		dlon = 2. * M_PI / nlon;
	} else if (full && offset) {
		dlat = M_PI / (double) nlat;
		dlon = 2. * M_PI / nlon;
	} else if (!full && offset) {
		dlat = M_PI / (double)(2 * (nlat - 1) + 1);
		dlon = 2. * M_PI / nlon;
	}

    assert(dlat > 0); assert(dlon > 0);

	for (int i = 0; i < nlat; ++i)
	{
		for (int j = 0; j < nlon; ++j)
		{
			if (!offset) {
				phi    = -0.5 * M_PI + i * dlat;
				lambda = j * dlon;
			} else if (offset && !full) {
				phi    = i * dlat;
				lambda = j * dlon;
			} else if (offset && full) {
				phi    = (dlat - M_PI) * 0.5 + (double)i * dlat;
				lambda = j * dlon;
			}

			if (rel[i * nlon + j] > 0)
			{
				rel[i * nlon + j] = 1.0 * rel[i * nlon + j] / rel_max;
			}
			else
			{
				rel[i * nlon + j] = 0.0;
			}
			cor[i * nlon + j] = rel[i * nlon + j] + 2 * sin (phi);
		}
	}

	fprintf(stderr, "done\n");
}

#undef OPTIONAL

void usage(const Config & config, const char * name)
{
	fprintf(stderr, "%s ...\n"
		"-c config_file\n", name);
	config.help();
	exit(1);
}

void calc_barvortex_forcing(Config & config, int argc, char *argv[])
{
	string config_name = "barvortex.ini";

	ConfigSkeleton s;
	s.data["general"]["T"]    = make_pair(ConfigSkeleton::OPTIONAL, "total time");

	s.data["sp"]["nlat"]      = make_pair(ConfigSkeleton::OPTIONAL, "latitude");
	s.data["sp"]["nlon"]      = make_pair(ConfigSkeleton::OPTIONAL, "longitude");
	s.data["sp"]["relief"]    = make_pair(ConfigSkeleton::REQUIRED, "relief (text format)");

	config.set_skeleton(s);

	for (int i = 0; i < argc; ++i) {
		if (!strcmp(argv[i], "-c")) {
			if (i == argc - 1) {
				usage(config, argv[0]);
			}
			config_name = argv[i + 1];
		}

		if (!strcmp(argv[i], "-h")) {
			usage(config, argv[0]);
		}
	}

	config.open(config_name);
	config.rewrite(argc, argv);
	if (!config.validate()) {
		usage(config, argv[0]);
	}

	BarVortex::Conf conf;
	double R   [[maybe_unused]]= 6.371e+6;
	double H   [[maybe_unused]]= 5000;
	double omg = 2.*M_PI/24./60./60.; // ?
	double T0  [[maybe_unused]]= 1./omg;

	int part_of_the_day = config.get("fdm", "part_of_the_day", 192);
	conf.k1    = config.get("fdm", "k1", 1.0);
	conf.k2    = config.get("fdm", "k2", 1.0);
	conf.tau   = 2 * M_PI / (double) part_of_the_day;
	conf.sigma = config.get("fdm", "sigma", 1.14e-2);
	conf.mu    = config.get("fdm", "mu", 1.77e-6);
	conf.nlat  = config.get("fdm", "nlat", 24);
	conf.nlon  = config.get("fdm", "nlon", 32);
	conf.theta = config.get("fdm", "theta", 0.5);

    string relief_fn = config.gets("fdm", "relief");
	string u_m_fn    = config.gets("fdm", "u_m");
	string v_m_fn    = config.gets("fdm", "v_m");

	conf.cor    = 0;
	conf.rp     = 0;
	conf.cor2   = 0;

	conf.filter = config.get("fdm", "filter", 1);
    conf.full   = config.get("fdm", "full", 0);
    int offset  = config.geti("fdm", "offset");

	int n = conf.n_phi * conf.n_la;

	double t [[maybe_unused]]= 0;
	double T [[maybe_unused]]= 2 * M_PI * 1000.0; //1445;//300 * 2.0 * M_PI;;

	vector < double > cor(n);
	vector < double > rel(n);

	vector < double > u_m(n); // TODO: load this
	vector < double > v_m(n); // TODO: load this
	vector < double > psi_m(n);

	int nlat     = conf.n_phi;
	int nlon     = conf.n_la;

	SLaplacian lapl(conf.n_phi, conf.n_la, !!conf.full);
	SJacobian jac(conf.n_phi, conf.n_la, !!conf.full);
	SVorticity vor(conf.n_phi, conf.n_la, !!conf.full);

	vor.calc(&psi_m[0], &u_m[0], &v_m[0]);

	load_relief(&cor[0], &rel[0], nlat, nlon, conf.full, offset, relief_fn);

	conf.cor2 = &cor[0];
	conf.rp2  = 0; //&f[0];

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

	BarVortex bv(conf);

	Variance < double > var(n);
}

int main(int argc, char ** argv)
{
	Config config;

	fprintf(stderr, "#cmd:");
	for (int i = 0; i < argc; ++i) {
		fprintf(stderr, "%s ", argv[i]);
	}
	fprintf(stderr, "\n");

	try {
    	calc_barvortex_forcing(config, argc, argv);
	} catch (std::exception & e) {
		fprintf(stderr, "exception: %s\n", e.what());
		usage(config, argv[0]);
	}
}
