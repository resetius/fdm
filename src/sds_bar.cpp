/*$Id$*/

/* Copyright (c) 2005, 2006 Alexey Ozeritsky
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by Alexey Ozeritsky.
 * 4. The name of the author may not be used to endorse or promote products
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
/**
 * уравнение баротропного вихря на сфере
 * \f[\frac{\partial \delta du}{\partial t} 
      + \rho J(u, \delta u + l + h) = 
	  -\sigma \delta u + \mu \delta^2 u + f (u)
   \f]
 * \f[
    \Delta u(\phi, \lambda, t) := \frac{1}{cos(\phi)} 
     \frac{\partial}{\partial \phi} cos(\phi) \frac{\partial u}{\partial \phi} +
    + \frac{1}{cos^2 \phi} \frac{\partial ^2 u}{\partial ^2 \lambda}
	\f]
 * \f$\phi \in [0,\pi/2]\f$ - широта
 * \f$\lambda \in [0,2\pi]\f$ - долгота
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <time.h>
#include <string.h>
#include <vector>

#include "asp_lib.h"
#include "sds_bar.h"

using namespace asp;

using namespace std;
using namespace SDS;

#include "sds_sphere_data.h"

//макросы для доступа к значениям собственных функций

#define _VM(i, j) VM[ ( i ) * n_la + ( j ) ]

//макросы для нахождения отступа в массиве для точки пространства
#define  pOff(i, j) ( i ) * n_la + ( j )
#define _pOff(i, j) ( i ) * d->n_la + ( j )
#define SQRT_M_1_PI 0.56418958354775629

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*________ задача Баротропного вихря на сфере - реализация ________________*/

#include "sds_bar_impl.h"

BarVortex::BarVortex(BarVortexConf&conf_) :
	d(new Private(conf_))
{
	d->p = this;
}

BarVortex::~BarVortex()
{
	delete d;
}

static void stat(int nn, double * d, char * mask)
{
	int i;

	//максимум, минимум, матожидание
	double max = 0, min = 0, med, c;
	int count = 0;

	for (i = 0; i < nn; ++i) {
		if (mask[i] == 1) {
			max = min = d[i];
			break;
		}
	}

	med = 0;
	for (i = 0; i < nn; ++i) {
		if (mask[i] == 1) {
			c = d[i];
			if (max < c) max = c;
			if (min > c) min = c;
			med += c;
			++ count;
		}
	}

	med /= (double)count;
	printf("st:   max=%lf\n", max);
	printf("st:   min=%lf\n", min);
	printf("st:   med=%lf\n", med);
}

static void printfmasked(const double *A, const char * mask, 
						 const char * fname,
						 int n_phi, int n_la)
{
	FILE * f = fopen(fname, "w");
	if (!f) return;

	for (int i = 0; i < n_phi; ++i) {
		for (int j = 0; j < n_la; ++j) {
			if (mask[pOff(i, j)]) {
				fprintf(f, "%.16le ", A[pOff(i, j)]);
			} else {
				fprintf(f, "%.16le ", 0.0);
			}
		}
		fprintf(f, "\n");
	}
	fclose(f);
}

void BarVortex::S_step(double *h1, const double *h)
{
#ifdef _BARVORTEX_PURE_IM
	d->S_step_im(h1, h);
#else
	d->S_step(h1, h);
#endif

	if (d->conf->filter) {
		d->lapl->filter(h1, h1);
	}
}

void BarVortex::S(double *h1, const double *h) {
	memcpy(h1, h, d->n_la * d->n_phi * sizeof(double));

	for (int i = 0; i < d->conf->steps; i++) {
#ifdef _BARVORTEX_PURE_IM
		d->S_step_im(h1, h1);
#else
		d->S_step(h1, h1);
#endif
	}
}

void BarVortex::LT_step(double *h1, const double *h, const double *z)
{
	if (d->conf->filter) {
		d->lapl->filter(h1, h);
		d->L_step_t(h1, h1, z);
	} else {
		d->L_step_t(h1, h, z);
	}
}

void BarVortex::L_step(double *h1, const double *h, const double * z)
{
#ifdef _BARVORTEX_LINEAR_PURE_IM
	d->L_step_im(h1, h, z);
#else
	d->L_step(h1, h, z);
#endif

	if (d->conf->filter) {
		d->lapl->filter(h1, h1);
	}
}

void BarVortex::L_1_step(double *h1, const double *h, const double * z)
{
	if (d->conf->filter) {
		d->lapl->filter_1(h1, h);
	}

	//SDS::L_1_step(h1, h1, z); return;
#ifdef _BARVORTEX_LINEAR_PURE_IM
	if (d->conf->filter) {
		d->L_1_step_im(h1, h1, z);
	} else {
		d->L_1_step_im(h1, h, z);
	}
#else
	if (d->conf->filter) {
		d->L_1_step_im(h1, h1, z);
	} else {
		d->L_1_step_im(h1, h, z);
	}
#endif
}

double BarVortex::scalar(const double *x, const double *y, int n)
{
	return d->scalar(x, y, n);
}

double BarVortex::norm(const double *x, int n)
{
	return d->norm(x, n);
}

double BarVortex::dist(const double * x, const double * y, int n)
{
	vector < double > d(n);
	vector_diff(&d[0], x, y, n);
	return norm(&d[0], n);
}

double BarVortex::phi(int i)
{
	return d->PHI[i];
}

double BarVortex::lambda(int j)
{
	return d->LA[j];
}

void BarVortex::calc_rp(double * rp, const double * u)
{
	d->calc_rp(rp, u);
}

void BarVortex::reset()
{
	d->reset();
}

