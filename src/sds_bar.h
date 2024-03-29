#ifndef _SDS_BAR_H
#define _SDS_BAR_H
/* Copyright (c) 2003, 2004, 2005, 2006, 2022 Alexey Ozeritsky
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
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
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

#include "asp_macros.h"

//______________ ключи Баротропного вихря на сфере ______________________

#define _BARVORTEX_PURE_IM    //чисто неявная схема (u с крышкой под f - итерации)
#define _BARVORTEX_LINEAR_PURE_IM
#define _BARVORTEX_IM_MAX_IT 3000
//#define _BARVORTEX_IM_MAX_IT 1
#define _BARVORTEX_IM_EPS 1e-10
//#define _BARVORTEX_IM_EPS 1e-9
#define _BARVORTEX_ARAKAWA

namespace SDS {

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/**
 * уравнение баротропного вихря на сфере
 * \f[\frac{\partial \delta du}{\partial t} 
      + \rho J(u, \delta u + l + h) = 
	  -\sigma \delta u + \mu \delta^2 u + f (u)
   \f]
 * \f[
    \delta u(\phi, \lambda, t) := \frac{1}/{cos(\phi)} 
     \frac{\partial}{\partial \phi} cos(phi) \frac{\partial u}{\partial \phi} +
    + \frac{1}{cos^2 \phi} \frac{\partial ^2 u}{\partial ^2 \lambda}
	\f]
 * \f$\phi \in [0,\pi/2]\f$ - широта
 * \f$\lambda \in [0,2\pi]\f$ - долгота
 */

class FDM_API BarVortex
{
private:
	class Private;
	Private * d;

public:
	struct FDM_API Conf
	{
		typedef double (*rp_t ) (double phi, double lambda, 
			double t, const Conf * conf);

		int nlat;      //!<разбиение (широта)
		int nlon;      //!<разбиение (долгота).
		int full;      //!<использовать полную сферу?
		int filter;

		double tau;    //!<шаг счёта.
		double sigma;  //!<параметры задачи.
		double mu;     //!<параметры задачи.
		double k1;
		double k2;
		double theta;  //!<параметр в полунеявной схеме.

		rp_t rp;
		rp_t cor;

		double * rp2;
		double * cor2;

		double * rp3; // rp = rp(2) + rp3

		int & n_phi;
		int & n_la;

		Conf()
			: rp(0), cor(0), rp2(0), cor2(0), rp3(0), n_phi(nlat), n_la(nlon) {}
	};

	virtual ~BarVortex();
	void S_step(double *h1, const double *h);
	void S_step(double *h1, const double *h, double t)
	{
		S_step(h1, h);
	}

	void L_step(double *h1, const double *h, const double *z);
	void LT_step(double *h1, const double *h, const double *z);
	void L_1_step(double *h1, const double *h, const double *z);

	BarVortex(const Conf&);
	
	double scalar(const double *x, const double *y, int n = 0 /*unused*/);
	double norm(const double *x, int n = 0 /*unused*/);
	double dist(const double * x, const double * y, int n = 0 /*unused*/);
	void calc_rp(double * rp, const double * u);

	double phi(int i);
	double lambda(int i);

	void reset();

	const Conf & config() const;
};

}
#endif //_SDS_BAR_H

