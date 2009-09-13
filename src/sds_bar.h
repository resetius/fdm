#ifndef _SDS_BAR_H
#define _SDS_BAR_H
/*$Id$*/

/* Copyright (c) 2003, 2004, 2005, 2006 Alexey Ozeritsky
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

//______________ ключи Баротропного вихря на сфере ______________________

#define _BARVORTEX_PURE_IM    //чисто неявная схема (u с крышкой под f - итерации)
//#define _BARVORTEX_LINEAR_PURE_IM
#define _BARVORTEX_IM_MAX_IT 3000
//#define _BARVORTEX_IM_MAX_IT 1
#define _BARVORTEX_IM_EPS 1e-10
//#define _BARVORTEX_IM_EPS 1e-9
#define _BARVORTEX_ARAKAWA

#define _BARVORTEX_FILTER //?

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
struct BarVortexConf 
{
	int steps;     //!<шагов
	double tau;    //!<шаг счёта.
	double theta;  //!<параметр в полунеявной схеме.
	double sigma;  //!<параметры задачи.
	double mu;     //!<параметры задачи.
    int n_phi;     //!<разбиение (широта)
    int n_la;      //!<разбиение (долгота).
	int full;      //!<использовать полную сферу?
	double k1;
	double k2;

	double rho;
	double omg;

	typedef double (*rp_t ) (double phi, double lambda, BarVortexConf * conf);
	rp_t rp;
	rp_t cor;
};

class BarVortex
{
private:
	class Private;
	Private * d;

public:
	typedef BarVortexConf conf_t;

	virtual ~BarVortex();
	void S(double *h1, const double *h);
	void S_step(double *h1, const double *h);
	void L_step(double *h1, const double *h, const double *z);
	void LT_step(double *h1, const double *h, const double *z);
	void L_1_step(double *h1, const double *h, const double *z);

	std::string name() { return "BarVortex"; }

	BarVortex(BarVortexConf&);
	
	double scalar(const double *x, const double *y, int n);
	double norm(const double *x, int n);
	double dist(const double * x, const double * y, int n);

	double phi(int i);
	double lambda(int i);
};

}
#endif //_SDS_BAR_H

