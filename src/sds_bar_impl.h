#ifndef _SDS_BAR_IMPL_H
#define _SDS_BAR_IMPL_H
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

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*________ задача Баротропного вихря на сфере - реализация ________________*/

#include <assert.h>

/*!
   Функторы линеаризировнного Якобиана
 */
template < typename Jac >
class J_functor {
	Jac * impl_;
	int n1_;
	int n2_;

public:
	J_functor(Jac * impl, int n1, int n2): 
	  impl_(impl), n1_(n1), n2_(n2) {}

	/*! берется сумма трех якобианов,
	    чтобы разделить по J(u, b + cor) на сумму

		J(p1, p2, i, j) + 
		J(p3, p4, i, j) + 
		J(p5, p6, i, j)
	*/
	double operator () (
		const double *h, const double  *cor, 
		const double *z, const double *omg /*h_lapl*/, 
		const double *z_lapl,
		int i, int j)
	{
		double jac = 0;
		//return 0;
#ifndef _BARVORTEX_ARAKAWA
		jac += impl_->J1(h, cor, i, j);    //psi, cor
		jac += impl_->J1(z, omg, i, j);    //z, omg (psi_lapl)
		jac += impl_->J1(h, z_lapl, i, j); //psi, z_lapl
#else
		jac += impl_->J(h, cor, i, j);    //psi, cor
		jac += impl_->J(z, omg, i, j);    //z, omg (psi_lapl)
		jac += impl_->J(h, z_lapl, i, j); //psi, z_lapl
#endif
		return jac;
	}

	void calc(double *h1,
		const double *h, const double  *cor, 
		const double *z, const double *omg /*h_lapl*/, 
		const double *z_lapl)
	{
		int off;
		for (int i = 0; i < n1_; ++i) {
			for (int j = 0; j < n2_; ++j) {
				off = i * n2_ + j;
				h1[off] = (*this)(h, cor, z, omg, z_lapl, i, j);
			}
		}
	}
};

template < typename Jac, typename Lapl >
class JT_functor {
	Jac  * impl_;
	Lapl * lapl_;
	int n1_;
	int n2_;
	int full_;

public:
	JT_functor(Jac * impl, Lapl * lapl, int n1, int n2, int full): 
	  impl_(impl), lapl_(lapl), n1_(n1), n2_(n2), full_(full) {}

	void calc(double *h1,
		const double *h, const double  *cor, 
		const double *z, const double *omg /*h_lapl*/, 
		const double *z_lapl)
	{
		assert(h1 != z_lapl);
		assert(h1 != h);

		int nn = n1_ * n2_;
		vector < double > p_lapl(nn);
		vector < double > tmp(nn);
		memset(h1, 0, nn * sizeof(double));
#ifdef _BARVORTEX_ARAKAWA
		impl_->JT(&tmp[0], h, &cor[0]);
		impl_->JT(&p_lapl[0], z, h);
		if (!full_) { memset(&p_lapl[0], 0, n2_ * sizeof(double)); }
		lapl_->lapl(&p_lapl[0], &p_lapl[0]);
		impl_->JT(h1, h, &z_lapl[0]);
#else
		impl_->J1T(&tmp[0], h, &cor[0], 1);
		impl_->J1T(&p_lapl[0], z, h, 2);
		if (!full_) { memset(&p_lapl[0], 0, n2_ * sizeof(double)); }
		lapl_->lapl(&p_lapl[0], &p_lapl[0]);
		impl_->J1T(h1, h, z_lapl, 1);
#endif
		if (!full_) { memset(&p_lapl[0], 0, n2_ * sizeof(double)); }
		vector_sum(h1, h1, &p_lapl[0], nn);
		vector_sum(h1, h1, &tmp[0], nn);
	}
};

template <typename Jac>
J_functor < Jac > make_J(Jac * impl, int n1, int n2)
{
	return J_functor < Jac > (impl, n1, n2);
}

template <typename Jac, typename Lapl >
JT_functor < Jac, Lapl > make_JT(Jac * impl, Lapl * lapl, int n1, int n2)
{
	return JT_functor < Jac, Lapl > (impl, lapl, n1, n2);
}

class BarVortex::Private: public SData {
public:
	BarVortex * p;
	BarVortexConf *conf; //!<конфиг

	int nn;
	vector < double > cor; //!<кориолис

	void setTau(double _tau) {
		tau   = _tau;
		tau_1 = 1.0 / _tau;

		forward_mult  = - conf->theta * conf->mu;
		forward_diag  = tau_1 + conf->theta * conf->sigma;

		backward_mult = (1.0 - conf->theta) * conf->mu;
		backward_diag = tau_1 - (1.0 - conf->theta) * conf->sigma;
	}

	Private(BarVortexConf& _conf)
		: SData(_conf.n_phi, _conf.n_la, _conf.full),
		p(0), conf(&_conf)
	{
		init();
		reset();
		setTau(conf->tau);

		nn = n_phi * n_la;
	}

	void reset() {
		nn = n_phi * n_la;
		cor.resize(nn);

		for (int i = 0; i < n_phi; ++i) {
			for (int j = 0; j < n_la; ++j) {
				cor[pOff(i, j)] = conf->cor(PHI[i], LA[j], conf);
			}
		}
	}

	/*!
	 \f[ J(u, v) = \frac{1}{cos(\phi)} 
	 \bigl( 
	   \frac{\partial u}{\partial \lambda} \frac{\partial v}{\partial \phi}
	  -\frac{\partial u}{\partial \phi} \frac{\partial v}{\partial \lambda}
	 \bigr)
	 \f]
	*/
	double Jacobian(const double *u, const double *v, int i, int j)
	{
#ifdef _BARVORTEX_ARAKAWA
		return jac->J(u, v, i, j);
#else
		return jac->J1(u, v, i, j);
#endif
	}

	void Jacobian(double * dst, const double *u, const double *v)
	{
		for (int i = 0; i < n_phi; ++i) {
			for (int j = 0; j < n_la; ++j) {
				dst[pOff(i, j)] = Jacobian(u, v, i, j);
			}
		}
	}

	/*!
	   Находит константную правую часть:
	   \f[
	     \frac{\omega^n}{\tau} + (1-\theta)L\omega^n+f(x,y)
	   \f]
	 */
	void take_B(double * dest, const double * omg, int have_rp) const
	{
		int    off;
		double rp       = 0.0;	
		double lpl;

		for (int i = 0; i < n_phi; ++i) {
			for (int j = 0; j < n_la; ++j) {
				off  = pOff(i, j);
				lpl  = lapl->lapl(omg, i, j);
				if (have_rp) rp  = conf->rp(PHI[i], LA[j], conf);

				dest[off] = omg[off] * tau_1 +
					(1. - conf->theta) * 
					   ( - conf->sigma * omg[off] + lpl * conf->mu) //оператор ~L
					+ rp;
			}
		}
	}

	/*!
	   неявная схема с внутренними итерациями
	   один шаг с правой частью \param f
	*/
	void S_step_im(double *dest, const double *h)
	{
//		int    off;
		vector < double > omg     (nn);
		vector < double > omg_old (nn);
		vector < double > psi     (nn);
		vector < double > psi_old (nn);
		vector < double > os      (nn);
		vector < double > B1      (nn);

		vector < double > bnd    (n_la);

		/*!находим \f$\omega_0 = \delta h_0\f$*/
		lapl->lapl(&omg_old[0], h);
		//?
		if (!full) { memset(&omg_old[0], 0, n_la * sizeof(double)); }

		/*!
		  в B1 кладем результат
	      \f[
	        \frac{\omega^n}{\tau} + (1-\theta)L\omega^n+f(x,y)
	      \f]
		 */
		take_B(&B1[0], &omg_old[0], 1);

		double norm;
		int it = 0;

		int s = (full) ? 0 : 1;

		memcpy(&psi[0], h, nn * sizeof(double));
		memcpy(&omg[0], &omg_old[0], nn * sizeof(double));
//		memcpy(bnd, omg_old, n_la * sizeof(double));

		memset(&bnd[0], 0, n_la * sizeof(double));

		while (true) {
			++it;
			memcpy(&psi_old[0], &psi[0], nn * sizeof(double));

			/*!
			  \f[
			    \frac{\omega^{n+1} + \omega^n}{2}
			  \f]
			 */
			vector_sum(&os[0], &omg[0], &omg_old[0], nn);
			vector_mult_scalar(&os[0], &os[0], 0.5, nn);

			/*!
			  \f[
			    \frac{\omega^{n+1} + \omega^n}{2}
			  \f]
			 */
			vector_sum(&psi[0], &psi[0], h, nn);
			vector_mult_scalar(&psi[0], &psi[0], 0.5, nn);

			//для установки краевого условия
			//начинаем от s!
			for (int i = s; i < n_phi; ++i) {
				for (int j = 0; j < n_la; ++j) {
					int off  = pOff(i, j);
					omg[off] = B1[off] -
						//0.0;
						conf->k1 * Jacobian(&psi[0], &os[0], i, j) + 
						conf->k2 * Jacobian(&psi[0], &cor[0], i, j);
				}
			}

			/*вычисляем u с крышкой*/
//			if (!full) { memset(omg, 0, n_la * sizeof(double));	}
			//_fprintfwmatrix("out/omg1.out", omg, n_la, n_phi,
			//	std::max(n_la, n_phi), "%23.16le ");
			
//здесь должно быть краевое условие laplace psi?
			if (!full) { 
				memcpy(&omg[0], &bnd[0], n_la * sizeof(double));
			}

			lapl->lapl_1(&omg[0], &omg[0], forward_mult, forward_diag);

			/*нашли омега, чтоб найти psi, надо обратить оператор лапласа*/
			if (!full) { 
				//memcpy(bnd, omg, n_la * sizeof(double));
				memset(&omg[0], 0, n_la * sizeof(double));	
			}
			//_fprintfwmatrix("out/omg2.out", omg, n_la, n_phi,
			//	std::max(n_la, n_phi), "%23.16le ");
			lapl->lapl_1(&psi[0], &omg[0]);

			//_fprintfwmatrix("out/psi.out", psi, n_la, n_phi,
			//	std::max(n_la, n_phi), "%23.16le ");

			norm = dist1(&psi[0], &psi_old[0], nn);
			//norm = dist2(psi, psi_old, nn);
			if (norm < _BARVORTEX_IM_EPS)
				break;
			if (it > _BARVORTEX_IM_MAX_IT) {
				printf("%s:%d:%s: it=%d, norm=%.16le\n", __FILE__, __LINE__, __FUNCTION__, it, norm);
				throw BadArgument("");
				//exit(1);
			}
		}

		memcpy(dest, &psi[0], nn * sizeof(double));
	}

	//явная схема
	void S_step(double *h1, const double *h, const double *F)
	{	
		vector < double > omg (nn);
		vector < double > B1(nn);

		int    off;
		double rp;
		double lpl;		

		/*!находим \f$\omega_0 = \delta h_0\f$*/
		lapl->lapl(&omg[0], h);
		if (!full) { memset(&omg[0], 0, n_la * sizeof(double)); }
		for (int i = 0; i < n_phi; ++i) {
			for (int j = 0; j < n_la; ++j) {
				off  = pOff(i, j);
				lpl  = lapl->lapl(&omg[0], i, j);
				rp  = F[off];
				rp -= conf->rho * (
						Jacobian(h, &omg[0], i, j) +
						Jacobian(h, &cor[0], i, j)
					);

				B1[off] = omg[off] * 
					(tau_1 - (1. - conf->theta) * conf->sigma)
					+ lpl * (1. - conf->theta) * conf->mu
					+ rp;
			}
		}

		/*вычисляем u с крышкой*/
		if (!full) { memset(&B1[0], 0, n_la * sizeof(double)); }
		lapl->lapl_1(&B1[0], &B1[0], forward_mult, forward_diag);
		/*нашли омега, чтоб найти h1, надо обратить оператор лапласа*/
		if (!full) { memset(&B1[0], 0, n_la * sizeof(double)); }
		lapl->lapl_1(h1, &B1[0]);
	}

	//неявная схема с внутренними итерациями
	void L_step_im(double *u1, const double *u_, 
		const double * z, int maxIt = _BARVORTEX_IM_MAX_IT)
	{
		vector < double > z_lapl(nn);
		
		vector < double > pt1(nn); //лаплас, умноженный на коэф
		vector < double > pt2(nn); //лаплас в квадрате, умноженный на коэф
		vector < double > pt3(nn); //якобиан, умноженный на коэф
		vector < double > omg(nn); //лаплас
		vector < double > u  (nn);
		vector < double > bnd(n_la);

		J_functor < SJacobian > funct1(jac, n_phi, n_la);

		memcpy(&u[0], u_, nn * sizeof(double));

		lapl->lapl(&z_lapl[0], z);
		lapl->lapl(&pt1[0], &u[0]); //первая часть - лаплас, умноженный на коэф, 
		//установка краевых условий

//?
		if (!full) { memset(&z_lapl[0], 0, n_la * sizeof(double)); }
		if (!full) { memset(&pt1[0], 0, n_la * sizeof(double)); }

		memcpy(&omg[0], &pt1[0], nn * sizeof(double)); //сохраняем лаплас - понадобится в pt3
		
		lapl->lapl(&pt2[0], &pt1[0]);
		//(1-\theta)*lapl^2
		vector_mult_scalar(&pt2[0], &pt2[0], 
			(1. - conf->theta) * conf->mu, nn);

		//(tau^{-1}-(1-\thetha)*sigma)*lapl
		vector_mult_scalar(&pt1[0], &pt1[0], 
			tau_1 - (1. - conf->theta) * conf->sigma, nn);

		int it = 0;
		double norm;

		vector < double > omg_n(nn);
		vector < double > u1_o (nn);

		memcpy(&u1[0], &u[0], nn * sizeof(double));
		memcpy(&omg_n[0], &omg[0], nn * sizeof(double));

//		memcpy(bnd, omg, n_la * sizeof(double));
		memset(&bnd[0], 0, n_la * sizeof(double));

		while (true) {
			++it;
			memcpy(&u1_o[0], &u1[0], nn * sizeof(double));

			vector_sum(&u1[0], &u1_o[0], &u[0], nn);
			vector_mult_scalar(&u1[0], &u1[0], 0.5, nn);

			vector_sum(&omg_n[0], &omg_n[0], &omg[0], nn);
			vector_mult_scalar(&omg_n[0], &omg_n[0], 0.5, nn);

			funct1.calc(&pt3[0], &u1[0], &cor[0], &z[0], &omg_n[0], &z_lapl[0]);
			vector_mult_scalar(&pt3[0], &pt3[0], -conf->rho, nn);

			memset(u1, 0, nn * sizeof(double));
			vector_sum(u1, u1, &pt1[0], nn);
			vector_sum(u1, u1, &pt2[0], nn);
			vector_sum(u1, u1, &pt3[0], nn);
			
			//if (!full) { memset(u1, 0, n_la * sizeof(double)); }

			if (!full) { memcpy(&u1[0], &bnd[0], n_la * sizeof(double)); }

			lapl->lapl_1(&omg_n[0], u1, forward_mult, forward_diag);

			if (!full) { memset(&omg_n[0], 0, n_la * sizeof(double)); }
			lapl->lapl_1(u1, &omg_n[0]);

			norm = dist1(&u1[0], &u1_o[0], nn);
			//norm = dist2(u1, u1_o, nn);
			//if (norm < maxIt)
			if (norm < _BARVORTEX_IM_EPS)
				break;
			if (it > maxIt) {
				::printf("%s:%d:%s: it=%d, norm=%.16le\n", __FILE__, __LINE__, __FUNCTION__, it, norm);
				break;
				//exit(1);
			}
		}
	}

	//неявная схема с внутренними итерациями
	void L_1_step_im(double *dest, const double *h_, 
		const double * z, int maxIt = _BARVORTEX_IM_MAX_IT)
	{
		int    off;
		vector < double > omg     (nn);
		vector < double > omg_old (nn);
		vector < double > psi     (nn);
		vector < double > psi_old (nn);
		vector < double > z_lapl  (nn);
		vector < double > os      (nn);
		vector < double > B1      (nn);	
		vector < double > h       (nn);
		//double rp       = 0.0;	
		//double k        = (1.0 - conf->theta);
		double lpl;

		memcpy(&h[0], h_, nn * sizeof(double));
		/*!находим \f$\omega_0 = \delta h_0\f$*/
		lapl->lapl(&omg_old[0], &h[0]);
		lapl->lapl(&z_lapl[0], &z[0]);
		//установка краевых условий
		if (!full) { memset(&z_lapl[0], 0, n_la * sizeof(double)); }
		if (!full) { memset(&omg_old[0], 0, n_la * sizeof(double)); }

		for (int i = 0; i < n_phi; ++i) {
			for (int j = 0; j < n_la; ++j) {
				off  = pOff(i, j);
				lpl  = lapl->lapl(&omg_old[0], i, j);

				B1[off] = omg_old[off] * 
					(tau_1 + ( conf->theta) * conf->sigma)
					+ lpl * ( - conf->theta) * conf->mu;
			}
		}

		double norm;
		int it = 0;

		memcpy(&psi[0], &h[0], nn * sizeof(double));
		memcpy(&omg[0], &omg_old[0], nn * sizeof(double));
		while (true) {
			++it;
			memcpy(&psi_old[0], &psi[0], nn * sizeof(double));

			vector_sum(&os[0], &omg[0], &omg_old[0], nn);
			vector_mult_scalar(&os[0], &os[0], 0.5, nn);

			vector_sum(&psi[0], &psi[0], &h[0], nn);
			vector_mult_scalar(&psi[0], &psi[0], 0.5, nn);

			for (int i = 0; i < n_phi; ++i) {
				for (int j = 0; j < n_la; ++j) {
					off      = pOff(i, j);
					omg[off] = B1[off] + 
						(conf->rho) * 
						(
							Jacobian(z, &os[0], i, j) + 
							Jacobian(&psi[0], &z_lapl[0], i, j) + 
							Jacobian(&psi[0], &cor[0], i, j)
						);
				}
			}

			/*вычисляем u с крышкой*/
			if (!full) { memset(&omg[0], 0, n_la * sizeof(double));	}
			lapl->lapl_1(&omg[0], &omg[0], backward_mult, backward_diag);
			/*нашли омега, чтоб найти h1, надо обратить оператор лапласа*/
			if (!full) { memset(&omg[0], 0, n_la * sizeof(double));	}
			lapl->lapl_1(&psi[0], &omg[0]);

			norm = dist1(&psi[0], &psi_old[0], nn);
			//norm = dist2(psi, psi_old, nn);
			if (norm < _BARVORTEX_IM_EPS)
				break;
			if (it > maxIt) {
				::printf("%s:%d:%s it=%d, norm=%.16le\n", __FILE__, __LINE__, __FUNCTION__, it, norm);
				//exit(1);
				break;
			}
		}

		memcpy(dest, &psi[0], nn * sizeof(double));
	}

	//явная схема
	void L_step(double *u1, const double *u, const double * z)
	{
		vector < double > z_lapl(nn);
		
		vector < double > pt1 (nn); //лаплас, умноженный на коэф
		vector < double > pt2 (nn); //лаплас в квадрате, умноженный на коэф
		vector < double > pt3 (nn); //якобиан, умноженный на коэф

		J_functor < SJacobian > funct1(jac, n_phi, n_la);

		lapl->lapl(&z_lapl[0], z);
		lapl->lapl(&pt1[0], u); //первая часть - лаплас, умноженный на коэф, 

		//установка краевых условий
		if (!full) { memset(&z_lapl[0], 0, n_la * sizeof(double)); }
		if (!full) { memset(&pt1[0], 0, n_la * sizeof(double)); }

		//умножаем позже, так как лаплас пока нужен
//		memset(pt3, 0, nn * sizeof(double));
		funct1.calc(&pt3[0], &u[0], &cor[0], z, &pt1[0], &z_lapl[0]);
		vector_mult_scalar(&pt3[0], &pt3[0], -conf->rho, nn);

//		memset(pt2, 0, nn * sizeof(double));
		lapl->lapl(&pt2[0], &pt1[0]);
		vector_mult_scalar(&pt2[0], &pt2[0], 
			(1. - conf->theta) * conf->mu, nn);

		vector_mult_scalar(&pt1[0], &pt1[0], 
			tau_1 - (1. - conf->theta) * conf->sigma, nn);

		memset(u1, 0, nn * sizeof(double));
		vector_sum(u1, u1, &pt1[0], nn);
		vector_sum(u1, u1, &pt2[0], nn);
		vector_sum(u1, u1, &pt3[0], nn);

		if (!full) { memset(u1, 0, n_la * sizeof(double)); }
		lapl->lapl_1(u1, u1, forward_mult, forward_diag);
		if (!full) { memset(u1, 0, n_la * sizeof(double)); }
		lapl->lapl_1(u1, u1);
	}

	//явная схема, сопряженная задача
	void L_step_t(double *v1, const double *v, const double * z)
	{
		vector < double > z_lapl(nn);
		
		vector < double > pt1 (nn); //лаплас, умноженный на коэф
		vector < double > pt2 (nn); //лаплас в квадрате, умноженный на коэф
		vector < double > pt3 (nn); //якобиан, умноженный на коэф

		JT_functor < SJacobian, SLaplacian > 
			funct2(jac, lapl, n_phi, n_la, full);

		lapl->lapl(&z_lapl[0], z);
		if (!full) { memset(&z_lapl[0], 0, n_la * sizeof(double)); }

		lapl->lapl_1(v1, v);
		lapl->lapl_1(v1, v1, forward_mult, forward_diag);

		lapl->lapl(&pt1[0], v1);
		if (!full) { memset(&pt1[0], 0, n_la * sizeof(double)); }
		vector_mult_scalar(&pt1[0], &pt1[0], 
			tau_1 - (1. - conf->theta) * conf->sigma, nn);

		lapl->lapl(&pt2[0], v1);
		if (!full) { memset(&pt2[0], 0, n_la * sizeof(double)); }
		lapl->lapl(&pt2[0], &pt2[0]);
		if (!full) { memset(&pt2[0], 0, n_la * sizeof(double)); }

		vector_mult_scalar(&pt2[0], &pt2[0], 
			(1. - conf->theta) * conf->mu, nn);

		funct2.calc(&pt3[0], v1, &cor[0], z, 0, &z_lapl[0]);
		vector_mult_scalar(&pt3[0], &pt3[0], -conf->rho, nn);

		memset(v1, 0, nn * sizeof(double));
		vector_sum(v1, v1, &pt1[0], nn);
		vector_sum(v1, v1, &pt2[0], nn);
		vector_sum(v1, v1, &pt3[0], nn);
		if (!full) { memset(v1, 0, n_la * sizeof(double)); }
	}

	//неявная схема, сопряженная задача
	void L_step_im_t(double *v1, const double *v, 
		const double * z, int maxIt = _BARVORTEX_IM_MAX_IT)
	{
		vector < double > z_lapl(nn);
		
		vector < double > pt1 (nn); //лаплас, умноженный на коэф
		vector < double > pt2 (nn); //лаплас в квадрате, умноженный на коэф
		vector < double > pt3 (nn); //якобиан, умноженный на коэф

		vector < double > v1_new(nn);
		vector < double > v1_old(nn);
		vector < double > tmp   (nn);

		JT_functor < SJacobian, SLaplacian > 
			funct2(jac, lapl, n_phi, n_la, full);

		lapl->lapl(&z_lapl[0], z);
		if (!full) { memset(&z_lapl[0], 0, n_la * sizeof(double)); }

		lapl->lapl_1(&tmp[0], v);
		lapl->lapl_1(v1, &tmp[0], forward_mult, forward_diag);

		lapl->lapl(&pt1[0], v1);
		if (!full) { memset(&pt1[0], 0, n_la * sizeof(double)); }
		vector_mult_scalar(&pt1[0], &pt1[0], 
			tau_1 - (1. - conf->theta) * conf->sigma, nn);

		lapl->lapl(&pt2[0], v1);
		if (!full) { memset(&pt2[0], 0, n_la * sizeof(double)); }
		lapl->lapl(&pt2[0], &pt2[0]);
		if (!full) { memset(&pt2[0], 0, n_la * sizeof(double)); }
		vector_mult_scalar(&pt2[0], &pt2[0], 
			(1. - conf->theta) * conf->mu, nn);

		//параметры явной схемы
		//__________________________________________________________
		vector < double > pt11   (nn); //лаплас, умноженный на коэф
		vector < double > pt21   (nn); //лаплас в квадрате, умноженный на коэф
		vector < double > pt31   (nn); //якобиан, умноженный на коэф
		vector < double > omg    (nn); //лаплас
		vector < double > omg_n  (nn);
		vector < double > u1_o   (nn);
		vector < double > u1     (nn);

		const double * u  = v;

		J_functor < SJacobian > funct1(jac, n_phi, n_la);
		lapl->lapl(&pt11[0], u); //первая часть - лаплас, умноженный на коэф, 
		//установка краевых условий		
		if (!full) { memset(&pt11[0], 0, n_la * sizeof(double)); }
		memcpy(&omg[0], &pt11[0], nn * sizeof(double)); //сохраняем лаплас - понадобится в pt3
		lapl->lapl(&pt21[0], &pt11[0]);
		vector_mult_scalar(&pt21[0], &pt21[0], 
			(1. - conf->theta) * conf->mu, nn);

		vector_mult_scalar(&pt11[0], &pt11[0], 
			tau_1 - (1. - conf->theta) * conf->sigma, nn);
		memcpy(&u1[0], u, nn * sizeof(double));
		memcpy(&omg_n[0], &omg[0], nn * sizeof(double));
		//__________________________________________________________


		//в неявной схеме pt1 и pt2 фиксированы, по pt3 идет итерирование

		double norm;
		int it = 0;
		memcpy(&v1_new[0], v1, nn * sizeof(double));
		while (true) {
			++it;

			//______________________________________________________
			//явная часть
			memcpy(&u1_o[0], &u1[0], nn * sizeof(double));

			vector_sum(&u1[0], &u1_o[0], &u[0], nn);
			vector_mult_scalar(&u1[0], &u1[0], 0.5, nn);

			vector_sum(&omg_n[0], &omg_n[0], &omg[0], nn);
			vector_mult_scalar(&omg_n[0], &omg_n[0], 0.5, nn);

			funct1.calc(&pt31[0], &u1[0], &cor[0], &z[0], &omg_n[0], &z_lapl[0]);
			vector_mult_scalar(&pt31[0], &pt31[0], -conf->rho, nn);

			memset(&u1[0], 0, nn * sizeof(double));
			vector_sum(&u1[0], &u1[0], &pt11[0], nn);
			vector_sum(&u1[0], &u1[0], &pt21[0], nn);
			vector_sum(&u1[0], &u1[0], &pt31[0], nn);
			if (!full) { memset(&u1[0], 0, n_la * sizeof(double)); }
			lapl->lapl_1(&omg_n[0], &u1[0], forward_mult, forward_diag);
			if (!full) { memset(&omg_n[0], 0, n_la * sizeof(double)); }
			lapl->lapl_1(&u1[0], &omg_n[0]);
			//______________________________________________________

			memcpy(&v1_old[0], &v1_new[0], nn * sizeof(double));
			lapl->lapl_1(&u1_o[0], &u1_o[0]);
			lapl->lapl_1(&u1_o[0], &u1_o[0], forward_mult, forward_diag);

			vector_sum(&v1_new[0], &u1_o[0], &v1[0], nn);
			vector_mult_scalar(&v1_new[0], &v1_new[0], 0.5, nn);

			funct2.calc(&pt3[0], &v1_new[0], &cor[0], z, 0, &z_lapl[0]);
			vector_mult_scalar(&pt3[0], &pt3[0], -conf->rho, nn);

			memset(&v1_new[0], 0, nn * sizeof(double));
			vector_sum(&v1_new[0], &v1_new[0], &pt1[0], nn);
			vector_sum(&v1_new[0], &v1_new[0], &pt2[0], nn);
			vector_sum(&v1_new[0], &v1_new[0], &pt3[0], nn);
			if (!full) { memset(&v1_new[0], 0, n_la * sizeof(double)); }

			norm = dist1(&v1_new[0], &v1_old[0], nn);
			//norm = dist2(v1_new, v1_old, nn);
			if (norm < _BARVORTEX_IM_EPS)
				break;			
			if (it > maxIt) {
				::printf("%s:%d:%s: it=%d, norm=%.16le\n", __FILE__, __LINE__, __FUNCTION__, it, norm);
				break;
				//exit(1);
			}
		}

		memcpy(&v1[0], &v1_new[0], nn * sizeof(double));
	}

	//явная схема
	//warning: оператор не является обратным к L_step, так как 
	//J берется от другого значения
	void L_1_step(double *u1, const double *u, const double * z)
	{
		vector < double > z_lapl(nn);
		
		vector < double > pt1 (nn); //лаплас, умноженный на коэф
		vector < double > pt2 (nn); //лаплас в квадрате, умноженный на коэф
		vector < double > pt3 (nn); //якобиан, умноженный на коэф

		J_functor < SJacobian > funct1(jac, n_phi, n_la);

		lapl->lapl(&z_lapl[0], z);
		lapl->lapl(&pt1[0], u); //первая часть - лаплас, умноженный на коэф, 

		//установка краевых условий
		if (!full) { memset(&z_lapl[0], 0, n_la * sizeof(double)); }
		if (!full) { memset(&pt1[0], 0, n_la * sizeof(double)); }

		//умножаем позже, так как лаплас пока нужен
		funct1.calc(&pt3[0], u, &cor[0], z, &pt1[0], &z_lapl[0]);
		vector_mult_scalar(&pt3[0], &pt3[0], conf->rho, nn);

		lapl->lapl(&pt2[0], &pt1[0]);
		vector_mult_scalar(&pt2[0], &pt2[0], 
			( - conf->theta) * conf->mu, nn);

		vector_mult_scalar(&pt1[0], &pt1[0], 
			tau_1 + ( conf->theta) * conf->sigma, nn);

		memset(u1, 0, nn * sizeof(double));
		vector_sum(u1, u1, &pt1[0], nn);
		vector_sum(u1, u1, &pt2[0], nn);
		vector_sum(u1, u1, &pt3[0], nn);

		if (!full) { memset(u1, 0, n_la * sizeof(double)); }
		lapl->lapl_1(u1, u1, backward_mult, backward_diag);
		if (!full) { memset(u1, 0, n_la * sizeof(double)); }
		lapl->lapl_1(u1, u1);
	}

	double scalar(const double *x, const double *y, int n)
	{
		return lapl->scalar(x, y);
	}

	double norm(const double *x, int n)
	{
		return sqrt(scalar(x, x, n));
	}

	void calc_rp(double * rp, const double * u)
	{
		vector < double > w(nn);
		vector < double > dw(nn);
		vector < double > jac1(nn);
		vector < double > jac2(nn);

		lapl->lapl(&w[0], &u[0]);  memset(&w[0], 0, n_la * sizeof(double));
		lapl->lapl(&dw[0], &w[0]); memset(&dw[0], 0, n_la * sizeof(double));

		jac->J(&jac1[0], &u[0], &w[0]);
		jac->J(&jac2[0], &u[0], &cor[0]);

		vector_sum(rp, &jac1[0], &jac2[0], nn);
		vector_sum1(rp, rp, &w[0], 1.0, conf->sigma, nn);
		vector_sum1(rp, rp, &dw[0], 1.0, -conf->mu, nn);

		memset(rp, 0, n_la * sizeof(double));
	}
};

#endif //_SDS_BAR_IMPL_H
