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
/*________ ������ ������������ ����� �� ����� - ���������� ________________*/

#include <assert.h>
#include <vector>

using namespace std;

/*!
   �������� ����������������� ��������
 */
template < typename Jac >
class J_functor {
	Jac * impl_;
	int n1_;
	int n2_;

public:
	J_functor(Jac * impl, int n1, int n2): 
	  impl_(impl), n1_(n1), n2_(n2) {}

	/*! ������� ����� ���� ���������,
	    ����� ��������� �� J(u, b + cor) �� �����

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
		double * p_lapl = new double[nn];
		double * tmp    = new double[nn];
		memset(h1, 0, nn * sizeof(double));
#ifdef _BARVORTEX_ARAKAWA
		impl_->JT(tmp, h, cor);
		impl_->JT(p_lapl, z, h);
		if (!full_) { memset(p_lapl, 0, n2_ * sizeof(double)); }
		lapl_->lapl(p_lapl, p_lapl);
		impl_->JT(h1, h, z_lapl);
#else
		impl_->J1T(tmp, h, cor, 1);
		impl_->J1T(p_lapl, z, h, 2);
		if (!full_) { memset(p_lapl, 0, n2_ * sizeof(double)); }
		lapl_->lapl(p_lapl, p_lapl);
		impl_->J1T(h1, h, z_lapl, 1);
#endif
		if (!full_) { memset(p_lapl, 0, n2_ * sizeof(double)); }
		vector_sum(h1, h1, p_lapl, nn);
		vector_sum(h1, h1, tmp, nn);
		delete [] p_lapl;
		delete [] tmp;
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

class Baroclin::Private: public SData {
public:
	Baroclin * p;
	BaroclinConf *conf; //!<������

	int nn;
	double *cor; //!<��������

	double *B1; //!<��������� �������	

	void setTau(double _tau) {
		tau   = _tau;
		tau_1 = 1.0 / _tau;

		forward_mult  = - conf->theta * conf->mu;
		forward_diag  = tau_1 + conf->theta * conf->sigma;

		backward_mult = (1.0 - conf->theta) * conf->mu;
		backward_diag = tau_1 - (1.0 - conf->theta) * conf->sigma;
	}

	Private(BaroclinConf& _conf)
		: SData(_conf.n_phi, _conf.n_la, _conf.full),
		p(0), conf(&_conf), cor(0), B1(0)
	{
		init();
		reset();
		setTau(conf->tau);

		nn = n_phi * n_la;
	}

	~Private() {
		if (B1)  delete [] B1;
		if (cor) delete [] cor;
	}

	void reset() {
		if (cor) { delete [] cor; cor = 0; }

		nn = n_phi * n_la;
		cor = new double[nn];

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
	   ������� ����������� ������ �����:
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
				if (have_rp) rp  = conf->rp1(PHI[i], LA[j], 0, conf);

				dest[off] = omg[off] * tau_1 +
					(1. - conf->theta) * 
					   ( - conf->sigma * omg[off] + lpl * conf->mu) //�������� ~L
					+ rp;
			}
		}
	}

	/*!
	   ������� ����� � ����������� ����������
	   ���� ��� � ������ ������ \param f
	   */
	void S_step_im(double *u11, double * u21, const double *u1, const double *u2, double t)
	{
		int sz = nn;
		double theta = conf->theta;
		double alpha = conf->alpha;
		double mu    = conf->mu;
		double sigma = conf->sigma;
		double mu1   = conf->mu1;
		double sigma1= conf->sigma1;
		double tau   = conf->tau;

		// ������ ����� 1:
		// -J(0.5(u1+u1), 0.5(w1+w1)+l+h) - J(0.5(u2+u2),w2+w2)+
		// + w1/tau - 0.5 (1-theta)sigma (w1-w2)+mu(1-theta)(\Delta w1)
		// ������ ����� 2:
		// -J(0.5(u1+u1), 0.5(w2+w2)) - J(0.5(u2+u2), 0.5(w1+w1)+l+h) -
		// - 0.5 (1-theta)sigma (w1 + w2) + (1-theta) mu \Delta w2
		// + w2/tau - alpha^2 u2/tau + alpha^2 J(0.5(u1+u1), 0.5(u2+u2)) -
		// - alpha^2 (1-theta) mu1 w2 +
		// + alpha^2 sigma1 (1-theta) u2 + alpha^2 f(phi, lambda)

		vector < double > w1(sz);
		vector < double > w2(sz);
		vector < double > dw1(sz);
		vector < double > dw2(sz);
		vector < double > FC(sz);
		vector < double > GC(sz);

		// next
		vector < double > u1_n(sz);
		vector < double > u2_n(sz);
		vector < double > w1_n(sz);
		vector < double > w2_n(sz);

		vector < double > u1_n1(sz);
		vector < double > u2_n1(sz);

		// tmp
		vector < double > tmp1(sz);
		vector < double > tmp2(sz);

		// jac
		vector < double > jac1(sz);
		vector < double > jac2(sz);
		vector < double > jac3(sz);

		//
		vector < double > F(sz);
		vector < double > G(sz);
		vector < double > Z(sz);

		lapl->lapl(&w1[0], &u1[0]);  memset(&w1[0], 0, n_la * sizeof(double));
		lapl->lapl(&w2[0], &u2[0]);  memset(&w2[0], 0, n_la * sizeof(double));

		lapl->lapl(&dw1[0], &w1[0]); memset(&dw1[0], 0, n_la * sizeof(double)); 
		lapl->lapl(&dw2[0], &w2[0]); memset(&dw2[0], 0, n_la * sizeof(double));

		vector_sum1(&FC[0], &w1[0], &w2[0], 
			-0.5 * (1.0 - theta) * sigma, 
			0.5 * (1.0 - theta) * sigma, sz);
		vector_sum1(&FC[0], &FC[0], &dw1[0], 1.0, mu * (1.0 - theta), sz);
		vector_sum1(&FC[0], &FC[0], &w1[0], 1.0, 1.0 / tau, sz);

		// w2/tau - 0.5 (1-theta)sigma (w1 + w2) + (1-theta) mu \Delta w2 -
		// - alpha^2 u2/tau - alpha^2 (1-theta) mu1 w2 + alpha^2 sigma1 (1-theta) u2
		vector_sum1(&GC[0], &w1[0], &w2[0],
			-0.5 * (1.0 - theta) * sigma, 
			-0.5 * (1.0 - theta) * sigma, sz);
		vector_sum1(&GC[0], &GC[0], &dw2[0], 1.0, mu * (1.0 - theta), sz);
		vector_sum1(&GC[0], &GC[0], &w2[0], 1.0, 1.0 / tau, sz);
		vector_sum1(&GC[0], &GC[0], &u2[0], 1.0, -alpha * alpha / tau, sz);
		vector_sum1(&GC[0], &GC[0], &w2[0], 1.0, -alpha * alpha * mu1 * (1-theta), sz);
		vector_sum1(&GC[0], &GC[0], &u2[0], 1.0, alpha * alpha * sigma1 * (1-theta), sz);

		memcpy(&u1_n[0], &u1[0], sz * sizeof(double));
		memcpy(&u2_n[0], &u2[0], sz * sizeof(double));
		memcpy(&w1_n[0], &w1[0], sz * sizeof(double));
		memcpy(&w2_n[0], &w2[0], sz * sizeof(double));

		for (int i = 0; i < n_phi; ++i) {
			for (int j = 0; j < n_la; ++j) {
				double phi = PHI[i];
				double la  = LA[j];
				int off  = pOff(i, j);
				if (conf->rp1) {
					FC[off] += conf->rp1(phi, la, t, conf);
				}
				if (conf->rp2) {
					GC[off] += alpha * alpha * conf->rp2(phi, la, t, conf);
				}
			}
		}

		for (int it = 0; it < 20; ++it) {
			// - J(0.5(u1+u1), 0.5(w1+w1)+l+h) - J(0.5(u2+u2),w2+w2)
			// J(0.5(u1+u1), 0.5(w1+w1)+l+h)
			vector_sum1(&tmp1[0], &u1[0], &u1_n[0], 1.0 - theta, theta, sz);
			vector_sum1(&tmp2[0], &w1[0], &w1_n[0], 1.0 - theta, theta, sz);
			vector_sum(&tmp2[0], &tmp2[0], &cor[0], sz);
			jac->J(&jac1[0], &tmp1[0], &tmp2[0]);
			//_fprintfwmatrix("jac.txt", &jac1[0], n_phi, n_la, n_la, "%.1le ");
			// J(0.5(u2+u2),w2+w2)
			vector_sum1(&tmp1[0], &u2[0], &u2_n[0], 1.0 - theta, theta, sz);
			vector_sum1(&tmp2[0], &w2[0], &w2_n[0], 1.0 - theta, theta, sz);
			jac->J(&jac2[0], &tmp1[0], &tmp2[0]);

			vector_sum1(&F[0], &jac1[0], &jac2[0], -1.0, -1.0, sz);

			// -J(0.5(u1+u1), 0.5(w2+w2)) - J(0.5(u2+u2), 0.5(w1+w1)+l+h) +
			// + alpha^2 J(0.5(u1+u1), 0.5(u2+u2))
			vector_sum1(&tmp1[0], &u1[0], &u1_n[0], 1.0 - theta, theta, sz);
			vector_sum1(&tmp2[0], &w2[0], &w2_n[0], 1.0 - theta, theta, sz);
			jac->J(&jac1[0], &tmp1[0], &tmp2[0]);
			vector_sum1(&tmp1[0], &u2[0], &u2_n[0], 1.0 - theta, theta, sz);
			vector_sum1(&tmp2[0], &w1[0], &w1_n[0], 1.0 - theta, theta, sz);
			vector_sum(&tmp2[0], &tmp2[0], &cor[0], sz);
			jac->J(&jac2[0], &tmp1[0], &tmp2[0]);
			vector_sum1(&tmp1[0], &u1[0], &u1_n[0], 1.0 - theta, theta, sz);
			vector_sum1(&tmp2[0], &u2[0], &u2_n[0], 1.0 - theta, theta, sz);
			jac->J(&jac3[0], &tmp1[0], &tmp2[0]);
			vector_sum1(&G[0], &jac1[0], &jac2[0], -1.0, -1.0, sz);
			vector_sum1(&G[0], &G[0], &jac3[0], 1.0, alpha * alpha, sz);

			//memcpy(&F[0], &FC[0], sz * sizeof(double));
			//memcpy(&G[0], &GC[0], sz * sizeof(double));
			vector_sum(&F[0], &F[0], &FC[0], sz);
			vector_sum(&G[0], &G[0], &GC[0], sz);

			memset(&F[0], 0, n_la * sizeof(double));
			memset(&G[0], 0, n_la * sizeof(double));

			lapl->baroclin_1(&w1_n[0], &w2_n[0], &u1_n1[0], &u2_n1[0],
				&F[0], &G[0], &Z[0], &Z[0], 
				-mu * theta, 1.0 / tau + 0.5 * theta * sigma,
				-mu * theta, 1.0 / tau + 0.5 * theta * sigma + alpha * alpha * theta * mu1,
				-1.0, 0.0,
				-1.0, 0.0,

				-0.5 * sigma * theta,
				 0.5 * sigma * theta,
				-alpha * alpha / tau - alpha * alpha * theta * sigma1,
				1.0,
				1.0
				);

//			lapl->lapl(&tmp1[0], &u1_n1[0]);  memset(&tmp1[0], 0, n_la * sizeof(double));
//			lapl->lapl_1(&tmp2[0], &w1_n[0]); memset(&tmp2[0], 0, n_la * sizeof(double));
//			lapl->lapl_1(&u2_n1[0], &w2_n[0]);

			double nr1 = p->dist(&u1_n1[0], &u1_n[0], sz);
			double nr2 = p->dist(&u2_n1[0], &u2_n[0], sz);
			double nr  = std::max(nr1, nr2);
			u1_n1.swap(u1_n);
			u2_n1.swap(u2_n);

			if (nr < 1e-8) {
				break;
			}
		}

		memcpy(u11, &u1_n[0], sz * sizeof(double));
		memcpy(u21, &u2_n[0], sz * sizeof(double));
	}

	//������� ����� � ����������� ����������
	void L_step_im(double *u1, const double *u_, 
		const double * z, int maxIt = _BARVORTEX_IM_MAX_IT)
	{
		double *z_lapl  = new double[nn];
		
		double *pt1 = new double[nn]; //������, ���������� �� ����
		double *pt2 = new double[nn]; //������ � ��������, ���������� �� ����
		double *pt3 = new double[nn]; //�������, ���������� �� ����
		double *omg = new double[nn]; //������
		double *u   = new double[nn];
		double * bnd    = new double[n_la];

		J_functor < SJacobian > funct1(jac, n_phi, n_la);

		memcpy(u, u_, nn * sizeof(double));

		lapl->lapl(z_lapl, z);
		lapl->lapl(pt1, u); //������ ����� - ������, ���������� �� ����, 
		//��������� ������� �������

//?
		if (!full) { memset(z_lapl, 0, n_la * sizeof(double)); }
		if (!full) { memset(pt1, 0, n_la * sizeof(double)); }

		memcpy(omg, pt1, nn * sizeof(double)); //��������� ������ - ����������� � pt3
		
		lapl->lapl(pt2, pt1);
		//(1-\theta)*lapl^2
		vector_mult_scalar(pt2, pt2, 
			(1. - conf->theta) * conf->mu, nn);

		//(tau^{-1}-(1-\thetha)*sigma)*lapl
		vector_mult_scalar(pt1, pt1, 
			tau_1 - (1. - conf->theta) * conf->sigma, nn);

		int it = 0;
		double norm;

		double * omg_n = new double[nn];
		double * u1_o  = new double[nn];

		memcpy(u1, u, nn * sizeof(double));
		memcpy(omg_n, omg, nn * sizeof(double));

//		memcpy(bnd, omg, n_la * sizeof(double));
		memset(bnd, 0, n_la * sizeof(double));

		while (true) {
			++it;
			memcpy(u1_o, u1, nn * sizeof(double));

			vector_sum(u1, u1_o, u, nn);
			vector_mult_scalar(u1, u1, 0.5, nn);

			vector_sum(omg_n, omg_n, omg, nn);
			vector_mult_scalar(omg_n, omg_n, 0.5, nn);

			funct1.calc(pt3, u1, cor, z, omg_n, z_lapl);
			vector_mult_scalar(pt3, pt3, -conf->rho, nn);

			memset(u1, 0, nn * sizeof(double));
			vector_sum(u1, u1, pt1, nn);
			vector_sum(u1, u1, pt2, nn);
			vector_sum(u1, u1, pt3, nn);
			
			//if (!full) { memset(u1, 0, n_la * sizeof(double)); }

			if (!full) { memcpy(u1, bnd, n_la * sizeof(double)); }

			lapl->lapl_1(omg_n, u1, forward_mult, forward_diag);

			if (!full) { memset(omg_n, 0, n_la * sizeof(double)); }
			lapl->lapl_1(u1, omg_n);

			norm = dist1(u1, u1_o, nn);
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

		delete [] z_lapl; delete [] u;
		delete [] pt1; delete [] pt2; delete [] pt3;
		delete [] omg; delete [] omg_n; delete [] u1_o;

		delete [] bnd;
	}

	//������� ����� � ����������� ����������
	void L_1_step_im(double *dest, const double *h_, 
		const double * z, int maxIt = _BARVORTEX_IM_MAX_IT)
	{
		int    off;
		double * omg    = new double[nn];
		double *omg_old = new double[nn];
		double *psi     = new double[nn];
		double *psi_old = new double[nn];
		double *z_lapl  = new double[nn];
		double *os      = new double[nn];
		double *B1      = new double[nn];	
		double *h       = new double[nn];
		//double rp       = 0.0;	
		//double k        = (1.0 - conf->theta);
		double lpl;

		memcpy(h, h_, nn * sizeof(double));
		/*!������� \f$\omega_0 = \delta h_0\f$*/
		lapl->lapl(omg_old, h);
		lapl->lapl(z_lapl, z);
		//��������� ������� �������
		if (!full) { memset(z_lapl, 0, n_la * sizeof(double)); }
		if (!full) { memset(omg_old, 0, n_la * sizeof(double)); }

		for (int i = 0; i < n_phi; ++i) {
			for (int j = 0; j < n_la; ++j) {
				off  = pOff(i, j);
				lpl  = lapl->lapl(omg_old, i, j);

				B1[off] = omg_old[off] * 
					(tau_1 + ( conf->theta) * conf->sigma)
					+ lpl * ( - conf->theta) * conf->mu;
			}
		}

		double norm;
		int it = 0;

		memcpy(psi, h, nn * sizeof(double));
		memcpy(omg, omg_old, nn * sizeof(double));
		while (true) {
			++it;
			memcpy(psi_old, psi, nn * sizeof(double));

			vector_sum(os, omg, omg_old, nn);
			vector_mult_scalar(os, os, 0.5, nn);

			vector_sum(psi, psi, h, nn);		
			vector_mult_scalar(psi, psi, 0.5, nn);

			for (int i = 0; i < n_phi; ++i) {
				for (int j = 0; j < n_la; ++j) {
					off      = pOff(i, j);
					omg[off] = B1[off] + 
						(conf->rho) * 
						(
							Jacobian(z, os, i, j) + 
							Jacobian(psi, z_lapl, i, j) + 
							Jacobian(psi, cor, i, j)
						);
				}
			}

			/*��������� u � �������*/
			if (!full) { memset(omg, 0, n_la * sizeof(double));	}
			lapl->lapl_1(omg, omg, backward_mult, backward_diag);
			/*����� �����, ���� ����� h1, ���� �������� �������� �������*/
			if (!full) { memset(omg, 0, n_la * sizeof(double));	}
			lapl->lapl_1(psi, omg);

			norm = dist1(psi, psi_old, nn);
			//norm = dist2(psi, psi_old, nn);
			if (norm < _BARVORTEX_IM_EPS)
				break;
			if (it > maxIt) {
				::printf("%s:%d:%s it=%d, norm=%.16le\n", __FILE__, __LINE__, __FUNCTION__, it, norm);
				//exit(1);
				break;
			}
		}

		memcpy(dest, psi, nn * sizeof(double));
		delete [] psi_old; delete [] psi;
		delete [] B1;  delete [] os;
		delete [] omg; delete [] omg_old;
		delete [] z_lapl; delete [] h;
	}

	//����� �����
	void L_step(double *u1, const double *u, const double * z)
	{
		double *z_lapl  = new double[nn];
		
		double *pt1 = new double[nn]; //������, ���������� �� ����
		double *pt2 = new double[nn]; //������ � ��������, ���������� �� ����
		double *pt3 = new double[nn]; //�������, ���������� �� ����

		J_functor < SJacobian > funct1(jac, n_phi, n_la);

		lapl->lapl(z_lapl, z);
		lapl->lapl(pt1, u); //������ ����� - ������, ���������� �� ����, 

		//��������� ������� �������
		if (!full) { memset(z_lapl, 0, n_la * sizeof(double)); }
		if (!full) { memset(pt1, 0, n_la * sizeof(double)); }

		//�������� �����, ��� ��� ������ ���� �����
//		memset(pt3, 0, nn * sizeof(double));
		funct1.calc(pt3, u, cor, z, pt1, z_lapl);
		vector_mult_scalar(pt3, pt3, -conf->rho, nn);

//		memset(pt2, 0, nn * sizeof(double));
		lapl->lapl(pt2, pt1);
		vector_mult_scalar(pt2, pt2, 
			(1. - conf->theta) * conf->mu, nn);

		vector_mult_scalar(pt1, pt1, 
			tau_1 - (1. - conf->theta) * conf->sigma, nn);

		memset(u1, 0, nn * sizeof(double));
		vector_sum(u1, u1, pt1, nn);
		vector_sum(u1, u1, pt2, nn);
		vector_sum(u1, u1, pt3, nn);

		if (!full) { memset(u1, 0, n_la * sizeof(double)); }
		lapl->lapl_1(u1, u1, forward_mult, forward_diag);
		if (!full) { memset(u1, 0, n_la * sizeof(double)); }
		lapl->lapl_1(u1, u1);

		delete [] z_lapl;
		delete [] pt1; delete [] pt2; delete [] pt3;
	}

	//����� �����, ����������� ������
	void L_step_t(double *v1, const double *v, const double * z)
	{
		double *z_lapl  = new double[nn];
		
		double *pt1 = new double[nn]; //������, ���������� �� ����
		double *pt2 = new double[nn]; //������ � ��������, ���������� �� ����
		double *pt3 = new double[nn]; //�������, ���������� �� ����

		JT_functor < SJacobian, SLaplacian > 
			funct2(jac, lapl, n_phi, n_la, full);

		lapl->lapl(z_lapl, z);
		if (!full) { memset(z_lapl, 0, n_la * sizeof(double)); }

		lapl->lapl_1(v1, v);
		lapl->lapl_1(v1, v1, forward_mult, forward_diag);

		lapl->lapl(pt1, v1);
		if (!full) { memset(pt1, 0, n_la * sizeof(double)); }
		vector_mult_scalar(pt1, pt1, 
			tau_1 - (1. - conf->theta) * conf->sigma, nn);

		lapl->lapl(pt2, v1);
		if (!full) { memset(pt2, 0, n_la * sizeof(double)); }
		lapl->lapl(pt2, pt2);
		if (!full) { memset(pt2, 0, n_la * sizeof(double)); }

		vector_mult_scalar(pt2, pt2, 
			(1. - conf->theta) * conf->mu, nn);

		funct2.calc(pt3, v1, cor, z, 0, z_lapl);
		vector_mult_scalar(pt3, pt3, -conf->rho, nn);

		memset(v1, 0, nn * sizeof(double));
		vector_sum(v1, v1, pt1, nn);
		vector_sum(v1, v1, pt2, nn);
		vector_sum(v1, v1, pt3, nn);
		if (!full) { memset(v1, 0, n_la * sizeof(double)); }

		delete [] z_lapl; delete [] pt1; delete [] pt2; delete [] pt3;
	}

	//������� �����, ����������� ������
	void L_step_im_t(double *v1, const double *v, 
		const double * z, int maxIt = _BARVORTEX_IM_MAX_IT)
	{
		double *z_lapl  = new double[nn];
		
		double *pt1 = new double[nn]; //������, ���������� �� ����
		double *pt2 = new double[nn]; //������ � ��������, ���������� �� ����
		double *pt3 = new double[nn]; //�������, ���������� �� ����

		double * v1_new = new double[nn];
		double * v1_old = new double[nn];
		double * tmp    = new double[nn];

		JT_functor < SJacobian, SLaplacian > 
			funct2(jac, lapl, n_phi, n_la, full);

		lapl->lapl(z_lapl, z);
		if (!full) { memset(z_lapl, 0, n_la * sizeof(double)); }

		lapl->lapl_1(tmp, v);
		lapl->lapl_1(v1, tmp, forward_mult, forward_diag);

		lapl->lapl(pt1, v1);
		if (!full) { memset(pt1, 0, n_la * sizeof(double)); }
		vector_mult_scalar(pt1, pt1, 
			tau_1 - (1. - conf->theta) * conf->sigma, nn);

		lapl->lapl(pt2, v1);
		if (!full) { memset(pt2, 0, n_la * sizeof(double)); }
		lapl->lapl(pt2, pt2);
		if (!full) { memset(pt2, 0, n_la * sizeof(double)); }
		vector_mult_scalar(pt2, pt2, 
			(1. - conf->theta) * conf->mu, nn);

		//��������� ����� �����
		//__________________________________________________________
		double *pt11   = new double[nn]; //������, ���������� �� ����
		double *pt21   = new double[nn]; //������ � ��������, ���������� �� ����
		double *pt31   = new double[nn]; //�������, ���������� �� ����
		double *omg    = new double[nn]; //������
		double * omg_n = new double[nn];
		double * u1_o  = new double[nn];
		double * u1    = new double[nn];

		const double * u  = v;

		J_functor < SJacobian > funct1(jac, n_phi, n_la);
		lapl->lapl(pt11, u); //������ ����� - ������, ���������� �� ����, 
		//��������� ������� �������		
		if (!full) { memset(pt11, 0, n_la * sizeof(double)); }
		memcpy(omg, pt11, nn * sizeof(double)); //��������� ������ - ����������� � pt3
		lapl->lapl(pt21, pt11);
		vector_mult_scalar(pt21, pt21, 
			(1. - conf->theta) * conf->mu, nn);

		vector_mult_scalar(pt11, pt11, 
			tau_1 - (1. - conf->theta) * conf->sigma, nn);
		memcpy(u1, u, nn * sizeof(double));
		memcpy(omg_n, omg, nn * sizeof(double));
		//__________________________________________________________


		//� ������� ����� pt1 � pt2 �����������, �� pt3 ���� ������������

		double norm;
		int it = 0;
		memcpy(v1_new, v1, nn * sizeof(double));
		while (true) {
			++it;

			//______________________________________________________
			//����� �����
			memcpy(u1_o, u1, nn * sizeof(double));

			vector_sum(u1, u1_o, u, nn);
			vector_mult_scalar(u1, u1, 0.5, nn);

			vector_sum(omg_n, omg_n, omg, nn);
			vector_mult_scalar(omg_n, omg_n, 0.5, nn);

			funct1.calc(pt31, u1, cor, z, omg_n, z_lapl);
			vector_mult_scalar(pt31, pt31, -conf->rho, nn);

			memset(u1, 0, nn * sizeof(double));
			vector_sum(u1, u1, pt11, nn);
			vector_sum(u1, u1, pt21, nn);
			vector_sum(u1, u1, pt31, nn);
			if (!full) { memset(u1, 0, n_la * sizeof(double)); }
			lapl->lapl_1(omg_n, u1, forward_mult, forward_diag);
			if (!full) { memset(omg_n, 0, n_la * sizeof(double)); }
			lapl->lapl_1(u1, omg_n);
			//______________________________________________________

			memcpy(v1_old, v1_new, nn * sizeof(double));
			lapl->lapl_1(u1_o, u1_o);
			lapl->lapl_1(u1_o, u1_o, forward_mult, forward_diag);

			vector_sum(v1_new, u1_o, v1, nn);
			vector_mult_scalar(v1_new, v1_new, 0.5, nn);

			funct2.calc(pt3, v1_new, cor, z, 0, z_lapl);
			vector_mult_scalar(pt3, pt3, -conf->rho, nn);

			memset(v1_new, 0, nn * sizeof(double));
			vector_sum(v1_new, v1_new, pt1, nn);
			vector_sum(v1_new, v1_new, pt2, nn);
			vector_sum(v1_new, v1_new, pt3, nn);
			if (!full) { memset(v1_new, 0, n_la * sizeof(double)); }

			norm = dist1(v1_new, v1_old, nn);
			//norm = dist2(v1_new, v1_old, nn);
			if (norm < _BARVORTEX_IM_EPS)
				break;			
			if (it > maxIt) {
				::printf("%s:%d:%s: it=%d, norm=%.16le\n", __FILE__, __LINE__, __FUNCTION__, it, norm);
				break;
				//exit(1);
			}
		}

		memcpy(v1, v1_new, nn * sizeof(double));

		delete [] z_lapl; delete [] pt1; delete [] pt2; delete [] pt3;
		delete [] v1_old; delete [] v1_new;

		delete [] tmp;
		delete [] pt11; delete [] pt21; delete [] pt31; delete [] omg;
	}

	//����� �����
	//warning: �������� �� �������� �������� � L_step, ��� ��� 
	//J ������� �� ������� ��������
	void L_1_step(double *u1, const double *u, const double * z)
	{
		double *z_lapl  = new double[nn];
		
		double *pt1 = new double[nn]; //������, ���������� �� ����
		double *pt2 = new double[nn]; //������ � ��������, ���������� �� ����
		double *pt3 = new double[nn]; //�������, ���������� �� ����

		J_functor < SJacobian > funct1(jac, n_phi, n_la);

		lapl->lapl(z_lapl, z);
		lapl->lapl(pt1, u); //������ ����� - ������, ���������� �� ����, 

		//��������� ������� �������
		if (!full) { memset(z_lapl, 0, n_la * sizeof(double)); }
		if (!full) { memset(pt1, 0, n_la * sizeof(double)); }

		//�������� �����, ��� ��� ������ ���� �����
		funct1.calc(pt3, u, cor, z, pt1, z_lapl);
		vector_mult_scalar(pt3, pt3, conf->rho, nn);

		lapl->lapl(pt2, pt1);
		vector_mult_scalar(pt2, pt2, 
			( - conf->theta) * conf->mu, nn);

		vector_mult_scalar(pt1, pt1, 
			tau_1 + ( conf->theta) * conf->sigma, nn);

		memset(u1, 0, nn * sizeof(double));
		vector_sum(u1, u1, pt1, nn);
		vector_sum(u1, u1, pt2, nn);
		vector_sum(u1, u1, pt3, nn);

		if (!full) { memset(u1, 0, n_la * sizeof(double)); }
		lapl->lapl_1(u1, u1, backward_mult, backward_diag);
		if (!full) { memset(u1, 0, n_la * sizeof(double)); }
		lapl->lapl_1(u1, u1);

		delete [] z_lapl;
		delete [] pt1; delete [] pt2; delete [] pt3;
	}

	double scalar(const double *x, const double *y, int n)
	{
		return lapl->scalar(x, y);
	}

	double norm(const double *x, int n)
	{
		return sqrt(scalar(x, x, n));
	}
};

#endif //_SDS_BAR_IMPL_H
