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
					if (alpha == 0) {
						GC[off] += conf->rp2(phi, la, t, conf);
					} else {
						GC[off] += alpha * alpha * conf->rp2(phi, la, t, conf);
					}
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
				-mu * theta, 1.0 / tau + 0.5 * theta * sigma + 
				 alpha * alpha * theta * mu1,

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
	void L_step_im(double * u11, double * u21, 
		const double * u1, const double * u2,
		const double * z1, const double * z2)
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
		// - J(z1, 0.5(w1+w1)) - J(u1, L(z1)+l+h) -
		// - J(z2, 0.5(w2+w2)) - J(0.5(u2+u2), L(z2)) +
		// + w1/tau - 0.5 (1-theta)sigma (w1-w2)+mu(1-theta)(L w1)
		// ������ ����� 2:
		// - J(z1, 0.5(w2+w2)) - J(0.5(u1+u1), L(z2)) -
		// - J(0.5(u2+u2), L(z1)+l+h) - J(z2, 0.5(w1+w1)) -
		// - 0.5 (1-theta)sigma (w1 + w2) + (1-theta) mu L w2
		// + w2/tau - alpha^2 u2/tau +
		// + alpha^2 J(z1, 0.5(u2+u2)) + alpha^2 J(0.5(u1+u1), z2) -
		// - alpha^2 (1-theta) mu1 w2 +
		// + alpha^2 sigma1 (1-theta) u2 

		vector < double > w1(sz);
		vector < double > w2(sz);
		vector < double > dw1(sz);
		vector < double > dw2(sz);
		vector < double > dz1(sz);
		vector < double > dz2(sz);
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

		lapl->lapl(&dz1[0], &z1[0]); memset(&dz1[0], 0, n_la * sizeof(double));
		lapl->lapl(&dz2[0], &z2[0]); memset(&dz2[0], 0, n_la * sizeof(double));

		// w1/tau - 0.5 (1-theta)sigma(w1-w2) + mu(1-theta)(L w1)
		vector_sum1(&FC[0], &w1[0], &w2[0], 
			-0.5 * (1.0 - theta) * sigma, 0.5 * (1.0 - theta) * sigma, sz);
		vector_sum1(&FC[0], &FC[0], &dw1[0], 1.0, mu * (1.0 - theta), sz);
		vector_sum1(&FC[0], &FC[0], &w1[0], 1.0, 1.0 / tau, sz);

		// w2/tau - 0.5 (1-theta)sigma (w1 + w2) + (1-theta) mu L w2 -
		// - alpha^2 u2/tau - alpha^2 (1-theta) mu1 w2 + alpha^2 sigma1 (1-theta) u2
		vector_sum1(&GC[0], &w1[0], &w2[0],
			-0.5 * (1.0 - theta) * sigma, -0.5 * (1.0 - theta) * sigma, sz);
		vector_sum1(&GC[0], &GC[0], &dw2[0], 1.0, mu * (1.0 - theta), sz);
		vector_sum1(&GC[0], &GC[0], &w2[0], 1.0, 1.0 / tau, sz);
		vector_sum1(&GC[0], &GC[0], &u2[0], 1.0, -alpha * alpha / tau, sz);
		vector_sum1(&GC[0], &GC[0], &w2[0], 1.0, -alpha * alpha * mu1 * (1-theta), sz);
		vector_sum1(&GC[0], &GC[0], &u2[0], 1.0, alpha * alpha * sigma1 * (1-theta), sz);

		memcpy(&u1_n[0], &u1[0], sz * sizeof(double));
		memcpy(&u2_n[0], &u2[0], sz * sizeof(double));
		memcpy(&w1_n[0], &w1[0], sz * sizeof(double));
		memcpy(&w2_n[0], &w2[0], sz * sizeof(double));

		for (int it = 0; it < 1; ++it) {
			// - J(0.5(u1+u1), L(z1)+l+h) - J(z1, 0.5(w1+w1)) -
			// - J(z2,0.5(w2+w2)) - J(0.5(u2+u2),L(z2))

			// J(0.5(u1+u1), L(z1)+l+h)
			vector_sum1(&tmp1[0], &u1[0], &u1_n[0], 1.0 - theta, theta, sz);
			vector_sum(&tmp2[0], &dz1[0], &cor[0], sz);
			jac->J(&jac1[0], &tmp1[0], &tmp2[0]);

			// J(z1, 0.5(w1+w1))
			vector_sum1(&tmp1[0], &w1[0], &w1_n[0], 1.0 - theta, theta, sz);
			jac->J(&jac2[0], &z1[0], &tmp1[0]);
			vector_sum1(&F[0], &jac1[0], &jac2[0], -1.0, -1.0, sz);

			// J(z2,0.5(w2+w2))
			vector_sum1(&tmp1[0], &w2[0], &w2_n[0], 1.0 - theta, theta, sz);
			jac->J(&jac1[0], &z2[0], &tmp1[0]);
			vector_sum1(&F[0], &F[0], &jac1[0], 1.0, -1.0, sz);

			// J(0.5(u2+u2),L(z2))
			vector_sum1(&tmp1[0], &u2[0], &u2_n[0], 1.0 - theta, theta, sz);
			jac->J(&jac1[0], &tmp1[0], &dz2[0]);
			vector_sum1(&F[0], &F[0], &jac1[0], 1.0, -1.0, sz);

			// - J(z1, 0.5(w2+w2)) - J(0.5(u1+u1), L(z2)) -
			// - J(0.5(u2+u2), L(z1)+l+h) - J(z2, 0.5(w1+w1)) +
			// + alpha^2 J(z1, 0.5(u2+u2)) + alpha^2 J(0.5(u1+u1), z2))

			// J(z1, 0.5(w2+w2))
			vector_sum1(&tmp1[0], &w2[0], &w2_n[0], 1.0 - theta, theta, sz);
			jac->J(&jac1[0], &dz1[0], &tmp1[0]);

			// J(0.5(u1+u1), L(z2))
			vector_sum1(&tmp1[0], &u1[0], &u1_n[0], 1.0 - theta, theta, sz);
			jac->J(&jac2[0], &dz2[0], &tmp1[0]);
			vector_sum1(&G[0], &jac1[0], &jac2[0], -1.0, -1.0, sz);

			// J(0.5(u2+u2), L(z1)+l+h)
			vector_sum1(&tmp1[0], &u2[0], &u2_n[0], 1.0 - theta, theta, sz);
			vector_sum(&tmp2[0], &dz1[0], &cor[0], sz);
			jac->J(&jac1[0], &tmp1[0], &tmp2[0]);
			vector_sum1(&G[0], &G[0], &jac1[0], 1.0, -1.0, sz);

			// J(z2, 0.5(w1+w1))
			vector_sum1(&tmp1[0], &w1[0], &w1_n[0], 1.0 - theta, theta, sz);
			jac->J(&jac1[0], &z2[0], &tmp1[0]);
			vector_sum1(&G[0], &G[0], &jac1[0], 1.0, -1.0, sz);

			// alpha^2 J(z1, 0.5(u2+u2))
			vector_sum1(&tmp1[0], &u2[0], &u2_n[0], 1.0 - theta, theta, sz);
			jac->J(&jac1[0], &z1[0], &tmp1[0]);
			vector_sum1(&G[0], &G[0], &jac1[0], 1.0, alpha * alpha, sz);

			// alpha^2 J(0.5(u1+u1), z2))
			vector_sum1(&tmp1[0], &u1[0], &u1_n[0], 1.0 - theta, theta, sz);
			jac->J(&jac1[0], &tmp1[0], &z2[0]);
			vector_sum1(&G[0], &G[0], &jac1[0], 1.0, alpha * alpha, sz);

			vector_sum(&F[0], &F[0], &FC[0], sz);
			vector_sum(&G[0], &G[0], &GC[0], sz);
			memset(&F[0], 0, n_la * sizeof(double));
			memset(&G[0], 0, n_la * sizeof(double));

			lapl->baroclin_1(&w1_n[0], &w2_n[0], &u1_n1[0], &u2_n1[0],
				&F[0], &G[0], &Z[0], &Z[0], 

				-mu * theta, 1.0 / tau + 0.5 * theta * sigma,
				-mu * theta, 1.0 / tau + 0.5 * theta * sigma + 
				 alpha * alpha * theta * mu1,

				-1.0, 0.0,
				-1.0, 0.0,

				-0.5 * sigma * theta,
				 0.5 * sigma * theta,
				-alpha * alpha / tau - alpha * alpha * theta * sigma1,
				1.0,
				1.0
				);

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
	void L_1_step_im(double * u11, double * u21, 
		const double * u1, const double * u2,
		const double * z1, const double * z2)
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
		//  J(z1, 0.5(w1+w1)) + J(u1, L(z1)+l+h) +
		// + J(z2, 0.5(w2+w2)) + J(0.5(u2+u2), L(z2)) +
		// + w1/tau + 0.5 theta sigma (w1-w2) - mu theta (L w1)
		// ������ ����� 2:
		//  J(z1, 0.5(w2+w2)) + J(0.5(u1+u1), L(z2)) +
		// + J(0.5(u2+u2), L(z1)+l+h) - J(z2, 0.5(w1+w1)) +
		// + 0.5 theta sigma (w1 + w2) - theta mu L w2 +
		// + w2/tau - alpha^2 u2/tau 
		// - alpha^2 J(z1, 0.5(u2+u2)) - alpha^2 J(0.5(u1+u1), z2) +
		// + alpha^2 theta mu1 w2 -
		// - alpha^2 sigma1 theta u2 

		vector < double > w1(sz);
		vector < double > w2(sz);
		vector < double > dw1(sz);
		vector < double > dw2(sz);
		vector < double > dz1(sz);
		vector < double > dz2(sz);
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

		lapl->lapl(&dz1[0], &z1[0]); memset(&dz1[0], 0, n_la * sizeof(double));
		lapl->lapl(&dz2[0], &z2[0]); memset(&dz2[0], 0, n_la * sizeof(double));

		// w1/tau + 0.5 theta sigma(w1-w2) - mu theta(L w1)
		vector_sum1(&FC[0], &w1[0], &w2[0], 
			0.5 * theta * sigma, -0.5 * theta * sigma, sz);
		vector_sum1(&FC[0], &FC[0], &dw1[0], 1.0, -mu * theta, sz);
		vector_sum1(&FC[0], &FC[0], &w1[0], 1.0, 1.0 / tau, sz);

		// w2/tau + 0.5 theta sigma (w1 + w2) - theta mu L w2 -
		// - alpha^2 u2/tau + alpha^2 theta mu1 w2 - alpha^2 sigma1 theta u2
		vector_sum1(&GC[0], &w1[0], &w2[0],
			0.5 * theta * sigma, 0.5 * theta * sigma, sz);
		vector_sum1(&GC[0], &GC[0], &dw2[0], 1.0, -mu * theta, sz);
		vector_sum1(&GC[0], &GC[0], &w2[0], 1.0, 1.0 / tau, sz);
		vector_sum1(&GC[0], &GC[0], &u2[0], 1.0, -alpha * alpha / tau, sz);
		vector_sum1(&GC[0], &GC[0], &w2[0], 1.0, alpha * alpha * mu1 * theta, sz);
		vector_sum1(&GC[0], &GC[0], &u2[0], 1.0, -alpha * alpha * sigma1 * theta, sz);

		memcpy(&u1_n[0], &u1[0], sz * sizeof(double));
		memcpy(&u2_n[0], &u2[0], sz * sizeof(double));
		memcpy(&w1_n[0], &w1[0], sz * sizeof(double));
		memcpy(&w2_n[0], &w2[0], sz * sizeof(double));

		for (int it = 0; it < 10; ++it) {
			//  J(0.5(u1+u1), L(z1)+l+h) + J(z1, 0.5(w1+w1)) +
			// + J(z2,0.5(w2+w2)) + J(0.5(u2+u2),L(z2))

			// J(0.5(u1+u1), L(z1)+l+h)
			vector_sum1(&tmp1[0], &u1[0], &u1_n[0], theta, 1.0 - theta, sz);
			vector_sum(&tmp2[0], &dz1[0], &cor[0], sz);
			jac->J(&jac1[0], &tmp1[0], &tmp2[0]);

			// J(z1, 0.5(w1+w1))
			vector_sum1(&tmp1[0], &w1[0], &w1_n[0], theta, 1.0 - theta, sz);
			jac->J(&jac2[0], &z1[0], &tmp1[0]);
			vector_sum(&F[0], &jac1[0], &jac2[0], sz);

			// J(z2,0.5(w2+w2))
			vector_sum1(&tmp1[0], &w2[0], &w2_n[0], theta, 1.0 - theta, sz);
			jac->J(&jac1[0], &z2[0], &tmp1[0]);
			vector_sum(&F[0], &F[0], &jac1[0], sz);

			// J(0.5(u2+u2),L(z2))
			vector_sum1(&tmp1[0], &u2[0], &u2_n[0], theta, 1.0 - theta, sz);
			jac->J(&jac1[0], &tmp1[0], &dz2[0]);
			vector_sum(&F[0], &F[0], &jac1[0], sz);

			//  J(z1, 0.5(w2+w2)) + J(0.5(u1+u1), L(z2)) +
			// + J(0.5(u2+u2), L(z1)+l+h) + J(z2, 0.5(w1+w1)) -
			// - alpha^2 J(z1, 0.5(u2+u2)) - alpha^2 J(0.5(u1+u1), z2))

			// J(z1, 0.5(w2+w2))
			vector_sum1(&tmp1[0], &w2[0], &w2_n[0], theta, 1.0 - theta, sz);
			jac->J(&jac1[0], &dz1[0], &tmp1[0]);

			// J(0.5(u1+u1), L(z2))
			vector_sum1(&tmp1[0], &u1[0], &u1_n[0], theta, 1.0 - theta, sz);
			jac->J(&jac2[0], &dz2[0], &tmp1[0]);
			vector_sum(&G[0], &jac1[0], &jac2[0], sz);

			// J(0.5(u2+u2), L(z1)+l+h)
			vector_sum1(&tmp1[0], &u2[0], &u2_n[0], theta, 1.0 - theta, sz);
			vector_sum(&tmp2[0], &dz1[0], &cor[0], sz);
			jac->J(&jac1[0], &tmp1[0], &tmp2[0]);
			vector_sum(&G[0], &G[0], &jac1[0], sz);

			// J(z2, 0.5(w1+w1))
			vector_sum1(&tmp1[0], &w1[0], &w1_n[0], theta, 1.0 - theta, sz);
			jac->J(&jac1[0], &z2[0], &tmp1[0]);
			vector_sum(&G[0], &G[0], &jac1[0], sz);

			// alpha^2 J(z1, 0.5(u2+u2))
			vector_sum1(&tmp1[0], &u2[0], &u2_n[0], theta, 1.0 - theta, sz);
			jac->J(&jac1[0], &z1[0], &tmp1[0]);
			vector_sum1(&G[0], &G[0], &jac1[0], 1.0, -alpha * alpha, sz);

			// alpha^2 J(0.5(u1+u1), z2))
			vector_sum1(&tmp1[0], &u1[0], &u1_n[0], theta, 1.0 - theta, sz);
			jac->J(&jac1[0], &tmp1[0], &z2[0]);
			vector_sum1(&G[0], &G[0], &jac1[0], 1.0, -alpha * alpha, sz);

			vector_sum(&F[0], &F[0], &FC[0], sz);
			vector_sum(&G[0], &G[0], &GC[0], sz);

			memset(&F[0], 0, n_la * sizeof(double));
			memset(&G[0], 0, n_la * sizeof(double));

			lapl->baroclin_1(&w1_n[0], &w2_n[0], &u1_n1[0], &u2_n1[0],
				&F[0], &G[0], &Z[0], &Z[0], 

				mu * (1-theta), 1.0 / tau - 0.5 * (1.0-theta) * sigma,
				mu * (1-theta), 1.0 / tau - 0.5 * (1.0-theta) * sigma - alpha * alpha * (1.0-theta) * mu1,

				-1.0, 0.0,
				-1.0, 0.0,

				 0.5 * sigma * (1.0-theta),
				-0.5 * sigma * (1.0-theta),
				-alpha * alpha / tau + alpha * alpha * (1.0-theta) * sigma1,
				1.0,
				1.0
				);

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
