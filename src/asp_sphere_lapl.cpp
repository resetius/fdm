/*$Id$*/

/* Copyright (c) 2003, 2004, 2005, 2007 Alexey Ozeritsky
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
 * Оператор Лапласа на сфере и полусфере
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sstream>
#include <vector>

#ifndef _WIN32
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/wait.h>
#endif

#include "asp_excs.h"
#include "asp_macros.h"
#include "asp_misc.h"
#include "asp_sphere_lapl.h"
#include "asp_fft.h"
#include "asp_gauss.h"

#ifndef min
#define min(a, b)  (((a) < (b)) ? (a) : (b))
#endif

#ifndef max
#define max(a, b)  (((a) > (b)) ? (a) : (b))
#endif

//#define _SLAPL_DEBUG //включает запись в файлы промежуточных результатов - медленно работает

//макросы для доступа к значениям собственных функций
#define _VM(i, j) VM[ ( i ) * n_la + ( j ) ]

//макросы для нахождения отступа в массиве для точки пространства
#define  pOff(i, j) ( i ) * n_la + ( j )
#define _pOff(i, j) ( i ) * d->n_la + ( j )
#define SQRT_M_1_PI 0.56418958354775629

using namespace std;
using namespace asp;

/*чтобы постоянно не выделять память*/
class conv_diags {
public:
	double* A;
	double* B;
	double* C;
	double* RP;
	conv_diags(int n) {
		A = new double[n];/*                     */
		B = new double[n];/*    diags            */
		C = new double[n];/*                     */
		RP = new double[n];/*    right part      */
	}

	~conv_diags() {
		delete [] A;
		delete [] B;
		delete [] C;
		delete [] RP;
	}
};

#ifdef _WIN32
static double get_time()
{
	return (double) time(0);
}

static double get_full_time()
{
	return (double) time(0);
}
#else
static double get_time()
{
	struct rusage buf;
	getrusage(RUSAGE_SELF, &buf);

	return buf.ru_utime.tv_sec*100.0+buf.ru_utime.tv_usec/10000.0;
}

static double get_full_time()
{
	struct timeval buf;
	gettimeofday (&buf,0);

	return buf.tv_sec*100.0+buf.tv_usec/10000.0;
}
#endif

class SLaplacian::Private: public SSteps
{
public:
	typedef SLaplacian::Private self;

	//int fft; //!<степень для быстрого преобразования Фурье
	fft * ft;

	double *B1;
    /**
     * предварительное вычисление часто используемых данных
     * в разы ускоряет счет, но памяти ест много
     */
    double* LM;  //!<собственные значения сферического лапласа
    double* VM;  //!<значения собственных функций сферического лапласа
	double* COS; //!<значения косинусов по широте

	double *Fc; //!<коэф-та фильтрации

	bool use_fft;

	vector < conv_diags * > d1;
	double * MF;

	Private(int _n_phi, int _n_la, bool _full)
		:  SSteps(_n_phi, _n_la, _full),
		ft(0), B1(0), LM(0), VM(0),
		COS(0), Fc(0), MF(0)
	{
		checkFFT();

		COS = new double[2 * (n_phi + 1)];

		LM  = new double[n_la];
		for (int i = 0; i < n_la; i++) {
			LM[i] = L_m(i);
		}

		double phi;
		//double sum = 0;
		for (int i = 0; i < n_phi; i++) {
			if (full)
				phi =(d_phi - M_PI) * 0.5 + (double)i * d_phi;
			else
				phi = (double)i * d_phi;
			COS[2 * i]     = cos(phi);
			//sum           += COS[2 * i];
			COS[2 * i + 1] = cos(phi + 0.5 * d_phi);
		}

		//printf("sum=%lf\n", sum);

		if (!use_fft) {
			VM = new double[n_la * n_la];

			for (int m = 0; m < n_la; m++) {
				for (int j = 0; j < n_la; j++) {
					_VM(m, j) = V_m(m, j);
				}
			}
		}

		MF = new double[n_phi * n_la];
		d1.push_back(new conv_diags(n_phi));
		init_filter();
	}

	void init_filter() {
		//double p   = 1.0 / cos(69.0 / 180.0 * M_PI);
		//double p   = 1.0 / cos(5.0 / 180.0 * M_PI);
		//double p   = 1.0 / cos(69.0 / 180.0 * M_PI);
		double p   = 1.0 / cos(36.0 / 180.0 * M_PI);
		//double p = 1.0;
		//double p = d_la / d_phi;

		Fc = new double[n_phi * n_la];

		for (int i = 0; i < n_phi; ++i) {
			//?
			double c = cos(phi(i));
			for (int m = 1; m < n_la; ++m) {
				double l = c * p / sin(la(m) * 0.5);
				Fc[m * n_phi + i] = min(1.0, l);
			}
		}
	}

	double phi(int i ){
		if (full) {
			return (d_phi - M_PI) * 0.5 + (double)i * d_phi;
		} else {
			return (double)i * d_phi;
		}
	}

	double la(int j) {
		return (double)j * d_la;
	}

	~Private()
	{
		delete [] LM;
		delete [] VM;
		delete [] B1;
		delete [] COS;
		if (ft) FFT_free(ft);
		delete [] MF;
		delete [] Fc;

		for (vector < conv_diags * >::iterator it  = d1.begin();
			it != d1.end(); ++it)
		{
			delete *it;
		}
	}

	/*!определение возможности использования быстрого преобразования
	   Фурье*/
	void checkFFT()
	{
//#ifndef _FFTW
		use_fft = ft = FFT_init(FFT_PERIODIC, n_la);
//#endif
	}

	/*! собственные значения сферического лапласа
	 * Самарский-Николаев, страница 69, формула 30
	 * \param m
	 * \return
	 */
	double L_m(int m) {
		_DEBUG_ASSERT_RANGE(0, n_la - 1, m);

		return 4. / d_la2 * ipow(sin((double) m * d_la * 0.5), 2);
	}

	/*! собственные функции сферического лапласа
	 * Самарский-Николаев, страница 71, формула 33
	 * \param m
	 * \param i
	 * \return
	 */
	double V_m(int m, int i) {
		_DEBUG_ASSERT_RANGE(0, n_la - 1, m);

		if (m == 0 || m == n_la / 2)
			return sqrt(0.5 * M_1_PI) *
			cos(2. * (double) m * M_PI *
			(double) i / (double) (n_la));
		else if (1 <= m && m <= n_la / 2 - 1)
			return sqrt(M_1_PI) *
			cos(2. * (double) m *
			M_PI * (double) i / (double) (n_la));
		else //if (n_la / 2 + 1 <= m && m <= n_la - 1)
			return sqrt(M_1_PI) *
			sin(2. * (double) (n_la - m) *
			M_PI * (double) i / (double) (n_la));
	}

    /*! перевод разложения по C[m][i] в сеточное разложение
     * \param U
     * \param X
     * \param Y
     */
    void XYtoU(double* U, const double* X)
	{
		if (use_fft) {
			double *s = new double[n_la];
			for (int i = 0; i < n_phi; ++i) {
				for (int m = n_la - 1; m >= 0; --m) {
					s[m] = X[m * n_phi + i];
				}
#ifndef _FFTW
				pFFT_2(&U[pOff(i, 0)], s, SQRT_M_1_PI, ft);
#else
				pFFT_fftw(&U[pOff(i, 0)], s, SQRT_M_1_PI, n_la);
#endif
			}
			delete [] s;
		} else {
			for (int i = 0; i < n_phi; i++) {
				for (int j = 0; j < n_la; j++) {
					double sum = 0.0;
					double vm;
					for (int m = 0; m < n_la; m++) {
						vm = _VM(m, j);
						sum += X[m * n_phi + i] * vm;
						//sum += X[m * n_phi + i] * _VM(m, j);
					}
					U[pOff(i, j)] = sum;
				}
			}
		}
		/*if (!full) {
			//на экваторе ноль
			for (int j = 0; j < n_la; j++) {
				U[pOff(0, j)] = 0.0;
			}
		}*/
	}

	/*! перевод сеточного разложения в разложение по C[m][i]
	 *  находит коеффициенты Фурье
     * \param X
     * \param Y
     * \param U
     */
	void UtoXY(double *X, const double *U)
	{
		double sum;

		if (use_fft) {
			double * s = new double[n_la];
			double dx  = d_la * SQRT_M_1_PI;
			for (int i = n_phi - 1; i >= 0; --i) {
//#ifndef _FFTW
				pFFT_2_1(s, &U[pOff(i, 0)], dx, ft);
//#else
//not checked
//				pFFT_1_fftw(s, &U[pOff(i, 0)], dx, n_la);
//#endif
				for (int m = n_la - 1; m >= 0; --m) {
					X[m * n_phi + i] = s[m];
				}
			}
			delete [] s;
		} else {
			for (int i = 0; i < n_phi; i++) {
				for (int m = 0 ; m < n_la; m++) {
					sum = 0.0;
					for (int j = 0; j < n_la; j++) {
						sum += U[pOff(i, j)] * _VM(m, j);
					}
					sum *= d_la;
					X[m * n_phi + i] = sum;
				}
			}
		}
	}

	/*!компонента по lambda*/
	double sphere_laplacian_lambda_ij(const double * M, int i, int j)
	{
		_DEBUG_ASSERT_RANGE(0, n_phi - 1, i);
		_DEBUG_ASSERT_RANGE(0, n_la - 1, j);

		double pt2 = 0.0;

		double rho_1 = 1.0 / COS[2 * i];
		double d_la2_rho2_1;

		//часть по lambda
		{
			d_la2_rho2_1 = d_la2_1 * rho_1 * rho_1;
			/* lambda кордината циклическая, надо рассмотреть условия склейки */
			pt2 += (M[pOff(i, (j + 1) % n_la)]
					- 2. * M[pOff(i, j)] +
					M[pOff(i, (n_la + j - 1) % n_la)]);
			pt2 *= d_la2_rho2_1;
		}
		return pt2;
	}

	/*!компонента по phi*/
	double sphere_laplacian_phi_ij(const double * M, int i, int j) {
		_DEBUG_ASSERT_RANGE(0, n_phi - 1, i);
		_DEBUG_ASSERT_RANGE(0, n_la - 1, j);

		/*результат*/
		double pt1 = 0.0;

		double rho_1 = 1.0 / COS[2 * i];
		//double rho_1 = 1.0;

		/* выделяем отдельно концевые случаи */
		if (0 < i && i < n_phi - 1) {
			pt1  = COS[2 * i + 1] * (M[pOff(i + 1, j)] - M[pOff(i, j)]);
			pt1 -= COS[2 * i - 1] * (M[pOff(i, j)] - M[pOff(i - 1, j)]);
			pt1 *= d_phi2_1 * rho_1;
		} else if (i == n_phi - 1) {
			pt1  = -COS[2 * i - 1] * (M[pOff(i, j)] - M[pOff(i - 1, j)]);
			pt1 *= d_phi2_1 * rho_1;
		} else if (i == 0 && full) {
			pt1  = COS[2 * i + 1] * (M[pOff(i + 1, j)] - M[pOff(i, j)]);
			pt1 *= d_phi2_1 * rho_1;
		} else {
			//pt1 = pt2 = 0.0;
			pt1  = COS[2 * i + 1] * (M[pOff(i + 1, j)] - M[pOff(i, j)]);
			pt1 -= cos(0 - 0.5 * d_phi)/*COS[2 * i - 1]*/ * (M[pOff(i, j)]);
			pt1 *= d_phi2_1 * rho_1;
		}

		return pt1;
	}

	/*! находит компоненту i,j сферического лапласиана
	 * Самарский-Николаев, стр 569 (полярная система координат)
	 * \param M матрица значений на сфере (хранение M[i * n_la + j])
	 * \param i широта
	 * \param j долгота
	 * \return (i, j) компонента лапласиана
	 */
	double sphere_laplacian_ij(const double * M, int i, int j) {
		_DEBUG_ASSERT_RANGE(0, n_phi - 1, i);
		_DEBUG_ASSERT_RANGE(0, n_la - 1, j);

		/*результат*/
		double delta = 0;
		double pt1 = 0.0, pt2 = 0.0;

		double rho_1 = 1.0 / COS[2 * i];
		double d_la2_rho2_1;

		//часть по lambda
		{
			d_la2_rho2_1 = d_la2_1 * rho_1 * rho_1;
			/* lambda кордината циклическая, надо рассмотреть условия склейки */
			pt2 += (M[pOff(i, (j + 1) % n_la)]
					- 2. * M[pOff(i, j)] +
					M[pOff(i, (n_la + j - 1) % n_la)]);
			pt2 *= d_la2_rho2_1;
		}

		/* выделяем отдельно концевые случаи */
		if (0 < i && i < n_phi - 1) {
			pt1  = COS[2 * i + 1] * (M[pOff(i + 1, j)] - M[pOff(i, j)]);
			pt1 -= COS[2 * i - 1] * (M[pOff(i, j)] - M[pOff(i - 1, j)]);
			pt1 *= d_phi2_1 * rho_1;
		} else if (i == n_phi - 1) {
			pt1  = -COS[2 * i - 1] * (M[pOff(i, j)] - M[pOff(i - 1, j)]);
			pt1 *= d_phi2_1 * rho_1;
		} else if (i == 0 && full) {
			pt1  = COS[2 * i + 1] * (M[pOff(i + 1, j)] - M[pOff(i, j)]);
			pt1 *= d_phi2_1 * rho_1;
		} else {
			//pt1 = pt2 = 0.0;
			pt1  = COS[2 * i + 1] * (M[pOff(i + 1, j)] - M[pOff(i, j)]);
			pt1 -= cos(0 - 0.5 * d_phi)/*COS[2 * i - 1]*/ * (M[pOff(i, j)]);
			pt1 *= d_phi2_1 * rho_1;
		}

		delta = pt1 + pt2;
		return delta;
	}

	/*! считает матрицу лапласиана
	 * \param M2 результат
	 * \param M1 исходная значение
	 */
	void sphere_laplacian_matrix(double* M2, const double* M1, int rank = 0, int threads = 1)
	{
		const double * S = M1;

		if (rank == 0)
		if (M1 == M2) {
			if (!B1) {B1 = new double[n_phi*n_la];}
			memcpy(B1, M1, n_phi*n_la * sizeof(double));
			S = B1;
		}

		for (int i = n_phi - 1; i >= 0; --i) {
			for (int j = n_la - 1; j >= 0; --j) {
				M2[pOff(i, j)] = sphere_laplacian_ij(S, i, j);
			}
		}
	}

	/*!компонента по lambda*/
	void sphere_laplacian_lambda_matrix(double* M2, const double* M1) {
		const double * S = M1;
		if (M1 == M2) {
			if (!B1) {B1 = new double[n_phi*n_la]; NOMEM(B1);}
			memcpy(B1, M1, n_phi*n_la * sizeof(double));
			S = B1;
		}

		for (int i = n_phi - 1; i >= 0; --i) {
			for (int j = n_la - 1; j >= 0; --j) {
				M2[pOff(i, j)] = sphere_laplacian_lambda_ij(S, i, j);
			}
		}
	}

	/*!компонента по phi*/
	void sphere_laplacian_phi_matrix(double* M2, const double* M1) {
		const double * S = M1;
		if (M1 == M2) {
			if (!B1) {B1 = new double[n_phi*n_la]; NOMEM(B1);}
			memcpy(B1, M1, n_phi*n_la * sizeof(double));
			S = B1;
		}

		for (int i = n_phi - 1; i >= 0; --i) {
			for (int j = n_la - 1; j >= 0; --j) {
				M2[pOff(i, j)] = sphere_laplacian_phi_ij(S, i, j);
			}
		}
	}

	//!переносит краевые условия в правую часть
	void prepare_right_part(double * M, double * mult_factor,
		bool flag, int bc)
	{
		int i = 1;
		double k = (flag) ? mult_factor[0] : mult_factor[1];
		double rho_1 = 1.0 / COS[2 * i] * COS[2 * i - 1];

		if (bc == BC_DIRICHLET) {
			for (int j = n_la - 1; j >= 0; --j) {
				M[pOff(1, j)] -= k * rho_1 * d_phi2_1 * M[pOff(0, j)];
			}
		} else {
			for (int j = n_la - 1; j >= 0; --j) {
				M[pOff(1, j)] -= k * rho_1 * d_phi_1 * M[pOff(0, j)];
			}
		}
	}

	/*! обратный лаплас на сфере
	 * \param M           начальное условие и ответ, на краях значения краевого условия
	 * \param mult_factor параметры для подправленного лапласа
	 * \param diag
	 * \param flag        размерность множителя (1 или вектор)
	 * \param bc          граничные условия
	 */
	void conv_sphere_laplace(double* M,
		double * mult_factor,
		double * diag, bool flag, int bc,
		int rank = 0, int threads = 1)
	{
		int i, m;
		double rho_1;
		double d_phi2_rho_1;
		double cos_phi1_d_phi2_rho;
		double cos_phi2_d_phi2_rho;

		conv_diags * d2 = d1[rank];

//		if (d1 == 0 && rank == 0) d1 = new conv_diags(n_phi);
//		if (MF == 0 && rank == 0) MF = new double[n_phi * n_la];

		if (!full && rank == 0) {
			//переносит краевые условия в правую часть
			prepare_right_part(M, mult_factor, flag, bc);
		}

		UtoXY(MF, M);

#ifdef _SLAPL_DEBUG
		if (full) {
			_fprintfwmatrix("out/M1_full.out", M, n_phi, n_la,
				max(n_phi, n_la), "%23.16le ");
			_fprintfwmatrix("out/MF1_full.out", MF, n_la, n_phi,
				max(n_phi, n_la), "%23.16le ");
		} else {
			_fprintfwmatrix("out/M1_half.out", M, n_phi, n_la,
				max(n_phi, n_la), "%23.16le ");
			_fprintfwmatrix("out/MF1_half.out", MF, n_la, n_phi,
				max(n_phi, n_la), "%23.16le ");
		}
#endif
//#ifdef _DEBUG
//		if (full)
//			si = n_phi / 2 + 2;
//#endif
		//на полусфере в нулевой точке ничего не находим - там краевое условие
		int si = (full) ? 0 : 1;

		for (m = 0; m < n_la; m++) {
			for (i = si + 1; i < n_phi - 1; i++) {
				rho_1 = 1.0 / COS[2 * i];
				d_phi2_rho_1 = d_phi2_1 * rho_1;
				cos_phi1_d_phi2_rho = COS[2 * i - 1] * d_phi2_rho_1;
				cos_phi2_d_phi2_rho = COS[2 * i + 1] * d_phi2_rho_1;


				d2->A[i - 1] = cos_phi1_d_phi2_rho;
				d2->B[i]     = -cos_phi2_d_phi2_rho
					           -cos_phi1_d_phi2_rho
					           -LM[m] * rho_1 * rho_1;
				d2->C[i]     = cos_phi2_d_phi2_rho;
				d2->RP[i]    = MF[m * n_phi + i];
			}

			i = si;
			rho_1 = 1.0 / COS[2 * i];
			d_phi2_rho_1 = d_phi2_1 * rho_1; //1/phi/phi/rho
			d2->RP[i] = MF[m * n_phi + i];

			if (full) {
				d2->B[i] = (-LM[m] * rho_1 * rho_1 -
					//cos(phi(i) - 0.5 * d_phi) * d_phi2_rho_1-
				           COS[2 * i + 1] * d_phi2_rho_1);
				d2->C[i] = COS[2 * i + 1] * d_phi2_rho_1;
			} else {
				d2->B[i] = -LM[m] * rho_1 * rho_1 -
				           COS[2 * i + 1] * d_phi2_rho_1 -
						   COS[2 * i - 1] * d_phi2_rho_1;
						   //cos(0 - 0.5 * d_phi) * d_phi2_rho_1;
				d2->C[i] = COS[2 * i + 1] * d_phi2_rho_1;
				//d1->B[i]  = 1.0;
				//d1->C[i]  = 0.0;
			}

			i = (n_phi - 1);
			rho_1 = 1.0 / COS[2 * i];
			d_phi2_rho_1 = d_phi2_1 * rho_1;

			d2->A[n_phi - 2] = COS[2 * i - 1] * d_phi2_rho_1;
			d2->B[n_phi - 1] = (-LM[m] * rho_1 * rho_1 -
				               COS[2 * i - 1] * d_phi2_rho_1);
			d2->RP[i] = MF[m * n_phi + i];

			/*для неявной схемы меняем диагональный элемент*/
			if (flag) {
				for (i = 0; i < n_phi; ++i) {
					d2->B[i] *= *mult_factor;
					d2->B[i] += *diag;
				}
				for (i = 0; i < n_phi - 1; ++i) {
					d2->A[i] *= *mult_factor;
					d2->C[i] *= *mult_factor;
				}
			} else {
				for (i = 0; i < n_phi; ++i) {
					d2->B[i] *= mult_factor[i];
					d2->B[i] += diag[i];
				}
				for (i = 0; i < n_phi - 1; ++i) {
					d2->A[i] *= mult_factor[i];
					d2->C[i] *= mult_factor[i];
				}
			}

			if (!full && bc == BC_NEUMANN) {
				d2->B[0] *= 0.5;
				//d1->B[0] += d_phi2_1; //?
			}

#ifdef _SLAPL_DEBUG
			{
				ostringstream str;
				FILE * f;
				string suf = (full)?"full":"half";
				str << "out/conv_" << m << "_" << suf << ".out";
				f = fopen(str.str().c_str(), "w");
				fprintf(f, "RP:\n");
				fprintfvector(f, d1->RP, n_phi, n_phi, "%.10le ");
				fprintf(f, "\nA:\n");
				fprintfvector(f, d1->A, n_phi, n_phi, "%.10le ");
				fprintf(f, "\nB:\n");
				fprintfvector(f, d1->B, n_phi, n_phi, "%.10le ");
				fprintf(f, "\nC:\n");
				fprintfvector(f, d1->C, n_phi, n_phi, "%.10le ");
				fclose(f);
			}
#endif
			if (full) {
				//int s = n_phi/2-1;
				int s = 0;
				solve_tdiag_linear(&d2->RP[s], &d2->A[s], &d2->B[s], &d2->C[s], n_phi - s);
			} else {
				int s = si;
				solve_tdiag_linear(&d2->RP[s], &d2->A[s], &d2->B[s], &d2->C[s], n_phi - s);
				//solve_tdiag_linear(d1->RP, d1->A, d1->B, d1->C, n_phi - si);
			}

			memcpy(&MF[m * n_phi + si], &d2->RP[si], (n_phi - si) * sizeof(double));
//			for (i = si; i < n_phi; i++) {
//				MF[m * n_phi + i] = d1->RP[i];
//			}
		}

		if (full && rank == 0) {
			//ортогонализуем единице
			memset(&MF[0], 0, n_phi * sizeof(double));
		}

		XYtoU(M, MF);

		//if (!full && bc == BC_NEUMANN) {
		//	double k = (flag) ? mult_factor[0] : mult_factor[1];

		//	for (int j = n_la - 1; j >= 0; --j) {
		//		M[pOff(0, j)] = rho * d_phi * M[pOff(0, j)] + M[pOff(1, j)] / k;
		//	}
		//}

#ifdef _SLAPL_DEBUG
		if (full) {
			_fprintfwmatrix("out/M2_full.out", M, n_phi, n_la,
				max(n_phi, n_la), "%23.16le ");
			_fprintfwmatrix("out/MF2_full.out", MF, n_la, n_phi,
				max(n_phi, n_la), "%23.16le ");
		} else {
			_fprintfwmatrix("out/M2_half.out", M, n_phi, n_la,
				max(n_phi, n_la), "%23.16le ");
			_fprintfwmatrix("out/MF2_half.out", MF, n_la, n_phi,
				max(n_phi, n_la), "%23.16le ");
		}
#endif
	}


	void add_diag_submatrix_to_matrix(double * A, double * RP, int n, double * F, 
		double lm, int off, 
		double mult, double diag)
	{
		int i = 0, j = 0, i1 = 0;
		double rho_1;
		double d_phi2_rho_1;
		double cos_phi1_d_phi2_rho;
		double cos_phi2_d_phi2_rho;

		for (i = off + 2; i < off + n_phi - 1; i++)
		{
			j  = i - 1;
			i1 = i - off;
			rho_1 = 1.0 / COS[2 * i1];
			d_phi2_rho_1 = d_phi2_1 * rho_1;
			cos_phi1_d_phi2_rho = COS[2 * i1 - 1] * d_phi2_rho_1;
			cos_phi2_d_phi2_rho = COS[2 * i1 + 1] * d_phi2_rho_1;

			//down
			A[j * n + j-1] =  mult * cos_phi1_d_phi2_rho;
			//middle
			A[j * n + j]   = (- cos_phi2_d_phi2_rho
				           - cos_phi1_d_phi2_rho
				           - lm * rho_1 * rho_1) * mult
				           + diag;
			//up
			A[j * n + j+1]  = mult * cos_phi2_d_phi2_rho;
			RP[j]           = F[i1];
		}

		i = off + 1; j = i - 1; i1 = i - off;
		rho_1 = 1.0 / COS[2 * i1];
		d_phi2_rho_1 = d_phi2_1 * rho_1;
		RP[j] = F[i1];

		A[j * n + j] = mult * (-lm * rho_1 * rho_1 -
			       COS[2 * i1 + 1] * d_phi2_rho_1 -
			       COS[2 * i1 - 1] * d_phi2_rho_1) + diag;
		A[j * n + j + 1] = mult * COS[2 * i1 + 1] * d_phi2_rho_1;

		i = (n_phi - 1) + off; j = i - 1; i1 = i - off;

		rho_1 = 1.0 / COS[2 * i1];
		d_phi2_rho_1 = d_phi2_1 * rho_1;

		A[j * n + j - 1] = mult * COS[2 * i1 - 1] * d_phi2_rho_1;
		A[j * n + j] = mult * (-lm * rho_1 * rho_1 -
			               COS[2 * i1 - 1] * d_phi2_rho_1) + diag;
		RP[j] = F[i1];
	}

	void add_diag_to_matrix(double * A, int n, 
		int x_off, int y_off, double diag)
	{
		int i = 0;
		for (i = 0; i < n_phi - 1; i++)
		{
			A[(i + y_off) * n + i + x_off] = diag;
		}
	}

	void conv_baroclin(
		double * oW1, double * oW2,
		double * oU1, double * oU2,

		const double * W1, const double * W2,
		const double * U1, const double * U2,

		double mult1,     double diag1,
		double mult2,     double diag2,
		double mult3,     double diag3,
		double mult4,     double diag4,

		double diag_w2,   
		double diag_w1,
		double diag_u2,
		double diag_w1_2, 
		double diag_w2_2)
	{
		int m;
		int n = n_phi - 1;

		vector < double > FW1(n_la * n_phi);
		vector < double > FW2(n_la * n_phi);

		vector < double > FU1(n_la * n_phi);
		vector < double > FU2(n_la * n_phi);

		vector < double > A(4 * n * 4 * n);
		vector < double > RP(4 * n);
		vector < double > X(4 * n);

		if (W1) UtoXY(&FW1[0], W1);
		if (W2) UtoXY(&FW2[0], W2);
		if (U1) UtoXY(&FU1[0], U1);
		if (U2) UtoXY(&FU2[0], U2);

		for (m = 0; m < n_la; m++) 
		{
			// n_phi-1 x n_phi-1 laplacian submatrix
			double * fw1 = &FW1[m * n_phi];
			double * fw2 = &FW2[m * n_phi];
			double * fu1 = &FU1[m * n_phi];
			double * fu2 = &FU2[m * n_phi];

			add_diag_submatrix_to_matrix(&A[0], &RP[0], 4 * n, fw1, LM[m], 0,     mult1, diag1);
			add_diag_submatrix_to_matrix(&A[0], &RP[0], 4 * n, fw2, LM[m], n,     mult2, diag2);
			add_diag_submatrix_to_matrix(&A[0], &RP[0], 4 * n, fu1, LM[m], 2 * n, mult3, diag3);
			add_diag_submatrix_to_matrix(&A[0], &RP[0], 4 * n, fu2, LM[m], 3 * n, mult4, diag4);

			add_diag_to_matrix(&A[0], 4 * n, n, 0,     diag_w2);
			add_diag_to_matrix(&A[0], 4 * n, 0, n,     diag_w1);
			add_diag_to_matrix(&A[0], 4 * n, 3 * n, n, diag_u2);
			add_diag_to_matrix(&A[0], 4 * n, 0, 2 * n, diag_w1_2);
			add_diag_to_matrix(&A[0], 4 * n, n, 3 * n, diag_w2_2);

			// тут надо решить систему уравнений A X = RP
			// извлечь из X четыре вектора и совершить обратное преобразование

			//_fprintfwmatrix("A.txt", &A[0], 4 * n, 4 * n, 4 * n, "%.1le ");
			gauss(&A[0], &RP[0], &X[0], 4 * n);

			memcpy(&FW1[m * n_phi + 1], &X[0],     n * sizeof(double));
			memcpy(&FW2[m * n_phi + 1], &X[n],     n * sizeof(double));
			memcpy(&FU1[m * n_phi + 1], &X[2 * n], n * sizeof(double));
			memcpy(&FU2[m * n_phi + 1], &X[3 * n], n * sizeof(double));
		}

		XYtoU(oW1, &FW1[0]); 
		memset(oW1, 0, n_la * sizeof(double));
		XYtoU(oW2, &FW2[0]); 
		memset(oW2, 0, n_la * sizeof(double));
		XYtoU(oU1, &FU1[0]); 
		memset(oU1, 0, n_la * sizeof(double));
		XYtoU(oU2, &FU2[0]); 
		memset(oU2, 0, n_la * sizeof(double));
	}
};

SLaplacian::~SLaplacian()
{
	delete d;
}

SLaplacian::SLaplacian(int n_phi, int n_la, bool full)
	: d(new Private(n_phi, n_la, full))
{
}

double SLaplacian::lapl(const double *M, int i, int j)
{
	return d->sphere_laplacian_ij(M, i, j);
}

void SLaplacian::lapl(double *Dest, const double *M)
{
	d->sphere_laplacian_matrix(Dest, M);
}

void SLaplacian::lapl_la(double *Dest, const double *M)
{
	d->sphere_laplacian_lambda_matrix(Dest, M);
}

void SLaplacian::lapl_phi(double *Dest, const double *M)
{
	d->sphere_laplacian_phi_matrix(Dest, M);
}

void SLaplacian::lapl_1(double * Dest, const double * Source, double mult, double diag, int bc)
{
	if (Dest != Source) memcpy(Dest, Source, d->n_la * d->n_phi * sizeof(double));

	d->conv_sphere_laplace(Dest, &mult, &diag, true, bc);
}

void SLaplacian::baroclin_1(
							double * oW1, double * oW2, 
							double * oU1, double * oU2,

							const double * W1, const double * W2, 
							const double * U1, const double * U2,

							double mult1, double diag1,
							double mult2, double diag2,
							double mult3, double diag3,
							double mult4, double diag4,

							double diag_w2, 
							double diag_w1,
							double diag_u2,
							double diag_w1_2, 
							double diag_w2_2)
{
	// нулевые граничные условия!
	d->conv_baroclin(oW1, oW2, oU1, oU2, 
		W1, W2, U1, U2, 
		mult1, diag1, mult2, diag2, mult3, diag3, mult4, diag4,
		diag_w2, diag_w1, diag_u2, diag_w1_2, diag_w2_2);
}

void SLaplacian::lapl_1(double * Dest, const double * Source, double * mult, double * diag, int bc)
{
	if (Dest != Source) memcpy(Dest, Source, d->n_la * d->n_phi * sizeof(double));

	d->conv_sphere_laplace(Dest, mult, diag, false, bc);
}

double SLaplacian::scalar(const double *u, const double *v)
{
	double rho;
	double sum = 0;
	for (int i = 0; i < d->n_phi; ++i) {
		rho = d->COS[2 * i];
		for (int j = 0; j < d->n_la; ++j) {
			sum += rho * u[_pOff(i, j)] * v[_pOff(i, j)];
		}
	}
	return sum;
}

void SLaplacian::filter(double *Dest, const double * Source)
{
	d->UtoXY(d->MF, Source);

	double * X = d->MF;
	for (int i = 0; i < d->n_phi; ++i) {
		for (int m = 1; m < d->n_la; ++m) {
			X[m * d->n_phi + i] *= d->Fc[m * d->n_phi + i];
		}
	}

	d->XYtoU(Dest, d->MF);
}

void SLaplacian::filter_1(double *Dest, const double * Source)
{
	d->UtoXY(d->MF, Source);

	double * X = d->MF;
	for (int i = 0; i < d->n_phi; ++i) {
		for (int m = 1; m < d->n_la; ++m) {
			X[m * d->n_phi + i] /= d->Fc[m * d->n_phi + i];
		}
	}

	d->XYtoU(Dest, d->MF);
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~Внутренние проверки ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void SLaplacian::check() {
	//check0(); //печать параметров
	//check1(); //проверка Lapl на известный ответ
	//check2(); //XYtoU/UtoXY
	//check3(); //самосопряженность
	check4(); //Lapl(Lapl_1)
}

void SLaplacian::check4()
{
	int nn = d->n_la * d->n_phi;
	double * C = new double[nn];
	double * A = new double[nn];
	double * B = new double[nn];


	printf("checking Lapl(Lapl_1)\n");
	random_initialize(A, nn);

	int n1 = d->n_phi;
	int n2 = d->n_la;

	double t1 = get_full_time();
	lapl(B, A);
	vector_mult_scalar(B, B, 10, nn);
	double t2 = get_full_time();

	printf("  Lapl Work time:%lf\n", (t2 - t1));

//	_fprintfwmatrix("out/check4_start.out", A, n1, n2, max(n1, n2), "%.3le ");
//	_fprintfwmatrix("out/check4_laplas.out", B, n1, n2, max(n1, n2), "%.3le ");

	t1 = get_full_time();
	//установка краевого условия
	memcpy(B, A, d->n_la * sizeof(double));
	//vector_mult_scalar(B, B, 10, d->n_la);
	lapl_1(B, B, 10);
	t2 = get_full_time();
	printf("  Lapl_1 Work time:%lf\n", (t2 - t1));

//	_fprintfwmatrix("out/check4_laplas_1.out", B, n1, n2, max(n1, n2), "%.3le ");
	printf("  |Lapl(Lapl_1)|=%.16le\n",dist1(B, A, nn));

	delete [] A; delete [] B; delete [] C;
}

/*!проверка на самосопряженность*/
void SLaplacian::check3() {
	int nn = d->n_la * d->n_phi;
	double * u  = new double[nn];
	double * u1 = new double[nn];
	double * v  = new double[nn];
	double * v1 = new double[nn];

	double nr1, nr2;

	printf("checking conjugacy ... \n");

	random_initialize(u, nn);
	random_initialize(v, nn);

	printf("  checking conjugacy of \\Delta_{\\lambda}... \n");

	lapl_la(u1, u);
	lapl_la(v1, v);
	nr1 = scalar(u1, v);
	nr2 = scalar(u, v1);
	printf("    (Lu,v)=%.16le\n", nr1);
	printf("    (u,LTv)=%.16le\n", nr2);
	printf("    |(Lu,v)-(u,LTv)|=%.16le\n", fabs(nr1 - nr2));

	printf("  checking conjugacy of \\Delta_{\\phi}... \n");

	lapl_phi(u1, u);
	lapl_phi(v1, v);
	nr1 = scalar(u1, v);
	nr2 = scalar(u, v1);
	printf("    (Lu,v)=%.16le\n", nr1);
	printf("    (u,LTv)=%.16le\n", nr2);
	printf("    |(Lu,v)-(u,LTv)|=%.16le\n", fabs(nr1 - nr2));

	printf("  checking conjugacy of full \\Delta... \n");

	lapl(u1, u);
	lapl(v1, v);
	nr1 = scalar(u1, v);
	nr2 = scalar(u, v1);
	printf("    (Lu,v)=%.16le\n", nr1);
	printf("    (u,LTv)=%.16le\n", nr2);
	printf("    |(Lu,v)-(u,LTv)|=%.16le\n", fabs(nr1 - nr2));

	delete [] u; delete [] u1; delete [] v; delete [] v1;
}

/*!проверка функций XYtoU/UtoXY на обратимость*/
void SLaplacian::check2() {
	int nn = d->n_la * d->n_phi;
	double * u = new double[nn];
	double * v = new double[nn];
	double * w = new double[nn];

	printf("checking XYtoU(UtoXY), UtoXY(XYtoU) ...\n");

	random_initialize(u, nn);
	d->UtoXY(v, u);
	d->XYtoU(w, v);

	printf("  |XYtoU(UtoXY)|=%le\n", dist1(u, w, nn));

	d->XYtoU(v, u);
	d->UtoXY(w, v);
	printf("  |UtoXY(XYtoU)|=%le\n", dist1(u, w, nn));

	delete [] u;
	delete [] w;
}

void SLaplacian::check_approx()
{
	check1();
}

void SLaplacian::check1()
{
	int nn = d->n_la * d->n_phi;
	double * u = new double[nn];
	double *uu = new double[nn];
	double * v = new double[nn];

	printf("checking lapl of known func ... \n");
	//static double * buf = 0;
	//if (buf == 0) {
	//	buf = new double[nn * 4];
	//	for (int i = 0; i < 4 * nn; ++i) {
	//		//buf[i] = (double)rand() /(double)RAND_MAX;
	//		buf[i] = 1;
	//	}
	//}

	//int k = 0;
	//if (d->full) {
	//for (int i = d->n_phi/2; i < d->n_phi; i++) {
	//	for (int j = 0; j < d->n_la; j++) {
	//		//u[_pOff(i, j)] = sin(d->la(j)) * sin(2.0 * d->phi(i));
	//		u[_pOff(i, j)] = buf[k++];
	//	}
	//}

	//for (int i = 0; i < d->n_phi/2; i++) {
	//	for (int j = 0; j < d->n_la; j++) {
	//		//u[_pOff(i, j)] = sin(d->la(j)) * sin(2.0 * d->phi(i));
	//		u[_pOff(i, j)] = buf[k++];
	//		//u[_pOff(i, j)] = 0; //k++;
	//	}
	//}

	//} else {
	//for (int i = 0; i < d->n_phi; i++) {
	//	for (int j = 0; j < d->n_la; j++) {
	//		//u[_pOff(i, j)] = sin(d->la(j)) * sin(2.0 * d->phi(i));
	//		u[_pOff(i, j)] = buf[k++];
	//	}
	//}
	//}

	printf("Known func is \\psi = sin(\\lambda) * sin(2 \\phi)\n");

	for (int i = 0; i < d->n_phi; i++) {
		for (int j = 0; j < d->n_la; j++) {
			u[_pOff(i, j)] = sin(d->la(j)) * sin(2.0 * d->phi(i));
			//u[_pOff(i, j)] = sin(d->la(j)) * sin(2.0 * d->phi(i));
			//u[_pOff(i, j)] = sin(d->la(j)) + sin(2.0 * d->phi(i));
			//u[_pOff(i, j)] = buf[k++];
			//u[_pOff(i, j)] = buf[k++];
		}
	}

	printf("Lapl of \\psi is -6.0 * sin(\\lambda) * sin(2 \\phi)\n");
	for (int i = 0; i < d->n_phi; i++) {
		for (int j = 0; j < d->n_la; j++) {
			uu[_pOff(i, j)] = -6.0 * sin(d->la(j)) *
				sin(2.0 * d->phi(i));
		}
	}

	lapl(v, u);

	/*if (d->full) {
		for (int j = 0; j < d->n_la; ++j) {
			v[_pOff(d->n_phi/2-1,j)]=0;
		}
	}*/

	int n1 = d->n_phi;
	int n2 = d->n_la;
	if (d->full) {
		_fprintfwmatrix("out/start_full.out", u, n1, n2, max(n1, n2), "%.3le ");
		_fprintfwmatrix("out/laplas_full.out", v, n1, n2, max(n1, n2), "%.3le ");
		_fprintfwmatrix("out/laplas_real_full.out", uu, n1, n2, max(n1, n2), "%.3le ");
	} else {
		_fprintfwmatrix("out/start_half.out", u, n1, n2, max(n1, n2), "%.3le ");
		_fprintfwmatrix("out/laplas_half.out", v, n1, n2, max(n1, n2), "%.3le ");
		_fprintfwmatrix("out/laplas_real_half.out", uu, n1, n2, max(n1, n2), "%.3le ");
	}

	double norm_koef = sqrt(d->d_phi * d->d_la);
	printf("  lapl nr2=%le\n", norm_koef * dist2(uu, v, d->n_la * d->n_phi));
	printf("  lapl nr2 (without phi=0)=%le\n", norm_koef * dist2(&uu[d->n_la], &v[d->n_la], d->n_la * (d->n_phi - 1)));
	printf("  lapl inside nr1=%le\n", _matrix_inside_diff(uu, v, d->n_phi, d->n_la));
	printf("  h1=%.16le\n", d->d_phi);
	printf("  h2=%.16le\n", d->d_la);
	printf("  h1^2=%.16le\n", d->d_phi2);
	printf("  h2^2=%.16le\n", d->d_la2);
	printf("  h1^2+h2^2=%.16le\n", d->d_la2 + d->d_phi2);

	//для обращения нужно краевое условие
	if (!d->full) {
		for (int j = 0; j < d->n_la; ++j) {
			v[_pOff(0, j)] = 0;
		}
	} else {
		//для обращения нужна ортогональность единице
		//0'я гармоника должна быть равна нулю
	}

	lapl_1(v, v);
	printf("  |Lapl(Lapl_1)|=%.16le\n",dist1(v, u, nn));
	if (d->full) {
		_fprintfwmatrix("out/laplas_1_full.out", v, n1, n2, max(n1, n2), "%.3le ");
	} else {
		_fprintfwmatrix("out/laplas_1_half.out", v, n1, n2, max(n1, n2), "%.3le ");
	}

	delete [] u; delete [] uu;
	delete [] v;
}

void SLaplacian::check0()
{
	printf("checking data ... \n");
	printf("  n_la=%d\n", d->n_la);
	printf("  n_phi=%d\n", d->n_phi);
	printf("  use_fft=%d\n", d->use_fft);
	printf("  full=%d\n", d->full);
}
