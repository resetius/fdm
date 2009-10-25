#include <math.h>
#include <string.h>
#include <vector>

#include "asp_excs.h"
#include "asp_macros.h"
#include "asp_misc.h"
#include "asp_lapl.h"
#include "asp_fft.h"
#include "asp_gauss.h"

//макросы для доступа к значениям собственных функций
#define _VM(i, j)  VM[( i )* n_y + ( j )]
#define __VM(i, j) d->VM[( i )* d->n_y + ( j )]
//макросы для нахождения отступа в массиве для точки пространства
#define  pOff(i, j) ( i ) * (n_y + 1) + ( j )
#define _pOff(i, j) ( i ) * (d->n_y + 1) + ( j )

using namespace asp;
using namespace std;

class conv_diags {
public:
	std::vector < double > A;
	std::vector < double > B;
	std::vector < double > C;
	std::vector < double > RP;

	conv_diags(int n): A(n), B(n), C(n), RP(n) 
	{
	}

	~conv_diags() {}
};

class Laplacian2D::Private: public SqSteps
{
	/* предварительное вычисление часто используемых данных
	 * в разы ускоряет счет, но памяти ест много */
	vector < double > LM; /*!<собственные значения лапласа на прямоугольнике*/
	vector < double > VM; /*!<значения собственных функций лапласа на прямоугольнике*/

	int fft;      //!<степень для FFT метода
	bool use_fft; //!<используем FFT

	double vm_mult; //!<множитель перед синусами - собственными функциями
	conv_diags * d1;
	int np;

public:
	Private(double lx, double ly, int nx, int ny)
		: SqSteps(lx, ly, nx, ny), d1(0)
	{
		fft = 0;
		int _n = n_y;
		while (_n % 2 == 0) {
			_n /= 2; fft++;
		}

		np = n_x + 1;

		use_fft = (_ipow(2, fft) == n_y);
		vm_mult = sqrt(2. / ly);


        LM.resize(n_y);

		for (int i = 1; i < n_y; i++) {
			LM[i] = L_m(i);
		}

        VM.resize(n_y * (n_y+1));

		for (int m = 1; m < n_y; m++) {
			for (int j = 0; j < n_y; j++) {
				_VM(m, j) = V_m(m, j);
			}
		}
	}

	~Private() {
		if (d1) delete d1;
	}

    /*! перевод сеточного разложения в разложение по C[m][i]
     * \param X
     * \param Y
     * \param U
     */
    void UtoXY(double* X, const double* U) {
		if (use_fft) {
			vector < double >s(n_y + 1);
			for (int i = 0; i <= n_x; i++) {
				FFT(&s[0], &U[pOff(i, 0)], &VM[0], d_y * vm_mult, n_y, fft);
				for (int m = 1; m < n_y; m++) {
					X[(m - 1) * np + i] = s[m];
				}
			}
		} else {
			double sum;
			for (int m = 1; m < n_y; m++) {
				for (int i = 0; i <= n_x; i++) {
					sum = 0.0;
					for (int j = 1; j <= n_y - 1; j++) {
						sum += U[pOff(i, j)] * _VM(m, j);
					}
					sum *= d_y;
					X[(m - 1) * np + i] = sum;
				}
			}
		}
    }

    /*! перевод разложения по C[m][i] в сеточное разложение
     * \param U
     * \param X
     * \param Y
     */
    void XYtoU(double* U, const double* X) {
		if (use_fft) {			
			vector < double > s (n_y + 1);
			for (int i = 0; i <= n_x; i++) {
				s[0]  = 0.0;
				for (int m = 1; m < n_y; m++) {
					s[m] = X[(m - 1) * np + i];
				}
				s[n_y] = 0.0;

				FFT(&U[pOff(i, 0)], &s[0], &VM[0], 1.0 * vm_mult, n_y, fft);
			}
		} else {
			double sum;
			for (int i = 0; i <= n_x; i++) {
				for (int j = 1; j < n_y; j++) {
					sum = 0.0;
					for (int m = 1; m < n_y; m++) {
						sum += X[(m - 1) * np + i] * _VM(m, j);
					}
					U[pOff(i, j)] = sum;
				}
			}
		}

		//на краях ноль
		//for (int j = 0; j <= n_y; j++) {
		//	U[pOff(0, j)]   = 0.0;
		//	U[pOff(n_x, j)] = 0.0;
		//}

		//for (int i = 0; i <= n_x; i++) {
		//	U[pOff(i, 0)]   = 0.0;
		//	U[pOff(i, n_y)] = 0.0;
		//}
    }

    /* собственные значения.
	 * Самарский-Николаев, страница 63 внизу
     * \param m
     * \return
     */
    double L_m(int m) {
		_DEBUG_ASSERT_RANGE(1, n_y - 1, m);

        return 4. / d_y2 *
              ipow(sin(M_PI * m * d_y * 0.5 / ly), 2);
    }

    /* собственные функции.
	 * Самарский-Николаев, страница 64, формула 7
     * \param m
     * \param j
     * \return
     */
    double V_m(int m, int j) {
		_DEBUG_ASSERT_RANGE(1, n_y - 1, m);
		_DEBUG_ASSERT_RANGE(0, n_y, j);

        return sqrt(2. / ly) * sin(M_PI * m * j / n_y);
    }

    /*! Оператор лаплас на прямоугольнике, значения находятся только во внутренних точках
     * \param A - значения функции в прямоугольнике
     * \param i - координата по x
     * \param j - координата по y
     * \return - значение оператора в точке (i, j)
	 * Самарский-Николаев, страница 234 внизу
     */
    double laplacian_ij(const double* A, int i, int j) {
		_DEBUG_ASSERT_RANGE(0, n_x, i);
		_DEBUG_ASSERT_RANGE(0, n_y, j);

        double ret = 0.0;
		if (0 < i && i < n_x) {
			ret += (A[pOff(i + 1, j)] 
				- 2. * A[pOff(i, j)] 
				+ A[pOff(i - 1, j)]) * d_x2_1;
		} else if (i == 0) {
			ret += (A[pOff(i + 1, j)] 
				- 2. * A[pOff(i, j)] 
				) * d_x2_1;
		} else if (i == n_x) {
			ret += (
				- 2. * A[pOff(i, j)] 
				+ A[pOff(i - 1, j)]) * d_x2_1;
		}

		if (0 < j && j < n_y) {
			ret += (A[pOff(i, j + 1)] 
				- 2. * A[pOff(i, j)] 
				+ A[pOff(i, j - 1)]) * d_y2_1;
		} else if (j == 0) {
			ret += (A[pOff(i, j + 1)] 
				- 2. * A[pOff(i, j)] 
				) * d_y2_1;
		} else if (j == n_y) {
			ret += (
				- 2. * A[pOff(i, j)] 
				+ A[pOff(i, j - 1)]) * d_y2_1;
		}
        return ret;
    }

    /* матрица оператора Лапласа на прямоугольника
     * \param M2 - ответ
     * \param M1 - исходная функция
     */
    void laplacian_matrix(double* M2, const double* M1) {
		for (int i = 0; i <= n_x; i++) {
            for (int j = 0; j <= n_y; j++) {
                M2[pOff(i, j)] = laplacian_ij(M1, i, j);
            }
		}

		//на краях нули
		/*for (int j = 0; j <= n_y; j++) {
			M2[pOff(0, j)]   = 0.0;
			M2[pOff(n_x, j)] = 0.0;
		}

		for (int i = 0; i <= n_x; i++) {
			M2[pOff(i, 0)]   = 0.0;
			M2[pOff(i, n_y)] = 0.0;
		}*/
	}


    /*! обратный лаплас в прямоугольнике
     * \param M     -     начальное условие и ответ
     * \param mult_factor параметры для подправленного лапласа
     * \param diag  - диагональная добавка
     */
	void conv_laplace(double* M,
		double mult_factor,
		double diag) 
	{
        if (d1 == 0) d1 = new conv_diags(n_x+1);
		vector < double > MF(np * (n_y + 1));

		//на Y краях ноль
		/*for (int j = 0; j <= n_y; j++) {
			M[pOff(0, j)]   = 0.0;
			M[pOff(n_x, j)] = 0.0;
		}*/

		//на X краях ноль
		for (int i = 0; i <= n_x; i++) {
			M[pOff(i, 0)]   = 0.0;
			M[pOff(i, n_y)] = 0.0;
		}

		//use_fft = 0;
		UtoXY(&MF[0], M);

        for (int m = 1; m < n_y; m++) {
            for (int i = 1; i <= n_x - 1; i++) {
                d1->A[i - 1] = d_x2_1; //1/d_x2
                d1->B[i]     = -2. * d_x2_1 - LM[m];//-2. / d_x2 - LM[m];
                d1->C[i]     = d_x2_1;
				d1->RP[i]    = MF[(m - 1) * np + i];
            }

            d1->B[0]  = -2 * d_x2_1 - LM[m];//1.0;
            d1->C[0]  = d_x2_1;//0.0;
            d1->RP[0] = MF[(m - 1) * np + 0]; //0.0

			d1->B[n_x]     = -2 * d_x2_1 - LM[m];//1.0;
			d1->A[n_x - 1] = d_x2_1;//0.0;
			d1->RP[n_x]    = MF[(m - 1) * np + n_x]; //0.0

            /*для неявной схемы меняем диагональный элемент*/
            for (int i = 0; i <= n_x; i++) {
                d1->B[i] *= mult_factor;
                d1->B[i] += diag;
			}
			for (int i = 0; i < n_x; i++) {
                d1->A[i] *= mult_factor;
                d1->C[i] *= mult_factor;
            }

			solve_tdiag_linear(&d1->RP[0], &d1->A[0], &d1->B[0], &d1->C[0], n_x + 1);

			memcpy(&MF[(m - 1) * (n_x + 1)], &d1->RP[0], (n_x + 1) * sizeof(double));
        }

		XYtoU(M, &MF[0]);
    }
};

Laplacian2D::Laplacian2D(double l_x, double l_y, int n_x, int n_y)
{
	d = new Private(l_x, l_y, n_x, n_y);
}

Laplacian2D::~Laplacian2D()
{
	delete d;
}

void Laplacian2D::lapl(double * Dest, const double * M)
{
	if (Dest != M) {
		d->laplacian_matrix(Dest, M);
	} else {
		int nn = (d->n_x + 1) * (d->n_y + 1);
		vector < double > tmp(nn);
		d->laplacian_matrix(&tmp[0], M);
		memcpy(Dest, &tmp[0], nn * sizeof(double));
	}
}

double Laplacian2D::lapl(const double * M, int i, int j)
{
	return d->laplacian_ij(M, i, j);
}

void Laplacian2D::lapl_1(double * Dest, const double * Source, 
		double mult, double diag)
{
	if (Dest != Source) memcpy(Dest, Source, 
		(d->n_x + 1) * (d->n_y + 1) * sizeof(double));
	d->conv_laplace(Dest, mult, diag);
}


Laplacian::Laplacian(double l_x_, int n_x_, int type_)
	: l_x(l_x_), n_x(n_x_), type(type_)
{
	d_x    = l_x / n_x;
	d_x2_1 = 1.0 / d_x / d_x;
}

void Laplacian::lapl(double * Dest, const double * M)
{
	double * tmp;
	if (Dest == M) {
		tmp = (double*)malloc(n_x * sizeof(double));
	} else {
		tmp = Dest;
	}
	int i;
	for (i = 0; i <= n_x; ++i) {
		tmp[i] = lapl(M, i);
	}

	if (Dest == M) {
		memcpy(Dest, tmp, n_x * sizeof(double));
		free(tmp);
	}
}

double Laplacian::lapl(const double * A, int i)
{
	_DEBUG_ASSERT_RANGE(0, n_x, i);

	double ret = 0.0;
	switch (type) {
	case PERIODIC:
		ret += (A[(i + 1) % (n_x + 1)] - 2. * A[i] + 
			A[(n_x + i) % (n_x + 1)]) * d_x2_1;
		break;
	case ZERO_COND:
	default:
		if (0 < i && i < n_x) {
			ret += (A[i + 1] - 2. * A[i] + A[i - 1]) * d_x2_1;
		} else if (i == 0) {
			ret += (A[i + 1] - 2. * A[i]) * d_x2_1;
		} else if (i == n_x) {
			ret += ( - 2. * A[i] + A[i - 1]) * d_x2_1;
		}
	}

	/*if (i == 0 || i == n_x)
		ret = 0;*/
	return ret;
}
