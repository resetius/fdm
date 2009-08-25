#ifndef _ASP_TEMPL_HPP
#define _ASP_TEMPL_HPP
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
   Базовые шаблоны
 */

#include <time.h>
#include <stdlib.h>
#ifdef _MPI_BUILD
#include <mpi.h>
#endif

typedef unsigned int uint;

/*!
   измерение времени работы \param Iter в секундах
 */
template < class Iter >
double measure_time(Iter iter)
{
	time_t t1 = time(0);
	iter();
	time_t t2 = time(0);
	return (double)(t2 - t1);
}

/*!
   измерение времени работы \param Iter в миллисекундах
 */
template < class Iter >
double measure_time_ms(Iter iter)
{
	time_t t1 = clock();
	iter();
	time_t t2 = clock();
	return (double) (t2 - t1) / (double) CLOCKS_PER_SEC;
}

/*!
   измерение времени работы \param Iter в миллисекундах
   для MPI программы
 */
#ifdef _MPI_BUILD
template < class Iter >
double measure_time_ms_mpi(Iter iter)
{
	double t1 = MPI_Wtime();
	iter();
	double t2 = MPI_Wtime();
	return (t2 - t1);
}
#endif

/*! самоинициализирующийся вектор */
template <class T, class Init >
class sivector: public std::vector < T > {
	Init * init;
public:
	sivector():init(0) {}

	sivector(Init *_init) {
		init = _init;
	}

	void setInit(Init * _init)
	{
		init = _init;
	}

	void clear() {
		init->destroy(*this);
		std::vector < T >::clear();
	}

	~sivector() {
		init->destroy(*this);
		delete init;
	}

	T&operator[] (uint i) {
		if (std::vector < T > ::empty() || i > std::vector < T > ::size() - 1) {
			if (init) init->init(*this, i);
		}
		return std::vector < T >::at(i);
	}

	T&at(uint i) {
		if (std::vector < T > ::empty() || i > std::vector < T > ::size() - 1) {
			if (init) init->init(*this, i);
		}
		return std::vector < T >::at(i);
	}
};

/*! целочисленная степень */
template < typename T >
T ipow(const T & x, int n) {
    T otv = x;
	int i;
    for (i = 1; i < n; i++)
        otv *= x;
    return otv;
}

template < typename T >
void inverse_general_matrix_my(T *Dest, T * Source, int n) 
{
    int i, j, k;
    int r;
	T * A = new T[n * n];
	double * X;
	int need_to_free_X = 0;

	memcpy(A, Source, n * n * sizeof(double));
	
	if (Dest == Source) {
		asp::load_identity_matrix(&X, n);
		need_to_free_X = 1;
	} else {
		X = Dest;
		asp::make_identity_matrix(X, n);
	}

    for (j = 0; j < n; j++) {
    	double a;
     /*поиск главного значения*/
        findmain(&r,j,n,A);
        swop(r,j,n,A);
        swop2(r,j,n,X);

        a = A[j * n + j];
        for (i = j + 1; i < n; i++) {
            double ba = A[i * n + j] / a;

            for (k = j; k < n; k++)
                A[i * n + k] -= A[j * n + k] * ba;

            for (k = 0; k < n; k++)
                X[i * n + k] -= X[j * n + k] * ba;
        }
    }

    for (j = n - 1; j >= 0; j--) {
        double a = A[j * n + j];
        for (i = j - 1; i >= 0; i--) {
            double ba = A[i * n + j] / a;
            for (k = n - 1; k >= j; k--)
                A[i * n + k] -= A[j * n + k] * ba;

            for (k = n - 1; k >= 0; k--)
                X[i * n + k] -= X[j * n + k] * ba;
        }
    }

    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            X[i * n + j] *= 1.0 / A[i * n + i];
        }
    }

	if (need_to_free_X) {
		memcpy(Dest, X, n * n * sizeof(double));
		free(X);
	}
	delete [] A;
}

#endif //_ASP_TEMPL_HPP
