/* Copyright (c) 2005, 2006, 2022 Alexey Ozeritsky
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

/**
 *  Сферический Якобиан
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include <vector>

#include "asp_excs.h"
#include "asp_macros.h"
#include "asp_misc.h"
#include "asp_sphere_lapl.h"
#include "asp_sphere_jac.h"

using namespace asp;
using namespace std;

//макросы для нахождения отступа в массиве для точки пространства
#define  pOff(i, j) ( i ) * n_la + ( j )
#define _pOff(i, j) ( i ) * d->n_la + ( j )

//#define _JACOBIAN_TEST //если определено, то не домножается на 1/r
                       //проверка относительно обычного скалярного произведения
#define REGISTER_J( name , func ) \
	void name(double * dst, const double *u, const double *v) \
	{ \
		vector < double > tmp(nn); \
		for (int i = 0; i < n_phi; ++i) { \
			for (int j = 0; j < n_la; ++j) { \
				tmp[pOff(i, j)] = func(u, v, i, j); \
			} \
		} \
		memcpy(dst, &tmp[0], nn * sizeof(double)); \
	}

class SJacobian::Private: public SSteps  {
public:
	vector < double > COS; //!<значения косинусов по широте
	int nn;
	Private(int _n_phi, int _n_la, bool _full)
		:  SSteps(_n_phi, _n_la, _full),
		COS(0)
	{
		COS.resize(2 * (n_phi + 1));
		nn = n_phi * n_la;

		double phi;
		for (int i = 0; i < n_phi; i++) {
			if (full)
				phi =(d_phi - M_PI) * 0.5 + (double)i * d_phi;
			else
				phi = (double)i * d_phi;
			COS[2 * i]     = cos(phi);
			COS[2 * i + 1] = cos(phi + 0.5 * d_phi);
		}
	}

	/*!
	   \param i__1 - i-1
	   \param i    - i
	   \param i_1  - i+1
	   \param j__1 - j-1
	   \param j    - j
	   \param j_1  - j+1
	 */
	void find_i_j(
		int & i__1, int & i, int & i_1, 
		int & j__1, int & j, int & j_1)
	{
		_DEBUG_ASSERT_RANGE(0, n_phi - 1, i);
		_DEBUG_ASSERT_RANGE(0, n_la - 1, j);
		int Nx = n_phi;
		int Ny = n_la;

		i_1  = (i == Nx - 1) ? i : i + 1;
		i__1 = (i == 0) ? ((full) ? i : 0) : i - 1;
		j_1  = (j + 1) % Ny;
		j__1 = (Ny + j - 1) % Ny;
	}

	double rho1(int i)
	{
#ifdef _JACOBIAN_TEST
		return 1.0;
#else
		return 1.0 / COS[2 * i];
#endif
	}

	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	/*~~~~~~~~~~~~~~~~~~~~~~~~~~ J1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	/*!часть по phi*/
	double J1_phi(const double *u, const double *v, int i, int j)
	{
		_DEBUG_ASSERT_RANGE(0, n_phi - 1, i);
		_DEBUG_ASSERT_RANGE(0, n_la - 1, j);

		double rho_1 = rho1(i);
		int Nx = n_phi;
		int Ny = n_la;

		double j1_phi = 0;
		int i_1, i__1, j_1, j__1;
		find_i_j(i__1, i, i_1, j__1, j, j_1);

		if (0 < i && i < Nx - 1) {
			j1_phi -= (u[pOff(i_1, j)] - u[pOff(i__1, j)])
					*(v[pOff(i, j_1)] - v[pOff(i, j__1)]);
		} else if (full && i == 0) {
			j1_phi -= (u[pOff(i_1, j)] - u[pOff(i__1, (j + Ny/2) % Ny)])
					*(v[pOff(i, j_1)] - v[pOff(i, j__1)]);
		} else if (i == Nx - 1) {
			//zero i
			//j1_phi -= 0;

			//zero i+1
			//j1_phi -= ( - u[pOff(i__1, j)])
			//		*(v[pOff(i, j_1)] - v[pOff(i, j__1)]);

			j1_phi -= (u[pOff(i_1, (j + Ny/2) % Ny)] - u[pOff(i__1, j)])
					*(v[pOff(i, j_1)] - v[pOff(i, j__1)]);
		} else { //i == 0
			j1_phi -= (u[pOff(i_1, j)] /*- u[pOff(i__1, j)]*/)
					*(v[pOff(i, j_1)] - v[pOff(i, j__1)]);
		}

		j1_phi *= 0.25 * d_la_1 * d_phi_1 * rho_1;
		//j1_phi *= 0.5 * d_phi_1 * rho_1;
		return j1_phi;
	}

	REGISTER_J(J1_phi, J1_phi);

	/*!часть по lambda*/
	double J1_la(const double *u, const double *v, int i, int j)
	{
		_DEBUG_ASSERT_RANGE(0, n_phi - 1, i);
		_DEBUG_ASSERT_RANGE(0, n_la - 1, j);

		double rho_1 = rho1(i);

		int Nx = n_phi;
		int Ny = n_la;

		double j1_la = 0;
		int i_1, i__1, j_1, j__1;
		find_i_j(i__1, i, i_1, j__1, j, j_1);

		if (0 < i && i < Nx - 1) {
			j1_la += (u[pOff(i, j_1)] - u[pOff(i, j__1)])
					* (v[pOff(i_1, j)] - v[pOff(i__1, j)]);
		} else if (full && i == 0) {
			j1_la += (u[pOff(i, j_1)] - u[pOff(i, j__1)])
					* (v[pOff(i_1, j)] - v[pOff(i__1, (j + Ny/2) % Ny)]);
		} else if (i == Nx - 1) {
			//zero i
			//j1_la += 0;

			//zero i+1
			//j1_la += (u[pOff(i, j_1)] - u[pOff(i, j__1)])
			//		* ( - v[pOff(i__1, j)]);

			//double u1 = u[pOff(i, j_1)];
			//double u2 = u[pOff(i, j__1)];
			//double v1 = v[pOff(i_1, (j + Ny/2) % Ny)];
			//double v2 = v[pOff(i__1, j)];

			j1_la += (u[pOff(i, j_1)] - u[pOff(i, j__1)])
					* (v[pOff(i_1, (j + Ny/2) % Ny)] - v[pOff(i__1, j)]);
		} else {
			j1_la += (u[pOff(i, j_1)] - u[pOff(i, j__1)])
					* (v[pOff(i_1, j)] /*- v[pOff(i__1, j)]*/);
		}

		j1_la *= 0.25 * d_la_1 * d_phi_1 * rho_1;
		//j1_la *= 0.5 * d_la_1 * rho_1;
		//j1_la *= 0.5 * d_phi_1 * rho_1;
		return j1_la;
	}

	REGISTER_J(J1_la, J1_la);

	double J1(const double *u, const double *v, int i, int j)
	{
		return J1_la(u, v, i, j) + J1_phi(u, v, i, j);
	}

	REGISTER_J(J1, J1);

	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~J3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	double J3_la(const double *u, const double *v, int i, int j)
	{
		_DEBUG_ASSERT_RANGE(0, n_phi - 1, i);
		_DEBUG_ASSERT_RANGE(0, n_la - 1, j);

		double j3_la = 0.0;
		double rho_1 = rho1(i);
		int Nx = n_phi;
		int Ny = n_la;
		double koef     = rho_1;
		double la_koef  = 0.5 * d_la_1;
		double phi_koef = 0.5 * d_phi_1;

		int i_1; //i+1;
		int i__1;//i-1;
		int j_1; //j+1;
		int j__1;//j-1;

		find_i_j(i__1, i, i_1, j__1, j, j_1);

		if (0 < i && i < Nx - 1) {
			j3_la += (v[pOff(i_1, j_1)] - v[pOff(i__1, j_1)]) 
					* u[pOff(i, j_1)];
			j3_la -= (v[pOff(i_1, j__1)] - v[pOff(i__1, j__1)])
					* u[pOff(i, j__1)];
		} else if (full && i == 0) {
			//int k_1  = (j + Ny/2 - 1) % Ny;
			//int k__1 = (j + Ny/2 + 1) % Ny;
			int k_1  = (j + Ny/2 + 1) % Ny;
			int k__1 = (j + Ny/2 - 1) % Ny;

			j3_la += (v[pOff(i_1, j_1)] - v[pOff(i__1, k_1)]) 
					* u[pOff(i, j_1)];
			j3_la -= (v[pOff(i_1, j__1)] - v[pOff(i__1, k__1)])
					* u[pOff(i, j__1)];
		} else if (i == Nx - 1) {

			//zero i
			//j3_la += 0.0;

			//zero i + 1
			//j3_la += ( - v[pOff(i__1, j_1)]) 
			//		* u[pOff(i, j_1)];
			//j3_la -= ( - v[pOff(i__1, j__1)])
			//		* u[pOff(i, j__1)];

			int k_1  = (j + Ny/2 + 1) % Ny;
			int k__1 = (j + Ny/2 - 1) % Ny;

			j3_la += (v[pOff(i_1, k_1)] - v[pOff(i__1, j_1)]) 
					* u[pOff(i, j_1)];
			j3_la -= (v[pOff(i_1, k__1)] - v[pOff(i__1, j__1)])
					* u[pOff(i, j__1)];
		} else {
			j3_la += (v[pOff(i_1, j_1)]/* - v[pOff(i__1, j_1)]*/) 
					* u[pOff(i, j_1)];

			j3_la -= (v[pOff(i_1, j__1)]/* - v[pOff(i__1, j__1)]*/)
					* u[pOff(i, j__1)];
		}

		j3_la *= phi_koef * la_koef * koef;
		return j3_la;
	}

	REGISTER_J(J3_la, J3_la);

	double J3_phi(const double *u, const double *v, int i, int j)
	{
		_DEBUG_ASSERT_RANGE(0, n_phi - 1, i);
		_DEBUG_ASSERT_RANGE(0, n_la - 1, j);

		double j3 = 0.0;
		double rho_1 = rho1(i);
		int Nx = n_phi;
		int Ny = n_la;
		double koef     = rho_1;
		double la_koef  = 0.5 * d_la_1;
		double phi_koef = 0.5 * d_phi_1;

		int i_1; //i+1;
		int i__1;//i-1;
		int j_1; //j+1;
		int j__1;//j-1;

		find_i_j(i__1, i, i_1, j__1, j, j_1);

		if (0 < i && i < Nx - 1) {
			j3 -= (v[pOff(i_1, j_1)] - v[pOff(i_1, j__1)])
				* u[pOff(i_1, j)];

			j3 += (v[pOff(i__1, j_1)] - v[pOff(i__1, j__1)])
				* u[pOff(i__1, j)];
		} else if (full && i == 0) {
			//южный полюс
			int k    = (j + Ny/2    ) % Ny;
			int k_1  = (j + Ny/2 - 1) % Ny;
			int k__1 = (j + Ny/2 + 1) % Ny;

			j3 -= (v[pOff(i_1, j_1)] - v[pOff(i_1, j__1)])
				* u[pOff(i_1, j)];

			j3 += (v[pOff(i__1, k_1)] - v[pOff(i__1, k__1)])
				* u[pOff(i__1, k)];
		} else if (i == Nx - 1) {
			//северный полюс
			//zero i
			//j3 += 0;

			//zero i+1
			//j3 += (v[pOff(i__1, j_1)] - v[pOff(i__1, j__1)])
			//	* u[pOff(i__1, j)];

			// changed !
			int k    = (j + Ny/2    ) % Ny;
			int k_1  = (j + Ny/2 - 1) % Ny;
			int k__1 = (j + Ny/2 + 1) % Ny;

			j3 -= (v[pOff(i_1, k_1)] - v[pOff(i_1, k__1)])
				* u[pOff(i_1, k)];

			j3 += (v[pOff(i__1, j_1)] - v[pOff(i__1, j__1)])
				* u[pOff(i__1, j)];
		} else { //i == 0
			j3 -= (v[pOff(i_1, j_1)] - v[pOff(i_1, j__1)])
				* u[pOff(i_1, j)];

			/*j3 += (v[pOff(i__1, j_1)] - v[pOff(i__1, j__1)])
				* u[pOff(i__1, j)];*/
		}

		j3 *= phi_koef * la_koef * koef;
		return j3;
	}

	REGISTER_J(J3_phi, J3_phi);

	double J3(const double *u, const double *v, int i, int j) {
//		if (i == n_phi - 1) {
//			return J1_phi(u, v, i, j) + J1_la(u, v, i, j);
//		} else {
			return J3_phi(u, v, i, j) + J3_la(u, v, i, j);
//		}
	}

	REGISTER_J(J3, J3);

	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~J2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

	double J2_la(const double *u, const double *v, int i, int j)
	{
		_DEBUG_ASSERT_RANGE(0, n_phi - 1, i);
		_DEBUG_ASSERT_RANGE(0, n_la - 1, j);

		double j2_la = 0.0;
		double rho_1 = rho1(i);
		int Nx = n_phi;
		int Ny = n_la;
		double koef     = rho_1;
		double la_koef  = 0.5 * d_la_1;
		double phi_koef = 0.5 * d_phi_1;

		int i_1; //i+1;
		int i__1;//i-1;
		int j_1; //j+1;
		int j__1;//j-1;

		find_i_j(i__1, i, i_1, j__1, j, j_1);

		if (0 < i && i < Nx - 1) {
			j2_la -= (u[pOff(i_1, j_1)] - u[pOff(i__1, j_1)]) *
					(v[pOff(i, j_1)]);
			j2_la += (u[pOff(i_1, j__1)] - u[pOff(i__1, j__1)]) *
					(v[pOff(i, j__1)]);

		} else if (full && i == 0) {
			//int k_1  = (j + Ny/2 - 1) % Ny;
			//int k__1 = (j + Ny/2 + 1) % Ny;
			int k_1  = (j + Ny/2 + 1) % Ny;
			int k__1 = (j + Ny/2 - 1) % Ny;

			double u1 = u[pOff(i_1, j_1)];
			double u2 = u[pOff(i__1, k_1)];
			double v1 = v[pOff(i, j_1)];
			double s1 = (u1 - u2) * v1;

			double u11 = u[pOff(i_1, j__1)];
			double u21 = u[pOff(i__1, k__1)];
			double v2  = v[pOff(i, j__1)];
			double s2  = (u11 - u21) * v2;

			j2_la -= (u[pOff(i_1, j_1)] - u[pOff(i__1, k_1)]) *
					(v[pOff(i, j_1)]);
			j2_la += (u[pOff(i_1, j__1)] - u[pOff(i__1, k__1)]) *
					(v[pOff(i, j__1)]);
		} else if (i == Nx - 1) {
			//zero i
			//j2_la -= 0;

			//zero i+1
			//j2_la -= ( - u[pOff(i__1, j_1)]) *
			//		(v[pOff(i, j_1)]);
			//j2_la += ( - u[pOff(i__1, j__1)]) *
			//		(v[pOff(i, j__1)]);

			int k_1  = (j + Ny/2 + 1) % Ny;
			int k__1 = (j + Ny/2 - 1) % Ny;

			double u1 = u[pOff(i__1, k_1)];
			double u2 = u[pOff(i_1, j_1)];
			double v1 = v[pOff(i, j_1)];
			double s1 = ((-0.5) * u1 - (-1.5) * u2) * v1;

			double u11 = u[pOff(i__1, k__1)];
			double u21 = u[pOff(i_1, j__1)];
			double v2  = v[pOff(i, j__1)];
			double s2  = ((-0.5) * u11 - (-1.5) * u21) * v2;

			j2_la -= (u[pOff(i_1, k_1)] - u[pOff(i__1, j_1)]) *
					(v[pOff(i, j_1)]);
					//(v[pOff(i, j)]);
			j2_la += (u[pOff(i_1, k__1)] - u[pOff(i__1, j__1)]) *
					(v[pOff(i, j__1)]);
					//(v[pOff(i, j)]);

//			printf("la: %.16le \n", j2_la);
		} else { //i == 0
			j2_la -= (u[pOff(i_1, j_1)] /*- u[pOff(i__1, j_1)]*/) *
					(v[pOff(i, j_1)]);
			j2_la += (u[pOff(i_1, j__1)] /*- u[pOff(i__1, j__1)]*/) *
					(v[pOff(i, j__1)]);
		}

		j2_la *= phi_koef * la_koef * koef;
		return j2_la;
	}

	REGISTER_J(J2_la, J2_la);

	double J2_phi(const double *u, const double *v, int i, int j)
	{
		_DEBUG_ASSERT_RANGE(0, n_phi - 1, i);
		_DEBUG_ASSERT_RANGE(0, n_la - 1, j);

		double j2 = 0.0;
		double rho_1 = rho1(i);
		int Nx = n_phi;
		int Ny = n_la;
		double koef     = rho_1;
		double la_koef  = 0.5 * d_la_1;
		double phi_koef = 0.5 * d_phi_1;

		int i_1; //i+1;
		int i__1;//i-1;
		int j_1; //j+1;
		int j__1;//j-1;

		find_i_j(i__1, i, i_1, j__1, j, j_1);

		if (0 < i && i < Nx - 1) {
			j2 += (u[pOff(i_1, j_1)] - u[pOff(i_1, j__1)]) *
				(v[pOff(i_1, j)]);
			j2 -= (u[pOff(i__1, j_1)] - u[pOff(i__1, j__1)]) *
				(v[pOff(i__1, j)]);
		} else if (full && i == 0) {
			int k    = (j + Ny/2    ) % Ny;
			int k_1  = (j + Ny/2 + 1) % Ny; //- ?
			int k__1 = (j + Ny/2 - 1) % Ny; //+ ?

			j2 += (u[pOff(i_1, j_1)] - u[pOff(i_1, j__1)]) *
				(v[pOff(i_1, j)]);
			j2 -= (u[pOff(i__1, k_1)] - u[pOff(i__1, k__1)]) *
				(v[pOff(i__1, k)]);
		} else if (i == Nx - 1) {

			//zero i
			//j2 -= 0;

			//zero i+1
			/*j2 -= (u[pOff(i__1, j_1)] - u[pOff(i__1, j__1)]) *
				(v[pOff(i__1, j)]);*/

			// changed !
			int k    = (j + Ny/2    ) % Ny;
			int k_1  = (j + Ny/2 - 1) % Ny; //- ?
			int k__1 = (j + Ny/2 + 1) % Ny; //+ ?

			double u1 = u[pOff(i_1, k_1)];
			double u2 = u[pOff(i_1, k__1)];
			double v1 = v[pOff(i_1, k)];

			double u12 = u[pOff(i__1, j_1)];
			double u13 = u[pOff(i__1, j__1)];
			double v2  = v[pOff(i__1, j)];
			
			//j2 += (u[pOff(i_1, k_1)] - u[pOff(i_1, k__1)]) *
			//	(v[pOff(i_1, k)]);

			double s1 = (u[pOff(i_1, k_1)] - u[pOff(i_1, k__1)]) *
				(v[pOff(i_1, k)]);
			
			j2 += s1;

			double s2 = (u[pOff(i__1, j_1)] - u[pOff(i__1, j__1)]) *
				(v[pOff(i__1, j)]);

			j2 -= s2;

//			printf("phi: %.16le \n", j2);
			//j2 -= (u[pOff(i__1, j_1)] - u[pOff(i__1, j__1)]) *
			//	(v[pOff(i__1, j)]);
		} else { //i == 0
			j2 += (u[pOff(i_1, j_1)] - u[pOff(i_1, j__1)]) *
				(v[pOff(i_1, j)]);
			/*j2 -= (u[pOff(i__1, j_1)] - u[pOff(i__1, j__1)]) *
				(v[pOff(i__1, j)]);*/
		}

		j2 *= phi_koef * la_koef * koef;
		return j2;
	}

	REGISTER_J(J2_phi, J2_phi);

	double J2(const double *u, const double *v, int i, int j) {
//		if (i == n_phi - 1) {
//			return J1_phi(u, v, i, j) + J1_la(u, v, i, j);
//		} else {
			return J2_phi(u, v, i, j) + J2_la(u, v, i, j);
//		}
	}

	REGISTER_J(J2, J2);

	/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	/*~~~~~~~~~~~~~~~~~~~~~~J, схема Аракавы ~~~~~~~~~~~~~~~~~~~~~~~~~*/
	double J(const double *u, const double *v, int i, int j) {
		return 0.3333333333333333333 * (
			J1(u, v, i, j) +
			J2(u, v, i, j) +
			J3(u, v, i, j));
	}

	REGISTER_J(J, J);

	/*!сопряжение по первому аргументу*/
	double JT1(const double *u, const double *v, int i, int j) {
		return 0.3333333333333333333 * (
			- J3(u, v, i, j)
			- J2(u, v, i, j)
			- J1(u, v, i, j));
	}

	REGISTER_J(JT1, JT1);

	/*!сопряжение по второму аргументу*/
	double JT2(const double *u, const double *v, int i, int j) {
		return 0.3333333333333333333 * (
			- J2(u, v, i, j)
			- J1(u, v, i, j)
			- J3(u, v, i, j));
	}

	REGISTER_J(JT2, JT2);
};

SJacobian::SJacobian(int n_phi, int n_la, bool full) 
	: d(new Private(n_phi, n_la, full))
{
}

SJacobian::~SJacobian()
{
	delete d;
}

double SJacobian::J1(const double *u, const double *v, int i, int j)
{
	return d->J1(u, v, i, j);
}

void SJacobian::J1(double *dest, const double *u, const double *v)
{
	return d->J1(dest, u, v);
}

double SJacobian::J1T(const double *u, const double *v, 
					  int i, int j, int k)
{
	if (k == 1) {
		return -d->J3(u, v, i, j);
	} else { //k == 2
		return -d->J2(u, v, i, j);
	}
}

void SJacobian::J1T(double *dest, const double *u, const double *v, 
					int k)
{
	if (k == 1) {
		d->J3(dest, u, v);
		vector_mult_scalar(dest, dest, -1.0, d->nn);
	} else { //k == 2
		d->J2(dest, u, v);
		vector_mult_scalar(dest, dest, -1.0, d->nn);
	}
}

double SJacobian::J2(const double *u, const double *v, int i, int j)
{
	return d->J2(u, v, i, j);
}

void SJacobian::J2(double *dest, const double *u, const double *v)
{
	return d->J2(dest, u, v);
}

double SJacobian::J2T(const double *u, const double *v, 
					  int i, int j, int k)
{
	if (k == 1) {
		return -d->J2(u, v, i, j);
	} else { //k == 2
		return -d->J1(u, v, i, j);
	}
}

void SJacobian::J2T(double *dest, const double *u, const double *v, 
					int k)
{
	if (k == 1) {
		d->J2(dest, u, v);
		vector_mult_scalar(dest, dest, -1.0, d->nn);
	} else { //k == 2
		d->J1(dest, u, v);
		vector_mult_scalar(dest, dest, -1.0, d->nn);
	}
}

double SJacobian::J3(const double *u, const double *v, int i, int j)
{
	return d->J3(u, v, i, j);
}

void SJacobian::J3(double *dest, const double *u, const double *v)
{
	return d->J3(dest, u, v);
}

double SJacobian::J3T(const double *u, const double *v, 
					  int i, int j, int k)
{
	if (k == 1) {
		return -d->J1(u, v, i, j);
	} else { //k == 2
		return -d->J3(u, v, i, j);
	}
}

void SJacobian::J3T(double *dest, const double *u, const double *v, 
					int k)
{
	if (k == 1) {
		d->J1(dest, u, v);
		vector_mult_scalar(dest, dest, -1.0, d->nn);
	} else { //k == 2
		d->J3(dest, u, v);
		vector_mult_scalar(dest, dest, -1.0, d->nn);
	}
}

double SJacobian::J(const double *u, const double *v, int i, int j)
{
	return d->J(u, v, i, j);
}

void SJacobian::J(double *dest, const double *u, const double *v)
{
	d->J(dest, u, v);
}

double SJacobian::JT(const double *u, const double *v, int i, int j)
{
	return d->JT1(u, v, i, j);
}

void SJacobian::JT(double *dest, const double *u, const double *v)
{
	d->JT1(dest, u, v);
}

double SJacobian::scalar(const double *u, const double *v)
{
	double rho;
	double sum = 0;
	int i, j;

	for (i = 0; i < d->n_phi; ++i) {
		rho = d->COS[2 * i];
		for (j = 0; j < d->n_la; ++j) {
			sum += rho * u[_pOff(i, j)] * v[_pOff(i, j)];
		}
	}
	return sum;
}

void SJacobian::JV2T(double * dest, const double * u, const double * v)
{
	JV2(dest, u, v);
	vector_mult_scalar(dest, dest, -1.0, d->nn);
}

void SJacobian::JV2(double * dest, const double * u, const double * v)
{
	int i, j;
	double J1, J2, J3;
	double J1a, J2a, J3a;

	int Nx = d->n_la;
	int Ny = d->n_phi;

	double Hx = d->d_la;
	double Hy = d->d_phi;

	double * ko = dest;
	const double * ps = u;
	const double * om = v;
	double KORIOLIS_KEY = 1.0;

	{


//shema Arakavy
//AO u->ps v->om; u_ij->ps_ji

		for (i = 0; i < Nx; i++)
			for (j = 0; j < Ny; j++)
				ko[i+j*Nx] = 0.;

		for (i = 1; i < Nx - 1; i++)
			for (j = 1; j < Ny - 1; j++)
			{
				J1 = (ps[i+1+j*Nx] - ps[i-1+j*Nx]) * (om[i+ (j+1) *Nx] - om[i+ (j-1) *Nx]) -
				     (ps[i+ (j+1) *Nx] - ps[i+ (j-1) *Nx]) * (om[i+1+j*Nx] - om[i-1+j*Nx]);
				J1 = J1 / 2. / Hx / 2. / Hy;

				J1a = (ps[i+1+j*Nx] - ps[i-1+j*Nx]) * (om[i+ (j+1) *Nx] - om[i+ (j-1) *Nx]) -
				      (ps[i+ (j+1) *Nx] - ps[i+ (j-1) *Nx]) * (om[i+1+j*Nx] - om[i-1+j*Nx]);
				J1a = J1a / 2. / Hx / 2. / Hy;


				J2 = (ps[i+1+ (j+1) *Nx] - ps[i-1+ (j+1) *Nx]) * om[i+ (j+1) *Nx] -
				     (ps[i+1+ (j-1) *Nx] - ps[i-1+ (j-1) *Nx]) * om[i+ (j-1) *Nx] -
				     (ps[i+1+ (j+1) *Nx] - ps[i+1+ (j-1) *Nx]) * om[i+1+j*Nx] +
				     (ps[i-1+ (j+1) *Nx] - ps[i-1+ (j-1) *Nx]) * om[i-1+j*Nx];
				J2 = J2 / 2. / Hx / 2. / Hy;

				J2a = (ps[i+1+ (j+1) *Nx] - ps[i-1+ (j+1) *Nx]) * om[i+ (j+1) *Nx] -
				      (ps[i+1+ (j-1) *Nx] - ps[i-1+ (j-1) *Nx]) * om[i+ (j-1) *Nx] -
				      (ps[i+1+ (j+1) *Nx] - ps[i+1+ (j-1) *Nx]) * om[i+1+j*Nx] +
				      (ps[i-1+ (j+1) *Nx] - ps[i-1+ (j-1) *Nx]) * om[i-1+j*Nx];
				J2a = J2a / 2. / Hx / 2. / Hy;


				J3 = (om[i+1+ (j+1) *Nx] - om[i+1+ (j-1) *Nx]) * ps[i+1+j*Nx] -
				     (om[i-1+ (j+1) *Nx] - om[i-1+ (j-1) *Nx]) * ps[i-1+j*Nx] -
				     (om[i+1+ (j+1) *Nx] - om[i-1+ (j+1) *Nx]) * ps[i+ (j+1) *Nx] +
				     (om[i+1+ (j-1) *Nx] - om[i-1+ (j-1) *Nx]) * ps[i+ (j-1) *Nx];

				J3 = J3 / 2. / Hx / 2. / Hy;

				J3a = (om[i+1+ (j+1) *Nx] - om[i+1+ (j-1) *Nx]) * ps[i+1+j*Nx] -
				      (om[i-1+ (j+1) *Nx] - om[i-1+ (j-1) *Nx]) * ps[i-1+j*Nx] -
				      (om[i+1+ (j+1) *Nx] - om[i-1+ (j+1) *Nx]) * ps[i+ (j+1) *Nx] +
				      (om[i+1+ (j-1) *Nx] - om[i-1+ (j-1) *Nx]) * ps[i+ (j-1) *Nx];

				J3a = J3a / 2. / Hx / 2. / Hy;



				ko[i+j*Nx] = 1. / cos (Hy * (j) ) * KORIOLIS_KEY * (J1 + J2 + J3) / 3.; //  (J1a+J2a+J3a)/3.;
				//ko[i+j*Nx] = 1. / cos (Hy * (j) ) * KORIOLIS_KEY * J2;
			}



		j = Ny - 1;

		for (i = 1; i < Nx - 1; i++)
		{
			double j3_la;
			double j3_phi;

			J1 = (ps[ (i+1) %Nx+j*Nx] - ps[ (i-1) %Nx+j*Nx]) * (om[ (i+Nx/2) %Nx + (j) *Nx] - om[ (i) %Nx+ (j-1) *Nx]) -
			     (ps[ (i+Nx/2) %Nx+ (j) *Nx] - ps[ (i) %Nx+ (j-1) *Nx]) * (om[ (i+1) %Nx+j*Nx] - om[ (i-1) %Nx+j*Nx]);
			J1 = J1 / 2. / Hx / 2. / Hy;

			J1a = (ps[i+1+j*Nx] - ps[i-1+j*Nx]) * (om[i+ (j+1) *Nx] - om[i+ (j-1) *Nx]) -
			      (ps[i+ (j+1) *Nx] - ps[i+ (j-1) *Nx]) * (om[i+1+j*Nx] - om[i-1+j*Nx]);
			J1a = J1a / 2. / Hx / 2. / Hy;


			J2 = (ps[ (i-1+Nx/2) %Nx+ (j) *Nx] - ps[ (i+1+Nx/2) %Nx+ (j) *Nx]) * om[ (i+Nx/2) %Nx+ (j) *Nx] -
			     (ps[ (i+1) %Nx+ (j-1) *Nx] - ps[ (i-1) %Nx+ (j-1) *Nx]) * om[ (i) %Nx+ (j-1) *Nx] -
			     (ps[ (i+1+Nx/2) %Nx+ (j) *Nx] - ps[ (i+1) %Nx+ (j-1) *Nx]) * om[ (i+1) %Nx+j*Nx] +
			     (ps[ (i-1+Nx/2) %Nx+ (j) *Nx] - ps[ (i-1) %Nx+ (j-1) *Nx]) * om[ (i-1) %Nx+j*Nx];
			J2 = J2 / 2. / Hx / 2. / Hy;

			J2a = (ps[i+1+ (j+1) *Nx] - ps[i-1+ (j+1) *Nx]) * om[i+ (j+1) *Nx] -
			      (ps[i+1+ (j-1) *Nx] - ps[i-1+ (j-1) *Nx]) * om[i+ (j-1) *Nx] -
			      (ps[i+1+ (j+1) *Nx] - ps[i+1+ (j-1) *Nx]) * om[i+1+j*Nx] +
			      (ps[i-1+ (j+1) *Nx] - ps[i-1+ (j-1) *Nx]) * om[i-1+j*Nx];
			J2a = J2a / 2. / Hx / 2. / Hy;


			J3 = (om[ (i+1+Nx/2) %Nx+ (j) *Nx] - om[ (i+1) %Nx+ (j-1) *Nx]) * ps[ (i+1) %Nx+j*Nx] -
			     (om[ (i-1+Nx/2) %Nx+ (j) *Nx] - om[ (i-1) %Nx+ (j-1) *Nx]) * ps[ (i-1) %Nx+j*Nx] -
			     (om[ (i-1+Nx/2) %Nx+ (j) *Nx] - om[ (i+1+Nx/2) %Nx+ (j) *Nx]) * ps[ (i+Nx/2) %Nx+ (j) *Nx] +
			     (om[ (i+1) %Nx+ (j-1) *Nx] - om[ (i-1) %Nx+ (j-1) *Nx]) * ps[ (i) %Nx+ (j-1) *Nx];

			j3_la  = (om[ (i+1+Nx/2) %Nx+ (j) *Nx] - om[ (i+1) %Nx+ (j-1) *Nx]) * ps[ (i+1) %Nx+j*Nx] -
			     (om[ (i-1+Nx/2) %Nx+ (j) *Nx] - om[ (i-1) %Nx+ (j-1) *Nx]) * ps[ (i-1) %Nx+j*Nx];
			j3_phi = -
			     (om[ (i+1+Nx/2) %Nx+ (j) *Nx] - om[ (i-1+Nx/2) %Nx+ (j) *Nx]) * ps[ (i+Nx/2) %Nx+ (j) *Nx] +
			     (om[ (i+1) %Nx+ (j-1) *Nx] - om[ (i-1) %Nx+ (j-1) *Nx]) * ps[ (i) %Nx+ (j-1) *Nx];

			J3 = J3 / 2. / Hx / 2. / Hy;

			J3a = (om[i+1+ (j+1) *Nx] - om[i+1+ (j-1) *Nx]) * ps[i+1+j*Nx] -
			      (om[i-1+ (j+1) *Nx] - om[i-1+ (j-1) *Nx]) * ps[i-1+j*Nx] -
			      (om[i+1+ (j+1) *Nx] - om[i-1+ (j+1) *Nx]) * ps[i+ (j+1) *Nx] +
			      (om[i+1+ (j-1) *Nx] - om[i-1+ (j-1) *Nx]) * ps[i+ (j-1) *Nx];

			J3a = J3a / 2. / Hx / 2. / Hy;



			ko[i+j*Nx] = 1. / cos (Hy * (j) ) * KORIOLIS_KEY * (J1 + J2 + J3) / 3.; //  (J1a+J2a+J3a)/3.;
			//ko[i+j*Nx] = 1. / cos (Hy * (j) ) * KORIOLIS_KEY * J2;

		}


		i = 0;
		for (j = 1; j < Ny - 1; j++)
		{
			J1 = (ps[i+1+j*Nx] - ps[Nx-1+j*Nx]) * (om[i+ (j+1) *Nx] - om[i+ (j-1) *Nx]) -
			     (ps[i+ (j+1) *Nx] - ps[i+ (j-1) *Nx]) * (om[i+1+j*Nx] - om[Nx-1+j*Nx]);
			J1 = J1 / 2. / Hx / 2. / Hy;


			J2 = (ps[i+1+ (j+1) *Nx] - ps[Nx-1+ (j+1) *Nx]) * om[i+ (j+1) *Nx] -
			     (ps[i+1+ (j-1) *Nx] - ps[Nx-1+ (j-1) *Nx]) * om[i+ (j-1) *Nx] -
			     (ps[i+1+ (j+1) *Nx] - ps[i+1+ (j-1) *Nx]) * om[i+1+j*Nx] +
			     (ps[Nx-1+ (j+1) *Nx] - ps[Nx-1+ (j-1) *Nx]) * om[Nx-1+j*Nx];
			J2 = J2 / 2. / Hx / 2. / Hy;


			J3 = (om[i+1+ (j+1) *Nx] - om[i+1+ (j-1) *Nx]) * ps[i+1+j*Nx] -
			     (om[Nx-1+ (j+1) *Nx] - om[Nx-1+ (j-1) *Nx]) * ps[Nx-1+j*Nx] -
			     (om[i+1+ (j+1) *Nx] - om[Nx-1+ (j+1) *Nx]) * ps[i+ (j+1) *Nx] +
			     (om[i+1+ (j-1) *Nx] - om[Nx-1+ (j-1) *Nx]) * ps[i+ (j-1) *Nx];

			J3 = J3 / 2. / Hx / 2. / Hy;




			ko[i+j*Nx] = 1. / cos (Hy * (j) ) * KORIOLIS_KEY * (J1 + J2 + J3) / 3.; //  (J1a+J2a+J3a)/3.;
			//ko[i+j*Nx] = 1. / cos (Hy * (j) ) * KORIOLIS_KEY * J2;
		}



		i = Nx - 1;
		for (j = 1; j < Ny - 1; j++)
		{
			J1 = (ps[0+j*Nx] - ps[i-1+j*Nx]) * (om[i+ (j+1) *Nx] - om[i+ (j-1) *Nx]) -
			     (ps[i+ (j+1) *Nx] - ps[i+ (j-1) *Nx]) * (om[0+j*Nx] - om[i-1+j*Nx]);
			J1 = J1 / 2. / Hx / 2. / Hy;


			J2 = (ps[0+ (j+1) *Nx] - ps[i-1+ (j+1) *Nx]) * om[i+ (j+1) *Nx] -
			     (ps[0+ (j-1) *Nx] - ps[i-1+ (j-1) *Nx]) * om[i+ (j-1) *Nx] -
			     (ps[0+ (j+1) *Nx] - ps[0+ (j-1) *Nx]) * om[0+j*Nx] +
			     (ps[i-1+ (j+1) *Nx] - ps[i-1+ (j-1) *Nx]) * om[i-1+j*Nx];
			J2 = J2 / 2. / Hx / 2. / Hy;



			J3 = (om[0+ (j+1) *Nx] - om[0+ (j-1) *Nx]) * ps[0+j*Nx] -
			     (om[i-1+ (j+1) *Nx] - om[i-1+ (j-1) *Nx]) * ps[i-1+j*Nx] -
			     (om[0+ (j+1) *Nx] - om[i-1+ (j+1) *Nx]) * ps[i+ (j+1) *Nx] +
			     (om[0+ (j-1) *Nx] - om[i-1+ (j-1) *Nx]) * ps[i+ (j-1) *Nx];

			J3 = J3 / 2. / Hx / 2. / Hy;


			ko[i+j*Nx] = 1. / cos (Hy * (j) ) * KORIOLIS_KEY * (J1 + J2 + J3) / 3.; //  (J1a+J2a+J3a)/3.;
			//ko[i+j*Nx] = 1. / cos (Hy * (j) ) * KORIOLIS_KEY * J2;
		}


		j = Ny - 1;
		i = 0;
		{
			J1 = (ps[ (i+1) %Nx+j*Nx] - ps[ (Nx-1) %Nx+j*Nx]) * (om[ (i+Nx/2) %Nx + (j) *Nx] - om[ (i) %Nx+ (j-1) *Nx]) -
			     (ps[ (i+Nx/2) %Nx+ (j) *Nx] - ps[ (i) %Nx+ (j-1) *Nx]) * (om[ (i+1) %Nx+j*Nx] - om[ (Nx-1) %Nx+j*Nx]);
			J1 = J1 / 2. / Hx / 2. / Hy;

			J2 = (ps[ (i-1+Nx/2) %Nx+ (j) *Nx] - ps[ (Nx+1+Nx/2) %Nx+ (j) *Nx]) * om[ (i+Nx/2) %Nx+ (j) *Nx] -
			     (ps[ (i+1) %Nx+ (j-1) *Nx] - ps[ (Nx-1) %Nx+ (j-1) *Nx]) * om[ (i) %Nx+ (j-1) *Nx] -
			     (ps[ (i+1+Nx/2) %Nx+ (j) *Nx] - ps[ (i+1) %Nx+ (j-1) *Nx]) * om[ (i+1) %Nx+j*Nx] +
			     (ps[ (Nx-1+Nx/2) %Nx+ (j) *Nx] - ps[ (Nx-1) %Nx+ (j-1) *Nx]) * om[ (Nx-1) %Nx+j*Nx];
			J2 = J2 / 2. / Hx / 2. / Hy;


			J3 = (om[ (i+1+Nx/2) %Nx+ (j) *Nx] - om[ (i+1) %Nx+ (j-1) *Nx]) * ps[ (i+1) %Nx+j*Nx] -
			     (om[ (Nx-1+Nx/2) %Nx+ (j) *Nx] - om[ (Nx-1) %Nx+ (j-1) *Nx]) * ps[ (Nx-1) %Nx+j*Nx] -
			     (om[ (i-1+Nx/2) %Nx+ (j) *Nx] - om[ (Nx+1+Nx/2) %Nx+ (j) *Nx]) * ps[ (i+Nx/2) %Nx+ (j) *Nx] +
			     (om[ (i+1) %Nx+ (j-1) *Nx] - om[ (Nx-1) %Nx+ (j-1) *Nx]) * ps[ (i) %Nx+ (j-1) *Nx];

			J3 = J3 / 2. / Hx / 2. / Hy;



			ko[i+j*Nx] = 1. / cos (Hy * (j) ) * KORIOLIS_KEY * (J1 + J2 + J3) / 3.; //  (J1a+J2a+J3a)/3.;
			//ko[i+j*Nx] = 1. / cos (Hy * (j) ) * KORIOLIS_KEY * J2;

		}


		j = Ny - 1;
		i = Nx - 1;

		{
			J1 = (ps[ (i+1) %Nx+j*Nx] - ps[ (i-1) %Nx+j*Nx]) * (om[ (i+Nx/2) %Nx + (j) *Nx] - om[ (i) %Nx+ (j-1) *Nx]) -
			     (ps[ (i+Nx/2) %Nx+ (j) *Nx] - ps[ (i) %Nx+ (j-1) *Nx]) * (om[ (i+1) %Nx+j*Nx] - om[ (i-1) %Nx+j*Nx]);
			J1 = J1 / 2. / Hx / 2. / Hy;

			J1a = (ps[i+1+j*Nx] - ps[i-1+j*Nx]) * (om[i+ (j+1) *Nx] - om[i+ (j-1) *Nx]) -
			      (ps[i+ (j+1) *Nx] - ps[i+ (j-1) *Nx]) * (om[i+1+j*Nx] - om[i-1+j*Nx]);
			J1a = J1a / 2. / Hx / 2. / Hy;


			J2 = (ps[ (i-1+Nx/2) %Nx+ (j) *Nx] - ps[ (i+1+Nx/2) %Nx+ (j) *Nx]) * om[ (i+Nx/2) %Nx+ (j) *Nx] -
			     (ps[ (i+1) %Nx+ (j-1) *Nx] - ps[ (i-1) %Nx+ (j-1) *Nx]) * om[ (i) %Nx+ (j-1) *Nx] -
			     (ps[ (i+1+Nx/2) %Nx+ (j) *Nx] - ps[ (i+1) %Nx+ (j-1) *Nx]) * om[ (i+1) %Nx+j*Nx] +
			     (ps[ (i-1+Nx/2) %Nx+ (j) *Nx] - ps[ (i-1) %Nx+ (j-1) *Nx]) * om[ (i-1) %Nx+j*Nx];
			J2 = J2 / 2. / Hx / 2. / Hy;

			J2a = (ps[i+1+ (j+1) *Nx] - ps[i-1+ (j+1) *Nx]) * om[i+ (j+1) *Nx] -
			      (ps[i+1+ (j-1) *Nx] - ps[i-1+ (j-1) *Nx]) * om[i+ (j-1) *Nx] -
			      (ps[i+1+ (j+1) *Nx] - ps[i+1+ (j-1) *Nx]) * om[i+1+j*Nx] +
			      (ps[i-1+ (j+1) *Nx] - ps[i-1+ (j-1) *Nx]) * om[i-1+j*Nx];
			J2a = J2a / 2. / Hx / 2. / Hy;


			J3 = (om[ (i+1+Nx/2) %Nx+ (j) *Nx] - om[ (i+1) %Nx+ (j-1) *Nx]) * ps[ (i+1) %Nx+j*Nx] -
			     (om[ (i-1+Nx/2) %Nx+ (j) *Nx] - om[ (i-1) %Nx+ (j-1) *Nx]) * ps[ (i-1) %Nx+j*Nx] -
			     (om[ (i-1+Nx/2) %Nx+ (j) *Nx] - om[ (i+1+Nx/2) %Nx+ (j) *Nx]) * ps[ (i+Nx/2) %Nx+ (j) *Nx] +
			     (om[ (i+1) %Nx+ (j-1) *Nx] - om[ (i-1) %Nx+ (j-1) *Nx]) * ps[ (i) %Nx+ (j-1) *Nx];

			J3 = J3 / 2. / Hx / 2. / Hy;

			J3a = (om[i+1+ (j+1) *Nx] - om[i+1+ (j-1) *Nx]) * ps[i+1+j*Nx] -
			      (om[i-1+ (j+1) *Nx] - om[i-1+ (j-1) *Nx]) * ps[i-1+j*Nx] -
			      (om[i+1+ (j+1) *Nx] - om[i-1+ (j+1) *Nx]) * ps[i+ (j+1) *Nx] +
			      (om[i+1+ (j-1) *Nx] - om[i-1+ (j-1) *Nx]) * ps[i+ (j-1) *Nx];

			J3a = J3a / 2. / Hx / 2. / Hy;



			ko[i+j*Nx] = 1. / cos (Hy * (j) ) * KORIOLIS_KEY * (J1 + J2 + J3) / 3.; //  (J1a+J2a+J3a)/3.;
			//ko[i+j*Nx] = 1. / cos (Hy * (j) ) * KORIOLIS_KEY * J2;

		}




	}

	return ;
}
