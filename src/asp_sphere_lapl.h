#ifndef _SPHERE_LAPL_H
#define _SPHERE_LAPL_H
/*$Id$*/

/* Copyright (c) 2003, 2004, 2005, 2007, 2010 Alexey Ozeritsky
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

/*!
  Шаги на сфере и полусфере
  суффикс _1 у переменных означает переменная в -1 степени
 */
struct SSteps {
	bool full;      //!<полная сфера ?
	int n_phi;      //!<шагов по phi (широте)
	int n_la;       //!<шагов по lambda (долготе)

	double d_phi;   //!<шаг по широте
	double d_phi_1;
	double d_la;    //!<шаг по долготе
	double d_la_1;
	double d_phi2;  //!<2 * шаг по широте
	double d_phi2_1;
	double d_la2;   //!<2 * шаг по долготе
	double d_la2_1;

	SSteps(int _n_phi, int _n_la, bool _full = false) 
	: full(_full), n_phi(_n_phi), n_la(_n_la) 
	{
		d_la    = 2. * M_PI / ((double) n_la);
		d_la_1  = 1.0 / d_la;
		if (full)
			d_phi = M_PI / (double) n_phi;
		else
			d_phi = M_PI / (double)(2 * (n_phi - 1) + 1);
		d_phi_1 = 1.0 / d_phi;

		d_phi2   = d_phi * d_phi;
		d_phi2_1 = 1.0 / d_phi2;
		d_la2    = d_la * d_la;
		d_la2_1  = 1.0 / d_la2;
	}
};


/*!
  Дивергенция поля (u, v)
  dv(i,j) = 1/cos*[ d(cost*v(i,j))/dtheta + d(u(i,j))/dlambda ]
 */
class SDiv: public SSteps {
public:
	/*!
	   Конструктор
	   \param n_phi - число шагов по широте
	   \param n_la  - число шагов по долготе
	   \param full  - использовать полную сферу или полусферу?
	*/
	SDiv(int n_phi, int n_la, bool full = false);
	~SDiv();

	void calc(double * dest, const double * u, const double * v);
};

/*!
  Завихренность поля (u, v)
  vt(i,j) =  [-dv/dlambda + d(cost*u)/dtheta]/cost
 */
class SVorticity: public SSteps {
public:
	/*!
	   Конструктор
	   \param n_phi - число шагов по широте
	   \param n_la  - число шагов по долготе
	   \param full  - использовать полную сферу или полусферу?
	*/
	SVorticity(int n_phi, int n_la, bool full = false);
	~SVorticity();

	void calc(double * dest, const double * u, const double * v);
};

/*!
   Оператор Лапласа на единичной сфере
   \f[
    \delta u(\phi, \lambda, t) := \frac{1}{cos(\phi)}
     \frac{\partial}{\partial \phi} cos(\phi) \frac{\partial u}{\partial \phi} +
    + \frac{1}{cos^2 \phi} \frac{\partial ^2 u}{\partial ^2 \lambda}
	\f]
   \f$\phi \in [0,\pi/2]\f$ - широта
   \f$\lambda \in [0,2\pi]\f$ - долгота
*/
class SLaplacian {
	class Private;
	Private *d;

public:
	//!граничные условия
	enum {
		BC_DIRICHLET = 1, //!< условие Дирихле (значение функции)
		BC_NEUMANN   = 2, //!< условие Неймана (значение производной)
	};

	/*!
	   Конструктор
	   \param n_phi - число шагов по широте
	   \param n_la  - число шагов по долготе
	   \param full  - использовать полную сферу или полусферу?
	*/
	SLaplacian(int n_phi, int n_la, bool full = false);
	~SLaplacian();

	/*!
	   Значение оператора в точке (i, j)
	   \param M - значения функции от которой находим оператор Лапласа
	*/
	double lapl(const double * M, int i, int j);

	/*!
	   Значения оператора Лапласа во всех точках сетки
	   \param Dest - ответ
	   \param M - значения функции от которой находим оператор Лапласа
	*/
	void lapl(double * Dest, const double * M);

	/*! компонента по lambda */
	void lapl_la(double * Dest, const double * M);

	/*! компонента по phi */
	void lapl_phi(double * Dest, const double * M);

	/*!
	   Значения обратного оператора Лапласа во всех точках сетки
	   \param Dest - ответ
	   \param M - значения функции от которой находим обратный оператор Лапласа
	   \param mult - множитель
	   \param diag - диагональная добавка	   
	*/
	void lapl_1(double * Dest, const double * Source, double mult = 1.0, double diag = 0.0, int bc = BC_DIRICHLET);

	/*!
	  Значения обратного оператора Лапласа во всех точках сетки.
	  вызывает lapl_1(Dest, Source, 1.0, 0.0)
	  Добавлено для совместимости. 
	   \param Dest - ответ
	   \param M - значения функции от которой находим обратный оператор Лапласа	  
	 */
	void solve(double * Dest, const double * Source)
	{
		lapl_1(Dest, Source, 1.0, 0.0);
	}

	/*!
	   Значения обратного оператора Лапласа во всех точках сетки
	   \param Dest - ответ
	   \param M - значения функции от которой находим обратный оператор Лапласа
	   \param mult - множитель (для каждого i)
	   \param diag - диагональная добавка (для каждого i)  
	*/
	void lapl_1(double * Dest, const double * Source, double * mult, double * diag, int bc = BC_DIRICHLET);
	void baroclin_1(
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
		double diag_w2_2);

	/*!скалярное произведение в сферических координатах*/
	double scalar(const double *u, const double *v);

	void filter(double *Dest, const double * Source);
	void filter_1(double *Dest, const double * Source);

	double phi(int i);
	double lambda(int i);
};

#endif //_SPHERE_LAPL_H

