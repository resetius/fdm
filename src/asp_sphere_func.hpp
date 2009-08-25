#ifndef _SPHERE_FUNC_HPP
#define _SPHERE_FUNC_HPP
/*$Id: asp_sphere_func.hpp 1311 2006-09-18 09:38:55Z manwe $*/

/* Copyright (c) 2006 Alexey Ozeritsky
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

#include <vector>
#include <map>

using namespace std;

/*! полиномы */
template < typename T >
class Poly {
protected:
	T arg_;
	map < int, T > nodes_;	

public:

	Poly (): arg_(0) {}

	Poly (const map < int, T > &nodes):
		arg_(0), nodes_(nodes) {}


	Poly (const T &x): arg_(x) {
	}

	Poly ( const Poly < T > & p ):
	arg_(p.arg_), nodes_(p.nodes_)
	{
	}

	Poly < T > & operator = (const Poly < T > & p) {
		nodes_ = p.nodes_;
		arg_   = p.arg_;
		return *this;
	}

	virtual ~Poly() {};

	void addNode(int value, int deg) {
		nodes_[deg] = value;
	}

	T arg() const {
		return arg_;
	}

	T result(const T & x) {	
		T res = nodes_[0];
		T xx;
		typename map < int, T >::const_iterator it  = nodes_.begin();
		typename map < int, T >::const_iterator ite = nodes_.end();

		if (it->first == 0)
			++it;
		for (;it != ite; ++it) {
			xx = ipow(x, it->first);
			res += it->second * xx;
		}
		return res;
	}

	template < typename X >
	Poly & operator *=(const X & x) {
		typename map < int, T >::iterator it  = nodes_.begin();
		typename map < int, T >::iterator ite = nodes_.end();
		for (;it != ite; ++it) {
			it->second *= x;
		}
		return *this;
	}

	map < int, T > coefs() const {
		return nodes_;
	}

	Poly & operator -= (const Poly & p) {
		map < int, T > cs = p.coefs();
		typename map < int, T >::const_iterator it  = cs.begin();
		typename map < int, T >::const_iterator ite = cs.end();		

		for (;it != ite; ++it) {
			nodes_[it->first] -= it->second;
		}
		return *this;
	}

	Poly & operator += (const Poly & p) {
		typename map < int, T >::iterator it  = nodes_.begin();
		typename map < int, T >::iterator ite = nodes_.end();
		map < int, T > cs = p.coefs();

		for (;it != ite; ++it) {
			it->second += cs[it->first];
		}
		return *this;
	}

	//производная порядка m
	Poly diff(int m) const {
		if (m == 0) {
			return Poly < T > (nodes_);
		}

		typename map < int, T >::const_iterator it  = nodes_.begin();
		typename map < int, T >::const_iterator ite = nodes_.end();
		map < int, T > newnodes;

		for (;it != ite; ++it) {
			int n = it->first;
			//узел обнуляется
			if (n - m < 0) continue;

			T mult = static_cast<T>(n);
			for (int i = 1; i < m; ++i) {
				mult *= static_cast<T>(n - i);
			}
			mult *= it->second;
			newnodes[n - m] = mult;
		}
		
		return Poly < T > (newnodes);
	}

	operator T () {
		return result(arg_);
	}

	//умножаем на x ^ n
	void degUp(int n = 1) 
	{
		map < int, T > newnodes;
		typename map < int, T >::const_iterator it  = nodes_.begin();
		typename map < int, T >::const_iterator ite = nodes_.end();

		for (;it != ite; ++it)
		{
			newnodes[it->first + n] = it->second;
		}
		nodes_ = newnodes;
	}
};

/*!полиномы Лежандра
 \f[
 P_i = \frac{1}{n!2^n} \frac{d^n}{dx^n}(x^2-1)^n
 \f]
 \f[
 P_0(x) = 1, P_1(x) = x, P_2(x) = 1/2(3x^2-1), P_3(x) = 1/2(5x^3-3x)
 \f]
 \f[
 P_{n+1} = \frac{1}{n+1}((2n+1)xP_n(x)-nP_{n-1}(x)
 \f]
*/
template < typename T >
class Lezh: public Poly < T >
{
	int n_;

public:
	Lezh(const T & x, int n): Poly < T > (x), n_(n) { init(); }
	Lezh(int n): n_(n) { init(); }

	Lezh < T > & operator = (const Lezh < T > & p) {
		Poly < T >::nodes_ = p.nodes_;
		Poly < T >::arg_   = p.arg_;
		return *this;
	}

	Lezh < T > & operator = (const Poly < T > & p) {
		Poly < T >::nodes_ = p.coefs();
		Poly < T >::arg_   = p.arg();
		return *this;
	}

	void init() {
		switch (n_) {
		case 0:
			init_0();
			break;
		case 1:
			init_1();
			break;
		default:
			init_n();
		}
	}

	void init_n() {
		Lezh < T > l1(n_ - 1);
		l1 *= static_cast<T>(2 * n_ - 1);
		l1.degUp();

		Lezh < T > l2(n_ - 1);
		l2 *= static_cast<T>(n_ - 1);
		l1 -= l2;
		l1 *= static_cast<T>(1) / static_cast<T>(n_);
		Poly < T >::nodes_ = l1.coefs();
	}

	void init_0() {
		Poly < T >::addNode(1, 0);
	}

	void init_1() {
		Poly < T >::addNode(1, 1);
	}
};

/*!
 функции Лежандра
 \f[
 P_n^m(x) = (1-x)^{m/2}\frac{d^mP_n}{dx^m}
 \f]
 */
template < typename T >
class LezhFunc {
	int n_;
	int m_;
	Lezh < T > l_;
	Poly < T > p_;

public:
	LezhFunc(int n, int m = 0): n_(n), m_(m), l_(n) {}

	void setM(int m) {
		if (m == m_)
			return;
		p_ = l_.diff(m);
		m_ = m;
	}

	T result (const T & x, int m) {
		setM(m);
		return result(x);
	}

	T result (const T & x) {
		T res;
		if (m_ % 2 == 0) {
			res  = ipow(x, m_ / 2);
			res *= p_.result(x);
		} else {
			res  = ipow(x, m_);
			res  = sqrt(res);
			res *= p_.result(x);
		}
		return res;
	}
};

/*!
 сферические функции
 \f[
 Yn(\phi, \lambda) = 
 \sum_{m=0}^{n}(A_{nm}cos m\lambda+B_{nm} sin m\lambda)P_n^m(sin \phi)
 \f]
 */
template < typename T >
class Yn {
	int n_;
	LezhFunc < T > lf;

public:
	Yn(int n): n_(n), lf(n) {}

	T result(const T & phi, const T & la) {
		T res = 0;

		for (int m = 0; m <= n_; ++m) {
			lf.setM(m);
			res += (cos(m * la) + sin(m * la)) * lf.result(sin(phi));
		}
		return res;
	}
};
#endif //_SPHERE_FUNC_HPP
