/*$Id$*/

/* Copyright (c) 2004, 2005 Alexey Ozeritsky
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
 * вычисление синуса и косинуса и экспоненты с точностью более 16 знаков.
 */
 
#include <cstdlib>
#include <iostream>

using namespace std;

#include "asp_mp.h"

//математические константы
mpf *Pi;
mpf *Half_Pi;
mpf *Two_Pi;
mpf *E;
mpf *Eps;      //машинный нуль
mpf *Identity; //единица
mpf *Ln10;     //натуральный логарифм десяти
mpf *Ln2;      //натуральный логарифм двойки

/**
 * целочисленная степень
 */
mpf pow(const mpf &t, int i) {
	mpf_t r;
	mpf_init(r);
	mpf_pow_ui(r, t.get_mpf_t(), i);
	return mpf(r);
}


/**
 * натуральный логарифм
 * считаем как ряд
 * ln(x) = (x-1) - (x-1)^2/2 + (x-1)^3/3 - (x-1)^4/4 + ...
 * область сходимости ряда -1 < x - 1 < 1, 
 * нормализуем x так, что -0.5 < x - 1 < 0.5, то есть
 * 0.5 < x < 1.5
 */
mpf log(const mpf &r) {
	mpf ret(0.0);
	mpf x(r); //t неизменяемо, поэтому вводим новую переменную
	/**
	 * нормализуем x так, что 0.5 < x < 1.5
	 */
	int j = 0;
	mpf t(0.5);
	while (cmp(x, t) < 0) {
		x = x * (*E);
		j--;
    }
    t = 1.5;
    while (cmp(x, t) > 0) {
    	x = x / *E;
    	j++;
    }
    
    x = x - 1;

    int i = 1;
    mpf p(1);
    while (1) {
    	p = p * x; // (x-1)^i
    	t = p / i; // (x-1)^i / i
    	
    	if (cmp(abs(t), *Eps) < 0) break;
    	if (i % 2 == 0) {
    		ret = ret - t;
    	} else {
    		ret = ret + t;
    	}
    	i++;
    }
    
   	ret = ret + j;
   	return ret;
}

/**
 * десятичный логарифм
 * log10(arg) = log(arg)/log(10)
 */
mpf log10(const mpf &r) {
	return log(r) / *Ln10;
}

/**
 * вычисление синуса sin(x)
 * убираем период 2pi, то есть
 * рассматриваем x как x = n *2pi + r
 * далее вычисляем 
 * sin(r) = x - x^3/3 + ... + (-1)^n x^{2n+1}/(2n+1)!+...
 */
mpf sin(const mpf&x) {
	/**
	 * отдельно рассматриваем случаи
	 * x = 0
	 */
	if (cmp(abs(x), *Eps) < 0) {
		return mpf(0);
	}
	/**
	 * находим разложение
	 * x = n*2pi + r
	 */
	mpf q = x / *Two_Pi;    //n + r/(2pi)
	mpf factor = floor(q);  //n
	q = factor * (*Two_Pi); //n*2pi
	mpf r = x - q;          //r
	if (cmp(abs(r), *Eps) < 0) {
		return mpf(0);
	}	
	
	//находим sin(r) = x - x^3/3 + ... + (-1)^n x^{2n+1}/(2n+1)!+...
	mpf ret(0);
	mpf rr(1);
	mpf fact(1);
	mpf term;
	int sign = -1, i = 0;
	while (1) {
		i++;
		rr   = rr * r;    //x^i
		fact = fact * i;  //i!
		if (i % 2 == 0) continue;
		term = rr / fact; //x^{2n+1}/(2n+1)!
		if (cmp(abs(term), *Eps) < 0) break;
		sign = -sign;
		if (sign > 0 ) {
			ret = ret + term;
		} else {
			ret = ret - term;
		}
	}
	
	return ret;
}

/**
 * вычисление косинуса cos(x)
 * убираем период 2pi, то есть
 * рассматриваем x как x = n *2pi + r
 * далее вычисляем 
 * cos(r) = 1 - x^2/2! + ... + (-1)^n x^{2n}/(2n)!+...
 */
mpf cos(const mpf&x) {
	/**
	 * отдельно рассматриваем случаи
	 * x = 0
	 */
	if (cmp(abs(x), *Eps) < 0) {
		return mpf(1);
	}
	/**
	 * находим разложение
	 * x = n*2pi + r
	 */
	mpf q = x / *Two_Pi;    //n + r/(2pi)
	mpf factor = floor(q);  //n
	q = factor * (*Two_Pi); //n*2pi
	mpf r = x - q;          //r
	if (cmp(abs(r), *Eps) < 0) {
		return mpf(1);
	}	
	
	//находим cos(r) = 1 - x^2/2! + ... + (-1)^n x^{2n}/(2n)!+...
	mpf ret(1);
	mpf rr(1);
	mpf fact(1);
	mpf term;
	int sign = 1, i = 0;
	while (1) {
		i++;
		rr   = rr * r;    //x^i
		fact = fact * i;  //i!
		if (i % 2 != 0) continue;
		term = rr / fact; //x^{2n+1}/(2n+1)!
		if (cmp(abs(term), *Eps) < 0) break;
		sign = -sign;
		if (sign > 0 ) {
			ret = ret + term;
		} else {
			ret = ret - term;
		}
	}
	
	return ret;
}

/**
 * вычисление e^x,
 * рассматриваем x как x = n ln2 + r, тогда
 * exp(x) = 2^n * exp(r).
 * exp(r) находим по формуле Тейлора
 */
mpf exp(const mpf&x) {
	/**
	 * отдельно рассматриваем случаи
	 * x = 0 и x = 1
	 */
	if (cmp(abs(x), *Eps) < 0) {
		return mpf(1);
	} else if (cmp(abs(x - *Identity), *Eps) < 0) {
		return *E;
	}
	
	/**
	 * находим разложение
	 * x = n ln2 + r
	 */
	mpf q = x / *Ln2;     //n + r/ln2
	mpf power = floor(q); //n
	q = power * (*Ln2);   //n ln2
	mpf r = x - q;            //r
	//вычисляем exp(r) = 1 + r/1 + r^2/2! + ... + r^2/i! + ...
	int i = 1;
	mpf rr(1);
	mpf fact(1);
	mpf er(1);
	mpf term;
	while (1) {
		rr   = rr * r;    //r^i
		fact = fact * i;  //i!
		term = rr / fact; //r^i/i!
		if (cmp(abs(term), *Eps) < 0) break;
		er = er + term;   //exp(r)
		i++;
	}
	
	//exp(x) = 2^n * exp(r).
	mpf ret;
	int n = (int)(power.get_d());
	if (n > 0) {
		ret = er * pow(mpf(2), n);
	} else {
		ret = er / pow(mpf(2), n);
	}
	return ret;
}

/*находит машинный нуль*/
mpf mp_machine_epsilon() {
	mpf e(1.0),e1;
	mpf identity(1.0);
	do {
		e=e/2.0;e1=e+1.0;
	}
	while(cmp(e1,identity)>0);
	return e;
}

/**
 * устанавливает точность и инициализирует
 * математические константы
 */
void mp_init(unsigned long int prec) {
	mpf_set_default_prec(prec);
	
	Pi       = new mpf();
	Half_Pi  = new mpf();
	Two_Pi   = new mpf();
	E        = new mpf(0.0);
	Eps      = new mpf(mp_machine_epsilon());
	Identity = new mpf(1.0);
	Ln10     = new mpf();
	Ln2      = new mpf();
	
	mpf a(1.0);
	mpf b(4.0);

	/**
	 * ищем E как сумму ряда
	 * E = \sum_i 1/i!
	 */
	int i = 1;
	while (1) {
		if (cmp(a, *Eps) < 0) break;
		*E = *E + a;
		a = a / i;
		i++;
	}
	
	/**
	 * ищем Pi по формуле 
	 * Bailey, Borwein and Plouffe
     * Pi = \sum_{i=0}^\infty (1/16)^i *
     *       * [4/(8i+1) - 2/(8i+4) - 1/(8i+5) - 1/(8i+6)]
	 */
	i = 0;
	while (1) {
		b = 4;
		b = b / (8 * i + 1); /* 4/(8n+1) */
		
		a = 2;
		a = a / (8 * i + 4); /* 2/(8n+4) */
		b = b - a;
		
		a = 1;
		a = a / (8 * i + 5); /* 1/(8n+5) */
		b = b - a;

		a = 1;
		a = a / (8 * i + 6); /* 1/(8n+6) */
		b = b - a;

		a = 16;
		a = pow(a, i); /* 16^n */

		b = b / a;

		if (cmp(b, *Eps) < 0) break;
		*Pi = *Pi + b;
		i++;
	}
	
	*Half_Pi = *Pi / 2;
	*Two_Pi  = *Pi * 2;
	*Ln10    = log(mpf(10));
	*Ln2     = log(mpf(2));
}
