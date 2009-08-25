#ifndef _ASP_SINGLETON_H
#define _ASP_SINGLETON_H
/*$Id$*/

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

/**
 * Логгер и фабрика объектов
 */

#include <stdio.h>
#include <stdarg.h>
#include <algorithm>
#include <vector>
#include <map>
#include <sstream>

#include "asp_misc.h"

#ifdef __cplusplus
#define LOG1(X)                  Logger::Instance().printf((X))
#define LOG2(X, Y)               Logger::Instance().printf((X), (Y))
#define LOG3(X, Y, Z)            Logger::Instance().printf((X), (Y), (Z))
#define LOG4(X1, X2, X3, X4)     Logger::Instance().printf((X1), (X2), (X3), (X4))
#define LOG5(X1, X2, X3, X4, X5) Logger::Instance().printf((X1), (X2), (X3), (X4), (X5))
#endif

class Logger {
public:
	static Logger & Instance()
	{
		static Logger log;
		return log;
	}

	void setFile(FILE * f)
	{
		logs.push_back(f);
	}

	void printf(const char * format, ...)
	{
		std::vector < FILE * >::iterator i, s = logs.begin(), e = logs.end();
		for (i = s; i < e; ++i)
		{
			va_list l;
			va_start(l, format);
			vfprintf(*i, format, l); fflush(*i);
			va_end(l);
		}
	}

	void clear()
	{
		std::for_each(logs.begin(), logs.end(), fclose);
		logs.clear();
	}

	void printfvector(double *h, int nn, int n, const char * format)
	{
		std::vector < FILE * >::iterator i, s = logs.begin(), e = logs.end();
		for (i = s; i < e; ++i) {
			asp::fprintfvector(*i, h, nn, n, format);
		}
	}
private:
	std::vector < FILE * > logs;

	Logger()
	{
		logs.push_back(stdout);
	}

	~Logger()
	{
		std::for_each(logs.begin(), logs.end(), fclose);
	}

	Logger(const Logger &);
	Logger &operator = (const Logger &);
};

template <
	class IdentifierType,
	class ProductType
>
class DefaultFactoryError
{
public:
	class Exception: public std::exception
	{
	public:
		Exception(const IdentifierType & id)
			: unknownId_(id)
		{
		}

		~Exception() throw()
		{
		}

		virtual const char * what() const throw()
		{
			std::ostringstream ostr;
			ostr << "Unknown type : " << unknownId_;
			return ostr.str().c_str();
		}

		const IdentifierType GetId()
		{
			return unknownId_;
		}

	private:
		IdentifierType unknownId_;
	};

protected:
	static ProductType * OnUnknownType(const IdentifierType & id)
	{
		throw Exception(id);
	}
};

template <
	class AbstractProduct,
	typename IdentifierType,
	typename ProductCreator,
	template <typename, class> class FactoryErrorPolicy = DefaultFactoryError
>
class Factory: public FactoryErrorPolicy < IdentifierType, AbstractProduct >
{
public:
	bool Register(const IdentifierType & id, ProductCreator creator)
	{
		return assiciations_.insert(typename AssocMap::value_type(id, creator)).second;
	}

	bool Unregister(const IdentifierType & id)
	{
		return assiciations_.erase(id) == 1;
	}

	AbstractProduct * CreateObject(const IdentifierType & id)
	{
		typename AssocMap::const_iterator i = assiciations_.find(id);
		if (i != assiciations_.end())
		{
			return (i->second)();
		}
		return FactoryErrorPolicy < IdentifierType, AbstractProduct >::OnUnknownType(id);
	}

protected:
	typedef std::map<IdentifierType, ProductCreator> AssocMap;
	AssocMap assiciations_;
};
#endif //_ASP_SINGLETON_H
