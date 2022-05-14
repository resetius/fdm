#ifndef _SDS_EXCS
#define _SDS_EXCS
/* Copyright (c) 2004, 2005, 2022 Alexey Ozeritsky
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
*   contributors may be used to endorse or promote products derived from
*   this software without specific prior written permission.
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
 * Исключения.
 * Генерация исключительных ситуаций в программе.
 */

#include <string>
#include <sstream>
#include <exception>

/**
* базовый класс
*/
class Exception: public std::exception {
protected:
	std::string name_;

public:
	Exception() {}
	Exception(const char * name): name_(name) {}
	Exception(const char *file, int line)
	{
		std::ostringstream out;
		out << file << ":" << line <<": Exception";
		name_ = out.str();
	}

	Exception(const char *name, const char * file, int line)
	{
		std::ostringstream out;
		out << file << ":" << line << " " << name;
		name_ = out.str();
	}

	virtual ~Exception() throw() {}

	const char * what() const throw() {
		return name_.c_str();
	}

	virtual std::string toString() {
		return name_;
	}
};

/**
* переполнение индекса
*/
class IndexOutOfRange: public Exception {
public:
	IndexOutOfRange(int l, int r, int idx)
	{
		std::ostringstream out;
		out << "IndexOutOfRange, index=" << idx << " range=["<<l<<","<<r<<"]";
		name_ = out.str();
	}

	virtual ~IndexOutOfRange() throw() {};
};

/**
* неверный аргумент функции
*/
class BadArgument: public Exception {
public:
	BadArgument(const char *name): Exception(name) {}
	BadArgument(const char *file, int line)
	{
		std::ostringstream out;
		out << file << ":" << line <<": Bad Argument";
		name_ = out.str();
	}

	BadArgument(const char *s, const char * file, int line)
		:Exception(s, file, line)
	{
	}

	virtual ~BadArgument() throw() {}
};

/**
* нереализовано
*/
class NotImplemented: public Exception {
public:
	NotImplemented(const char *name): Exception(name) {}
	NotImplemented(const char *file, int line)
	{
		std::ostringstream out;
		out << file << ":" << line <<": function not implemented";
		name_ = out.str();
	}

	NotImplemented(const char *name, const char * file, int line)
	{
		std::ostringstream out;
		out << file << ":" << line <<": " << name << " not implemented";
		name_ = out.str();
	}

	virtual ~NotImplemented() throw() {}
};

/**
* полудинамическая система не линеаризована
*/
class NotLinearized: public NotImplemented {
public:
	NotLinearized(const char *s, const char * file, int line)
		:NotImplemented(s,file,line)
	{
	}

	virtual ~NotLinearized() throw() {}
};

class NotFound: public BadArgument {
public:
	NotFound(const char *n, const char *f, int l)
		:BadArgument(n, f, l)
	{
		name_ = std::string("not found: ") + name_;
	}

	NotFound(const char *n)
		:BadArgument(n)
	{
		name_ = std::string("not found: ") + name_;
	}

	virtual ~NotFound() throw() {}
};
#endif // _SDS_EXCS
