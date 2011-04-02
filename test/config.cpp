/* Copyright (c) 2010 Alexey Ozeritsky
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
 * 3. The name of the author may not be used to endorse or promote products
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

#include <stdio.h>
#include <string.h>
#include <sstream>
#include <stdexcept>

#include "config.h"

using namespace std;

template < typename A, typename B >
A lexical_cast (const B & b)
{
	A a;
	std::stringstream str;
	str.setf(ios::scientific, ios::floatfield);
	str.precision(16);
	str << b;
	str >> a;
	return a;
}

Config::Config()
{
}

Config::~Config()
{
}

void Config::open(const std::string & fn)
{
	filename = fn;

	FILE * f = fopen(fn.c_str(), "rb");
	if (!f) {
		return;
	}

	load(f);

	fclose(f);
}

void Config::load (FILE * f)
{
#define BUF_SZ 4096
	char buf[BUF_SZ];
	char section[BUF_SZ];
	const char * sep = " =\t";
	char * comment;

	section_t * cur = 0;

	while (fgets (buf, BUF_SZ, f) )
	{
		if (sscanf (buf, "[%4000s]", section) == 1)
		{
			section[strlen (section) - 1] = 0;
			//fprintf(stderr, "section -> %s\n", section);
			cur = &data[section];
		}
		else if (*buf == '#' || *buf == ';')
		{
			continue;
		}
		else if (cur)
		{
			char * k = 0, * v = 0;
			if (buf[strlen(buf) - 1] == '\n') buf[strlen(buf) - 1] = 0;
			if (buf[strlen(buf) - 1] == '\r') buf[strlen(buf) - 1] = 0;
			comment = strstr(buf, ";");
			if (comment) {
				*comment = 0;
			}

			k = strtok (buf, sep);
			v = strtok (0, sep);

			//fprintf(stderr, "%s:%s\n", k, v);
			if (k && v)
			{
				(*cur) [k] = v;
			}
		}
	}
}

void Config::rewrite (int argc, char ** argv)
{
	for (int i = 1; i < argc; ++i)
	{
		// --section_name:option_name=value
		string s = argv[i];
		if (s.find ("--") != 0)
		{
			continue;
		}
		s = s.substr (2);
		size_t pos = s.find (":");
		if (pos == string::npos)
		{
			continue;
		}
		string section_name = s.substr (0, pos);
		s = s.substr (pos + 1);
		pos = s.find ("=");
		if (pos == string::npos)
		{
			continue;
		}
		string option_name = s.substr (0, pos);
		string value = s.substr (pos + 1);

		data[section_name][option_name] = value;
	}
}

bool Config::validate() const
{
	for (ConfigSkeleton::data_t::const_iterator it = skeleton.data.begin();
		it != skeleton.data.end(); ++it)
	{
		string section_name = it->first;
		data_t::const_iterator tt = data.find(section_name);
		if (tt == data.end()) {
			continue;
		}

		for (ConfigSkeleton::section_t::const_iterator jt = 
			it->second.begin(); jt != it->second.end(); ++jt)
		{
			if (tt->second.find(jt->first) == tt->second.end())
			{
				if (jt->second.first & ConfigSkeleton::REQUIRED) {
					fprintf(stderr, "%s:%s REQUIRED\n", section_name.c_str(),
						jt->first.c_str());
					return false;
				} else {
					continue;
				}
			}

			// TODO: check data type and ranges here
		}
	}

	return true;
}

void Config::help() const
{
	for (ConfigSkeleton::data_t::const_iterator it = skeleton.data.begin();
		it != skeleton.data.end(); ++it)
	{
		string section_name = it->first;

		for (ConfigSkeleton::section_t::const_iterator jt = 
			it->second.begin(); jt != it->second.end(); ++jt)
		{
			fprintf(stderr, "--%s:%s -- ", section_name.c_str(), jt->first.c_str());
			if (jt->second.first & ConfigSkeleton::REQUIRED) {
				fprintf(stderr, "REQUIRED -- ");
			} else {
				fprintf(stderr, "OPTIONAL -- ");
			}
			fprintf(stderr, "%s\n", jt->second.second.c_str());
		}
	}
}

void Config::save() const
{
	FILE * f = fopen(filename.c_str(), "wb");
	if (!f) return;
	print(f);
	fclose(f);
}

void Config::print(FILE * f) const
{
	for (data_t::const_iterator it = data.begin(); it != data.end(); ++it)
	{
		fprintf(f, "[%s]\n", it->first.c_str());
		for (data_t::mapped_type::const_iterator jt = it->second.begin();
			jt != it->second.end(); ++jt)
		{
			fprintf(f, "%s = %s\n", jt->first.c_str(), jt->second.c_str());
		}
		fprintf(f, "\n");
	}
}

bool ConfigSkeleton::is_required(const std::string & section, const std::string & prm)
{
	if (data.find(section) == data.end()) {
		return false;
	}

	if (data[section].find(prm) == data[section].end()) {
		return false;
	}

	return data[section][prm].first & ConfigSkeleton::REQUIRED;
}

double Config::get(const std::string & section, const std::string & name, double def)
{
	// TODO: check type and range
	return lexical_cast < double > (get(section, name, lexical_cast < string > (def)));
}

double Config::getd(const std::string & section, const std::string & name)
{
	return lexical_cast < double > (gets(section, name));
}

int Config::get(const std::string & section, const std::string & name, int def)
{
	// TODO: check type and range
	return lexical_cast < int > (get(section, name, lexical_cast < string > (def)));
}

int Config::geti(const std::string & section, const std::string & name)
{
	return lexical_cast < int > (gets(section, name));
}

std::string Config::get(const std::string & section, const std::string & name, const std::string & def)
{
	if (skeleton.is_required(section, name)) {
		stringstream str;
		str << "cannot use with default\n";
		str << "parameter " << section << ":" << name << " is required";
		throw runtime_error(str.str());
	}

	if (data[section].find(name) == data[section].end()) {
		data[section][name] = def;
	}

	return data[section][name];
}

std::string Config::gets(const std::string & section, const std::string & name)
{
	if (data.find(section) == data.end()) {
		stringstream str;
		str << "section " << section << " not found";
		throw runtime_error(str.str());
	}

	if (data[section].find(name) == data[section].end()) {
		stringstream str;
		str << "parameter " << section << ":" << name << " not found";
		throw runtime_error(str.str());
	}

	return data[section][name];
}
