#ifndef GIDROSTABLE_CONFIG_H
#define GIDROSTABLE_CONFIG_H
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

#include <map>
#include <string>

struct ConfigSkeleton
{
	enum Flags
	{
		OPTIONAL = 0,
		REQUIRED = 1
	};

	// parameter_name -> flags, docstring
	typedef std::map < std::string, std::pair < Flags, std::string > > section_t;
	// section_name -> section
	typedef std::map < std::string, section_t > data_t;

	data_t data;

	bool is_required(const std::string & section, const std::string & prm);
};

class Config
{
	typedef std::map < std::string, std::string > section_t;
	typedef std::map < std::string, section_t > data_t;

	data_t data;
	std::string filename;
	ConfigSkeleton skeleton;

	void load(FILE * f);

public:
	Config();
	~Config();

	void set_skeleton(const ConfigSkeleton & s) { skeleton = s; }
	void rewrite(int argc, char ** argv);
	bool validate() const;
	void help() const;

	void open(const std::string & filename);
	void save() const;
	void print(FILE * f) const;

	double get(const std::string & section, const std::string & name, double def);
	double getd(const std::string & section, const std::string & name);

	int get(const std::string & section, const std::string & name, int def);
	int geti(const std::string & section, const std::string & name);

	std::string get(const std::string & section, const std::string & name, const std::string & def);
	std::string gets(const std::string & section, const std::string & name);
};

#endif /* GIDROSTABLE_CONFIG_H */

