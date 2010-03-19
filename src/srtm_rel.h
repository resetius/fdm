#ifndef SRTM_REL_H
#define SRTM_REL_H

#include <string>

class ReliefLoader
{
	struct Data;
	Data * d;

	bool loaded;
	std::string path;

public:
	ReliefLoader(const std::string & path);
	~ReliefLoader();

	void get(double * out, long nlat, long nlon, bool full, bool offset);
};

#endif /* SRTM_REL_H */

