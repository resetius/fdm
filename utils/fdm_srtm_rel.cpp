#define _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>

#include <vector>
#include <string>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#ifdef WIN32
#include <windows.h>
#define htons(x) \
	(((((unsigned short)(x)) >>8) & 0xff) | \
                ((((unsigned short)(x)) & 0xff)<<8))
#else
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <stdint.h>
#endif

#include "fdm_srtm_rel.h"

using namespace std;

#ifdef WIN32
	typedef __int16 int16_t;
	typedef HANDLE file_t;
#else
	typedef int file_t;
#endif

struct ReliefLoader::Data {
	int16_t * d_;

#ifdef WIN32
	HANDLE fd_;
	HANDLE map_;
#else
	int fd_;
	size_t sz_;
#endif

	Data(): d_(0), 
		fd_((file_t)-1)
	{
	}

	~Data()
	{
#ifdef WIN32
		if (fd_ >= (file_t)-1) CloseHandle(fd_);
		if (map_) CloseHandle(map_);
		if (d_) UnmapViewOfFile(d_);
		map_ = 0;
#else
		if (d_) munmap(d_, sz_);
		if (fd_ >= 0) close(fd_);
#endif
		d_  = 0;
		fd_ = (file_t)-1;
	}

	bool load(const char * path) {
#ifdef WIN32
		fd_ = CreateFile(path, GENERIC_READ, NULL, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

		if (fd_ < 0) {
			return false;
		}

		map_ = CreateFileMapping(fd_, NULL, PAGE_READONLY, NULL, NULL, path);
		if (!map_) {
			CloseHandle(fd_);
			return false;
		}

		d_ = (int16_t *)MapViewOfFile(map_, FILE_MAP_READ, NULL, NULL, NULL);
		if (!d_) {
			CloseHandle(fd_);
			CloseHandle(map_);
			return false;
		}

		return true;
#else
		struct stat buf;
		fd_ = open(path, O_RDONLY);
		if (fd_ < 0) {
			return false;
		}

		if (stat(path, &buf) == 0) {
			sz_ = buf.st_size;
		} else {
			close(fd_); fd_ = -1;
			return false;
		}

		assert(sz_ >= 2 * 43200 * 21600);

		d_ = (int16_t*)mmap(0, sz_, PROT_READ, MAP_SHARED, fd_, 0);
		if (d_ == (void*)-1) {
			close(fd_); fd_ = -1;
			return false;
		}

		return true;
#endif
	}

	double get(double u, double v) {
		int i, j;
		assert(-M_PI / 2 <= u && u <= M_PI / 2);
		assert(0 <= v && v<= 2 * M_PI);
		j = (int)((v * (43200.0 - 1.0)) / 2.0 / M_PI);
		i = (int)((u + M_PI / 2) * (21600.0 - 1.0) / M_PI);
		i = 21599 - i;

		return (double)((int16_t)htons(d_[i * 43200 + j]));
	}
};


ReliefLoader::ReliefLoader(const string & path): 
	d(new Data), loaded(false), path(path)
{

}

ReliefLoader::~ReliefLoader()
{
	delete d;
}

// offset - отступить на полшага от полюса?
void ReliefLoader::get(double * out, long nlat, long nlon, bool full, bool offset)
{
	if (!loaded) {
		loaded = d->load(path.c_str());
	}

	if (!loaded) {
		fprintf(stderr, "%s not found\n", path.c_str());
		exit(1);
		return;
	}

	double dlat;
	double dlon;
	double phi;
	double lambda;

	if (full && !offset) {
		dlat = M_PI / (nlat - 1);
		dlon = 2. * M_PI / nlon;
	} else if (!full && !offset) {
		dlat = M_PI / (nlat - 1) / 2;
		dlon = 2. * M_PI / nlon;
	} else if (full && offset) {
		dlat = M_PI / (double) nlat;
		dlon = 2. * M_PI / nlon;
	} else if (!full && offset) {
		dlat = M_PI / (double)(2 * (nlat - 1) + 1);
		dlon = 2. * M_PI / nlon;
	}

	int i, j;

	for (i = 0; i < nlat; ++i)
	{
		for (j = 0; j < nlon; ++j)
		{
			if (!offset) {
				phi    = -0.5 * M_PI + i * dlat;
				lambda = j * dlon;
			} else if (offset && !full) {
				phi    = i * dlat;
				lambda = j * dlon;
			} else if (offset && full) {
				phi    = (dlat - M_PI) * 0.5 + (double)i * dlat;
				lambda = j * dlon;
			}

			out[i * nlon + j] = d->get(phi, lambda);
		}
	}
}

#ifdef TEST

void usage(const char * argv)
{
	fprintf(stderr, "%s topo30 [-nlat nlat] [-nlon nlon] [-o output] [-fmt bin|txt] [-type half|full]");
	exit(1);
}

int main(int argc, char ** argv)
{
	string output = "output.rel";
	int nlat = 32;
	int nlon = 24;
	int full = 0;
	int bin  = 0;

	if (argc < 2) {
		usage(argv[0]);
	}

	ReliefLoader loader(argv[1]);

	for (int i = 2; i < argc; ++i)
	{
		if (!strcmp(argv[i], "-nlat")) {
			if (i == argc - 1) {
				usage(argv[0]);
			}

			nlat = atoi(argv[i + 1]);
		} else if (!strcmp(argv[i], "-nlon")) {
			if (i == argc - 1) {
				usage(argv[0]);
			}

			nlon = atoi(argv[i + 1]);
		} else if (!strcmp(argv[i], "-o")) {
			if (i == argc - 1) {
				usage(argv[0]);
			}

			output = argv[i + 1];
		} else if (!strcmp(argv[i], "-fmt")) {
			if (i == argc - 1) {
				usage(argv[0]);
			}

			if (!strcmp(argv[i + 1], "bin")) {
				bin = 1;
			} else if (!strcmp(argv[i + 1], "txt")) {
				bin = 0;
			} else {
				usage(argv[0]);
			}
		} else if (!strcmp(argv[i], "-type")) {
			if (i == argc - 1) {
				usage(argv[0]);
			}
			if (!strcmp(argv[i + 1], "full")) {
				full = 1;
			} else if (!strcmp(argv[i + 1], "half")) {
				full = 0;
			} else {
				usage(argv[0]);
			}
		}
	}

	fprintf(stderr, "nlat=%d\n", nlat);
	fprintf(stderr, "nlon=%d\n", nlon);
	fprintf(stderr, "full=%d\n", full);
	fprintf(stderr, "bin=%d\n", bin);
	fprintf(stderr, "output=%s\n", output.c_str());

	vector < double > rel(nlat * nlon);
	loader.get(&rel[0], nlat, nlon, full, true);

	FILE * f = fopen(output.c_str(), "wb");
	if (!f) {
		usage(argv[0]);
	}

	if (bin) {
		fwrite(&rel, 1, nlat * nlon * sizeof(double), f);
	} else {
		for (int i = 0; i < nlat; ++i)
		{
			for (int j = 0; j < nlon; ++j)
			{
				fprintf(f, "%.16le ", rel[i * nlon + j]);
			}
			fprintf(f, "\n");
		}
	}

	fclose(f);

	return 0;
}
#endif
