/* Copyright (c) 2005 Alexey Ozeritsky
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
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <unistd.h>
#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#endif

#include <sys/stat.h>
#include <sys/types.h>

#include "asp_readdir.h"

typedef unsigned int uint;
using namespace std;

#ifdef _WIN32
int readFolder(vector < string > &folders , const string &path)
{
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	char DirSpec[MAX_PATH + 1];  // directory specification
	DWORD dwError;

	//printf ("Target directory is %s.\n", argv[1]);
	//strncpy (DirSpec, argv[1], strlen(argv[1])+1);
	strncpy (DirSpec, path.c_str(), path.length());
	DirSpec[path.length()] = 0;
	strncat (DirSpec, "\\*", 3);

	hFind = FindFirstFile(DirSpec, &FindFileData);

	if (hFind == INVALID_HANDLE_VALUE) 
	{
		//printf ("Invalid file handle. Error is %u\n", GetLastError());
		return -1;
	} 
	else 
	{
		//printf ("First file name is %s\n", FindFileData.cFileName);
		string folder = FindFileData.cFileName;
		if (folder != "." && folder != "..")
			folders.push_back(folder);

		while (FindNextFile(hFind, &FindFileData) != 0) 
		{
			//printf ("Next file name is %s\n", FindFileData.cFileName);
			folder = FindFileData.cFileName;
			if (folder != "." && folder != "..")
				folders.push_back(folder);
		}

		dwError = GetLastError();
		FindClose(hFind);
		if (dwError != ERROR_NO_MORE_FILES) 
		{
			//printf ("FindNextFile error. Error is %u\n", dwError);
			return -1;
		}
	}
	return 0;
}
#else
int readFolder(vector < string > &folders , const string &path)
{
	DIR *dp;
	struct dirent *dir_entry;
	struct stat stat_info;

	if((dp = opendir(path.c_str())) == NULL) {
		//fprintf(stderr,"cannot open directory: %s\n", dir);
		return -1;
	}
	//chdir(dir);
	while((dir_entry = readdir(dp)) != NULL) {
		lstat(dir_entry->d_name,&stat_info);
		if(S_ISDIR(stat_info.st_mode)) {
			/* Directory, but ignore . and .. */
			if(strcmp(".",dir_entry->d_name) == 0 ||
				strcmp("..",dir_entry->d_name) == 0)
				continue;
			string folder = dir_entry->d_name;
			folders.push_back(folder);
			//printf("%*s%s/\n",indent,"",dir_entry->d_name);
			/* Recurse at a new indent level */
			//ScanDir(dir_entry->d_name,indent+4);
		} else {
			string folder = dir_entry->d_name;
			folders.push_back(folder);
		}
		//else printf("%*s%s\n",indent,"",dir_entry->d_name);
	}
	//chdir("..");
	closedir(dp);
	return 0;
}
#endif

void makeDir(const char * str)
{
#ifdef _WIN32
	_mkdir(str);
#else
	mkdir(str, 0755);
#endif
}

void initFolder(const string &folder)
{
	vector < string > subfolders;

	int id     = -1;
	int old_id = -1;
	do {
		old_id = id;
		id = folder.find('/', old_id + 1);
		if (id > 0) {
			subfolders.push_back(folder.substr(
				0, id));
		} else {
			if (folder.length() - 1 > (uint)old_id) {
				subfolders.push_back(folder.substr(
					0, folder.length()));
			} else if (old_id == -1 && id == -1) {
				subfolders.push_back(folder);
			}
		}
	} while (id >= 0);
	vector < string >::iterator i;
#ifdef _WIN32
	for (i = subfolders.begin(); i != subfolders.end(); i++) {
		_mkdir((*i).c_str());
	}
#else
	for (i = subfolders.begin(); i != subfolders.end(); i++) {
		mkdir((*i).c_str(), 0755);
	}
#endif
}

#ifdef _WIN32
string getCurDir() {
    DWORD   cchCurDir;
    LPTSTR  lpszCurDir;
    TCHAR   tchBuffer[MAX_PATH + 1];
    DWORD   nSize;


    lpszCurDir = tchBuffer;
    cchCurDir = MAX_PATH;

    nSize = GetCurrentDirectory(cchCurDir, lpszCurDir);

	string dir = lpszCurDir;
	return dir;
}
#else
string getCurDir() {
	char buf[32768];
	getcwd(buf, 32768);
	string dir = buf;
	return dir;
}
#endif
