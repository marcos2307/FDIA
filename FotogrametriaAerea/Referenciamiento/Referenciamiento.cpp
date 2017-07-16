#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

struct ImageInfo 
{
	string name;
	double latitude;
	double longitude;
	double height;
	double yaw;
	double pitch;
	double roll;
};

vector < string > split(string line, char separator = ' ');
int main(int argc, char** argv)
{
	vector< ImageInfo> Iminfo;
	vector < string > word;
	string line;
	ifstream info;
	info.open("C:\\Users\\marcos\\Desktop\\Investigacion\\FiunaCitec70\\Fotobruto\\location.txt");
	if (info.is_open())
	{
		//elimina la primera linea(titulos del archivo)
		getline(info, line);
		line.clear();

		//lee cada linea del archivo
		while (getline(info, line))
		{
			word = split(line);
			cout << line << endl;

			ImageInfo Im;
			Im.name += word[0];
			Im.latitude = stod(word[1]);
			Im.longitude = stod(word[2]);
			Im.height = stod(word[3]);
			Im.yaw = stod(word[4]);
			Im.pitch = stod(word[5]);
			Im.roll = stod(word[6]);
			Iminfo.push_back(Im);
			word.clear();

		}
		info.close();
	}

	else cout << "Unable to open file";

	system("pause");
	return 0;
}
vector < string > split(string line, char separator)
{
	vector < string > word(1);
	int pos = 0;
	cout << "line.size: " << line.size() << endl;
	for (size_t i = 0; i < line.size(); i++)
	{
		if (line[i] == ' ') 
		{
			word.resize(word.size() + 1);
			++pos;
		}
		word[pos] += line[i];
	}
	return word;
}