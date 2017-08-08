#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

static const double DEG_TO_RAD = 0.017453292519943295769236907684886;
static const double EARTH_RADIUS_IN_METERS = 6372797.560856;

class ImageInfo 
{
	string name;
	double latitude;
	double longitude;
	double height;
	double yaw;
	double pitch;
	double roll;
public:
	ImageInfo(string name,
		double latitude,
		double longitude,
		double height,
		double yaw,
		double pitch,
		double roll) {
		this->name += name;
		this->latitude = latitude;
		this->longitude = longitude;
		this->height = height;
		this->yaw = yaw;
		this->pitch = pitch;
		this->roll = roll;
		
	}
	double ArcInRadians(ImageInfo to) {
		double latitudeArc = (latitude - to.latitude) * DEG_TO_RAD;
		double longitudeArc = (longitude - to.longitude) * DEG_TO_RAD;
		double latitudeH = sin(latitudeArc * 0.5);
		latitudeH *= latitudeH;
		double lontitudeH = sin(longitudeArc * 0.5);
		lontitudeH *= lontitudeH;
		double tmp = cos(latitude*DEG_TO_RAD) * cos(to.latitude*DEG_TO_RAD);
		return 2.0 * asin(sqrt(latitudeH + tmp*lontitudeH));

	}
	double DistanceInMeters(ImageInfo to) {
		return EARTH_RADIUS_IN_METERS*ArcInRadians(to);
	}

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

			ImageInfo Im(word[0], stod(word[1]), stod(word[2]), stod(word[3]), stod(word[4]), stod(word[5]), stod(word[6]));
			Iminfo.push_back(Im);
			word.clear();

		}

		info.close();
	}

	else cout << "Unable to open file" << endl;

	for (int i = 1; i < Iminfo.size(); ++i)
	{
		cout << Iminfo[i-1].DistanceInMeters(Iminfo[i]) << endl;
	}
	

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




