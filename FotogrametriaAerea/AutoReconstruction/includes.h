#pragma once
//includes de libreria estandar
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

//includes de opencv
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;

struct ImageInfo //usada en getImageInfo
{
	string name;
	double latitude;
	double longitude;
	double height;
	double yaw;
	double pitch;
	double roll;
};
