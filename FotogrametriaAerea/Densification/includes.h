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

//se incluyen las constantes
#include "defines.h" 

struct ImageInfo //usada en getImageInfo
{
	string name;
	double latitude;
	double longitude;
	double height;
	double yaw;
	double pitch;
	double roll;
	
	double arcInRadians(ImageInfo to) {
		double latitudeArc = (latitude - to.latitude) * DEG_TO_RAD;
		double longitudeArc = (longitude - to.longitude) * DEG_TO_RAD;
		double latitudeH = sin(latitudeArc * 0.5);
		latitudeH *= latitudeH;
		double lontitudeH = sin(longitudeArc * 0.5);
		lontitudeH *= lontitudeH;
		double tmp = cos(latitude*DEG_TO_RAD) * cos(to.latitude*DEG_TO_RAD);
		return 2.0 * asin(sqrt(latitudeH + tmp*lontitudeH));

	}
	double distanceInMeters(ImageInfo to) {
		return EARTH_RADIUS_IN_METERS*arcInRadians(to);
	}
};

class Camera 
{
private:
	vector <Point3d> points = vector <Point3d>(5);
	double w;
	double h;
	Mat rCam = Mat::Mat(Size(3, 3), CV_64FC1);
	Mat tCam = Mat::Mat(Size(1, 3), CV_64FC1);
public:
	Camera(Mat rCam = Mat::eye(3,3,CV_64FC1), Mat tCam = Mat(Size(1, 3), CV_64FC1, { 0, 0, 0 }), double w = 4, double h = 3)
	{
		this->w = w;
		this->h = h;
		this->points[0] = (Point3d(0, 0, 0));
		this->points[1] = (Point3d(w / 2, h / 2, 2*w));
		this->points[2] = (Point3d(w / 2, -h / 2, 2*w));
		this->points[3] = (Point3d(-w / 2, -h / 2, 2*w));
		this->points[4] = (Point3d(-w / 2, h / 2, 2*w));
		this->tCam = Mat(Size(1, 3), CV_64FC1, { 0, 0, 0 });
		this->rCam = Mat::eye(3, 3, CV_64FC1);
		rotoTrans(rCam, tCam);
	}

	vector<Point3d> getPoints(void) 
	{
		return points;
	}
	Mat getR(void)
	{
		return rCam.clone();
	}
	Mat getT(void)
	{
		return tCam.clone();
	}
	
	void rotoTrans(Mat R, Mat t)
	{
		rCam = R * rCam;
		tCam = tCam + t;
		Mat Temp = Mat::Mat(Size(1, 3), CV_64FC1);
		Temp = Mat(points, CV_64FC1).reshape(1);
		Temp =  (R*Temp.t()).t();
		for (int i = 0; i < 5; ++i)
		{
			Temp.row(i) = Temp.row(i) + t.t();
			cout << Temp.row(i) << endl;
		}
		Temp.reshape(3).clone().copyTo(points);
	}

};