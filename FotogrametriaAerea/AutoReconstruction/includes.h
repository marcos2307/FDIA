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

class Camera 
{
private:
	vector <Point3d> points = vector <Point3d>(5);
	double w;
	double h;
	Mat rCam = Mat::Mat(Size(3, 3), CV_64FC1);
	Mat tCam = Mat::Mat(Size(1, 3), CV_64FC1);
public:
	Camera(Mat rCam = Mat::eye(3,3,CV_64FC1), Mat tCam = Mat(Size(1, 3), CV_64FC1, { 0, 0, 0 }), double w = 0.04, double h = 0.03)
	{
		this->w = w;
		this->h = h;
		cout << "entro" << endl;
		this->points[0] = (Point3d(0, 0, 0));
		this->points[1] = (Point3d(w / 2, h / 2, 0.02));
		this->points[2] = (Point3d(w / 2, -h / 2, 0.02));
		this->points[3] = (Point3d(-w / 2, -h / 2, 0.02));
		this->points[4] = (Point3d(-w / 2, h / 2, 0.02));
		this->tCam = tCam.clone();
		this->rCam = rCam.clone();
	}

	vector<Point3d> getPoints(void) 
	{
		return points;
	}
	Mat getR(void)
	{
		return rCam;
	}
	Mat getT(void)
	{
		return tCam;
	}
	
	void rotoTrans(Mat R, Mat t)
	{
		rCam = R * rCam;
		tCam = tCam + t;
		Mat Temp = Mat::Mat(Size(1, 3), CV_64FC1);
		Temp = Mat(points, CV_64FC1).reshape(1);
		cout <<"Temp: "<< Temp << endl;
		cout << "Temp.Type: " << Temp.type() << " R: " << R.type() << endl;
		Temp =  (R*Temp.t()).t();
		for (int i = 0; i < 5; ++i)
		{
			Temp.row(i) = Temp.row(i) + t.t();
		}
		cout << "R*Temp: " << Temp << endl;
		cout << "posttemp" << endl;
		Temp.reshape(3).copyTo(points);
	}

};