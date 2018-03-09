#include "stdafx.h"
#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define MAX_NUM_IMAGES 50


using namespace std;

vector <cv::Mat> readImages(cv::String folder, int flag);

int main(int argc, const char** argv)
{
	cv::String folder = argv[1];
	vector <cv::Mat> gray = readImages(folder, cv::IMREAD_GRAYSCALE);
	vector <cv::Mat> img = readImages(folder, cv::IMREAD_COLOR);

	//criterio de terminacion
	cv::TermCriteria criteria = cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.01);

	int M = 9, N = 6;
	vector < vector <cv::Vec3f> > objpoints;
	vector < vector <cv::Vec2f> > imgpoints;
	cv::Mat corners;
	bool ret;

	vector < cv::Vec3f> objp;
	for (int j = 0; j < N; ++j)
	{
		for (int i = 0; i < M; ++i)
		{
			objp.push_back(cv::Vec3f((float)i*2.58, (float)j*2.58, 0));
		}
	}
	//halla los corners
	cout << "finding corners..." << endl;
	for (vector< cv::Mat>::iterator it = gray.begin(); it != gray.end(); ++it)
	{
		ret = cv::findChessboardCorners(*it, cvSize(M, N), corners, cv::CALIB_CB_ADAPTIVE_THRESH);
		if (ret == true)
		{
			cout << "corners found.." << endl;
			objpoints.push_back(objp);
			cornerSubPix(*it, corners, cv::Size(12, 12), cv::Size(-1, -1), criteria);

			imgpoints.push_back(corners);
			//muestra los corners
			cout << "image.." << endl;
			cv::drawChessboardCorners(*(it - gray.begin() + img.begin()), cvSize(M, N), corners, ret);
			cv::namedWindow("img", cv::WINDOW_NORMAL);
			cv::imshow("img", *(it - gray.begin() + img.begin()));
			cv::waitKey(0);
			cv::destroyAllWindows();
		}
	}


	//calibra la camara
	cout << "calibrating camera..." << endl;
	cv::Mat K;
	cv::Mat dist;
	vector < cv::Mat > rvecs, tvecs;
	cv::calibrateCamera(objpoints, imgpoints, gray[0].size(), K, dist, rvecs, tvecs, 0, criteria);

	cv::FileStorage file("cam.xml", cv::FileStorage::WRITE, cv::String());
	file.write("K", K);
	file.write("dist", dist);
	//file.write("rvecs", rvecs);
	//file.write("tvecs", tvecs);


	cv::Mat d, u;
	d = img[12];
	cv::namedWindow("dist", cv::WINDOW_NORMAL);
	cv::namedWindow("undist", cv::WINDOW_NORMAL);
	cv::undistort(d, u, K, dist, cv::noArray());
	cv::imshow("dist", d);
	cv::waitKey(0);
	cv::imshow("undist", u);
	cv::waitKey(0);
	cv::destroyAllWindows();
	system("pause");
	return 0;
}

vector <cv::Mat> readImages(cv::String folder, int flag)
{
	vector<cv::String> filenames;
	glob(folder, filenames);
	vector<cv::Mat> images;
	for (size_t i = 0; i < filenames.size(); ++i)
	{
		cv::Mat src = imread(filenames[i], flag);

		if (!src.data)
		{
			cerr << "no image, my friend!!!" << endl;
		}
		else 
		{
			images.push_back(src);
		}
	}
	return images;
}