#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <fstream>
#include <algorithm>
#include "utilities.h"
#include "drawing.h"
#include "reconstructionUtils.h"
#include "densificationUtils.h"

using namespace std;
using namespace cv;

int main()
{
	//Variables
	//Images
	const String path = "C:\\Users\\stn\\Desktop\\marcos";
	const String camPath = "C:\\Users\\stn\\Desktop\\camParam\\cam.XML";
	const String inliersPath = "C:\\Users\\stn\\Desktop\\camParam\\inliers.XML";

	vector <Mat> images;
	vector <Mat> grayImages;

	//Image loading
	getImages(images, path);
	//Conversion to Gray
	color2Gray(images, grayImages);
	

	//Camera parameters
	Mat K, dist;

	cout << "Getting cam params...\n";
	getCameraParameters(camPath, K, dist);
	cout << "Cam Params ready...\n\n";

	//Inliers y matriz F
	Mat F, inliers1, inliers2;

	cv::FileStorage f(inliersPath, cv::FileStorage::READ, cv::String());
	cout << "cargar\n";
	f["in1"] >> inliers1;
	cout << "fin inliers\n";
	f["in2"] >> inliers2;
	cout << "fin inliers\n";
	f["F"] >> F;



	cout << "imprimir\n";

	imshow("hola", images.at(0));
	waitKey(0);

	return 0;
}

