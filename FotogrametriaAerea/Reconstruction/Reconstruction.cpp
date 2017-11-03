#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "utilities.h"
#include "drawing.h"
#include "matchingUtils.h"
#include "reconstructionUtils.h"
#include "algorithm"
#include "densificationUtils.h"

#define LOWE_RATIO 0.8f //Razon de Lowe
#define INF 2147483647  //Infinito

using namespace std;
using namespace cv;

void match(vector<Mat> descriptors, vector<KeyPoint> keyPoints1, vector<KeyPoint> keyPoints2, vector<DMatch> &goodMatches, vector<Point2f> &srcPoints, vector<Point2f> &dstPoints);
void detectComputeDescriptors(vector<Mat> grayImages, vector<vector<KeyPoint>> &keyPoints, vector<Mat> &descriptors, int treshold, int octaves, float patternScale);
void densification(vector<miMatch> &seed, vector<miMatch> &map, vector<Mat> images, vector<Mat> grayImages);

int main()
{
	//Variables
	//Images
	const String path = "C:\\Users\\stn\\Desktop\\marcos";
	const String camPath = "C:\\Users\\stn\\Desktop\\camParam\\cam.XML";

	vector <Mat> images;
	vector <Mat> grayImages;

	//Camera parameters
	Mat K, dist;

	//Key points' vectors
	vector<vector<KeyPoint>> keyPoints;

	//Descriptors' Matrices
	vector<Mat> descriptors;

	//Matched Points
	vector<DMatch> goodMatches;
	vector<Point2f> srcPoints, dstPoints;

	//Brisk parameters
	const int treshold = 30; //default = 30
	const int octaves = 5; //default = 3
	const float patternScale = 1.0f; //default = 1.0f

	//RANSAC parameters
	double reprojError = 4;
	int maxIters = 2000;
	double confidence = 0.98;

	//Image loading
	getImages(images, path);
	//Conversion to Gray
	color2Gray(images, grayImages);

	detectComputeDescriptors(grayImages, keyPoints, descriptors, treshold, octaves, patternScale);

	//Matching and filtering
	cout << "Matching Keypoints and filtering Matches...\n";
	match(descriptors, keyPoints.at(0), keyPoints.at(1), goodMatches, srcPoints, dstPoints);
	cout << "Finished matching\n";

	cout << "\nGetting Camera Parameters...\n";
	getCameraParameters(camPath, K, dist);
	cout << "Camera Parameters Ready\n";

	vector<uchar> mask;

	//Get essential Matrix
	cout << "\nGetting Essential Matrix and inliers' mask...\n";
	Mat E = findEssentialMat(srcPoints, dstPoints, K, RANSAC, confidence, reprojError, mask);
	cout << "Essential Matrix Ready\n";

	densification();

	vector<Vec3b> colors1, colors2;
	vector<Point2f> inliers1, inliers2;
	//Get inliers
	cout << "\nGetting inliers...\n";
	getInliers(srcPoints, dstPoints, mask, inliers1, inliers2);
	cout << "Inliers Ready\nGetting Inliers' colors...\n";
	
	//Get inliers' colors
	getColors(images.at(0), images.at(1), inliers1, inliers2, colors1, colors2);

	//customDrawMatches(images[0], inliers1, images[1], inliers2);

	Mat R, t;

	recoverPose(E, inliers1, inliers2, K, R, t, noArray());
	cout << "R t:" << endl << R << t << endl;
	Mat P1, P2;
	P1 = Mat::eye(Size(4,3), CV_64FC1);
	hconcat(R, t, P2);

	//Mat F = findFundamentalMat(inliers1, inliers2);

	cv::FileStorage file("C:\\Users\\stn\\Desktop\\camParam\\inliers.XML", cv::FileStorage::WRITE, cv::String());
	file.write("in1", inliers1);
	file.write("in2", inliers2);
	//file.write("F", F);

	Mat pts4D;

	cout << "Triangulating.." << endl;
	triangulatePoints(P1, P2, inliers1, inliers2, pts4D);
	//cout << pts4D << endl;
	pts4D = pts4D.t();
	Mat pts3D;
	convertPointsFromHomogeneous(pts4D, pts3D);

	//cout << pts3D << endl;
	cout << "Creating PLY file.." << endl;
	generatePLY("C:\\Users\\stn\\Desktop\\nube.ply", pts3D, colors1);

	return 0;
}

void match(vector<Mat> descriptors, vector<KeyPoint> keyPoints1, vector<KeyPoint> keyPoints2, vector<DMatch> &goodMatches, vector<Point2f> &srcPoints, vector<Point2f> &dstPoints)
{
	//Matching with Brute Force Matcher
	BFMatcher matcher(NORM_HAMMING);
	vector<vector<DMatch>> matches;
	cout << "\nMatching keypoints...\n";
	matcher.knnMatch(descriptors.at(0), descriptors.at(1), matches, 2);
	cout << "Matches found: " << matches.size() << "\n";

	vector<DMatch> matches1, matches2;

	goodMatches = loweCriteria(matches, LOWE_RATIO);

	cout << "Matching finished\n";
	cout << "\n" << goodMatches.size() << " matches found \n\n";

	//Get Keypoints
	cout << "Retrieving Keypoints...\n";

	retrieveKeyPoints(goodMatches, keyPoints1, keyPoints2, srcPoints, dstPoints);

	cout << "Keypoints retrieved\n\n";
}

void detectComputeDescriptors(vector<Mat> grayImages, vector<vector<KeyPoint>> &keyPoints, vector<Mat> &descriptors, int treshold, int octaves, float patternScale)
{
	//Key points detection
	Ptr<BRISK> detector = BRISK::create(treshold, octaves, patternScale);
	cout << "\nDetecting keypoints...\n";
	detector->detect(grayImages, keyPoints);
	cout << "Keypoints detected\n";

	//Descriptors computation
	cout << "\nComputing descriptors...\n";
	detector->compute(grayImages, keyPoints, descriptors);
	cout << "Finished descriptors computation\n";
}

double s(Point2f x, Mat M)
{
	Point2f up(x), down(x), left(x), right(x);
	up.y++;
	down.y--;
	left.x--;
	right.x++;
	double d[4] = { 0 };

	d[0] = abs(M.at<char>(up) - M.at<char>(x)) / (double)255;
	d[1] = abs(M.at<char>(down) - M.at<char>(x)) / (double)255;
	d[2] = abs(M.at<char>(left) - M.at<char>(x)) / (double)255;
	d[3] = abs(M.at<char>(right) - M.at<char>(x)) / (double)255;
	double max = 0;
	for (int i = 0; i < 4; ++i)
	{
		max = d[i] > max ? d[i] : max;
	}
	return max;
}

double ZNCC(Point2f p1, Point2f p2, Mat gray1, Mat gray2)
{
	Mat res;
	Rect r1(p1.x - 1, p1.y - 1, 5, 5);
	Rect r2(p2.x - 1, p2.y - 1, 5, 5);
	matchTemplate(gray1(r1), gray2(r2), res, TM_CCOEFF_NORMED);
	return res.at<float>(0);
}

void densification(vector<miMatch> &seed, vector<miMatch> &map, vector<Mat> images, vector<Mat> grayImages)
{
	//Datos de entrada para el algoritmo
	vector<miMatch> mapP1 = seed;
	vector<miMatch> mapP2 = seed;
	make_heap(seed.begin(), seed.end(), byQuality);
	make_heap(mapP1.begin(), mapP1.end(), byPoint1);
	make_heap(mapP2.begin(), mapP2.end(), byPoint2);

	while (seed.size() > 0)
	{
		pop_heap(seed.begin(), seed.end());

	}
}