#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include "utilities.h"
#include "conversions.h"
#include "drawing.h"
#include "visualization.h"
#include "operations.h"

using namespace std;
using namespace cv;

int main()
{
	//Variables
	//Images
	const String path = "C:\\Users\\stn\\Desktop\\ventana";
	vector <Mat> images;
	vector <Mat> grayImages;
	//Key points' vectors
	vector<vector<KeyPoint>> keyPoints;
	//Descriptors' Matrices
	vector<Mat> descriptors;
	//Brisk parameters
	const int treshold = 40; //default = 30
	const int octaves = 0; //default = 3
	const float patternScale = 1.0f; //default = 1.0f
									 //Function asociated variables
									 //Drawing variables
	int radius = 10; //Circles
	int thickness = 6; //Lines
	double tolerance = 0.02; //for match filtering

	//Image loading
	getImages(images, path);
	//Conversion to Gray
	color2Gray(images, grayImages);

	//Key points detection
	Ptr<BRISK> detector = BRISK::create(treshold, octaves, patternScale);
	cout << "\nDetecting keypoints...\n";
	detector->detect(grayImages, keyPoints);
	cout << "Keypoints detected\n";

	//Descriptors computation
	cout << "\nComputing descriptors...\n";
	detector->compute(grayImages, keyPoints, descriptors);
	cout << "Finished descriptors computation\n";

	//Matching with Brute Force Matcher
	BFMatcher matcher(NORM_HAMMING);
	vector<vector<DMatch>> matches;
	cout << "\nMatching keypoints...\n";
	matcher.knnMatch(descriptors.at(0), descriptors.at(1), matches, 2);
	cout << "Matches found: " << matches.size();

	for (int i = 0; i < matches.size(); ++i)
	{
		cout << matches[i][0].distance << "\t" << matches[i][1].distance << "\n";
	}


	vector<DMatch> goodMatches;
	for (int i = 0; i < matches.size(); ++i)
	{
		const float ratio = 0.6f; // As in Lowe's paper; can be tuned
		if (matches[i][0].distance < ratio * matches[i][1].distance)
		{
			goodMatches.push_back(matches[i][0]);
		}
	}

	//Sorting matches
	cout << "\n\nSorting matches...\n";
	sort(goodMatches.begin(), goodMatches.end(), comp);

	cout << "Matching finished\n";
	cout << "\n" << goodMatches.size() << " matches found \n\n";

	//Get Keypoints
	cout << "Retrieving Keypoints...\n";
	vector<Point2f> srcPoints;
	vector<Point2f> dstPoints;

	for (int i = 0; i < goodMatches.size(); i++)
	{
		srcPoints.push_back(keyPoints[0][goodMatches[i].queryIdx].pt);
		dstPoints.push_back(keyPoints[1][goodMatches[i].trainIdx].pt);
	}
	cout << "Keypoints retrieved\n\n";

	customDrawMatches(images.at(0), srcPoints, images.at(1), dstPoints, radius, thickness);

	vector<uchar> mask;
	vector<uchar> maskF;

	double reprojError = 1;
	int maxIters = 2000;
	double confidence = 0.99;

	cout << "Finding Homography...\n";

	Mat H = findHomography(srcPoints, dstPoints, RANSAC, reprojError, mask, maxIters, confidence);

	Mat F = findFundamentalMat(srcPoints, dstPoints, CV_FM_RANSAC, reprojError, confidence, maskF);

	cout << "Homography ready\n\n";

	//Get inliers
	cout << "Getting inliers...\n";
	vector<Point2f> inliers1;
	vector<Point2f> inliers2;

	vector<Point2f> inliers1F;
	vector<Point2f> inliers2F;

	for (int i = 0; i < mask.size(); i++)
	{
		if (mask.at(i) != 0)
		{
			inliers1.push_back(srcPoints[i]);
			inliers2.push_back(dstPoints[i]);
		}
	}

	//Inliers F Matrix
	for (int i = 0; i < maskF.size(); i++)
	{
		if (maskF.at(i) != 0)
		{
			inliers1F.push_back(srcPoints[i]);
			inliers2F.push_back(dstPoints[i]);
		}
	}

	cout << "Finished inliers\nInliers Found: " << inliers1.size() << "\n";
	cout << "Finished F inliers\nF Inliers Found: " << inliers1F.size() << "\n\n";
	destroyAllWindows();

	//Draw inliers
	cout << "\nDrawing matches...";
	customDrawMatches(images.at(0), inliers1, images.at(1), inliers2, radius, thickness);
	customDrawMatches(images.at(0), inliers1F, images.at(1), inliers2F, radius, thickness);

	destroyAllWindows();

	return 0;
}