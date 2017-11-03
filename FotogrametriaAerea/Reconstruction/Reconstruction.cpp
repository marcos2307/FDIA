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
#include "Region.h"

#define LOWE_RATIO 0.8f

using namespace std;
using namespace cv;

void match(vector<Mat> descriptors, vector<KeyPoint> keyPoints1, vector<KeyPoint> keyPoints2, vector<DMatch> &goodMatches, vector<Point2f> &srcPoints, vector<Point2f> &dstPoints);
void detectComputeDescriptors(vector<Mat> grayImages, vector<vector<KeyPoint>> &keyPoints, vector<Mat> &descriptors, int treshold, int octaves, float patternScale);
void densification();

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

void densification()
{
	vector< miMatch > seed, local;
	vector < miMatch > map;

	for (int i = 0; i < pts1i.size(); i++)
	{
		seed.push_back(miMatch(ZNCC(pts1i[i], pts2i[i], imag1, imag2), pts1i[i], pts2i[i]));
		map.push_back(miMatch(ZNCC(pts1i[i], pts2i[i], imag1, imag2), pts1i[i], pts2i[i]));
	}

	make_heap(seed.begin(), seed.end());
	int T = 25000;
	while (seed.size() != 0)
	{
		miMatch temp(seed.front().distance, seed.front().p1, seed.front().p2);
		std::pop_heap(seed.begin(), seed.end());
		seed.pop_back();
		double t = 0.01;
		if (temp.p1.x > 10 && temp.p1.x < imag1.cols - 10 && temp.p1.y > 10
			&& temp.p1.y < imag1.rows && temp.p2.x > 10 && temp.p2.x < imag1.cols - 10
			&& temp.p2.y > 10 && temp.p2.y < imag1.rows)
		{
			for (int i = -2; i < 3; ++i)
			{
				for (int j = -2; j < 3; ++j)
				{
					if (i != 0 && j != 0)
					{
						for (int k = i - 1 < -2 ? -2 : i - 1; k < i + 1 > 2 ? 2 : i + 1; ++k)
						{
							for (int l = j - 1 < -2 ? -2 : j - 1; l < j + 1 > 2 ? 2 : j + 1; ++l)
							{
								if (k != 0 && l != 0)
								{
									double x1 = temp.p1.x + i;
									double y1 = temp.p1.y + j;
									double x2 = temp.p2.x + k;
									double y2 = temp.p2.y + l;
									Point2f pt1 = Point2f(x1, y1);
									Point2f pt2 = Point2f(x2, y2);
									bool encontro = false;
									for (vector < miMatch >::iterator it = map.begin(); it != map.end(); ++it)
									{
										if (pt1 == it->p1 || pt2 == it->p2)
										{
											encontro = true;
										}
									}
									if (!encontro)
									{
										double d = ZNCC(pt1, pt2, imag1, imag2);
										double a = s(pt1, imag1);
										double b = s(pt2, imag2);
										if (a > t && b > t && d > 0.5)
										{
											local.push_back(miMatch(d, pt1, pt2));
										}
									}

								}
							}
						}
					}
				}
			}
		}

		cout << "local.size(): " << local.size() << endl;
		make_heap(local.begin(), local.end());
		while (local.size() != 0)
		{
			miMatch temp(local.front().distance, local.front().p1, local.front().p2);
			std::pop_heap(local.begin(), local.end());
			local.pop_back();
			bool encontro = false;
			for (vector < miMatch >::iterator it = map.begin(); it != map.end(); ++it)
			{
				if (temp.p1 == it->p1 || temp.p2 == it->p2)
				{
					encontro = true;
				}
			}
			if (!encontro)
			{
				seed.push_back(temp);
				push_heap(seed.begin(), seed.end());
				map.push_back(temp);
			}
		}
		if (map.size()>T)
		{
			T = T + 25000;
			images[0].copyTo(img1);
			images[1].copyTo(img2);
			for (int i = 0; i < map.size(); ++i)
			{
				Scalar color(rand() % 256, rand() % 256, rand() % 256);
				circle(img1, map[i].p1, 10, color, 6);
				circle(img2, map[i].p2, 10, color, 6);
			}
			cout << "map size: " << map.size() << endl;
			img1.copyTo(out(rect1));
			img2.copyTo(out(rect2));
			namedWindow("rect", CV_WINDOW_KEEPRATIO);
			imshow("rect", out);
			waitKey(0);
		}
	}



	cout << "map size: " << map.size() << endl;
	imwrite("res.PNG", out);
}