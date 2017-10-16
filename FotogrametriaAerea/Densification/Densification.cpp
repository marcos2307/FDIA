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

	/*Rect region_of_interest = Rect(3000, 3000, 1500, 1500);
	Mat image_roi = images[0](region_of_interest);

	namedWindow("Rect", WINDOW_FREERATIO);
	imshow("Rect", image_roi);
	waitKey(0);
	destroyAllWindows();*/
	

	//Camera parameters
	Mat K, dist;

	cout << "Getting cam params...\n";
	getCameraParameters(camPath, K, dist);
	cout << "Cam Params ready...\n\n";

	//Inliers y matriz F
	Mat F, inliers1, inliers2;

	cv::FileStorage f(inliersPath, cv::FileStorage::READ, cv::String());
	f["in1"] >> inliers1;
	f["in2"] >> inliers2;
	f["F"] >> F;


	//Rectificar Imagenes
	Mat H1, H2;
	stereoRectifyUncalibrated(inliers1, inliers2, F, grayImages[0].size(), H1, H2);

	Mat rGImag1, rGImag2, rImag1, rImag2;
	warpPerspective(grayImages[0], rGImag1, H1, grayImages[0].size());
	warpPerspective(grayImages[1], rGImag2, H2, grayImages[1].size());

	warpPerspective(images[0], rImag1, H1, images[0].size());
	warpPerspective(images[1], rImag2, H2, images[1].size());

	//Nuevas coordenadas de los inliers
	Mat newInliers1, newInliers2;

	perspectiveTransform(inliers1, newInliers1, H1);
	perspectiveTransform(inliers2, newInliers2, H2);

	//Dibujar inliers
	//customDrawMatches(rImag1, (vector<Point2f>)newInliers1, rImag2, (vector<Point2f>)newInliers2, "Matches", 20, 13, DRAW_ONE);

	vector<myMatch> seed, local;
	vector<myMatch> map;

	for (int i = 0; i < ((vector<Point2f>)inliers1).size(); i++)
	{
		seed.push_back(myMatch(((vector<Point2f>)inliers1)[i], ((vector<Point2f>)inliers2)[i], ZNCC(((vector<Point2f>)inliers1)[i], ((vector<Point2f>)inliers2)[i], rImag1, rImag2)));
		map.push_back(myMatch(((vector<Point2f>)inliers1)[i], ((vector<Point2f>)inliers2)[i], ZNCC(((vector<Point2f>)inliers1)[i], ((vector<Point2f>)inliers2)[i], rImag1, rImag2)));
	}

	make_heap(seed.begin(), seed.end());
	int T = 25000;
	while (seed.size() != 0)
	{
		myMatch temp(seed.front().point1, seed.front().point2, seed.front().distance);
		std::pop_heap(seed.begin(), seed.end());
		seed.pop_back();
		double t = 0.01;
		if (temp.point1.x > 10 && temp.point1.x < rImag1.cols - 10 && temp.point1.y > 10
			&& temp.point1.y < rImag1.rows && temp.point2.x > 10 && temp.point2.x < rImag1.cols - 10
			&& temp.point2.y > 10 && temp.point2.y < rImag1.rows)
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
									double x1 = (double)((int)temp.point1.x + i);
									double y1 = (double)((int)temp.point1.y + j);
									double x2 = (double)((int)temp.point2.x + k);
									double y2 = (double)((int)temp.point2.y + l);
									Point2f pt1 = Point2f(x1, y1);
									Point2f pt2 = Point2f(x2, y2);
									bool encontro = false;
									for (vector < myMatch >::iterator it = map.begin(); it != map.end(); ++it)
									{
										if (pt1 == it->point1 || pt2 == it->point2)
										{
											encontro = true;
										}
									}
									if (!encontro)
									{
										double d = ZNCC(pt1, pt2, rImag1, rImag2);
										double a = s(pt1, rImag1);
										double b = s(pt2, rImag2);
										if (a > t && b > t && d > 0.5)
										{
											local.push_back(myMatch(pt1, pt2, d));
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
			myMatch temp(local.front().point1, local.front().point2, local.front().distance);
			std::pop_heap(local.begin(), local.end());
			local.pop_back();
			bool encontro = false;
			for (vector < myMatch >::iterator it = map.begin(); it != map.end(); ++it)
			{
				if (temp.point1 == it->point1 || temp.point2 == it->point2)
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
		
	}



	cout << "map size: " << map.size() << endl;

	return 0;
}

