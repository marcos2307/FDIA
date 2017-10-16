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

	cout << "Inicio rectificacion\n\n";


	//Rectificar Imagenes
	Mat H1, H2;
	stereoRectifyUncalibrated(inliers1, inliers2, F, grayImages[0].size(), H1, H2);

	Mat rImag1, rImag2, rCImag1, rCImag2;
	warpPerspective(grayImages[0], rImag1, H1, grayImages[0].size());
	warpPerspective(grayImages[1], rImag2, H2, grayImages[1].size());

	warpPerspective(images[0], rCImag1, H1, images[0].size());
	warpPerspective(images[1], rCImag2, H2, images[1].size());

	//Nuevas coordenadas de los inliers
	Mat newInliers1, newInliers2;

	perspectiveTransform(inliers1, newInliers1, H1);
	perspectiveTransform(inliers2, newInliers2, H2);

	cout << "Fin rectificacion\n\n";

	Mat outputImg1, outputImg2, output;

	vector<myMatch> seed, local;
	vector<myMatch> mapPoint1, mapPoint2;

	for (int i = 0; i < ((vector<Point2f>)newInliers1).size(); i++)
	{
		if (((vector<Point2f>)newInliers1)[i].x > 10 && ((vector<Point2f>)newInliers1)[i].x < rImag1.cols - 10 &&
			((vector<Point2f>)newInliers1)[i].y > 10 && ((vector<Point2f>)newInliers1)[i].y < rImag1.rows - 10 &&
			((vector<Point2f>)newInliers2)[i].x > 10 && ((vector<Point2f>)newInliers2)[i].x < rImag2.cols - 10 &&
			((vector<Point2f>)newInliers2)[i].y > 10 && ((vector<Point2f>)newInliers2)[i].y < rImag2.rows - 10)
		{
			seed.push_back(myMatch(((vector<Point2f>)newInliers1)[i], ((vector<Point2f>)newInliers2)[i], ZNCC(((vector<Point2f>)newInliers1)[i], ((vector<Point2f>)newInliers2)[i], rImag1, rImag2)));
			mapPoint1.push_back(myMatch(((vector<Point2f>)newInliers1)[i], ((vector<Point2f>)newInliers2)[i], ZNCC(((vector<Point2f>)newInliers1)[i], ((vector<Point2f>)newInliers2)[i], rImag1, rImag2)));
			mapPoint2.push_back(myMatch(((vector<Point2f>)newInliers1)[i], ((vector<Point2f>)newInliers2)[i], ZNCC(((vector<Point2f>)newInliers1)[i], ((vector<Point2f>)newInliers2)[i], rImag1, rImag2)));
		}
	}

	cout << "Crear heaps\n\n";

	make_heap(seed.begin(), seed.end(), byDistance);
	make_heap(mapPoint1.begin(), mapPoint1.end(), byPoint1);
	make_heap(mapPoint2.begin(), mapPoint2.end(), byPoint2);

	cout << "Heaps Creados\n\n";

	int T = 25000;

	while (seed.size() != 0)
	{
		cout << "Inicia algoritmo\n\n";
		myMatch temp(seed.front().point1, seed.front().point2, seed.front().distance);
		std::pop_heap(seed.begin(), seed.end(), byDistance);
		seed.pop_back();
		double t = 0.01;

		if (temp.point1.x > 10 && temp.point1.x < rImag1.cols - 10 &&
			temp.point1.y > 10 && temp.point1.y < rImag1.rows - 10 &&
			temp.point2.x > 10 && temp.point2.x < rImag2.cols - 10 &&
			temp.point2.y > 10 && temp.point2.y < rImag2.rows - 10)
		{
			for (int i = -2; i < 3; ++i)
			{
				for (int j = -2; j < 3; ++j)
				{
					if (i != 0 && j != 0)
					{
						for (int k = i - 1 < -2 ? -2 : i - 1; k < i + 1 > 2 ? 2 : i + 1; ++k)
						{
							if (k != 0 && j != 0)
							{
								double x1 = ((int)temp.point1.x + i);
								double y1 = ((int)temp.point1.y + j);
								double x2 = ((int)temp.point2.x + k);
								double y2 = ((int)temp.point2.y + j);
								Point2f pt1 = Point2f(x1, y1);
								Point2f pt2 = Point2f(x2, y2);
								double d = ZNCC(pt1, pt2, rImag1, rImag2);

								myMatch tMatch(pt1, pt2, d);

								bool encontro = false;
									/*for (vector < myMatch >::iterator it = map.begin(); it != map.end(); ++it)
									{
										if (pt1 == it->point1 || pt2 == it->point2)
										{
											encontro = true;
										}
									}*/
								cout << "busqueda binaria\n\n";
								if (binary_search(mapPoint1.begin(), mapPoint1.end(), tMatch, compMatch1) || binary_search(mapPoint2.begin(), mapPoint2.end(), tMatch, compMatch2))
								{
									encontro = true;
								}

								if (!encontro)
								{
									double a = s(pt1, rImag1);
									double b = s(pt2, rImag2);
									if (a > t && b > t && d > 0.5)
									{
										local.push_back(tMatch);
									}
								}
							}
						}
					}
				}
			}
		}
		cout << "Fin de local\n\n";
		make_heap(local.begin(), local.end(), byDistance);
		while (local.size() != 0)
		{
			myMatch temp(local.front().point1, local.front().point2, local.front().distance);
			std::pop_heap(local.begin(), local.end(), byDistance);
			local.pop_back();
			bool encontro = false;
			/*for (vector < myMatch >::iterator it = map.begin(); it != map.end(); ++it)
			{
				if (temp.point1 == it->point1 || temp.point2 == it->point2)
				{
					encontro = true;
				}
			}*/

			if (binary_search(mapPoint1.begin(), mapPoint1.end(), temp, compMatch1) || binary_search(mapPoint2.begin(), mapPoint2.end(), temp, compMatch2))
			{
				encontro = true;
			}

			if (!encontro)
			{
				seed.push_back(temp);
				push_heap(seed.begin(), seed.end(), byDistance);
				mapPoint1.push_back(temp);
				mapPoint2.push_back(temp);
				push_heap(mapPoint1.begin(), mapPoint1.end(), byPoint1);
				push_heap(mapPoint2.begin(), mapPoint2.end(), byPoint2);
			}
		}

		if (mapPoint1.size()>T)
		{
			destroyAllWindows();
			T = T + 25000;
			rCImag1.copyTo(outputImg1);
			rCImag2.copyTo(outputImg2);
			for (int i = 0; i < mapPoint1.size(); ++i)
			{
				Scalar color(rand() % 256, rand() % 256, rand() % 256);
				circle(outputImg1, mapPoint1[i].point1, 10, color, 6);
				circle(outputImg2, mapPoint1[i].point2, 10, color, 6);
			}
			cout << "map size: " << mapPoint1.size() << endl;
			hconcat(outputImg1, outputImg2, output);
			namedWindow("Output", CV_WINDOW_KEEPRATIO);
			imshow("Output", output);
		}
		
	}

	waitKey(0);
	destroyAllWindows();

	cout << "map size: " << mapPoint1.size() << endl;

	imwrite("C:\\Users\\stn\\Desktop\\output.jpg", output);

	return 0;
}

