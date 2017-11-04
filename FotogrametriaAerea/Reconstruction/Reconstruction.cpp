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

using namespace std;
using namespace cv;

void match(vector<Mat> descriptors, vector<KeyPoint> keyPoints1, vector<KeyPoint> keyPoints2, vector<DMatch> &goodMatches, vector<Point2f> &srcPoints, vector<Point2f> &dstPoints);
void detectComputeDescriptors(vector<Mat> grayImages, vector<vector<KeyPoint>> &keyPoints, vector<Mat> &descriptors, int treshold, int octaves, float patternScale);
vector<miMatch> densification(vector<miMatch> &seed, vector<Mat> images, vector<Mat> grayImages);

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

	//cout << grayImages[0].cols << ", " << grayImages[0].rows << endl;
	//cout << grayImages[1].cols << ", " << grayImages[1].rows << endl;


	//vector<miMatch> mapP1;

	//cout << "crear map\n";
	//for (int i = 0; i < 10; i++)
	//{
	//	cout << i << endl;
	//	Point2f p1(4.4 + i, 6.4 + i);
	//	Point2f p2(4.3 + i, 6.8 + i);

	//	miMatch temp(p1, p2, ZNCC(p1, p2, grayImages));

	//	mapP1.push_back(temp);
	//}
	//cout << "fin map\n";
	//make_heap(mapP1.begin(), mapP1.end(), byPoint1);

	//Point2f p1(4.4, 6.4);
	//Point2f p2(4.3, 6.8);

	//miMatch newCandidate(p1, p2, ZNCC(p1, p2, grayImages));

	//if (!binary_search(mapP1.begin(), mapP1.end(), newCandidate, byPoint1i))
	//{
	//	//Si no existe almacenar en local
	//	cout << "NO HAY";
	//}
	//else
	//{
	//	cout << "HAY";
	//}




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

	

	vector<Vec3b> colors1, colors2;
	vector<Point2f> inliers1, inliers2;
	//Get inliers
	cout << "\nGetting inliers...\n";
	getInliers(srcPoints, dstPoints, mask, inliers1, inliers2);

	vector<miMatch> seed;

	for (int i = 0; i < inliers1.size(); i++)
	{
		miMatch temp(inliers1[i], inliers2[i],ZNCC(inliers1[i], inliers2[i],grayImages));
		seed.push_back(temp);
	}

	/*vector<miMatch> map = densification(seed, images, grayImages);

	inliers1.clear();
	inliers2.clear();

	for (int i = 0; i < map.size(); i++)
	{
		inliers1.push_back(map[i].point1);
		inliers2.push_back(map[i].point2);
	}*/
	
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

	/*cv::FileStorage file("C:\\Users\\stn\\Desktop\\camParam\\inliers.XML", cv::FileStorage::WRITE, cv::String());
	file.write("in1", inliers1);
	file.write("in2", inliers2);*/
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
	generatePLY("C:\\Users\\stn\\Desktop\\nube", pts3D, colors1);

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

vector<miMatch> densification(vector<miMatch> &seed, vector<Mat> images, vector<Mat> grayImages)
{
	//Parametros del algoritmo
	double t = 0.01;
	double minCorr = 0.5;
	double numberOfMatches = 5000;

	//Datos de entrada para el algoritmo
	vector<miMatch> mapP1 = seed;
	vector<miMatch> mapP2 = seed;
	vector<miMatch> allMatches = seed;
	make_heap(seed.begin(), seed.end(), byQuality);
	make_heap(mapP1.begin(), mapP1.end(), byPoint1);
	make_heap(mapP2.begin(), mapP2.end(), byPoint2);

	//Para graficar
	Mat result1, result2, result;
	int lastIndex = 0;
	images[0].copyTo(result1);
	images[1].copyTo(result2);

	while (seed.size() > 0)
	{
		//Tomar el mejor match
		miMatch temp(seed.front().point1, seed.front().point2, seed.front().quality);
		std::pop_heap(seed.begin(), seed.end(), byQuality);
		seed.pop_back();

		//Crear local para almacenar candidatos a Match
		vector<miMatch> local;

		//Buscar candidatos a match en el vecindario de la semilla
		//Buscar punto en la primera imagen
		//columnas
		for (int i = -2; i < 3; i++)
		{
			//filas
			for (int j = -2; j < 3; j++)
			{
				//Si no es el punto semilla
				if (i != 0 || j != 0)
				{
					//Buscar el match en la segunda imagen
					//columnas
					for (int k = (i > -2 ? -1 : 0); k < (i < 2 ? 2 : 1); k++)
					{
						//filas
						for (int l = (i > -2 ? -1 : 0); l < (i < 2 ? 2 : 1); l++)
						{
							//Si no es el match semilla
							if (k != 0 || l != 0)
							{
								Point2f point1(temp.point1.x + i, temp.point2.y + j);
								Point2f point2(temp.point2.x + i + k, temp.point2.y + j + l);

								//Verificar limites de imagen
								if (point1.x<0 || point1.x>grayImages[0].cols || point1.y<0 || point1.y>grayImages[0].rows ||
									point2.x<0 || point2.x>grayImages[1].cols || point2.y<0 || point2.y>grayImages[1].rows)
								{
									continue;
								}
								
								//Verificar textura
								if (s(point1, grayImages[0]) > t && s(point2, grayImages[1]) > t)
								{
									float quality = ZNCC(point1, point2, grayImages);
									//Verificar correlacion minima
									if (quality > minCorr)
									{
										//Crear el candidato a match
										miMatch newCandidate(point1, point2, quality);
										//Buscar si alguno de los puntos ya existe en Map
										if (!binary_search(mapP1.begin(), mapP1.end(), newCandidate, byPoint1i) && !binary_search(mapP2.begin(), mapP2.end(), newCandidate, byPoint2i))
										{
											//Si no existe almacenar en local
											local.push_back(newCandidate);
										}
									}
								}
							}
						}
					}
				}
			}
		}
		//Ordenar local segun la calidad de los matches
		make_heap(local.begin(), local.end(), byQuality);
		//cout << "bestCandidates:\n";
		while (local.size() > 0)
		{
			miMatch bestCandidate(local.front().point1, local.front().point2, local.front().quality);
			std::pop_heap(local.begin(), local.end(), byQuality);
			local.pop_back();

			//buscar si el candidato ya existe en map
			if (!binary_search(mapP1.begin(), mapP1.end(), bestCandidate, byPoint1i) && !binary_search(mapP2.begin(), mapP2.end(), bestCandidate, byPoint2i))
			{
				//bestCandidate.print();
				//Si no existe almacenar en seed y map
				seed.push_back(bestCandidate);
				mapP1.push_back(bestCandidate);
				mapP2.push_back(bestCandidate);

				//Para graficar
				allMatches.push_back(bestCandidate);

				//Reordenar map y seed
				make_heap(seed.begin(), seed.end(), byQuality);
				make_heap(mapP1.begin(), mapP1.end(), byPoint1);
				make_heap(mapP2.begin(), mapP2.end(), byPoint2);
			}
		}
		if (allMatches.size() >= numberOfMatches)
		{
			numberOfMatches *= 2;
			cout << "Matches: " << allMatches.size() << endl;

			for (lastIndex; lastIndex < allMatches.size(); lastIndex++)
			{
				Scalar color = Scalar(rand() % 256, rand() % 256, rand() % 256);
				circle(result1, allMatches[lastIndex].point1, 40, color, -1);
				circle(result2, allMatches[lastIndex].point2, 40, color, -1);
			}
			hconcat(result1, result2, result);

			namedWindow("result", WINDOW_FREERATIO);
			imshow("result", result);

			if (waitKey(10000) == 'f')
			{
				destroyAllWindows();
				break;
			}
			else
			{
				destroyAllWindows();
			}
		}
	}
	namedWindow("result", WINDOW_FREERATIO);
	imshow("result", result);
	waitKey(0);
	destroyAllWindows();
	vector<miMatch> map = mapP1;
	return map;
}