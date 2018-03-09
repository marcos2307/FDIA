// DisparityMap.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"

void matchTwoViews(vector <Mat> images, vector <Mat> grayImages, String camFolder);

int main(int argc, const char** argv)
{
	String imagesFolder = argv[1];
	String camFolder = argv[2];
	vector < Mat> img, gray;
	img = readImages(imagesFolder, IMREAD_COLOR);
	gray = readImages(imagesFolder, IMREAD_GRAYSCALE);

	cout << "Detecting and computing keypoints using BRISK.." << endl;

	matchTwoViews(img, gray, camFolder);

	system("pause");

	return 0;
}


void matchTwoViews(vector <Mat> images, vector <Mat> grayImages, String camFolder)
{
	Mat img1 = images[0];
	Mat img2 = images[1];
	Size imgSize = Size(img1.cols, img1.rows);
	vector < Mat> descriptor;
	vector < vector< KeyPoint > > kp(2, vector< KeyPoint >(POINTSQUANTITY));
	vector < vector< DMatch > >  matches;

	Ptr< BRISK > brisk = BRISK::create(40, 0, 1.0F);
	brisk->detect(grayImages, kp);
	brisk->compute(grayImages, kp, descriptor);

	cout << "Creating Matcher BruteForce-Hamming.." << endl;
	BFMatcher matcher(NORM_HAMMING);
	cout << "Matching and sorting matches.." << endl;
	
	matcher.knnMatch(descriptor[0], descriptor[1], matches, 2);

	cout << "Taking good matches.." << endl;
	vector< DMatch > good1;
	vector< Point2f > pts1, pts2;
	vector< Vec3b > color1, color2;
	int i;
	for (i = 0; i < matches.size(); ++i)
	{
		if (matches[i][0].distance < 0.6*matches[i][1].distance)
		{
			good1.push_back(matches[i][0]);
			pts2.push_back(kp.at(1)[matches[i][0].trainIdx].pt);
			color2.push_back(img2.at< Vec3b >(kp.at(1)[matches[i][0].trainIdx].pt));
			pts1.push_back(kp.at(0)[matches[i][0].queryIdx].pt); //bf.match(query, train)
			color1.push_back(img1.at< Vec3b >(kp.at(0)[matches[i][0].queryIdx].pt));
		}
	}
	cout << endl << good1.size() << endl;
	graficarMatches(img1, img2, pts1, pts2, ALL_IN_ONE);
	Mat F;
	vector <uchar> inliers;
	F = findFundamentalMat(pts1, pts2, CV_FM_RANSAC, 3.0, 0.99, inliers);
	cout << F << endl;
	Mat im1ep, im2ep;

	vector< Vec3f > lines1, lines2;
	computeCorrespondEpilines(pts1, 1, F, lines2);
	computeCorrespondEpilines(pts2, 2, F, lines1);
	graficarEpipolares(img1, img2, pts1, pts2, lines1, lines2, &im1ep, &im2ep);


	FileStorage f(camFolder, cv::FileStorage::READ, cv::String());
	Mat K, dist;
	f["K"] >> K;
	f["dist"] >> dist;
	cout << "K dist:" << endl << K << dist << endl;
	Mat img1u, img2u;
	undistort(img1, img1u, K, dist);
	undistort(img2, img2u, K, dist);

	namedWindow("1Undistorted", CV_WINDOW_KEEPRATIO);
	imshow("1Undistorted", img1u);
	waitKey(0);
	namedWindow("2Undistorted", CV_WINDOW_KEEPRATIO);
	imshow("2Undistorted", img2u);
	waitKey(0);

	Mat R, t;
	recoverPose(F, pts1, pts2, K, R, t, noArray());
	cout << "R t:" << endl << R << t << endl;
	Mat H1, H2;
	Mat imag1, imag2;
	stereoRectifyUncalibrated(pts1, pts2, F, imgSize, H1, H2);
	//warpPerspective(im1ep, imag1, H1, imgSize);
	warpPerspective(img1, imag1, H1, imgSize);

	//warpPerspective(im2ep, imag2, H2, imgSize);
	warpPerspective(img2, imag2, H2, imgSize);

	Mat out(imag1.rows, 2 * imag1.cols, CV_8UC3);
	Rect rect1 = Rect(0, 0, imag1.cols, imag1.rows);
	Rect rect2 = Rect(imag1.cols, 0, imag1.cols, imag1.rows);
	imag1.copyTo(out(rect1));
	imag2.copyTo(out(rect2));

	int rango = 100; // distancia maxima (en x) en pixeles entre imag1(i,j) imag2(i, j + d)...
	int wz = 9;

	Rect w1, w2;
	double n1, n2, distancia;
	int DD = 0.9;
	int im2x;
	vector <Point2i> points1, points2;
	for (int i = imag1.rows / 10; i < imag1.rows * 9 / 10; i = i+10) 
	{
		double distAnt = DD;
		for (int j = imag1.cols / 10; j < imag1.cols * 9 / 10; j = j+10)
		{
			w1 = Rect(j - wz, i - wz, 2 * wz + 1, 2 * wz + 1);
			for (int k = -rango ; k<=rango ; ++k)
			{
				w2 = Rect(j - wz + k, i - wz, 2 * wz + 1, 2 * wz + 1);
				n1 = sqrt(imag1(w1).dot(imag1(w1)));
				n2 = sqrt(imag2(w2).dot(imag2(w2)));
				distancia = imag1(w1).dot(imag2(w2))/ (n1*n2);
				if (distancia > distAnt)
				{
					distAnt = distancia;
					cout << distancia << endl;
					im2x = j - wz + k;
				}

			}
			if (distAnt > 0.9999)
			{
				points1.push_back(Point2i(j, i));
				points2.push_back(Point2i(im2x, i));
				Scalar color1(rand() % 256, rand() % 256, rand() % 256);
				circle(out(rect1), Point2i(j, i), POINT_SIZE, color1, -1, CV_AA);
				circle(out(rect2), Point2i(im2x, i), POINT_SIZE, color1, -1, CV_AA);
			}
		}
	}


	
	resize(out, out, Size(2000, 500));
	imwrite("rect.JPEG", out);
	namedWindow("rect", CV_WINDOW_KEEPRATIO);
	imshow("rect", out);
	waitKey(0);

	cvtColor(imag1, imag1, COLOR_BGR2GRAY);
	cvtColor(imag2, imag2, COLOR_BGR2GRAY);

	Ptr<StereoBM> s = StereoBM::create(512, 31);
	Mat d;
	s->compute(imag1, imag2, d);
	imwrite("disparityMap.JPG", d);
}

