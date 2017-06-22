// DisparityMap.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"

void matchTwoViews(Mat img1, Mat img2, String camFolder);

int main(int argc, const char** argv)
{
	String imagesFolder = argv[1];
	String camFolder = argv[2];
	vector < Mat> img, gray;
	img = readImages(imagesFolder, IMREAD_COLOR);
	gray = readImages(imagesFolder, IMREAD_GRAYSCALE);

	cout << "Detecting and computing keypoints using BRISK.." << endl;

	matchTwoViews(img[1], img[0], camFolder);


	system("pause");

	return 0;
}


void matchTwoViews(Mat img1, Mat img2, String camFolder)
{
	Size imgSize = Size(img1.cols, img1.rows);
	vector < Mat> descriptor(2);
	vector < vector< KeyPoint > > kp(2, vector< KeyPoint >(POINTSQUANTITY));
	vector < vector< DMatch > >  matches(1, vector< DMatch >(POINTSQUANTITY));

	Ptr< BRISK > brisk = BRISK::create(40, 3, 1.0F);
	brisk->detectAndCompute(img2, noArray(), kp[0], descriptor[0], false);
	brisk->detectAndCompute(img1, noArray(), kp[1], descriptor[1], false);

	cout << "Creating Matcher BruteForce-Hamming.." << endl;
	Ptr< DescriptorMatcher > matcher = DescriptorMatcher::create("BruteForce-Hamming");
	cout << "Matching and sorting matches.." << endl;

	matcher->match(descriptor[1], descriptor[0], matches[0], noArray());
	sort(matches[0].begin(), matches[0].end(), comp);


	cout << "Taking good matches.." << endl;
	vector< DMatch > good1;
	vector< Point2f > pts1, pts2;
	vector< Vec3b > color1, color2;
	double distAnt = INF;
	double Tprom = 0.0000;
	double delta1 = 3;
	double Nprom = 0.0;
	double delta2 = 15;
	int i = 0;
	for (vector< DMatch >::iterator it = matches[0].begin(); it != matches[0].end(); ++it)
	{
		double Dx = kp[0][it->trainIdx].pt.x - kp[1][it->queryIdx].pt.x;
		double Dy = kp[0][it->trainIdx].pt.y - kp[1][it->queryIdx].pt.y;
		double t = atan2l(Dy, Dx);
		double N2 = sqrt(Dx*Dx + Dy*Dy);

		double D1 = Tprom - t;
		double D2 = Nprom - N2;

		Tprom = (Tprom*(i)+t) / (i + 1);
		Nprom = (Nprom*(i)+N2) / (i + 1);
		//El promedio debe ser solo de los puntos correctos

		if (D1< 0) D1 = -D1;
		if (D2< 0) D2 = -D2;
		if (D1<(delta1* PI / 180) && D2<(delta2*Nprom / 100))
		{
			good1.push_back(*it);
			pts2.push_back(kp[0][it->trainIdx].pt);
			color2.push_back(img2.at< Vec3b >(kp[0][it->trainIdx].pt));
			pts1.push_back(kp[1][it->queryIdx].pt); //bf.match(query, train)
			color1.push_back(img1.at< Vec3b >(kp[1][it->queryIdx].pt));
		}
		distAnt = it->distance;
	}
	graficarMatches(img1, img2, pts1, pts2);
	Mat F;
	F = findFundamentalMat(pts1, pts2, CV_FM_RANSAC, 3.0, 0.99, noArray());
	cout << F << endl;
	//F = mi8Points(pts1, pts2);
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

	for (int i = 10; i < imag1.rows; i = i + 60)
	{
		line(imag1, Point(0, i), Point(imag1.cols - 1, i), Scalar(0, 0, 255), 3);
		line(imag2, Point(0, i), Point(imag1.cols - 1, i), Scalar(0, 0, 255), 3);
	}

	Mat out(imag1.rows, 2 * imag1.cols, CV_8UC3);
	Rect rect1 = Rect(0, 0, imag1.cols, imag1.rows);
	Rect rect2 = Rect(imag1.cols, 0, imag1.cols, imag1.rows);
	imag1.copyTo(out(rect1));
	imag2.copyTo(out(rect2));
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

