#include "functions.h"

using namespace std;
using namespace cv;

void reconstruct(vector <Mat> images, vector <Mat> grayImages, String camFolder);
int main(int argc, const char** argv)
{
	String imagesFolder = argv[1];
	String camFolder = argv[2];
	vector < Mat> img, gray;
	img = readImages(imagesFolder, IMREAD_COLOR);
	gray = readImages(imagesFolder, IMREAD_GRAYSCALE);

	cout << "Detecting and computing keypoints using BRISK.." << endl;

	reconstruct(img, gray, camFolder);


	system("pause");

	return 0;
}
void reconstruct(vector <Mat> images, vector <Mat> grayImages, String camFolder)
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
		if (matches[i][0].distance < 0.8*matches[i][1].distance)
		{
			good1.push_back(matches[i][0]);
			pts2.push_back(kp.at(1)[matches[i][0].trainIdx].pt);
			color2.push_back(img2.at< Vec3b >(kp.at(1)[matches[i][0].trainIdx].pt));
			pts1.push_back(kp.at(0)[matches[i][0].queryIdx].pt); //bf.match(query, train)
			color1.push_back(img1.at< Vec3b >(kp.at(0)[matches[i][0].queryIdx].pt));
		}
	}
	cout << endl << good1.size() << endl;

	Mat F;
	vector <uchar> inliers;
	F = findFundamentalMat(pts1, pts2, CV_FM_RANSAC, 4.0, 0.99, inliers);
	cout << F << endl;

	FileStorage f(camFolder, cv::FileStorage::READ, cv::String());
	Mat K, dist;
	f["K"] >> K;
	f["dist"] >> dist;
	cout << "K dist:" << endl << K << dist << endl;

	Mat R, t;
	recoverPose(F, pts1, pts2, K, R, t, noArray());
	cout << "R t:" << endl << R << t << endl;
	Mat P1, P2;
	P1 = Mat::eye(Size(4, 3), CV_64FC1);
	hconcat(R, t, P2);
	cout << "P1 P2:" << endl << P1 << endl << P2 << endl;
	Mat R1, R2, Q;
	stereoRectify(K, dist, K, dist, images[0].size(), R, t, R1, R2, P1, P2, Q);
	Mat pts4D;

	cout << "Triangulating.." << endl;
	triangulatePoints(P1, P2, pts1, pts2, pts4D);
	//cout << pts4D << endl;
	pts4D = pts4D.t();
	Mat pts3D;
	convertPointsFromHomogeneous(pts4D, pts3D);

	cout << pts3D << endl;
	cout << "Creating PLY file.." << endl;
	generatePLY("nube", pts3D, color1);

}