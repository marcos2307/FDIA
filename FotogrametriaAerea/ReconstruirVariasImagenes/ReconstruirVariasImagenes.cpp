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

	FileStorage f(camFolder, cv::FileStorage::READ, cv::String());
	Mat K, dist;
	f["K"] >> K;
	f["dist"] >> dist;
	cout << "K dist:" << endl << K << dist << endl;

	Size imgSize = Size(images[0].cols, images[0].rows);
	vector < Mat> descriptor;
	vector < vector< KeyPoint > > kp(500, vector< KeyPoint >(POINTSQUANTITY));

	Ptr< BRISK > brisk = BRISK::create(30, 3, 1.0F);
	brisk->detect(grayImages, kp);
	brisk->compute(grayImages, kp, descriptor);

	cout << "Creating Matcher BruteForce-Hamming.." << endl;
	BFMatcher matcher(NORM_HAMMING);
	cout << "Matching and sorting matches.." << endl;




	for (int i = 0; i < descriptor.size() - 1; ++i)
	{
		vector< Point2f > pts1, pts2;
		vector< DMatch > good1;
		vector < vector < DMatch > > M;
		pts1.clear();
		pts2.clear();
		good1.clear();
		M.clear();
		matcher.clear();
		matcher.knnMatch(descriptor[i], descriptor[i + 1], M, 2);
		cout << "Taking good matches.." << endl;
		for (int j = 0; j < M.size(); ++j)
		{
			if (M[j][0].distance < 0.8*M[j][1].distance)
			{
				good1.push_back(M[j][0]);
				pts2.push_back(kp.at(i + 1)[M[j][0].trainIdx].pt);
				pts1.push_back(kp.at(i)[M[j][0].queryIdx].pt); //bf.match(query, train)
			}
		}
		cout << "despues del for" << endl;
		//Mat F;
		vector <uchar> inlier;
		vector< Point2f > pts1i, pts2i;
		vector< Vec3b > color1, color2;
		//F = findFundamentalMat(pts1, pts2, CV_FM_RANSAC, 4.0, 0.99, inlier);

		Mat E;
		E = findEssentialMat(pts1, pts2, K, CV_RANSAC, 0.99, 3.0, inlier);

		cout << E << endl;

		for (int j = 0; j < inlier.size(); j++)
		{
			if (inlier.at(j) != 0)
			{
				pts2i.push_back(pts2[j]);
				color2.push_back(images[i + 1].at< Vec3b >(kp.at(i+1)[good1[j].trainIdx].pt));
				pts1i.push_back(pts1[j]); //bf.match(query, train)
				color1.push_back(images[i].at< Vec3b >(kp.at(i)[good1[j].queryIdx].pt));
			}
		}

		Mat R, t;
		recoverPose(E, pts1i, pts2i, K, R, t);
		cout << "R t:" << endl << R << t << endl;
		Mat P1, P2;
		P1 = Mat::eye(Size(4, 3), CV_64FC1);
		hconcat(R, t, P2);
		cout << "P1 P2:" << endl << P1 << endl << P2 << endl;
		Mat R1, R2, Q;
		stereoRectify(K, dist, K, dist, images[0].size(), R, t, R1, R2, P1, P2, Q);
		Mat pts4D;

		cout << "Triangulating.." << endl;
		triangulatePoints(P1, P2, pts1i, pts2i, pts4D);
		//cout << pts4D << endl;
		pts4D = pts4D.t();
		Mat pts3D;
		convertPointsFromHomogeneous(pts4D, pts3D);

		//cout << pts3D << endl;
		cout << "Creating PLY file.." << endl;
		generatePLY("nube" + to_string(i), pts3D, color1);

	}
}
