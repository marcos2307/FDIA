#include "functions.h"

using namespace std;
using namespace cv;

void reconstruct(vector <Mat> images, vector <Mat> grayImages, String camFile, vector <ImageInfo> info);
int main(int argc, const char** argv)
{
	String imagesFolder = argv[1];
	String camFile = argv[2];
	String infoFile = argv[3];
	vector < Mat> img, gray;  
	vector<cv::String> filenames;
	glob(imagesFolder, filenames);
	img = readImages(filenames, IMREAD_COLOR);
	gray = readImages(filenames, IMREAD_GRAYSCALE);

	vector < ImageInfo > imInfo = getImageInfo(infoFile);
	vector < ImageInfo > info;
	for (int i = 0; i < filenames.size(); ++i)
	{
		vector<string> l = split(filenames[i], '\\');
		for (int j = 0; j < imInfo.size(); ++j)
		{
			//cout << "fileName, Info.name: " << l[l.size() - 1] << ", " << imInfo[j].name << endl;
			if (l[l.size() - 1] == imInfo[j].name)
			{
				info.push_back(imInfo[j]);
				//cout << "i, j: " << i << ", " << j << endl;
			}
		}
	}
	reconstruct(img, gray, camFile, info);


	system("pause");

	return 0;
}
void reconstruct(vector <Mat> images, vector <Mat> grayImages, String camFile, vector < ImageInfo > info)
{
	cout << "Detecting and computing keypoints using BRISK.." << endl;
	Mat img1 = images[0];
	Mat img2 = images[1];
	Size imgSize = Size(img1.cols, img1.rows);
	vector < Mat> descriptor;
	vector < vector< KeyPoint > > kp(2, vector< KeyPoint >(POINTSQUANTITY));
	vector < vector< DMatch > >  matches;

	Ptr< BRISK > brisk = BRISK::create(30, 3, 1.0F);
	brisk->detect(grayImages, kp);
	brisk->compute(grayImages, kp, descriptor);

	cout << "Creating Matcher BruteForce-Hamming.." << endl;
	BFMatcher matcher(NORM_HAMMING);
	cout << "Matching and sorting matches.." << endl;

	matcher.knnMatch(descriptor[0], descriptor[1], matches, 2);

	cout << "Taking good matches.." << endl;
	vector< DMatch > good1;
	vector< Point2f > pts1, pts2;
	int i;
	for (i = 0; i < matches.size(); ++i)
	{
		if (matches[i][0].distance < 0.8*matches[i][1].distance)
		{
			good1.push_back(matches[i][0]);
			pts2.push_back(kp.at(1)[matches[i][0].trainIdx].pt);
			pts1.push_back(kp.at(0)[matches[i][0].queryIdx].pt); //bf.match(query, train)
		}
	}


	graficarMatches(img1, img2, pts1, pts2);

	cout << endl << good1.size() << endl;

	Mat F;
	vector <uchar> inlier;
	vector< Point2f > pts1i, pts2i;
	vector< Vec3b > color1, color2;
	

	FileStorage f(camFile, cv::FileStorage::READ, cv::String());
	Mat K, dist;
	f["K"] >> K;
	f["dist"] >> dist;
	cout << "K dist:" << endl << K << dist << endl;

	Mat E;
	E = findEssentialMat(pts1, pts2, K, CV_RANSAC, 0.99, 4.0, inlier);
	//F = findFundamentalMat(pts1, pts2, CV_FM_RANSAC, 4.0, 0.99, inlier);
	cout << "E:" << E << endl;

	for (int i = 0; i < inlier.size(); i++)
	{
		if (inlier.at(i) != 0)
		{
			pts2i.push_back(pts2[i]);
			color2.push_back(img2.at< Vec3b >(kp.at(1)[good1[i].trainIdx].pt));
			pts1i.push_back(pts1[i]); //bf.match(query, train)
			color1.push_back(img1.at< Vec3b >(kp.at(0)[good1[i].queryIdx].pt));
		}
	}

	graficarMatches(img1, img2, pts1i, pts2i);

	Mat P1, P2;
	Mat R1 = Mat::eye(Size(3,3), CV_64FC1);
	R1 = ypr2rm(info[0]);
	cout << "R1 de ypr2rm: " << R1 << endl;
	//rota la camara 90 grados alrededor de x ( para que el eje de la camara este en direccion de z negativa)
	Mat Rz = -Mat::eye(3, 3, CV_64FC1);
	Rz.at<double>(0, 0) = 1;
	R1 = R1*Rz;
	Mat t1 = Mat::zeros(3, 1, CV_64FC1);
	t1.at<double>(0, 2) = info[0].height;
	hconcat(R1, t1, P1);
	Camera C1;
	cout << "t1: " << t1 << endl << "P1: " << P1 << endl;
	C1.rotoTrans(R1, t1);
	
	Mat R, tr;
	recoverPose(E, pts1i, pts2i, K, R, tr);
	cout << "R*R1 t de la matriz Esencial:" << endl << R*R1 << tr << endl;
	cout << "R del GPS" << ypr2rm(info[1]) * Rz << ", " << endl;
	double dh = abs(info[1].height - info[0].height);
	Mat t = -(R1*tr); //tr es relativa a C1 por lo que debe transformarse a las coordenadas de tierra
	double S = 30; // dh / t.at<double>(2);
	cout << "lat1, lon1: " << info[0].latitude << ", " << info[0].longitude << endl;
	cout << "lat2, lon2: " << info[1].latitude << ", " << info[1].longitude << endl;
	cout << "distancia en metros entre camaras(escala): " << S << endl;
	Mat I1 = C1.getR().clone();
	Mat I2 = C1.getT().clone();
	Camera C2(I1, I2);
	cout << " posicion de C2: " << C2.getT()<< endl << " y R: " << C2.getR() << endl;
	cout << "t: " << t << endl << "S*t: " << S*t << endl;
	C2.rotoTrans(R, S*t);
	cout << "C1: (R t)" << endl << C1.getR() << endl << C1.getT() << endl;
	cout << "C2: (R t)" << endl << C2.getR() << endl << C2.getT() << endl;
	vector<Camera> cam;
	cam.push_back(C1);
	cam.push_back(C2);
	cout << "camara1con rototranslacion:" << Mat(C1.getPoints()) << endl;
	hconcat(C2.getR(), C2.getT(), P2);
	//Mat R1, R2, Q;
	//stereoRectify(K, dist, K, dist, images[0].size(), R, t, R1, R2, P1, P2, Q);
	Mat pts4D;
	cout << "P1 P2:" << endl << P1 << endl << P2 << endl;
	cout << "Triangulating.." << endl;
	triangulatePoints(K*P1, K*P2, pts1i, pts2i, pts4D);
	//cout << pts4D << endl;
	pts4D = pts4D.t();
	Mat pts3D;
	convertPointsFromHomogeneous(pts4D, pts3D);

	cout << "Creating PLY file.." << endl;
	generatePLY("Nube", pts3D, color1);
	//prueba
}