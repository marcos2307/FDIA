#include "functions.h"
#include <set>;

using namespace std;
using namespace cv;

pair<int, int> rellenar(int inicio1, int fin1, int inicio2, int fin2, int row, Mat img1, Mat img2, int ws);
pair<int, int> rellenar2(int inicio1, int fin1, int inicio2, int fin2, int row, Mat img1, Mat img2, int ws);
pair<int, int> rellenarLowe(int inicio1, int fin1, int inicio2, int fin2, int row, Mat img1, Mat img2, int ws);
double corr(int row, int x1, int x2, Mat img1, Mat img2, int halfwindowSize = 3);

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

	for (int i = 0; i < matches.size(); ++i)
	{
		if (matches[i][0].distance < 0.8*matches[i][1].distance)
		{
			good1.push_back(matches[i][0]);
			pts2.push_back(kp.at(1)[matches[i][0].trainIdx].pt);
			pts1.push_back(kp.at(0)[matches[i][0].queryIdx].pt); //bf.match(query, train)
		}
	}


	//graficarMatches(img1, img2, pts1, pts2);

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
	F = findFundamentalMat(pts1, pts2, CV_FM_RANSAC, 4.0, 0.99, inlier);
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


	Mat R, tr;
	recoverPose(E, pts1i, pts2i, K, R, tr);

	Mat H1, H2;
	Mat imag1, imag2;
	stereoRectifyUncalibrated(pts1i, pts2i, F, imgSize, H1, H2);

	warpPerspective(img1, imag1, H1, imgSize);
	warpPerspective(img2, imag2, H2, imgSize);
	
	Mat out(imag1.rows, 2 * imag1.cols, CV_8UC3);
	Rect rect1 = Rect(0, 0, imag1.cols, imag1.rows);
	Rect rect2 = Rect(imag1.cols, 0, imag1.cols, imag1.rows);

	imag1.copyTo(out(rect1));
	imag2.copyTo(out(rect2));
	namedWindow("rect", CV_WINDOW_KEEPRATIO);
	imshow("rect", out);
	waitKey(0);


	Mat i1, i2;

	cvtColor(imag1, i1, COLOR_BGR2GRAY);
	cvtColor(imag2, i2, COLOR_BGR2GRAY);

	set <int> x1, x2, x1t, x2t;
	int ws = 10;
	x1.insert(ws + 1);
	x1.insert(i1.cols - ws - 1);
	x2.insert(ws + 1);
	x2.insert(i2.cols - ws - 1);
	set<int>::iterator it1, it2, it1t, it2t;
	for (int i = 0; i < 5; ++i)
	{
		it2 = x2.begin();
		it1 = x1.begin();
		int initialSize = x1.size();
		for (int j = 0; j<initialSize - 1; ++j) 
		{
			int ini1, ini2;
			ini1 = *it1;
			ini2 = *it2;
			pair<int, int> p = rellenar2(ini1, *(++it1), ini2, *(++it2), 30, i1, i2, ws);
			x1t.insert(p.first);
			x2t.insert(p.second);
		}
		it2t = x2t.begin();
		it1t = x1t.begin();
		initialSize = x1t.size();
		for (int  j = 0; j < initialSize; ++j)
		{
			x1.insert(*it1t++);
			x2.insert(*it2t++);
		}
	}

	cout << "se encontraron " << x1.size() << " pares " << endl;
	
	it2 = x2.begin();
	it1 = x1.begin();
	int initialSize = x1.size();
	for (int j = 0; j<initialSize; ++j)
	{
		Scalar color(rand() % 256, rand() % 256, rand() % 256);
		circle(out(rect1), Point2i(*(it1++), 30), 8, color, -1, CV_AA);
		circle(out(rect2), Point2i(*(it2++), 30), 8, color, -1, CV_AA);
	}


	

	imwrite("rect.JPEG", out);
	namedWindow("rect", CV_WINDOW_KEEPRATIO);
	imshow("rect", out);
	waitKey(0);


	//cvtColor(imag1, imag1, COLOR_BGR2GRAY);
	//cvtColor(imag2, imag2, COLOR_BGR2GRAY);

	//Ptr<StereoBM> s = StereoBM::create(512, 3);
	//Mat d;
	//s->compute(imag1, imag2, d);
	//imwrite("disparityMap.JPG", d);
	
	
	//cout << "R*R1 t de la matriz Esencial:" << endl << R*R1 << tr << endl;
	//cout << "R del GPS" << ypr2rm(info[1]) * Rz << ", " << endl;
	//double dh = abs(info[1].height - info[0].height);
	//Mat t = -(R1*tr); //tr es relativa a C1 por lo que debe transformarse a las coordenadas de tierra
	//double S = 30; // dh / t.at<double>(2);
	//cout << "lat1, lon1: " << info[0].latitude << ", " << info[0].longitude << endl;
	//cout << "lat2, lon2: " << info[1].latitude << ", " << info[1].longitude << endl;
	//cout << "distancia en metros entre camaras(escala): " << S << endl;
	//Mat I1 = C1.getR().clone();
	//Mat I2 = C1.getT().clone();
	//Camera C2(I1, I2);
	//cout << " posicion de C2: " << C2.getT() << endl << " y R: " << C2.getR() << endl;
	//cout << "t: " << t << endl << "S*t: " << S*t << endl;
	//C2.rotoTrans(R, S*t);
	//cout << "C1: (R t)" << endl << C1.getR() << endl << C1.getT() << endl;
	//cout << "C2: (R t)" << endl << C2.getR() << endl << C2.getT() << endl;
	//vector<Camera> cam;
	//cam.push_back(C1);
	//cam.push_back(C2);
	//cout << "camara1con rototranslacion:" << Mat(C1.getPoints()) << endl;
	//hconcat(C2.getR(), C2.getT(), P2);
	////Mat R1, R2, Q;
	////stereoRectify(K, dist, K, dist, images[0].size(), R, t, R1, R2, P1, P2, Q);
	//Mat pts4D;
	//cout << "P1 P2:" << endl << P1 << endl << P2 << endl;
	//cout << "Triangulating.." << endl;
	//triangulatePoints(K*P1, K*P2, pts1i, pts2i, pts4D);
	////cout << pts4D << endl;
	//pts4D = pts4D.t();
	//Mat pts3D;
	//convertPointsFromHomogeneous(pts4D, pts3D);

	//cout << "Creating PLY file.." << endl;
	//generatePLY("Nube", pts3D, color1);
	//prueba
}

pair<int, int> rellenar(int inicio1, int fin1, int inicio2, int fin2, int row, Mat img1, Mat img2, int ws)
{
	if ((fin1 - inicio1) < ws || (fin2 - inicio2) < ws)
		return  pair<int, int>(inicio1, inicio2);
	//return the best match im1x-im2x across an epipolar scanline
	double distancia;
	int im1x, im2x;
	double Max, secondMax;
	Max = -1;
	secondMax = -1;
	for (int i = inicio1 + 1; i < fin1; i++)
	{
		for (int j = inicio2 + 1; j < fin2; j++)
		{
			distancia = corr(row, i, j, img1, img2, ws);
			if (distancia > Max)
			{
				secondMax = Max;
				Max = distancia;
				im1x = i;
				im2x = j;
			}
			else if (distancia > secondMax)
			{
				secondMax = distancia;
			}
		}
		/*if ((Max - secondMax) > 0.0001)
			return pair<int, int>(im1x, im2x);
		else 
			return pair<int, int>(inicio1, inicio2);*/
		return pair<int, int>(im1x, im2x);
	}
}

double corr(int row, int x1, int x2, Mat img1, Mat img2, int halfwindowSize) 
{

		int hws = halfwindowSize;
		Rect w1 = Rect(x1 - hws, row - hws, 2 * hws + 1, 2 * hws + 1);
		Rect w2 = Rect(x2 - hws, row - hws, 2 * hws + 1, 2 * hws + 1);
		double n1 = sqrt(img1(w1).dot(img1(w1)));
		double n2 = sqrt(img2(w2).dot(img2(w2)));
		if (n1 > 0.01 && n2 > 0.01)
			return  img1(w1).dot(img2(w2)) / (n1*n2);
		else
			return -1;
}

pair<int, int> rellenarLowe(int inicio1, int fin1, int inicio2, int fin2, int row, Mat img1, Mat img2, int ws) 
{
	if ((fin1 - inicio1) < ws || (fin2 - inicio2) < ws)
		return  pair<int, int>(inicio1, inicio2);
	Rect w1, w2;
	double n1, n2, distancia;
	int im1x, im2x;
	double Max, secondMax, lowe, loweMax;
	pair<int, int> p;
	Max = -1;
	secondMax = -1;
	lowe = 0;
	loweMax = 0;
	for (int i = inicio1 + 1; i < fin1; i++)
	{
		for (int j = inicio2 + 1; j < fin2; j++)
		{
			distancia = corr(row, i, j, img1, img2, ws);
			if (distancia > Max)
			{
				secondMax = Max;
				Max = distancia;
				im1x = i;
				im2x = j;
			}
			else if (distancia > secondMax  && n1 > 0.1 && n2 > 0.1)
			{
				secondMax = distancia;
			}
		}
		lowe = (Max - secondMax)/secondMax;
		if (lowe > loweMax)
		{
			loweMax = lowe;
			p = pair<int,int>(im1x, im2x);
		}
	}
	return p;
}

pair<int, int> rellenar2(int inicio1, int fin1, int inicio2, int fin2, int row, Mat img1, Mat img2, int ws)
{
	if((fin1 - inicio1 < ws) || (fin2 - inicio2 < ws))
		return  pair<int, int>(inicio1, inicio2);

	Mat im = img2(Rect(inicio2 + 1, row, (fin2 - inicio2) - 1, ws));
	Mat temp;
	Mat res;
	Point pmin, pmax;
	double min, max;
	double Max = -1;
	int x1 , x2;
	for (int i = inicio1 + 1; i < fin1 - 1; ++i) {
		temp = img1(Rect(i, row, ws, ws));
		matchTemplate(im, temp, res, TM_CCORR_NORMED);
		minMaxLoc(res, &min, &max, &pmin, &pmax);
		if (max > Max) 
		{
			x1 = i;
			x2 = pmax.x;
		}
	}
	return pair<int, int>(x1, x2);
}