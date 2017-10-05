#include "functions.h"
#include <list>

using namespace std;
using namespace cv;


struct miMatch
{
	double distance = INF;
	Point2f p1 = Point2f(0, 0);
	Point2f p2 = Point2f(0, 0);
	miMatch(double distance, Point2f p1, Point2f p2) 
	{
		this->distance = distance;
		this->p1 = p1;
		this->p2 = p2;
	}
	void print(void)
	{
		cout << "distance: " << distance << endl;
		cout << "p1: (" << p1.x << ", " << p1.y << ")" << endl;
		cout << "p2: (" << p2.x << ", " << p2.y << ")" << endl;
	}
};

pair<int, int> rellenar2(int inicio1, int fin1, int inicio2, int fin2, int row, Mat img1, Mat img2, int ws);
double s(Point2f x, Mat M);
double ZNCC(Point2f p1, Point2f p2, Mat gray1, Mat gray2);

inline bool operator<(const miMatch& a, const miMatch& b)
{
	return (a.distance < b.distance);
}

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
	Mat img1, img2;
	images[0].copyTo(img1);
	images[1].copyTo(img2);
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

	cout << "initial good points: " << good1.size() << endl;

	Mat F;
	vector <uchar> inlier;
	vector < Point2f > pts1i, pts2i;
	vector < double > distance;
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

	cout << "inliers: " << pts1i.size() << endl;

	Mat R, tr;
	recoverPose(E, pts1i, pts2i, K, R, tr);

	Mat H1, H2;
	Mat imag1, imag2;
	imag1 = grayImages[0];
	imag2 = grayImages[1];
	Mat out(img1.rows, 2 * img1.cols, CV_8UC3);
	Rect rect1 = Rect(0, 0, imag1.cols, imag1.rows);
	Rect rect2 = Rect(imag1.cols, 0, imag1.cols, imag1.rows);


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
		if(temp.p1.x > 10 && temp.p1.x < imag1.cols - 10 && temp.p1.y > 10 
			&& temp.p1.y < imag1.rows && temp.p2.x > 10 && temp.p2.x < imag1.cols - 10 
			&& temp.p2.y > 10 && temp.p2.y < imag1.rows)
		{ 
			for (int i = -2; i < 3; ++i) 
		{
			for (int j = -2; j < 3; ++j)
			{
				if (i != 0 && j != 0) 
				{
					for (int k = i-1 < -2? -2:i-1; k < i+1 > 2? 2: i+1; ++k)
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
								if(!encontro)
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
		if(map.size()>T)
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

pair<int, int> rellenar2(int inicio1, int fin1, int inicio2, int fin2, int row, Mat img1, Mat img2, int ws)
{
	if ((fin1 - inicio1 < ws) || (fin2 - inicio2 < ws))
		return  pair<int, int>(inicio1, inicio2);

	Mat im = img2(Rect(inicio2 + 1, row, (fin2 - inicio2) - 1, ws));
	Mat temp;
	Mat res;
	Point pmin, pmax;
	double min, max;
	double Max = -1;
	int x1, x2;
	for (int i = inicio1 + 1; i < fin1 - 1; ++i) {
		temp = img1(Rect(i, row, ws, ws));
		matchTemplate(im, temp, res, TM_CCOEFF_NORMED);
		minMaxLoc(res, &min, &max, &pmin, &pmax);
		if (max > Max)
		{
			x1 = i;
			x2 = pmax.x;
		}
	}
	return pair<int, int>(x1, x2);
}

double s(Point2f x, Mat M)
{
	Point2f up(x), down(x), left(x), right(x);
	up.y++;
	down.y--;
	left.x--;
	right.x++;
	double d[4] = { 0 };

	d[0] = abs(M.at<char>(up) - M.at<char>(x)) / (double)255;
	d[1] = abs(M.at<char>(down) - M.at<char>(x)) / (double)255;
	d[2] = abs(M.at<char>(left) - M.at<char>(x)) / (double)255;
	d[3] = abs(M.at<char>(right) - M.at<char>(x)) / (double)255;
	double max = 0;
	for (int i = 0; i < 4; ++i) 
	{
		max = d[i] > max ? d[i] : max;
	}
	return max;
}

double ZNCC(Point2f p1, Point2f p2, Mat gray1, Mat gray2) 
{
	Mat res;
	Rect r1(p1.x - 1, p1.y - 1, 3, 3);
	Rect r2(p2.x - 1, p2.y - 1, 3, 3);
	matchTemplate(gray1(r1), gray2(r2), res, TM_CCOEFF_NORMED);
	return res.at<float>(0);
}
