#include"functions.h"

vector <cv::Mat> readImages(cv::String folder, int flag)
{
	vector<cv::String> filenames;
	glob(folder, filenames);
	vector<cv::Mat> images;
	for (size_t i = 0; i < filenames.size(); ++i)
	{
		cv::Mat src = imread(filenames[i], flag);

		if (!src.data)
		{
			cerr << "no image, my friend!!!" << endl;
		}
		else
		{
			images.push_back(src);
		}
	}
	return images;
}


bool comp(DMatch& A, DMatch& B)
{
	return A.distance < B.distance;
}

void graficarEpipolares(Mat img1, Mat img2, vector<Point2f> pts1, vector<Point2f> pts2, vector<Vec3f> lines1, vector<Vec3f> lines2, Mat* imout1, Mat* imout2)
{
	cv::Mat outImg(img1.rows, img1.cols * 2, CV_8UC3);
	Rect rect1(0, 0, img1.cols, img1.rows);
	Rect rect2(img1.cols, 0, img1.cols, img1.rows);
	img1.copyTo(outImg(rect1));
	img2.copyTo(outImg(rect2));
	for (size_t i = 0; i < pts1.size(); i++)
	{

		Scalar color1(rand() % 256, rand() % 256, rand() % 256);
		Scalar color2(rand() % 256, rand() % 256, rand() % 256);
		line(outImg(rect1),
			Point(0, -lines1[i][2] / lines1[i][1]),
			Point(img1.cols, -(lines1[i][2] + lines1[i][0] * img1.cols) / lines1[i][1]),
			color1,
			THICKNESS);
		circle(outImg(rect1), pts1[i], POINT_SIZE, color1, -1, CV_AA);
		line(outImg(rect2),
			Point(0, -lines2[i][2] / lines2[i][1]),
			Point(img2.cols, -(lines2[i][2] + lines2[i][0] * img2.cols) / lines2[i][1]),
			color2,
			THICKNESS);
		circle(outImg(rect2), pts2[i], POINT_SIZE, color2, -1, CV_AA);

		*imout1 = outImg(rect1).clone();
		*imout2 = outImg(rect2).clone();
	}
	namedWindow("IMG", CV_WINDOW_NORMAL);
	imshow("IMG", outImg);
	waitKey(0);
}

void graficarMatches(Mat img1, Mat img2, vector<Point2f> pts1, vector<Point2f> pts2, int flag)
{
	cv::Mat outImg(img1.rows, img1.cols * 2, CV_8UC3);
	Rect rect1(0, 0, img1.cols, img1.rows);
	Rect rect2(img1.cols, 0, img1.cols, img1.rows);
	img1.copyTo(outImg(rect1));
	img2.copyTo(outImg(rect2));
	for (size_t i = 0; i < pts1.size(); i++)
	{
		if (flag == ONE_BY_ONE)
		{
			img1.copyTo(outImg(rect1));
			img2.copyTo(outImg(rect2));
		}

		Scalar color(rand() % 256, rand() % 256, rand() % 256);
		circle(outImg(rect1), pts1[i], POINT_SIZE, color, -1, CV_AA);
		circle(outImg(rect2), pts2[i], POINT_SIZE, color, -1, CV_AA);
		Point2f p2 = Point2f(pts2[i].x + img1.cols, pts2[i].y);
		line(outImg, pts1[i], p2, color, 5);
		if (flag == ONE_BY_ONE)
		{
			namedWindow("IMG", CV_WINDOW_NORMAL);
			imshow("IMG", outImg);
			waitKey(0);
		}
	}
	if (flag == ALL_IN_ONE)
	{
		namedWindow("IMG", CV_WINDOW_NORMAL);
		imshow("IMG", outImg);
		waitKey(0);
	}
}

void graficarLinea(Mat img, Vec3f linea, Scalar color)
{
	line(img,
		Point(0, -linea[2] / linea[1]),
		Point(img.cols, -(linea[2] + linea[0] * img.cols) / linea[1]),
		color,
		THICKNESS);
	namedWindow("Graficar Linea", CV_WINDOW_NORMAL);
	imshow("Graficar Linea", img);
	waitKey(0);
}

void generatePLY(String name, Mat pts3D, vector< Vec3b > color)
{
	ofstream f;
	f.open(name + ".ply");
	f << "ply" << endl;
	f << "format ascii 1.0" << endl;
	f << "element vertex " << pts3D.rows << endl;
	f << "property float x" << endl;
	f << "property float y" << endl;
	f << "property float z" << endl;
	f << "property uchar blue" << endl;
	f << "property uchar green" << endl;
	f << "property uchar red" << endl;
	f << "end_header" << endl;
	for (int i = 0; i < pts3D.rows; ++i)
	{
		const float* Mi = pts3D.ptr<float>(i);
		if (Mi[0] * Mi[0] + Mi[1] * Mi[1] + Mi[2] * Mi[2] < 2000)
		{
			f << Mi[0] << " " << Mi[1] << " " << Mi[2] << " " << (int)color[i][0] << " " << (int)color[i][1] << " " << (int)color[i][2] << endl;
		}
		else
		{
			f << 0 << " " << 0 << " " << 0 << " " << (int)color[i][0] << " " << (int)color[i][1] << " " << (int)color[i][2] << endl;

		}
	}

	f.close();
}

cv::Mat ypr2rm(double alpha, double betha, double gamma)
{
	Mat R = Mat(3, 3, CV_64F, 0.0);

	R.at<double>(0, 0) = cos(alpha)*cos(betha);
	R.at<double>(0, 1) = cos(alpha)*sin(betha)*sin(gamma) - sin(alpha)*cos(gamma);
	R.at<double>(0, 2) = cos(alpha)*sin(betha)*cos(gamma) + sin(alpha)*sin(gamma);
	R.at<double>(1, 0) = sin(alpha)*cos(betha);
	R.at<double>(1, 1) = sin(alpha)*sin(betha)*sin(gamma) + cos(alpha)*cos(gamma);
	R.at<double>(1, 2) = sin(alpha)*sin(betha)*cos(gamma) - cos(alpha)*sin(gamma);
	R.at<double>(2, 0) = sin(betha)*(-1);
	R.at<double>(1, 1) = cos(betha)*sin(gamma);
	R.at<double>(1, 2) = cos(betha)*cos(gamma);

	return R;
}
